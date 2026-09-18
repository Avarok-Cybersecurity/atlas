# SPDX-License-Identifier: AGPL-3.0-only

"""Encode each candidate as its own sequence, and keep the last hidden state.

THE SHAPE, AND WHY IT IS THE WHOLE POINT
----------------------------------------
v1 builds one prompt that lists every option and reads one hidden state from
it. This builds K prompts, each containing the state, the question and exactly
ONE candidate:

    State:
    <observation>
    Question: <question>
    Candidate: <this candidate>
    Decision:

Nothing in a candidate's prompt mentions the others, so nothing about its
encoding can depend on where it sat in a list. The cache is therefore
[N, K, D] instead of [N, D], and -- this is the useful consequence --
permuting a question's options is EXACTLY permuting the K axis of the cache.
Invariance becomes testable without re-running the backbone, and the test is
exact rather than statistical.

The price is K backbone passes per question. Every candidate shares the state
and question tokens, so a serving implementation would prefill that prefix once
and branch per candidate; SemIf measures 8.6x from precisely that reuse. Here
the passes are simply batched, since the cost is paid once and cached.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from reader import Reader

LEVEL_TEXT = {
    "none": "no realistic chance of collision",
    "low": "a small chance of collision",
    "medium": "a moderate chance of collision",
    "high": "a large chance of collision",
    "certain": "collision is essentially certain",
}


def candidate_text(ex: dict, label: str) -> str:
    kind = ex["kind"]
    if kind == "score":
        return LEVEL_TEXT[label]
    if kind == "choice":
        return f"moving {label} is the safest available move"
    if kind == "wall":
        return "that cell is a wall" if label == "yes" else "that cell is open"
    return "the move collides with a wall" if label == "yes" else "the move does not collide"


def state_text(ex: dict) -> str:
    s = (
        "A gridworld observation, 5x5, centred on the agent.\n"
        "'#' is a wall, '.' is open, 'A' is the agent.\n"
    )
    if ex["kind"] != "wall":
        s += (
            f"The agent's actuator follows the requested direction with probability "
            f"{ex['rho']:.2f}; otherwise it moves in a uniformly random one of the four "
            f"directions.\n"
        )
    return s + "\n" + ex["view"]


def prompts_for(ex: dict) -> list[str]:
    head = f"State:\n{state_text(ex)}\nQuestion: {ex['question']}\n"
    return [f"{head}Candidate: {candidate_text(ex, l)}\nDecision:" for l in ex["labels"]]


class RawReader(Reader):
    """Reader variant that reads a hidden state from arbitrary prompts.

    `Reader.read_batch` builds its own option-listing prompt, which is exactly
    what this design removes, so the prompt construction is replaced rather
    than reused.
    """

    @torch.no_grad()
    def hidden_for(self, prompts: list[str]) -> torch.Tensor:
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        out = self.model(**enc, output_hidden_states=True)
        # Left padding puts the final real token at -1 for every row.
        return out.hidden_states[-1][:, -1, :].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="/home/ms/models/qwen3-0.6b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64, help="candidate sequences per forward")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    rows = [json.loads(x) for x in open(a.data)]
    reader = RawReader(a.model, device=a.device)
    d = reader.hidden_size
    kmax = max(len(r["labels"]) for r in rows)
    n = len(rows)

    H = np.zeros((n, kmax, d), dtype=np.float32)
    MK = np.zeros((n, kmax), dtype=bool)
    TR = np.zeros((n, kmax), dtype=np.float32)
    OC = np.zeros(n, dtype=np.int64)
    kinds = sorted({r["kind"] for r in rows})
    KD = np.zeros(n, dtype=np.int32)

    # Flatten every (example, candidate) pair so batches are full even when
    # questions have different K.
    flat = [(i, j, p) for i, r in enumerate(rows) for j, p in enumerate(prompts_for(r))]
    for s in range(0, len(flat), a.batch):
        chunk = flat[s : s + a.batch]
        hs = reader.hidden_for([p for _, _, p in chunk]).cpu().numpy()
        for (i, j, _), h in zip(chunk, hs):
            H[i, j] = h
        if (s // a.batch) % 20 == 0:
            print(f"  {s}/{len(flat)} candidate sequences", flush=True)

    for i, r in enumerate(rows):
        k = len(r["labels"])
        MK[i, :k] = True
        TR[i, :k] = r["truth"]
        OC[i] = r["outcome"]
        KD[i] = kinds.index(r["kind"])

    np.savez_compressed(
        a.out, H=H, mask=MK, truth=TR, outcome=OC, kind=KD, kinds=np.array(kinds)
    )
    print(f"wrote {a.out}: n={n} kmax={kmax} d={d} ({len(flat)} candidate sequences)")


if __name__ == "__main__":
    main()
