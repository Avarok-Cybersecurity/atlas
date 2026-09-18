# SPDX-License-Identifier: AGPL-3.0-only

"""Run the backbone once, keep what the head needs.

The backbone is frozen, so its hidden state for an example never changes.
Running it per training step would make the objective comparison -- the actual
experiment -- cost a full forward pass per arm per epoch, for no information.

Caching also makes the comparison FAIR in a way that matters: every arm then
trains on byte-identical features, so a difference between `ce`, `brier` and
`paired` cannot be a difference in what the backbone happened to produce.

The cache holds, per example:
  h            the hidden state at the answer position   -> the head's input
  label_idx    compact ids into a shared embedding table -> the candidates
  base_logits  the LM head's scores for those labels     -> the untuned arm
  truth        the true distribution                     -> evaluation only
  outcome      the sampled outcome                       -> training

`truth` is cached but must never reach a loss. Training sees `outcome`; only
`evaluate.py` may read `truth`. Keeping them in one file is convenient and
also the obvious way to leak the answer, so the split is stated here and the
trainer never opens the `truth` array.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from reader import Reader, resolve_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl from sim.py")
    ap.add_argument("--model", default="/home/ms/models/qwen3-0.6b")
    ap.add_argument("--out", required=True, help="npz destination")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(a.data)]
    reader = Reader(a.model, device=a.device)
    d = reader.hidden_size

    # A compact vocabulary over every label token any example uses, so the
    # embedding table carried alongside is a handful of rows rather than the
    # model's whole 150k x 1024 matrix.
    token_of_label: dict[str, int] = {}
    for r in rows:
        ls = resolve_labels(reader.tok, r["labels"])
        for lab, tid in zip(r["labels"], ls.ids):
            if lab in token_of_label and token_of_label[lab] != tid:
                raise RuntimeError(f"label {lab!r} resolved to two ids")
            token_of_label[lab] = tid
    labels_sorted = sorted(token_of_label)
    compact = {lab: i for i, lab in enumerate(labels_sorted)}
    tok_ids = [token_of_label[lab] for lab in labels_sorted]

    emb_table = (
        reader.model.get_input_embeddings().weight[torch.tensor(tok_ids, device=a.device)]
        .float()
        .detach()
        .cpu()
        .numpy()
    )

    kinds = sorted({r["kind"] for r in rows})
    kind_of = {k: i for i, k in enumerate(kinds)}
    kmax = max(len(r["labels"]) for r in rows)
    n = len(rows)

    H = np.zeros((n, d), dtype=np.float32)
    BL = np.full((n, kmax), -1e30, dtype=np.float32)
    LI = np.zeros((n, kmax), dtype=np.int32)
    MK = np.zeros((n, kmax), dtype=bool)
    TR = np.zeros((n, kmax), dtype=np.float32)
    OC = np.zeros(n, dtype=np.int64)
    KD = np.zeros(n, dtype=np.int32)
    OR = np.zeros(n, dtype=bool)

    done = 0
    for s in range(0, n, a.batch):
        chunk = rows[s : s + a.batch]
        label_logits, hidden = reader.read_batch(chunk, want_hidden=True)
        H[s : s + len(chunk)] = hidden.cpu().numpy()
        for j, r in enumerate(chunk):
            i = s + j
            k = len(r["labels"])
            BL[i, :k] = label_logits[j].cpu().numpy()
            LI[i, :k] = [compact[lab] for lab in r["labels"]]
            MK[i, :k] = True
            TR[i, :k] = r["truth"]
            OC[i] = r["outcome"]
            KD[i] = kind_of[r["kind"]]
            # Only `score` has an order over its labels; saying so lets the
            # head's ordinal term apply where it is meaningful and nowhere else.
            OR[i] = r["kind"] == "score"
        done += len(chunk)
        if done % (a.batch * 10) == 0 or done == n:
            print(f"  cached {done}/{n}", flush=True)

    np.savez_compressed(
        a.out,
        H=H, base_logits=BL, label_idx=LI, mask=MK, truth=TR,
        outcome=OC, kind=KD, ordered=OR,
        emb=emb_table, labels=np.array(labels_sorted), kinds=np.array(kinds),
    )
    print(f"wrote {a.out}: n={n} d={d} kmax={kmax} labels={labels_sorted}")


if __name__ == "__main__":
    main()
