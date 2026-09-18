# SPDX-License-Identifier: AGPL-3.0-only

"""Run the released NanoJev checkpoint on the simulator.

`C-Tianyu/NanoJev` is Qwen3-0.6B with decision heads, and it is the closest
published thing to what this testbed builds. Two details from its config make
it the interesting one to measure:

  * It was trained on GRID NAVIGATION (`private_navigation_v3/coords_multi`),
    so unlike the DeBERTa encoder -- which scored at chance on the perception
    control because banking tickets are nothing like ASCII grids -- this is
    roughly in domain.
  * Its released objective is `teacher`: distillation from a teacher
    distribution, not the RLCD proper-scoring arms. The RLCD comparison lives
    in their docs as a separate experiment. So this measures a distilled
    decision model, which is worth stating before reading any calibration
    number off it.

ARCHITECTURE, AND WHY IT MATTERS HERE
-------------------------------------
NanoJev encodes each candidate as its OWN sequence -- prefix + "Candidate:
<text>\\nDecision:" -- runs the backbone over all of them, and scores each
final hidden state with a scalar head. For `choice` a set-attention layer then
mixes the candidates, under a key-padding mask.

That makes candidate scores permutation-invariant by construction: no candidate
is ever "the one listed first", because they never appear in a list. It is a
structural fix for the order sensitivity measured in
`test_order_sensitivity.py`, where our shared-hidden-state head inherits the
prompt's option ordering and flips 78% of answers. The price is K forward
passes per question instead of one.

Booleans use a single leaf and emit logits [0, z], so the distribution is over
(false, true) -- which is this simulator's label order too.

Licence: the NanoJev repository is MIT and the dataset records carry CC0-1.0;
the HF weights repo declares no licence tag.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

LEVEL_DESCRIPTIONS = {
    "none": "no realistic chance of collision",
    "low": "a small chance of collision",
    "medium": "a moderate chance of collision",
    "high": "a large chance of collision",
    "certain": "collision is essentially certain",
}


def load_decision_model_class(repo: str):
    """DecisionModel lives in a training script, not an importable package."""
    path = Path(repo) / "scripts" / "train_toy_decisions.py"
    spec = importlib.util.spec_from_file_location("nj_train", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nj_train"] = mod
    spec.loader.exec_module(mod)
    return mod.DecisionModel


def _coords_from_view(view: str) -> dict:
    """Recover walls and the agent position from the rendered 5x5 window.

    The render is the ground truth here, so this is lossless: '#' is a wall and
    'A' marks the agent, which the simulator always places at the centre.
    """
    rows = view.splitlines()
    walls = [[r, c] for r, line in enumerate(rows) for c, ch in enumerate(line) if ch == "#"]
    pos = next(
        ([r, c] for r, line in enumerate(rows) for c, ch in enumerate(line) if ch == "A"),
        [len(rows) // 2, len(rows[0]) // 2],
    )
    return {"game": "grid_navigation", "size": len(rows), "walls": walls, "position": pos}


def build_state(ex: dict, fmt: str = "ascii") -> str:
    """The observation, in one of two encodings.

    `ascii` matches what every other arm is shown, which is what makes the
    cross-model comparison fair. `coords` matches the shape NanoJev was trained
    on (environment_state with walls and position as integer pairs). Running
    both separates "cannot read this format" from "cannot do this task" -- and
    only the second would be a statement about the model.
    """
    if fmt == "coords":
        s = (
            "Grid navigation state, as coordinates. Rows and columns are "
            "0-indexed from the top left.\n"
            f"{json.dumps(_coords_from_view(ex['view']), ensure_ascii=False)}\n"
        )
    else:
        s = (
            "A gridworld observation, 5x5, centred on the agent.\n"
            "'#' is a wall, '.' is open, 'A' is the agent.\n"
        )
    if ex["kind"] != "wall":
        s += (
            f"The agent's actuator is unreliable: it follows the requested direction with "
            f"probability {ex['rho']:.2f}; otherwise it moves in a uniformly random one of the "
            f"four directions (which may coincidentally be the requested one).\n"
        )
    if fmt == "coords":
        return s
    return s + "\n" + ex["view"]


def build_leaves(ex: dict, tok, max_length: int, fmt: str = "ascii"):
    """Reproduce NanoJev's own example construction, verbatim in shape.

    Deviating here would measure prompt formatting rather than the model, so
    the segment strings match `load_examples` in their trainer exactly.
    """
    kind = ex["kind"]
    state = build_state(ex, fmt)
    if kind in ("noul", "wall"):
        typ = "boolean"
        texts = ["The proposition is true."]
        crit = (
            {"false": "that cell is open", "true": "that cell is a wall"}
            if kind == "wall"
            else {
                "false": "the move does not collide",
                "true": "the move ends in a collision with a wall",
            }
        )
    elif kind == "choice":
        typ = "choice"
        crit = {d: f"moving {d} is the safest available move" for d in ex["labels"]}
        texts = [f"{k}: {crit[k]}" for k in ex["labels"]]
    else:
        typ = "score"
        crit = None
        texts = [LEVEL_DESCRIPTIONS[x] for x in ex["labels"]]

    segments = [f"State:\n{state}\n", f"Question type: {typ}\nQuestion:\n{ex['question']}\n"]
    if typ == "boolean":
        for key, label in [("false", "False"), ("true", "True")]:
            segments[1] += f"{label} criterion: {crit[key]}\n"

    prefix = sum([tok.encode(t, add_special_tokens=False) for t in segments], [])
    leaves = [
        prefix + tok.encode(f"Candidate:\n{t}\nDecision:", add_special_tokens=False)
        + [tok.eos_token_id]
        for t in texts
    ]
    if max(map(len, leaves)) > max_length:
        raise ValueError(f"leaf exceeds max_length {max_length}; refusing to truncate silently")
    n_out = 2 if typ == "boolean" else len(texts)
    return {"type": typ, "leaf_tokens": leaves, "candidate_ids": [str(i) for i in range(n_out)]}


def main():
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="/home/ms/models/nanojev")
    ap.add_argument("--repo", required=True, help="clone of TianyuCodings/NanoJev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--state-format", default="ascii", choices=["ascii", "coords"],
                    help="ascii matches the other arms; coords matches NanoJev's training shape")
    a = ap.parse_args()

    root = Path(a.ckpt)
    run_config = json.loads((root / "config.json").read_text())
    max_length = int(run_config.get("max_length", 512))

    tok = AutoTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    body_config = AutoConfig.from_pretrained(str(root / "backbone_config"), local_files_only=True)
    body_config.use_cache = False
    body = AutoModel.from_config(body_config, attn_implementation="sdpa").float()

    DecisionModel = load_decision_model_class(a.repo)
    model = DecisionModel(body, run_config["set_head"])
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(root / "best.safetensors"), device="cpu"), strict=True)
    model.to(device=a.device, dtype=torch.float32).eval()
    print(f"loaded {run_config['model']} set_head={run_config['set_head']} "
          f"objective={run_config.get('objective')} state_format={a.state_format}")

    rows = [json.loads(x) for x in open(a.data)]
    t0 = time.time()
    with open(a.out, "w") as fh, torch.no_grad():
        for i, ex in enumerate(rows):
            item = build_leaves(ex, tok, max_length, a.state_format)
            logits, valid = model([item], tok.pad_token_id)
            k = len(ex["labels"])
            p = logits[0, :k].float().softmax(-1).tolist()
            fh.write(json.dumps({"idx": i, "kind": ex["kind"], "probs": p}) + "\n")
            if (i + 1) % 40 == 0:
                print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    dt = time.time() - t0
    print(f"wrote {a.out}: {len(rows)} decisions in {dt:.1f}s ({dt/len(rows)*1000:.0f} ms/decision)")


if __name__ == "__main__":
    main()
