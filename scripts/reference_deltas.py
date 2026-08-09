#!/usr/bin/env python
"""Emit reference LoRA deltas for the Atlas offline parity test (M0 exit gate).

For every (layer, module) pair in a PEFT adapter dir, loads lora_A [r, in] / lora_B [out, r],
reads r / lora_alpha / use_rslora from adapter_config.json (per-adapter, NEVER defaulted:
missing fields are fatal), computes

    scaling = lora_alpha / r            (use_rslora = false)
    scaling = lora_alpha / sqrt(r)      (use_rslora = true)
    delta   = scaling * (B.float() @ A.float())        # [out, in], fp32

and writes one safetensors file the Rust offline-parity test loads and compares against its own
loaded-A/B/scale product (tolerance <= 1e-2 relative error, elementwise on the fp32 delta).

Output keys (canonical module path = saved PEFT key with the "base_model.model." wrapper
stripped, e.g. "model.language_model.layers.3.self_attn.k_proj"):
    {path}.delta     fp32 [out, in]
    {path}.scaling   fp32 [1]
File-level metadata (strings): r, lora_alpha, use_rslora, scaling, num_modules.

Run:
  /home/ms/nemotron-diffusion-playground/.venv/bin/python scripts/reference_deltas.py \
      --adapter-dir test_data/lora-holo-tiny
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PEFT_WRAPPER_PREFIX = "base_model.model."
A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"


def canonical_path(key: str, suffix: str) -> str:
    assert key.endswith(suffix), key
    stem = key[: -len(suffix)]
    if stem.startswith(PEFT_WRAPPER_PREFIX):
        stem = stem[len(PEFT_WRAPPER_PREFIX):]
    return stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("/home/ms/atlas/.claude/worktrees/lora-mvp-e0877873/test_data/lora-holo-tiny"),
    )
    ap.add_argument("--out", type=Path, default=None,
                    help="output file (default {adapter-dir}/reference_deltas.safetensors)")
    args = ap.parse_args()
    out_path = args.out or (args.adapter_dir / "reference_deltas.safetensors")

    cfg = json.loads((args.adapter_dir / "adapter_config.json").read_text())
    # Scaling inputs are read per adapter and NEVER defaulted (locked decision).
    for field in ("r", "lora_alpha", "use_rslora"):
        if field not in cfg:
            raise SystemExit(f"adapter_config.json missing required field {field!r} "
                             f"(older PEFT may omit use_rslora; refuse rather than default)")
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise SystemExit("rank_pattern/alpha_pattern unsupported by the v0 oracle")
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    use_rslora = bool(cfg["use_rslora"])
    scaling = alpha / math.sqrt(r) if use_rslora else alpha / r

    tensors = load_file(str(args.adapter_dir / "adapter_model.safetensors"))
    a_keys = sorted(k for k in tensors if k.endswith(A_SUFFIX))
    b_keys = sorted(k for k in tensors if k.endswith(B_SUFFIX))
    stray = set(tensors) - set(a_keys) - set(b_keys)
    if stray:
        raise SystemExit(f"unrecognized adapter tensors (bidirectional-audit spirit): {stray}")
    a_by_path = {canonical_path(k, A_SUFFIX): tensors[k] for k in a_keys}
    b_by_path = {canonical_path(k, B_SUFFIX): tensors[k] for k in b_keys}
    if a_by_path.keys() != b_by_path.keys():
        raise SystemExit(f"unpaired A/B: {a_by_path.keys() ^ b_by_path.keys()}")

    out: dict[str, torch.Tensor] = {}
    print(f"r={r} alpha={alpha} use_rslora={use_rslora} -> scaling={scaling}")
    for path in sorted(a_by_path):
        a = a_by_path[path]          # [r, in]
        b = b_by_path[path]          # [out, r]
        assert a.shape[0] == r and b.shape[1] == r, (path, a.shape, b.shape)
        delta = scaling * (b.float() @ a.float())   # [out, in] fp32
        out[f"{path}.delta"] = delta.contiguous()
        out[f"{path}.scaling"] = torch.tensor([scaling], dtype=torch.float32)
        print(f"  {path}: A{tuple(a.shape)} {a.dtype} x B{tuple(b.shape)} -> "
              f"delta{tuple(delta.shape)} |delta|max={delta.abs().max():.5f}")

    metadata = {
        "r": str(r),
        "lora_alpha": repr(alpha),
        "use_rslora": str(use_rslora).lower(),
        "scaling": repr(scaling),
        "num_modules": str(len(a_by_path)),
    }
    save_file(out, str(out_path), metadata=metadata)
    print(f"wrote {len(out)} tensors -> {out_path}")
    return None


if __name__ == "__main__":
    sys.exit(main())
