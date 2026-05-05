#!/usr/bin/env python3
"""
Gemma-4 HuggingFace reference dump for Atlas divergence diagnosis.

Purpose
-------
Atlas-Spark's Gemma-4-31B (NVFP4) produces degenerate output starting around
token 3 on the Creative haiku prompt (e.g. "Crystals a a a a a ..."). Root
cause remains unknown after multiple diagnostic passes. This script produces
a reference forward pass from HuggingFace transformers (BF16) for the same
prompt so per-layer hidden states and next-token logits can be diff'd against
Atlas to isolate the first divergent layer.

How to run
----------
    # GPU (preferred, ~60 GB VRAM for 27B BF16):
    python3 tests/gemma4_hf_diff.py --model google/gemma-3-27b-it --device cuda

    # CPU fallback (very slow, only if GPUs busy):
    python3 tests/gemma4_hf_diff.py --model google/gemma-3-27b-it --device cpu

Arguments:
    --model    HF model id. Default: google/gemma-3-27b-it. Alt: google/gemma-2-27b-it.
               nvidia NVFP4 checkpoints are NOT directly BF16-loadable; use a
               Gemma-3/Gemma-2 BF16 checkpoint as reference (same family).
    --prompt   Default: "Write a haiku about the ocean."
    --output   JSON path. Default: /workspace/atlas/tests/gemma4_hf_reference.json
    --device   cpu | cuda. Default: cpu (safe when GPUs are busy; warns slow).
    --dtype    Default: bfloat16.

How to diff against Atlas
-------------------------
Atlas has a diagnostic env var `ATLAS_DIAG_GEMMA4_HIDDEN=1` that logs per-layer
hidden-state norms during decode. Run Atlas with that env var on the same
prompt, then compare layer-by-layer norms / first-8 values / abs-max indices
against the JSON this script writes.

Relevant Atlas files:
    /workspace/atlas/crates/spark-model/src/layers/qwen3_attention/trait_impl.rs
        - decode forward + `gemma4_diag_enabled()` instrumentation
    /workspace/atlas/crates/spark-model/src/weight_loader/gemma4.rs
        - Gemma-4 weight loading (NVFP4 dequant, scale handling)

CAVEAT
------
Gemma-3 and Gemma-4 share an architecture family but differ in details
(rotary scheme, norm placement, SWA pattern, tokenizer vocabulary). Small
divergences at matching layers are expected; a LARGE mismatch at layer N that
compounds after N is the bug-candidate signal. Use this reference as a sanity
anchor for shape/scale, not a bit-exact oracle.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-3-27b-it",
                    help="HF model id (default: google/gemma-3-27b-it). "
                         "Alt: google/gemma-2-27b-it.")
    ap.add_argument("--prompt", default="Write a haiku about the ocean.")
    ap.add_argument("--output", default="/workspace/atlas/tests/gemma4_hf_reference.json")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Default cpu (slow) since GPUs are often busy. Use cuda if free.")
    ap.add_argument("--dtype", default="bfloat16",
                    help="Torch dtype for model weights (default: bfloat16).")
    return ap.parse_args()


def resolve_dtype(name: str):
    import torch
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {list(mapping)}.")
    return mapping[key]


def main() -> int:
    args = parse_args()

    # Import heavy deps lazily so --help works without them installed.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = resolve_dtype(args.dtype)

    if args.device == "cpu":
        warnings.warn(
            "Running on CPU. A 27B BF16 forward pass on CPU will take many "
            "minutes to tens of minutes. Prefer --device cuda when a GPU is free.",
            RuntimeWarning,
        )
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: --device cuda requested but torch.cuda.is_available() is False.",
                  file=sys.stderr)
            return 2

    print(f"[info] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"[info] Loading model: {args.model} (dtype={args.dtype}, device={args.device})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
    )
    model.eval()

    # Apply chat template to match how Atlas serves the prompt.
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"[info] Templated prompt:\n{templated}\n")

    enc = tokenizer(templated, return_tensors="pt")
    input_ids = enc["input_ids"].to(args.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)

    token_ids_list = input_ids[0].tolist()
    token_strs = tokenizer.convert_ids_to_tokens(token_ids_list)

    print(f"[info] input_ids length: {len(token_ids_list)}")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    # outputs.hidden_states: tuple len = num_layers + 1 (embedding + each layer out)
    hidden_states = outputs.hidden_states
    num_layer_outputs = len(hidden_states)
    num_layers = num_layer_outputs - 1
    print(f"[info] Captured {num_layer_outputs} hidden-state tensors ({num_layers} transformer layers + embedding).")

    per_layer = []
    for layer_idx, hs in enumerate(hidden_states):
        # hs shape: [batch, seq_len, hidden_size]
        last_tok = hs[0, -1, :].to(torch.float32).cpu()
        norm_val = float(torch.linalg.vector_norm(last_tok).item())
        first_8 = [float(x) for x in last_tok[:8].tolist()]
        abs_vals = last_tok.abs()
        abs_max_idx = int(torch.argmax(abs_vals).item())
        abs_max = float(abs_vals[abs_max_idx].item())
        per_layer.append({
            "layer_idx": layer_idx,  # 0 = embedding output, 1..N = transformer layer N outputs
            "norm": norm_val,
            "first_8": first_8,
            "abs_max": abs_max,
            "abs_max_idx": abs_max_idx,
        })

    # Next-token logits: final layer logits at last position.
    logits = outputs.logits[0, -1, :].to(torch.float32).cpu()
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=10)
    top10 = []
    for p, tid in zip(top_probs.tolist(), top_ids.tolist()):
        top10.append({
            "token_id": int(tid),
            "token": tokenizer.decode([int(tid)]),
            "token_repr": tokenizer.convert_ids_to_tokens([int(tid)])[0],
            "prob": float(p),
            "logit": float(logits[int(tid)].item()),
        })

    result = {
        "model": args.model,
        "dtype": args.dtype,
        "device": args.device,
        "prompt": args.prompt,
        "templated_prompt": templated,
        "num_layers": num_layers,
        "num_hidden_states_captured": num_layer_outputs,
        "tokenization": {
            "input_ids": token_ids_list,
            "tokens": token_strs,
        },
        "per_layer": per_layer,
        "next_token_logits_top10": top10,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"[info] Wrote reference dump -> {out_path}")

    # Summary: per-layer norms, one per line.
    print("\n=== Per-layer last-token hidden-state L2 norms ===")
    print("layer_idx  norm            abs_max     abs_max_idx")
    for row in per_layer:
        print(f"{row['layer_idx']:>9d}  {row['norm']:>14.6f}  {row['abs_max']:>10.4f}  {row['abs_max_idx']:>11d}")

    print("\n=== Top-10 next-token predictions ===")
    for i, tok in enumerate(top10):
        print(f"{i+1:>2d}. id={tok['token_id']:<8d} prob={tok['prob']:.5f}  logit={tok['logit']:+.4f}  {tok['token_repr']!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
