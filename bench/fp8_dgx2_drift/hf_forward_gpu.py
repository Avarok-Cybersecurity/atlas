#!/usr/bin/env python3
"""HF forward on dgx2 GPU. Runs ONE snapshot at a time (CLI arg: bf16|fp8).

Loads the model directly onto the GB10 GPU (BF16), registers per-layer hooks,
forwards the 10382-token prompt from /tmp/atlas_tokens_dgx2.json, dumps
per-layer LAST-token hidden states to /workspace/atlas-dumps/fp8native_dgx2/.

  bf16  -> hf_bf16_L{0..39}.bin  (uses Qwen3.6-35B-A3B BF16 snapshot, the unquant ref)
  fp8   -> hf_fp8dq_L{0..39}.bin (uses FP8-dequanted BF16 snapshot, the ceiling ref)
"""
from __future__ import annotations

import gc
import json
import pathlib
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

BF16_SNAP = "/workspace/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
FP8DQ_SNAP = "/workspace/.cache/huggingface/Qwen3.6-35B-A3B-FP8-dequanted-BF16"
TOKENS_PATH = pathlib.Path("/tmp/atlas_tokens_dgx2.json")
OUT_DIR = pathlib.Path("/workspace/atlas-dumps/fp8native_dgx2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_f32(path: pathlib.Path, arr: np.ndarray, label: str):
    arr_f32 = np.ascontiguousarray(arr, dtype="<f4")
    path.write_bytes(arr_f32.tobytes())
    print(f"  wrote {path.name:35s} n={arr_f32.size:>7d}  {label}", flush=True)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("bf16", "fp8"):
        print("Usage: hf_forward_gpu.py {bf16|fp8}")
        sys.exit(2)
    mode = sys.argv[1]
    snap = BF16_SNAP if mode == "bf16" else FP8DQ_SNAP
    prefix = "hf_bf16" if mode == "bf16" else "hf_fp8dq"

    tok_data = json.loads(TOKENS_PATH.read_text())
    all_tokens = tok_data["all_tokens"]
    prompt_len = len(all_tokens) - len(tok_data.get("generated_tokens", []))
    prompt_tokens = all_tokens[:prompt_len]
    print(f"prompt_len={prompt_len}, last={prompt_tokens[-1]}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {mode} model from {snap}", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        snap,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)
    print(
        f"  GPU mem after load: {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated, "
        f"{torch.cuda.memory_reserved() / 1e9:.1f} GB reserved",
        flush=True,
    )

    captured: dict[int, np.ndarray] = {}

    def make_hook(li):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            last = h[0, -1, :].detach().float().cpu().numpy()
            captured[li] = last
            return out

        return hook

    hooks = []
    layers = model.model.layers
    n_layers = len(layers)
    print(f"[{time.strftime('%H:%M:%S')}] hooking {n_layers} layers", flush=True)
    for i in range(n_layers):
        hooks.append(layers[i].register_forward_hook(make_hook(i)))

    print(f"[{time.strftime('%H:%M:%S')}] forward — {prompt_len} tokens", flush=True)
    t0 = time.time()
    with torch.no_grad():
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device="cuda:0")
        _ = model(input_ids, use_cache=False, output_hidden_states=False)
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  forward done in {dt:.1f}s ({prompt_len / dt:.1f} tok/s)", flush=True)

    for h in hooks:
        h.remove()

    print(f"[{time.strftime('%H:%M:%S')}] writing per-layer dumps to {OUT_DIR}", flush=True)
    for i in sorted(captured.keys()):
        write_f32(OUT_DIR / f"{prefix}_L{i}.bin", captured[i], f"L{i} hidden[last_tok]")

    # Final logits sanity check
    final_hidden = torch.tensor(captured[n_layers - 1], dtype=torch.bfloat16, device="cuda:0").unsqueeze(0)
    with torch.no_grad():
        normed = model.model.norm(final_hidden)
        logits = model.lm_head(normed.unsqueeze(0)).squeeze().float().cpu().numpy()
    top10_idx = np.argsort(-logits)[:10]
    top10 = [(int(i), float(logits[i])) for i in top10_idx]
    print(f"  {prefix} top-10 logits: {top10}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] DONE")


if __name__ == "__main__":
    main()
