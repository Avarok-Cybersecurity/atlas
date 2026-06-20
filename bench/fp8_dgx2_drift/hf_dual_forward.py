#!/usr/bin/env python3
"""Run HF[BF16-unquant] AND HF[FP8->BF16] forward on a freshly-dumped Atlas
prompt-token sequence, emitting per-layer hidden states for both.

Output layout:
  /workspace/atlas-dumps/fp8native_dgx2/hf_bf16_L{0..39}.bin
  /workspace/atlas-dumps/fp8native_dgx2/hf_fp8dq_L{0..39}.bin

Inputs:
  /tmp/atlas_tokens_dgx2.json  — tokens dumped by Atlas via ATLAS_DFLASH_DEBUG_DUMP_FULL=1

Usage: python3 hf_dual_forward.py [bf16|fp8|both]
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


def forward(snap: str, prefix: str, prompt_tokens: list[int]):
    print(f"\n[{time.strftime('%H:%M:%S')}] === Forward pass on {snap}", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        snap,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)

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
    print(
        f"[{time.strftime('%H:%M:%S')}] registering hooks on {n_layers} layers",
        flush=True,
    )
    for i in range(n_layers):
        hooks.append(layers[i].register_forward_hook(make_hook(i)))

    prompt_len = len(prompt_tokens)
    print(
        f"[{time.strftime('%H:%M:%S')}] forward pass — {prompt_len} tokens",
        flush=True,
    )
    t0 = time.time()
    with torch.no_grad():
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        _ = model(input_ids, use_cache=False, output_hidden_states=False)
    dt = time.time() - t0
    print(f"  forward done in {dt:.1f}s ({prompt_len / dt:.1f} tok/s)", flush=True)

    for h in hooks:
        h.remove()

    print(
        f"[{time.strftime('%H:%M:%S')}] writing per-layer dumps to {OUT_DIR}",
        flush=True,
    )
    for i in sorted(captured.keys()):
        write_f32(OUT_DIR / f"{prefix}_L{i}.bin", captured[i], f"L{i} hidden[last_tok]")

    # Final logits for sanity check
    final_hidden = captured[n_layers - 1]
    final_norm_module = model.model.norm
    with torch.no_grad():
        h_t = torch.tensor(final_hidden, dtype=torch.bfloat16).unsqueeze(0)
        normed = final_norm_module(h_t).float().cpu().numpy().squeeze(0)
        h_t2 = torch.tensor(normed, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0)
        logits = model.lm_head(h_t2).float().cpu().numpy().squeeze()
    top10_idx = np.argsort(-logits)[:10]
    top10 = [(int(i), float(logits[i])) for i in top10_idx]
    print(f"  {prefix} top-10 logits: {top10}", flush=True)

    # Free model
    del model
    gc.collect()


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    print(f"[{time.strftime('%H:%M:%S')}] loading tokens from {TOKENS_PATH}", flush=True)
    tok_data = json.loads(TOKENS_PATH.read_text())
    all_tokens = tok_data["all_tokens"]
    generated = tok_data.get("generated_tokens", [])
    prompt_len = len(all_tokens) - len(generated)
    prompt_tokens = all_tokens[:prompt_len]
    print(
        f"  prompt_len={prompt_len}, last prompt tok={prompt_tokens[-1]} "
        f"(all={len(all_tokens)} - generated={len(generated)})",
        flush=True,
    )

    if mode in ("bf16", "both"):
        forward(BF16_SNAP, "hf_bf16", prompt_tokens)
    if mode in ("fp8", "both"):
        forward(FP8DQ_SNAP, "hf_fp8dq", prompt_tokens)

    print(f"\n[{time.strftime('%H:%M:%S')}] ALL DONE")


if __name__ == "__main__":
    main()
