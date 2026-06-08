#!/usr/bin/env python3
"""HF CPU forward on the dequanted FP8->BF16 snapshot.

Phase 2a-step-3. Loads `/workspace/.cache/huggingface/Qwen3.6-35B-A3B-FP8-dequanted-BF16/`
(produced by dequant_to_bf16.py), runs forward on Atlas's exact 18920-token
prompt (from /tmp/atlas_tokens.json), and dumps per-layer last-token hidden
states to /workspace/atlas-dumps/fp8dequant/hf_L{0..39}.bin matching the
Phase alpha layout. Then cosine_compare.py reads these against
/workspace/atlas-dumps/numdrift/hf_L*.bin (the unquantized BF16 reference)
to produce comparison A from the plan.

Reuses the exact pattern from /tmp/hf_qwen36_full_prefill_ref.py with two
diffs: source snapshot path + output dir. SSOT for forward logic.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

SNAP = "/workspace/.cache/huggingface/Qwen3.6-35B-A3B-FP8-dequanted-BF16"
TOKENS_PATH = pathlib.Path("/tmp/atlas_tokens.json")
OUT_DIR = pathlib.Path("/workspace/atlas-dumps/fp8dequant")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_f32(path: pathlib.Path, arr: np.ndarray, label: str):
    arr_f32 = np.ascontiguousarray(arr, dtype="<f4")
    path.write_bytes(arr_f32.tobytes())
    print(f"  wrote {path.name:30s} n={arr_f32.size:>7d}  {label}", flush=True)


def main() -> None:
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

    print(f"[{time.strftime('%H:%M:%S')}] loading model from {SNAP}", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        SNAP,
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
        write_f32(OUT_DIR / f"hf_L{i}.bin", captured[i], f"L{i} hidden[last_tok]")

    final_hidden = captured[n_layers - 1]
    final_norm_module = model.model.norm
    with torch.no_grad():
        h_t = torch.tensor(final_hidden, dtype=torch.bfloat16).unsqueeze(0)
        normed = final_norm_module(h_t).float().cpu().numpy().squeeze(0)
    write_f32(OUT_DIR / "hf_final_norm.bin", normed, "final_norm[last_tok]")

    with torch.no_grad():
        h_t = torch.tensor(normed, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0)
        logits = model.lm_head(h_t).float().cpu().numpy().squeeze()
    write_f32(OUT_DIR / "hf_logits.bin", logits, f"logits (vocab={logits.size})")

    top10_idx = np.argsort(-logits)[:10]
    top10 = [(int(i), float(logits[i])) for i in top10_idx]
    print(f"  HF[FP8->BF16] top-10 logits: {top10}", flush=True)
    print(
        f"  (Atlas reported top-10: [(9764, 21.5), (760, 20.875), (6820, 20.25), ...])",
        flush=True,
    )

    print(f"[{time.strftime('%H:%M:%S')}] DONE — dumps in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
