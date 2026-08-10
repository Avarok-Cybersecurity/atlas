#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Gemma-4 E2B HuggingFace oracle — per-layer hidden-state + logit dump.

Purpose
-------
Provide a bit-exact-ish BF16 reference forward pass for google/gemma-4-E2B-it
so Atlas (spark) can be diffed layer-by-layer during bring-up (per
DEBUGGING_METHODOLOGY.md). The text core (PLE, KV sharing, dual RoPE,
double-wide MLP) must match HF layer-for-layer before multimodal work starts.

How to run (on the GB10 host with the cached checkpoint)
--------------------------------------------------------
    python3 tests/e2b_oracle.py \
        --model google/gemma-4-E2B-it \
        --prompt "What is the capital of France?" \
        --out-dir /tmp/e2b_oracle --device cuda

Feeding exact token ids (the methodology's rule: never re-render the template):
    python3 tests/e2b_oracle.py --tokens-file /tmp/tokens.json --out-dir ...

Outputs (per prompt, in --out-dir):
    <slug>_tokens.json      input_ids + greedy next-token ids (N decode steps)
    <slug>_hidden_L{i}.npy  post-layer hidden states [seq, hidden] at prefill
    <slug>_logits.npy       next-token logits [vocab] after final layer (step 0)
    <slug>_ple_inputs.npy   per_layer_inputs [seq, 35, 256] (context+identity)

Note: the checkpoint ships k/v/k_norm weights in ALL 35 layers; with
num_kv_shared_layers=20 the HF runtime ignores layers 15-34's own K/V and
reuses layer 13 (sliding) / layer 14 (full) states. The oracle dumps whatever
HF computes — Atlas must match THAT, not the dead weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--tokens-file", default=None,
                    help="JSON with {\"input_ids\": [..]} to feed exact ids (overrides --prompt).")
    ap.add_argument("--out-dir", default="/tmp/e2b_oracle")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max-new-tokens", type=int, default=8,
                    help="Greedy decode steps for next-token logits (0 = prefill only).")
    ap.add_argument("--capture-ple", action="store_true",
                    help="Also dump the model's computed per_layer_inputs tensor.")
    return ap.parse_args()


def slugify(prompt: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in prompt)[:48]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    # GB10 ARM: triton's JIT can't link libcuda.so.1 from /lib/aarch64-linux-gnu.
    # Force the non-triton math attention backend for a pure reference dump.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    dev = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=dev, low_cpu_mem_usage=True
    )
    model.eval()

    lm = model.model.language_model  # Gemma4ForConditionalGeneration -> Gemma4Model -> text
    text_cfg = lm.config
    n_layers = text_cfg.num_hidden_layers
    hidden = text_cfg.hidden_size

    # Collect per-layer post-layer hidden states of the decoder.
    captured: dict[int, torch.Tensor] = {}

    def make_hook(i: int):
        def hook(_mod, _args, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[i] = hs.detach().float().cpu()
        return hook

    handles = [lm.layers[i].register_forward_hook(make_hook(i)) for i in range(n_layers)]

    if args.tokens_file:
        with open(args.tokens_file) as f:
            ids = json.load(f)["input_ids"]
        input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
        prompt_slug = Path(args.tokens_file).stem
    else:
        ids = tok(args.prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
        input_ids = ids
        prompt_slug = slugify(args.prompt)

    # Prefill forward: per-layer hidden states + per_layer_inputs.
    kwargs = {}
    if args.capture_ple:
        # Replicate the model's PLE precompute (token identity + context projection).
        pad_id = text_cfg.pad_token_id
        mm_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=dev)
        llm_ids = input_ids.clone()
        llm_ids = torch.where(mm_mask, torch.tensor(pad_id, device=dev), llm_ids)
        embeds = lm.embed_tokens(llm_ids) * (text_cfg.hidden_size**0.5)
        pad_emb = lm.embed_tokens.weight[pad_id].unsqueeze(0).unsqueeze(0)
        llm_embeds = torch.where(mm_mask.unsqueeze(-1), pad_emb, embeds)
        ple = lm.get_per_layer_inputs(llm_ids, llm_embeds)
        ple = lm.project_per_layer_inputs(llm_embeds, ple)
        captured_ple = ple.detach().float().cpu()

    out_after = lm(input_ids=input_ids, output_hidden_states=True)
    seq_len = input_ids.shape[1]

    # Final-layer logits (prefill step 0 next token).
    logits = model.lm_head(out_after.hidden_states[-1][:, -1, :]).float().cpu()
    next_id = int(logits.argmax(dim=-1).item())

    record = {
        "input_ids": input_ids.tolist()[0],
        "prompt": args.prompt if not args.tokens_file else f"from {args.tokens_file}",
        "next_token_0": next_id,
        "seq_len": seq_len,
        "hidden_size": hidden,
        "n_layers": n_layers,
    }
    with open(out / f"{prompt_slug}_tokens.json", "w") as f:
        json.dump(record, f, indent=2)
    np.save(out / f"{prompt_slug}_logits.npy", logits.numpy())
    for i in range(n_layers):
        hs = captured.get(i)
        if hs is not None:
            np.save(out / f"{prompt_slug}_hidden_L{i}.npy", hs[:, :, :].numpy())
    if args.capture_ple:
        np.save(out / f"{prompt_slug}_ple_inputs.npy", captured_ple.numpy())

    # Greedy decode: next-token ids for parity of the sampler path.
    decode_ids = []
    cur = input_ids
    for _ in range(max(args.max_new_tokens, 0)):
        out_step = model(cur, use_cache=True)
        nxt = int(out_step.logits[:, -1, :].argmax(dim=-1).item())
        decode_ids.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long, device=dev)], dim=1)
    record["greedy_next_ids"] = decode_ids
    with open(out / f"{prompt_slug}_tokens.json", "w") as f:
        json.dump(record, f, indent=2)

    for h in handles:
        h.remove()
    print(f"ORACLE_OK prompt={prompt_slug} seq={seq_len} next0={next_id} greedy={decode_ids}")
    print(f"  wrote {len(captured)} layer hidden states to {out}")


if __name__ == "__main__":
    sys.exit(main())
