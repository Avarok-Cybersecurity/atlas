#!/usr/bin/env python3
"""Prefill golden prompt 0; dump last-position top-8 logits.

HF greedy continuation starts `[1459, 387, ...]` (387 is the second token).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(os.environ.get("K3_TWIN", "/home/pidtom/k3-lab/refs/Kimi-K3-0.40B"))
PROMPT = "According to all known laws of aviation,"
HF_FIRST = 1459
HF_SECOND = 387


def main() -> None:
    tok = AutoTokenizer.from_pretrained(ROOT, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        ROOT, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()
    lm = getattr(model, "language_model", model)
    ids = tok(PROMPT, return_tensors="pt")
    ids = {k: v.to(device) for k, v in ids.items()}
    print("prompt_ids", ids["input_ids"][0].tolist(), flush=True)
    with torch.no_grad():
        out = lm(**ids)
        logits = out.logits[0, -1].float()
    vals, idx = torch.topk(logits, 8)
    top = [(int(i), float(v)) for i, v in zip(idx, vals)]
    pred = int(idx[0])
    print(
        "argmax",
        pred,
        "hf_first",
        HF_FIRST,
        "hf_second",
        HF_SECOND,
        "top-8",
        top,
        flush=True,
    )
    if pred != HF_FIRST:
        raise SystemExit(f"first token {pred} != {HF_FIRST}")


if __name__ == "__main__":
    main()
