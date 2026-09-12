#!/usr/bin/env python3
"""C1 goldens: HF greedy on inference-optimization/Kimi-K3-0.40B.

RST: freeze the prompt list. Known-bad is a mutated graph that must diverge.
Do not download official 1.56 TB shards.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(os.environ.get("K3_TWIN", "/home/pidtom/k3-lab/refs/Kimi-K3-0.40B"))
OUT = Path(os.environ.get("K3_GOLDEN_OUT", "/tmp/kimi-k3-0.40b-greedy.json"))
MAX_NEW = int(os.environ.get("K3_MAX_NEW", "128"))

PROMPTS = [
    "According to all known laws of aviation,",
    "The capital of France is",
    "Write a haiku about rain.",
    "2 + 2 =",
    "Once upon a time in a small village,",
    "def fibonacci(n):",
    "Explain gravity in one sentence.",
    "Hello, my name is",
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(ROOT, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ROOT, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()
    rows = []
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            lm = getattr(model, "language_model", model)
            ids = tok(prompt, return_tensors="pt")
            out = lm.generate(
                **ids, max_new_tokens=MAX_NEW, do_sample=False
            )
            rows.append(
                {
                    "i": i,
                    "prompt": prompt,
                    "token_ids": out[0].tolist(),
                    "text": tok.decode(out[0], skip_special_tokens=True),
                }
            )
            print(f"[{i}] tokens={out.shape[-1]} {rows[-1]['text'][:80]!r}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"max_new_tokens": MAX_NEW, "rows": rows}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
