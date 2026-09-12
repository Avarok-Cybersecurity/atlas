#!/usr/bin/env python3
"""RST: is HF C1 oracle path-dependent?

Path A: one forward of the full prompt (chunk_kda, q_len>1).
Path B: token-by-token with cache (fused_recurrent_kda, q_len=1).

If argmax disagrees, goldens from generate() (chunk prefill) are not the
same as a sequential CPU engine. Do not call that a KDA CPU bug yet.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(os.environ.get("K3_TWIN", "/home/pidtom/k3-lab/refs/Kimi-K3-0.40B"))
PROMPT = os.environ.get("K3_PROMPT", "Once upon a time in a small village,")


def top16(logits: torch.Tensor) -> list[tuple[int, float]]:
    vals, idx = torch.topk(logits.float(), 16)
    return [(int(i), float(v)) for i, v in zip(idx, vals)]


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
    prompt_ids = ids["input_ids"][0].tolist()
    print("prompt_ids", prompt_ids, flush=True)

    with torch.no_grad():
        out_a = lm(**ids)
        logits_a = out_a.logits[0, -1]
        print("A_chunk_argmax", int(logits_a.argmax()), "top16", top16(logits_a), flush=True)

        # Sequential: feed one token at a time with cache (q_len=1 → fused_recurrent).
        past = None
        logits_b = None
        for t in prompt_ids:
            step = torch.tensor([[t]], device=device)
            out_b = lm(input_ids=step, past_key_values=past, use_cache=True)
            past = out_b.past_key_values
            logits_b = out_b.logits[0, -1]
        print("B_recurrent_argmax", int(logits_b.argmax()), "top16", top16(logits_b), flush=True)

        a = int(logits_a.argmax())
        b = int(logits_b.argmax())
        print("A_vs_B", a, b, "match" if a == b else "DIVERGE", flush=True)
        for i in {a, b, 996, 60063}:
            print(
                "logit",
                i,
                "A",
                float(logits_a[i]),
                "B",
                float(logits_b[i]),
                flush=True,
            )


if __name__ == "__main__":
    main()
