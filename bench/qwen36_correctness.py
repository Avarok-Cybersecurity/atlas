#!/usr/bin/env python3
"""Quick correctness probe for Qwen3.5-35B-A3B-FP8 with ATLAS_FP8_MOE_COALESCED=1.

Asks 5 short coherence / math / code prompts. Fails if any response is empty,
repeats a 20-token pattern ≥3 times (fuzzy repetition regression), or contains
obvious corruption markers.

Run AFTER the TTFT bench to confirm the new kernel path produces sane output.
"""
import json
import re
import sys
import urllib.request

PROMPTS = [
    ("factual",     "What is the capital of Japan? Answer in one sentence."),
    ("math",        "Solve step by step: if a train travels 60 km in 45 minutes, "
                    "what is its speed in km/h?"),
    ("code",        "Write a Python function that returns the factorial of n. "
                    "Include a docstring. No explanation outside the code."),
    ("instruction", "List three common causes of cache-miss penalties in GPU kernels, "
                    "one per line, prefixed with a dash."),
    ("short_ctx",   "A=7, B=12. Return A+B as a JSON object with key 'sum'."),
]


def post(url: str, body: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def looks_repetitive(text: str, window: int = 20, repeats: int = 3) -> bool:
    tokens = text.split()
    if len(tokens) < window * repeats:
        return False
    tail = tokens[-window * repeats:]
    first = tail[:window]
    for i in range(1, repeats):
        seg = tail[i * window:(i + 1) * window]
        # approximate match; one token different is still "same pattern"
        diffs = sum(1 for a, b in zip(first, seg) if a != b)
        if diffs > 2:
            return False
    return True


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8888"
    model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.5-35B-A3B-FP8"

    failures = []
    for name, prompt in PROMPTS:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.0,
        }
        r = post(f"{url}/v1/chat/completions", body, timeout=240.0)
        text = r["choices"][0]["message"]["content"] or ""
        if not text.strip():
            failures.append(f"{name}: empty response")
            continue
        if looks_repetitive(text):
            failures.append(f"{name}: fuzzy-repetition pattern at tail")
            continue
        if re.search(r"(\w{5,})\1{4,}", text):
            failures.append(f"{name}: exact-token loop (5+ repeats)")
            continue
        print(f"[ok] {name}: {text.strip()[:100].replace(chr(10), ' ')}...")

    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
