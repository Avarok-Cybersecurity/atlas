#!/usr/bin/env python3
"""Quick Atlas benchmark — measures TTFT and decode tok/s with minimal overhead."""

import requests
import time
import sys
import json

HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost"
PORT = sys.argv[2] if len(sys.argv) > 2 else "8888"
URL = f"http://{HOST}:{PORT}/v1/chat/completions"
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 100

# Auto-detect model
try:
    models = requests.get(f"http://{HOST}:{PORT}/v1/models", timeout=5).json()
    MODEL = models["data"][0]["id"]
except Exception:
    MODEL = "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4"

print(f"Model: {MODEL}")
print(f"URL:   {URL}")
print(f"Runs:  {RUNS}, max_tokens={MAX_TOKENS}")
print()

prompts = [
    ("short", "What is the capital of France?"),
    ("medium", "The quick brown fox " * 50 + "Summarize."),
    ("long", "word " * 1000),
]

for label, prompt in prompts:
    ttfts, tpss, cts = [], [], []
    for i in range(RUNS):
        try:
            r = requests.post(URL, json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_TOKENS,
            }, timeout=120)
            d = r.json()
            u = d.get("usage", {})
            ttft = u.get("time_to_first_token_ms", 0)
            tps = u.get("response_token/s", 0)
            ct = u.get("completion_tokens", 0)
            pt = u.get("prompt_tokens", 0)
            ttfts.append(ttft)
            tpss.append(tps)
            cts.append(ct)
        except Exception as e:
            print(f"  {label} run {i+1}: ERROR {e}")
    if ttfts:
        avg_ttft = sum(ttfts) / len(ttfts)
        avg_tps = sum(tpss) / len(tpss)
        avg_ct = sum(cts) / len(cts)
        pt_str = f"~{pt}" if 'pt' in dir() else "?"
        print(f"  {label:>8} ({pt_str:>5} in): TTFT={avg_ttft:7.0f}ms  decode={avg_tps:5.1f} tok/s  ({RUNS} runs, avg {avg_ct:.0f} tokens)")
