#!/usr/bin/env python3
"""MTP-gate verification: single-stream decode tok/s over N runs.

Sequential (batch=1) requests so the MTP/verify path is exercised. Reports
per-run and aggregate decode throughput from the server's usage block.
"""
import sys
import time
import requests

HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost"
PORT = sys.argv[2] if len(sys.argv) > 2 else "8888"
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 200
URL = f"http://{HOST}:{PORT}/v1/chat/completions"

models = requests.get(f"http://{HOST}:{PORT}/v1/models", timeout=10).json()
MODEL = models["data"][0]["id"]
PROMPT = "Write a detailed explanation of how a four-stroke internal combustion engine works, covering each stroke in order."

print(f"model={MODEL} url={URL} runs={RUNS} max_tokens={MAX_TOKENS}", flush=True)
tpss, cts = [], []
for i in range(RUNS):
    t0 = time.time()
    r = requests.post(
        URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
        },
        timeout=300,
    )
    wall = time.time() - t0
    d = r.json()
    u = d.get("usage", {})
    ct = u.get("completion_tokens", 0)
    # Prefer server-reported decode rate; fall back to wall-derived.
    tps = u.get("response_token/s") or u.get("response_tokens_per_second")
    if not tps and ct:
        tps = ct / wall
    tpss.append(tps or 0.0)
    cts.append(ct)
    print(f"  run {i+1:2d}: decode={tps:6.2f} tok/s  tokens={ct}  wall={wall:5.2f}s", flush=True)

valid = [t for t in tpss if t > 0]
valid.sort()
n = len(valid)
mean = sum(valid) / n if n else 0
median = valid[n // 2] if n else 0
print(f"\nN={n}  mean={mean:.2f} tok/s  median={median:.2f} tok/s  "
      f"min={valid[0]:.2f}  max={valid[-1]:.2f}  avg_tokens={sum(cts)/len(cts):.0f}", flush=True)
