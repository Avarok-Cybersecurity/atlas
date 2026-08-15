#!/usr/bin/env python3
"""Decode-rate measurement that actually decodes.

The A/B harness's decode leg asked the model to count to 200 and got 49
tokens — it stopped early, so that number is a 49-token sample and its ~4%
delta is not separable from noise. This forces a long generation with
ignore_eos and repeats it, so the rate is measured over enough tokens for the
difference to mean something.

Usage: decode_probe.py <port> <tag> [reps]
"""
import json
import statistics
import sys
import time
import urllib.request

PORT, TAG = sys.argv[1], sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
GEN = 400


def once(i):
    body = {
        "model": "m",
        # Unique preamble per rep so a prefix-cache hit cannot shorten prefill
        # and flatter the decode rate.
        "messages": [{"role": "user", "content": f"[rep {i}] Write a detailed essay about paged KV caches."}],
        "max_tokens": GEN,
        "min_tokens": GEN,
        "ignore_eos": True,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read())
    wall = time.time() - t0
    u = d.get("usage", {})
    ctok = u.get("completion_tokens") or 0
    return ctok, wall, ctok / wall if wall else 0.0


rates = []
for i in range(REPS):
    ctok, wall, rate = once(i)
    rates.append(rate)
    print(f"  [{TAG}] rep{i}: {ctok} tok in {wall:.2f}s = {rate:.2f} tok/s", flush=True)
print(f"[{TAG}] median {statistics.median(rates):.2f} tok/s  "
      f"min {min(rates):.2f}  max {max(rates):.2f}  over {REPS} reps of {GEN} tokens")
