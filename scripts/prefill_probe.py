#!/usr/bin/env python3
"""Prefill throughput at C=1 and C=4, cold.

Concurrency is not optional here. FlashInfer-GDN FAILS OPEN — without its
library it silently falls back, and that costs 40-50% of prefill AT
CONCURRENCY while C=1 cannot see it. A C=1-only number would report "the flag
did nothing" and "the flag worked" identically.

Cold on purpose: every prompt carries a unique preamble so a prefix-cache hit
cannot be mistaken for prefill speed. Generation is capped at 8 tokens so the
wall is prefill, not decode.

Usage: prefill_probe.py <port> <tag> [reps]
"""
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PORT, TAG = sys.argv[1], sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

BODY = (
    "A paged KV cache stores per-layer key and value tensors in fixed-size blocks so that "
    "sequences of different lengths share one allocator without fragmentation. Each block "
    "holds a fixed number of token slots, and a per-sequence block table maps logical "
    "positions onto physical blocks. Prefill fills these blocks in chunks; decode appends "
    "one token per step and occasionally allocates a fresh block. "
)


def ask(uniq):
    prompt = f"[{TAG}-{uniq}-{time.time_ns()}] " + BODY * 24 + "\nSummarise in one sentence."
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    return d.get("usage", {}).get("prompt_tokens") or 0, time.time() - t0


def run(conc, rep):
    """Aggregate prefill tok/s: total prompt tokens / wall for the whole wave."""
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        out = list(ex.map(ask, [f"c{conc}r{rep}i{i}" for i in range(conc)]))
    wall = time.time() - t0
    ptok = sum(p for p, _ in out)
    return ptok, wall, ptok / wall if wall else 0.0


for conc in (1, 4):
    rates = []
    for rep in range(REPS):
        ptok, wall, rate = run(conc, rep)
        rates.append(rate)
        print(f"  [{TAG}] C={conc} rep{rep}: {ptok} prompt tok in {wall:.2f}s = {rate:.0f} tok/s",
              flush=True)
    print(f"[{TAG}] C={conc} MEDIAN {statistics.median(rates):.0f} tok/s "
          f"(min {min(rates):.0f}, max {max(rates):.0f})", flush=True)
