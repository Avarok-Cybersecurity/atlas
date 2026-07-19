#!/usr/bin/env python3
"""Continuation TTFT probe — measures the warm-TTFT that the GDN tail capture
(midchunk) targets: a SHARED prefix (cached) + a NEW tail (prefilled, SSM state
replayed from the prefix snapshot). This is the BFCL agentic multi-turn pattern
(system + turn1 cached, turn2 = continuation).

Usage: python3 probe_continuation.py <host:port> <model> <label> [n]
Writes <label>_continuation.json. Measures turn-2 (continuation) TTFT — lower = better.
"""
import json, sys, time, urllib.request, urllib.error

HOSTPORT, MODEL, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 4
URL = f"http://{HOSTPORT}/v1/chat/completions"
BASE = ("The quick brown fox jumps over the lazy dog. "
        "Inference engines optimize prefill and decode throughput on GB10 SM121. "
        "Quantization to NVFP4 reduces weight memory bandwidth for large language models. ")

def build(approx_tokens):
    reps = max(1, approx_tokens // 12)
    return (BASE * reps)[:approx_tokens * 5]

# Buckets: prefix length (the cached part) + a short NEW user turn (the tail to prefill)
BUCKETS = [
    ("cont_short",  build(500),  build(60)),    # 500-tok prefix + 60-tok tail
    ("cont_medium", build(2000), build(80)),    # 2000-tok prefix + 80-tok tail
    ("cont_long",   build(4000), build(120)),   # 4000-tok prefix + 120-tok tail
]

def send(messages, max_tokens=64):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = None; out = 0
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            for line in resp:
                if not line.startswith(b"data:"): continue
                chunk = line[5:].strip()
                if chunk == b"[DONE]": break
                try: obj = json.loads(chunk)
                except: continue
                c = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                if c:
                    if t_first is None: t_first = time.perf_counter()
                    out += 1
    except Exception as e:
        return None, str(e)
    return (t_first - t0) * 1000 if t_first else None, out

results = {}
print(f"=== continuation probe {LABEL}  host={HOSTPORT}  model={MODEL}  n={N} ===")
for name, prefix, tail in BUCKETS:
    ttfts = []
    # turn 1: cache the prefix (system + user=prefix). Measure its TTFT too (cold).
    cold_ttft, _ = send([{"role": "user", "content": prefix}], 16)
    print(f"  {name} turn1(cold cache-fill): TTFT={cold_ttft}ms")
    # turn 2..N: continuation = [user=prefix, assistant=short, user=tail] — prefix cached, tail is new
    for i in range(N):
        msgs = [{"role": "user", "content": prefix},
                {"role": "assistant", "content": "OK. I have read the context."},
                {"role": "user", "content": tail + " Based on the above, summarize in one sentence."}]
        ttft, out = send(msgs, 64)
        if ttft is not None:
            ttfts.append(ttft)
            print(f"  {name} turn2(continuation) #{i}: TTFT={ttft:.0f}ms")
        else:
            print(f"  {name} turn2 #{i}: {out}")
    results[name] = {"cold_fill_ttft_ms": cold_ttft, "cont_ttft_all": ttfts,
                     "cont_ttft_med_ms": sorted(ttfts)[len(ttfts)//2] if ttfts else None}
    print(f"  {name} continuation TTFT median: {results[name]['cont_ttft_med_ms']}ms")

import statistics
print("\n=== SUMMARY (continuation TTFT median, ms — lower=better) ===")
print(f"{'bucket':14} {'cold_fill':>10} {'cont_med':>10}")
for b, d in results.items():
    print(f"{b:14} {('%d'%d['cold_fill_ttft_ms']) if d['cold_fill_ttft_ms'] else '-':>10} {('%d'%d['cont_ttft_med_ms']) if d['cont_ttft_med_ms'] else '-':>10}")
out = {"label": LABEL, "host": HOSTPORT, "model": MODEL, "n": N, "summary": results}
json.dump(out, open(f"/workspace/{LABEL}_continuation.json", "w"), indent=2)
print(f"\nwrote /workspace/{LABEL}_continuation.json")
