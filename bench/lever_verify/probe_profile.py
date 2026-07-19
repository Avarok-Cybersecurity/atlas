#!/usr/bin/env python3
"""Fast A/B profiling probe for an Atlas serve: fire a controlled prompt battery
(streaming) and measure TTFT (time-to-first-token) + decode tok/s client-side,
bucketed by cold/warm × short/long. Lets an optimization be measured in minutes
instead of a 2.7h BFCL e2e.

Usage: python3 probe_profile.py <host:port> <model> <label> [n_per_bucket]
Writes <label>_profile.json. Prints a summary table.

Buckets:
  cold_short  ~200-token prompt, 128 out
  cold_medium ~1500-token prompt, 256 out  (near BFCL avg prompt len)
  cold_long   ~4000-token prompt, 256 out
  warm_medium same medium prompt repeated (prefix-cache hit on 2nd+)
"""
import json, sys, time, urllib.request, urllib.error

HOSTPORT, MODEL, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 4
URL = f"http://{HOSTPORT}/v1/chat/completions"

BASE = ("The quick brown fox jumps over the lazy dog. "
        "Inference engines optimize prefill and decode throughput on GB10 SM121. "
        "Quantization to NVFP4 reduces weight memory bandwidth for large language models. ")

def build_prompt(approx_tokens):
    # ~12 tokens per sentence; repeat to approximate target
    reps = max(1, approx_tokens // 12)
    return (BASE * reps)[:approx_tokens * 5]  # char-ish cap

PROMPTS = {
    "cold_short":  build_prompt(200),
    "cold_medium": build_prompt(1500),
    "cold_long":   build_prompt(4000),
}
MAXTOK = {"cold_short": 128, "cold_medium": 256, "cold_long": 256}

def stream_once(prompt, max_tokens):
    body = json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    out_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                if not line or not line.startswith(b"data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == b"[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except Exception:
                    continue
                choice = obj.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    if t_first is None:
                        t_first = time.perf_counter()
                    out_tokens += 1
    except urllib.error.URLError as e:
        return None, None, f"URLError: {e}"
    except Exception as e:
        return None, None, f"err: {e}"
    if t_first is None:
        return None, 0, "no-content"
    t_end = time.perf_counter()
    ttft_ms = (t_first - t0) * 1000
    decode_tok_s = (out_tokens - 1) / (t_end - t_first) if (out_tokens > 1 and t_end > t_first) else 0.0
    return ttft_ms, decode_tok_s, out_tokens

results = {}
print(f"=== probe {LABEL}  host={HOSTPORT}  model={MODEL}  n={N}/bucket ===")
for bucket, prompt in PROMPTS.items():
    ttfts, decs, ok = [], [], 0
    # warm_medium: reuse cold_medium prompt for 2nd+ requests (prefix cache)
    for i in range(N):
        if bucket == "warm_medium" and i == 0:
            continue  # first is cold; measure 2nd..N as warm
        ttft, dec, info = stream_once(prompt, MAXTOK[bucket])
        if ttft is not None:
            ttfts.append(ttft); decs.append(dec); ok += 1
        print(f"  {bucket} #{i}: TTFT={ttft}ms dec={'%.1f'%(dec) if dec else 0} tok/s out={info}")
    # warm_medium bucket: warm runs are i=1..N of the medium prompt
    if bucket == "cold_medium":
        # also do warm runs of the same prompt
        for i in range(1, N + 1):
            ttft, dec, info = stream_once(prompt, MAXTOK["cold_medium"])
            if ttft is not None:
                results.setdefault("warm_medium", {"ttft": [], "dec": []})
                results["warm_medium"]["ttft"].append(ttft)
                results["warm_medium"]["dec"].append(dec)
                print(f"  warm_medium #{i}: TTFT={ttft}ms dec={'%.1f'%(dec) if dec else 0} tok/s")
    results[bucket] = {"ttft": ttfts, "dec": decs}

import statistics
def med(x): return statistics.median(x) if x else None
summary = {}
print("\n=== SUMMARY (medians) ===")
print(f"{'bucket':14} {'n':>3} {'TTFT_med_ms':>12} {'decode_tok/s':>12}")
for b, d in results.items():
    t = d.get("ttft", []); c = d.get("dec", [])
    summary[b] = {"n": len(t), "ttft_med_ms": med(t), "decode_tok_s_med": med(c),
                  "ttft_all": t, "dec_all": c}
    print(f"{b:14} {len(t):>3} {('%d'%med(t)) if med(t) else '-':>12} {('%.1f'%med(c)) if med(c) else '-':>12}")

out = {"label": LABEL, "host": HOSTPORT, "model": MODEL, "n_per_bucket": N, "summary": summary}
with open(f"/workspace/{LABEL}_profile.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote /workspace/{LABEL}_profile.json")
