#!/usr/bin/env python3
"""DeepSeek-V4-Flash bench: coherence + cold throughput/TTFT + prefix-cache A/B.

Run against a live Atlas EP=2 server:
    python3 scripts/bench_deepseek_v4.py --url http://10.10.10.1:8888 --label fp8-kv

The KV-cache dtype and prefix-caching flag are properties of how the *server*
was launched (start-deepseek-ep2-redhat.sh KV_DTYPE=...); this script just
labels the run. MTP is not exercised — the public checkpoint ships no MTP
draft weights, so speculative decode is off server-side.
"""
import argparse
import json
import time
import urllib.request

MODEL = "RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8"


def chat(url, messages, max_tokens=128, temperature=0.0):
    body = json.dumps(
        {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
         "temperature": temperature, "stream": False}
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    wall = time.time() - t0
    msg = d["choices"][0]["message"]["content"]
    u = d.get("usage", {})
    gen = u.get("completion_tokens", 0)
    return {
        "text": msg,
        "wall_s": wall,
        "prompt_tokens": u.get("prompt_tokens", 0),
        "gen_tokens": gen,
        "ttft_ms": u.get("time_to_first_token_ms", 0.0),
        "tok_s": u.get("response_token/s", gen / max(wall, 1e-3)),
    }


def coherence(url):
    print("\n== Coherence ==")
    cases = [
        ("Capital of France?", "What is the capital of France? One word."),
        ("Count", "Count from 1 to 10, comma separated."),
        ("Math", "What is 17 * 23? Answer with just the number."),
        ("Haiku", "Write a haiku about the ocean."),
    ]
    ok = 0
    for name, prompt in cases:
        r = chat(url, [{"role": "user", "content": prompt}], max_tokens=64)
        snippet = r["text"].replace("\n", " ")[:80]
        print(f"  [{name}] {r['tok_s']:.1f} tok/s :: {snippet}")
        if r["text"].strip():
            ok += 1
    print(f"  coherence: {ok}/{len(cases)} non-empty")
    return ok == len(cases)


def cold_throughput(url):
    print("\n== Cold throughput / TTFT (150 tok) ==")
    r = chat(url, [{"role": "user",
                    "content": "Write a Python function to compute fibonacci with a docstring."}],
             max_tokens=150, temperature=0.7)
    print(f"  {r['tok_s']:.1f} tok/s, TTFT={r['ttft_ms']:.0f}ms, "
          f"gen={r['gen_tokens']}, prompt={r['prompt_tokens']}")
    return r


def prefix_cache_ab(url):
    """Send a long shared prefix twice. With prefix caching ON, the 2nd
    request's prefill is a cache hit → much lower TTFT."""
    print("\n== Prefix-cache A/B (same ~1.5k-token prefix twice) ==")
    filler = ("The following is a long technical document about distributed "
              "inference systems and memory hierarchies on the GB10 platform. ") * 60
    base = [{"role": "user", "content": filler + "\n\nSummarize the above in one sentence."}]
    cold = chat(url, base, max_tokens=40)
    warm = chat(url, base, max_tokens=40)
    print(f"  cold: TTFT={cold['ttft_ms']:.0f}ms prompt={cold['prompt_tokens']} tok")
    print(f"  warm: TTFT={warm['ttft_ms']:.0f}ms prompt={warm['prompt_tokens']} tok")
    if cold["ttft_ms"] > 0:
        speedup = cold["ttft_ms"] / max(warm["ttft_ms"], 1e-3)
        print(f"  warm/cold TTFT speedup: {speedup:.2f}x "
              f"({'cache HIT' if speedup > 1.3 else 'no clear hit'})")
    return {"cold_ttft_ms": cold["ttft_ms"], "warm_ttft_ms": warm["ttft_ms"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://10.10.10.1:8888")
    ap.add_argument("--label", default="run")
    args = ap.parse_args()
    print(f"=== DeepSeek-V4-Flash bench :: {args.label} :: {args.url} ===")
    out = {"label": args.label, "url": args.url}
    out["coherent"] = coherence(args.url)
    out["cold"] = cold_throughput(args.url)
    out["prefix_cache"] = prefix_cache_ab(args.url)
    print("\n=== SUMMARY ===")
    print(json.dumps({
        "label": args.label,
        "coherent": out["coherent"],
        "cold_tok_s": round(out["cold"]["tok_s"], 1),
        "cold_ttft_ms": round(out["cold"]["ttft_ms"], 0),
        "prefix_cold_ttft_ms": round(out["prefix_cache"]["cold_ttft_ms"], 0),
        "prefix_warm_ttft_ms": round(out["prefix_cache"]["warm_ttft_ms"], 0),
    }, indent=2))


if __name__ == "__main__":
    main()
