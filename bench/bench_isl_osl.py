#!/usr/bin/env python3
"""
Atlas Spark ISL/OSL sweep at batch=1.

Measures TTFT, TPOT, and decode throughput across 6 representative
ISL/OSL configurations (single request, no concurrency).

Usage:
    python3 bench_isl_osl.py                              # all 6 configs
    python3 bench_isl_osl.py --configs balanced_short      # one config
    python3 bench_isl_osl.py --url http://127.0.0.1:8888  # remote
    python3 bench_isl_osl.py --runs 5                      # more samples
"""
import argparse
import json
import sys
import time
from threading import Barrier
from urllib.request import urlopen, Request

FILLER_WORD = "The quick brown fox jumps over the lazy dog. "
PROMPT_SUFFIX = (
    "\n\nProvide a very detailed and comprehensive analysis. "
    "Do not stop early. Cover every aspect in depth."
)

TEST_CONFIGS = [
    (256, 256, "balanced_short", "Short chat"),
    (1024, 128, "prefill_1k", "Prefill 1K"),
    (1024, 1024, "balanced_1k", "Standard chat 1K"),
    (128, 1024, "decode_short", "Code generation"),
    (4096, 128, "prefill_4k", "Prefill 4K"),
    (4096, 1024, "balanced_4k", "Standard chat 4K"),
    (8192, 128, "prefill_8k", "Prefill 8K"),
    (8192, 1024, "balanced_8k", "RAG / document QA"),
    (16384, 128, "prefill_16k", "Prefill 16K"),
    (16384, 1024, "balanced_16k", "Long context 16K"),
    (32768, 128, "prefill_32k", "Prefill 32K"),
    (32768, 1024, "balanced_32k", "Long context 32K"),
    (65536, 128, "prefill_64k", "Prefill 64K"),
    (65536, 1024, "balanced_64k", "Long context 64K"),
    (131072, 128, "prefill_128k", "Prefill 128K"),
    (131072, 1024, "balanced_128k", "Long context 128K"),
]


def make_prompt(target_tokens: int) -> str:
    chars_needed = target_tokens * 4
    repeats = max(1, chars_needed // len(FILLER_WORD))
    filler = (FILLER_WORD * repeats)[:chars_needed]
    return f"Analyze the following text thoroughly:\n\n{filler}{PROMPT_SUFFIX}"


def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def send_request(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()

    req = Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.perf_counter()
    t_first = None
    t_last = None
    token_count = 0

    try:
        resp = urlopen(req, timeout=600)
    except Exception as e:
        return {"error": str(e)[:200]}

    for raw_line in resp:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        delta = chunk.get("choices", [{}])[0].get("delta", {})
        if delta.get("content"):
            t_now = time.perf_counter()
            if t_first is None:
                t_first = t_now
            t_last = t_now
            token_count += 1

    t_end = time.perf_counter()
    total = t_end - t_start
    ttft = (t_first - t_start) if t_first else total
    decode_time = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0

    if token_count > 1 and decode_time > 0:
        tpot = decode_time / (token_count - 1)
        decode_tok_s = (token_count - 1) / decode_time
    else:
        tpot = 0.0
        decode_tok_s = 0.0

    return {
        "ttft_ms": ttft * 1000,
        "tpot_ms": tpot * 1000,
        "decode_tok_s": decode_tok_s,
        "tokens": token_count,
        "total_s": total,
    }


def detect_model(url: str) -> str:
    try:
        resp = urlopen(f"{url}/v1/models", timeout=5)
        data = json.loads(resp.read().decode())
        return data["data"][0]["id"]
    except Exception as e:
        print(f"Cannot reach server at {url}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Atlas Spark ISL/OSL sweep (batch=1)")
    parser.add_argument("--url", default="http://localhost:8888")
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per config (median reported)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Filter by regime name (e.g. balanced_short decode_long)")
    parser.add_argument("--output", default=None,
                        help="JSON output path")
    args = parser.parse_args()

    configs = TEST_CONFIGS
    if args.configs:
        configs = [c for c in TEST_CONFIGS if c[2] in args.configs]
        if not configs:
            valid = [c[2] for c in TEST_CONFIGS]
            print(f"No matching configs. Valid: {valid}", file=sys.stderr)
            sys.exit(1)

    model = detect_model(args.url)
    print(f"Server:  {args.url}")
    print(f"Model:   {model}")
    print(f"Runs:    {args.runs} per config (median)")
    print(f"Configs: {len(configs)}")
    print()

    # Warmup
    if args.warmup > 0:
        print(f"Warming up ({args.warmup} requests)...")
        for i in range(args.warmup):
            r = send_request(args.url, model, "Hello!", 50)
            if "error" in r:
                print(f"  warmup {i + 1}: FAILED ({r['error'][:80]})")
            else:
                print(f"  warmup {i + 1}: {r['decode_tok_s']:.1f} tok/s, "
                      f"TTFT={r['ttft_ms']:.0f}ms")
        print()

    # Sweep
    results = []
    for isl, osl, regime, label in configs:
        prompt = make_prompt(isl)
        ttfts, tpots, tok_rates, token_counts = [], [], [], []

        for run_idx in range(args.runs):
            r = send_request(args.url, model, prompt, osl)
            if "error" in r:
                print(f"  {regime} run {run_idx + 1}: FAILED ({r['error'][:80]})")
                continue
            ttfts.append(r["ttft_ms"])
            tpots.append(r["tpot_ms"])
            tok_rates.append(r["decode_tok_s"])
            token_counts.append(r["tokens"])
            print(f"  {regime} run {run_idx + 1}: "
                  f"TTFT={r['ttft_ms']:.1f}ms  "
                  f"TPOT={r['tpot_ms']:.2f}ms  "
                  f"tok/s={r['decode_tok_s']:.1f}  "
                  f"tokens={r['tokens']}")

        if not ttfts:
            results.append({
                "isl": isl, "osl": osl, "regime": regime, "label": label,
                "status": "failed",
            })
            continue

        entry = {
            "isl": isl, "osl": osl, "regime": regime, "label": label,
            "status": "ok",
            "ttft_ms": {"p50": round(percentile(ttfts, 50), 1)},
            "tpot_ms": {"p50": round(percentile(tpots, 50), 2)},
            "decode_tok_s": {"p50": round(percentile(tok_rates, 50), 1)},
            "avg_tokens": round(sum(token_counts) / len(token_counts)),
            "runs": len(ttfts),
        }
        results.append(entry)
        print()

    # Summary table
    print("=" * 85)
    print(f"{'ISL/OSL SWEEP — Single Request (batch=1)':^85}")
    print("=" * 85)
    print(f"  {'Workload':<25} {'ISL/OSL':>10} {'TTFT p50':>10} "
          f"{'TPOT p50':>10} {'tok/s p50':>10} {'Tokens':>8}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}")

    for r in results:
        if r["status"] != "ok":
            print(f"  {r['label']:<25} {r['isl']}/{r['osl']:>5}   FAILED")
            continue
        print(
            f"  {r['label']:<25} "
            f"{r['isl']:>4}/{r['osl']:<5} "
            f"{r['ttft_ms']['p50']:>8.1f}ms "
            f"{r['tpot_ms']['p50']:>8.2f}ms "
            f"{r['decode_tok_s']['p50']:>10.1f} "
            f"{r['avg_tokens']:>8}"
        )

    print()

    # Markdown table
    print("### Markdown\n")
    print("| Workload | ISL/OSL | TTFT p50 | TPOT p50 | tok/s |")
    print("|---|---:|---:|---:|---:|")
    for r in results:
        if r["status"] != "ok":
            print(f"| {r['label']} | {r['isl']}/{r['osl']} | FAIL | | |")
            continue
        print(
            f"| {r['label']} "
            f"| {r['isl']}/{r['osl']} "
            f"| {r['ttft_ms']['p50']}ms "
            f"| {r['tpot_ms']['p50']}ms "
            f"| {r['decode_tok_s']['p50']} |"
        )
    print()

    # Save JSON
    if args.output:
        output = {
            "model": model, "server": args.url,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
