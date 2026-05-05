#!/usr/bin/env python3
"""
Atlas Spark Concurrency Sweep Benchmark — Latency-Throughput Curve

Same methodology as bench-nvfp4-concurrency.py (vLLM benchmark):
  6 ISL/OSL configs x 7 concurrency levels = 42 runs
  Streaming SSE, P50/P90/P99 for TTFT and TPOT, aggregate throughput.

Usage:
  python3 bench-atlas-concurrency.py                          # Run benchmark
  python3 bench-atlas-concurrency.py --compare VLLM.json      # Compare with vLLM results
"""

import asyncio
import json
import os
import sys
import time
import statistics
import argparse

# ---- Configuration ----
SERVER_HOST = os.environ.get("BENCH_HOST", "localhost")
PORT = int(os.environ.get("BENCH_PORT", "8888"))
URL_BASE = f"http://{SERVER_HOST}:{PORT}"
URL_CHAT = f"{URL_BASE}/v1/chat/completions"
URL_MODELS = f"{URL_BASE}/v1/models"
MODEL = None

WARMUP_REQUESTS = 3
REQUESTS_PER_LEVEL = int(os.environ.get("BENCH_REQUESTS_PER_LEVEL", "0"))
RESULTS_FILE = os.environ.get("BENCH_RESULTS_FILE",
                              "/workspace/atlas/bench-atlas-concurrency-results.json")

# SSM state pool = 32 slots. Slots leak when pool is exhausted (server bug),
# so cap concurrency well below pool size. conc=16 leaves headroom.
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]

# Max sequence length for the server (ISL+OSL must fit).
MAX_SEQ_LEN = int(os.environ.get("BENCH_MAX_SEQ_LEN", "4096"))

_ALL_CONFIGS = [
    (1024,  128,  "prefill_short",  "Summarization short (NVIDIA NIM 1000/200 class)"),
    (8192,  1024, "prefill_long",   "RAG / document (SemiAnalysis 8K/1K class)"),
    (256,   256,  "balanced_short", "Short chat baseline"),
    (1024,  1024, "balanced_long",  "Standard chat (SemiAnalysis/SGLang 1K/1K)"),
    (128,   1024, "decode_short",   "Code generation (NVIDIA NIM 200/1000 class)"),
    (1024,  8192, "decode_long",    "Long reasoning (SemiAnalysis 1K/8K class)"),
]

# Filter out configs that exceed max_seq_len (ISL+OSL > limit).
# Order by ISL ascending so shorter prefills run first (less GPU state risk).
TEST_CONFIGS = sorted(
    [(i, o, r, l) for i, o, r, l in _ALL_CONFIGS if i + o <= MAX_SEQ_LEN],
    key=lambda x: x[0],
)

FILLER_WORD = "The quick brown fox jumps over the lazy dog. "
PROMPT_SUFFIX = ("\n\nProvide a very detailed and comprehensive analysis. "
                 "Do not stop early. Cover every aspect in depth.")


def make_prompt(target_tokens: int) -> str:
    chars_needed = target_tokens * 4
    repeats = max(1, chars_needed // len(FILLER_WORD))
    return f"Analyze the following text thoroughly:\n\n{(FILLER_WORD * repeats)[:chars_needed]}{PROMPT_SUFFIX}"


def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[-1]
    return s[f] + (k - f) * (s[c] - s[f])


async def detect_model():
    global MODEL
    import urllib.request
    for i in range(180):
        try:
            with urllib.request.urlopen(f"{URL_MODELS}", timeout=3) as r:
                data = json.loads(r.read())
                MODEL = data["data"][0]["id"]
                return True
        except Exception:
            if i % 12 == 0:
                print(f"  Waiting for server at {SERVER_HOST}:{PORT}... ({i*5}s)")
            await asyncio.sleep(5)
    return False


async def send_streaming_request(session, prompt: str, max_tokens: int):
    import aiohttp
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first_token = None
    t_last_token = None
    completion_tokens = 0
    prompt_tokens = 0

    try:
        async with session.post(URL_CHAT, json=payload,
                                timeout=aiohttp.ClientTimeout(total=600)) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {"error": f"HTTP {resp.status}: {body[:200]}"}

            buffer = ""
            async for chunk in resp.content.iter_any():
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                        choices = event.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                now = time.perf_counter()
                                if t_first_token is None:
                                    t_first_token = now
                                t_last_token = now
                                completion_tokens += 1
                        usage = event.get("usage")
                        if usage:
                            completion_tokens = usage.get("completion_tokens", completion_tokens)
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        return {"error": str(e)[:200]}

    t_end = time.perf_counter()
    total_time = t_end - t_start
    ttft = (t_first_token - t_start) if t_first_token else total_time
    decode_end = t_last_token if t_last_token else t_end
    decode_time = (decode_end - t_first_token) if t_first_token else total_time
    tpot = decode_time / (completion_tokens - 1) if completion_tokens > 1 else 0.0

    return {
        "ttft_ms": ttft * 1000,
        "tpot_ms": tpot * 1000,
        "e2e_throughput": completion_tokens / total_time if total_time > 0 else 0.0,
        "decode_throughput": completion_tokens / decode_time if decode_time > 0 else 0.0,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_time_s": total_time,
        "decode_time_s": decode_time,
    }


async def benchmark_config(session, isl: int, osl: int, regime: str, label: str) -> list:
    prompt = make_prompt(isl)
    config_results = []

    for conc in CONCURRENCY_LEVELS:
        total_requests = REQUESTS_PER_LEVEL if REQUESTS_PER_LEVEL > 0 else max(8, conc * 2)
        all_results = []
        rounds = (total_requests + conc - 1) // conc
        t_wall_start = time.perf_counter()

        for rnd in range(rounds):
            batch_size = min(conc, total_requests - len(all_results))
            if batch_size <= 0:
                break
            tasks = [send_streaming_request(session, prompt, osl) for _ in range(batch_size)]
            batch = await asyncio.gather(*tasks)
            all_results.extend(batch)

        wall_time = time.perf_counter() - t_wall_start
        good = [r for r in all_results if "error" not in r]
        errors = [r for r in all_results if "error" in r]

        if not good:
            print(f"    conc={conc:>3}: ALL FAILED ({len(errors)} errors)")
            if errors:
                print(f"           {errors[0]['error'][:100]}")
            config_results.append({"concurrency": conc, "status": "failed",
                                   "errors": len(errors),
                                   "error_sample": errors[0]["error"][:200] if errors else ""})
            continue

        ttfts = [r["ttft_ms"] for r in good]
        tpots = [r["tpot_ms"] for r in good if r["tpot_ms"] > 0]
        total_out = sum(r["completion_tokens"] for r in good)
        total_in = sum(r["prompt_tokens"] for r in good)
        agg_tput = total_out / wall_time if wall_time > 0 else 0
        req_s = len(good) / wall_time if wall_time > 0 else 0

        result = {
            "concurrency": conc, "status": "ok",
            "total_requests": len(all_results), "successful": len(good), "errors": len(errors),
            "rounds": rounds, "wall_time_s": round(wall_time, 2),
            "ttft_ms": {"p50": round(percentile(ttfts, 50), 1),
                        "p90": round(percentile(ttfts, 90), 1),
                        "p99": round(percentile(ttfts, 99), 1),
                        "avg": round(statistics.mean(ttfts), 1),
                        "min": round(min(ttfts), 1), "max": round(max(ttfts), 1)},
            "tpot_ms": {"p50": round(percentile(tpots, 50), 2),
                        "p90": round(percentile(tpots, 90), 2),
                        "p99": round(percentile(tpots, 99), 2),
                        "avg": round(statistics.mean(tpots), 2) if tpots else 0,
                        "min": round(min(tpots), 2) if tpots else 0,
                        "max": round(max(tpots), 2) if tpots else 0},
            "aggregate_throughput_tok_s": round(agg_tput, 1),
            "per_request_e2e_tok_s": round(statistics.mean([r["e2e_throughput"] for r in good]), 1),
            "per_request_decode_tok_s": round(statistics.mean([r["decode_throughput"] for r in good]), 1),
            "requests_per_sec": round(req_s, 2),
            "total_output_tokens": total_out, "total_input_tokens": total_in,
            "avg_output_tokens": round(total_out / len(good), 0),
            "avg_input_tokens": round(total_in / len(good), 0),
        }
        config_results.append(result)
        print(f"    conc={conc:>3}: "
              f"TTFT p50={result['ttft_ms']['p50']:>7.0f}ms p99={result['ttft_ms']['p99']:>7.0f}ms | "
              f"TPOT p50={result['tpot_ms']['p50']:>6.1f}ms p99={result['tpot_ms']['p99']:>6.1f}ms | "
              f"Agg={agg_tput:>6.1f} tok/s | Req/s={req_s:>5.2f} | n={len(good)}/{len(all_results)}")

    return config_results


def print_summary(all_results: dict):
    print("=" * 90)
    print("  SUMMARY — Aggregate Throughput (tok/s) by Concurrency")
    print("=" * 90)
    print()
    conc_hdr = "".join(f"{c:>8}" for c in CONCURRENCY_LEVELS)
    for metric_name, metric_key, fmt in [
        ("Agg Throughput (tok/s)", "aggregate_throughput_tok_s", "{:>8.1f}"),
        ("TTFT P99 (ms)", ("ttft_ms", "p99"), "{:>8.0f}"),
        ("TPOT P99 (ms)", ("tpot_ms", "p99"), "{:>8.1f}"),
    ]:
        print(f"  {metric_name:<25} |{conc_hdr}")
        print(f"  {'-'*25}-+{'-'*len(conc_hdr)}")
        for data in all_results.values():
            label = f"{data['isl']}/{data['osl']} {data['regime'][:7]}"
            values = ""
            for sp in data["concurrency_sweep"]:
                if sp.get("status") == "ok":
                    if isinstance(metric_key, tuple):
                        v = sp[metric_key[0]][metric_key[1]]
                    else:
                        v = sp[metric_key]
                    values += fmt.format(v)
                else:
                    values += f"{'FAIL':>8}"
            print(f"  {label:<25} |{values}")
        print()


async def run_benchmark():
    import aiohttp

    print("=" * 90)
    print("  Atlas Spark — Concurrency Sweep (SLAI Scheduling)")
    print("  6 configs x 7 concurrency levels = 42 runs")
    print("=" * 90)
    print(f"\n  Server: {SERVER_HOST}:{PORT}  |  Concurrency: {CONCURRENCY_LEVELS}")
    for isl, osl, regime, label in TEST_CONFIGS:
        print(f"    {isl:>5}/{osl:<5} [{regime}] {label}")
    print()

    print("Waiting for server...")
    if not await detect_model():
        print(f"ERROR: Server not ready at {SERVER_HOST}:{PORT} after 15 minutes")
        sys.exit(1)
    print(f"Model: {MODEL}\n")

    print(f"Warming up ({WARMUP_REQUESTS} requests)...")
    async with aiohttp.ClientSession() as session:
        for i in range(WARMUP_REQUESTS):
            r = await send_streaming_request(session, "Hello! Tell me about AI.", 50)
            if "error" in r:
                print(f"  warmup {i+1}: FAILED ({r['error'][:80]})")
            else:
                print(f"  warmup {i+1}: {r['e2e_throughput']:.1f} tok/s, {r['completion_tokens']} tok")
    print()

    all_results = {}
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0, limit_per_host=0)
    ) as session:
        for isl, osl, regime, label in TEST_CONFIGS:
            print(f"  === {regime.upper()} ({isl}/{osl}) — {label} ===")
            results = await benchmark_config(session, isl, osl, regime, label)
            all_results[f"{isl}/{osl}_{regime}"] = {
                "isl": isl, "osl": osl, "regime": regime, "label": label,
                "concurrency_sweep": results,
            }
            print()

    print_summary(all_results)

    output = {
        "label": "atlas-slai",
        "model": MODEL,
        "server": f"{SERVER_HOST}:{PORT}",
        "concurrency_levels": CONCURRENCY_LEVELS,
        "warmup_requests": WARMUP_REQUESTS,
        "test_configs": [{"isl": i, "osl": o, "regime": r, "label": l}
                         for i, o, r, l in TEST_CONFIGS],
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")


def compare_results(vllm_path: str):
    atlas_path = RESULTS_FILE
    if not os.path.exists(atlas_path):
        print(f"ERROR: Atlas results not found at {atlas_path}")
        print("Run the benchmark first, then use --compare.")
        sys.exit(1)
    if not os.path.exists(vllm_path):
        print(f"ERROR: vLLM results not found at {vllm_path}")
        sys.exit(1)

    with open(atlas_path) as f:
        atlas = json.load(f)
    with open(vllm_path) as f:
        vllm = json.load(f)

    print("=" * 110)
    print("  Atlas Spark (SLAI) vs vLLM — Side-by-Side Comparison")
    print("=" * 110)
    print()

    for config_key, atlas_data in atlas["results"].items():
        vllm_data = vllm["results"].get(config_key)
        if not vllm_data:
            continue

        regime = atlas_data["regime"]
        isl, osl = atlas_data["isl"], atlas_data["osl"]
        print(f"  === {regime.upper()} ({isl}/{osl}) ===")
        print(f"  {'Conc':>4} | {'Atlas tok/s':>11} {'vLLM tok/s':>11} {'Ratio':>7} | "
              f"{'Atlas TPOT p50':>14} {'vLLM TPOT p50':>14} {'Lat Ratio':>10} | "
              f"{'Atlas TTFT p50':>14} {'vLLM TTFT p50':>14}")
        print(f"  {'-'*4}-+-{'-'*11}-{'-'*11}-{'-'*7}-+-"
              f"{'-'*14}-{'-'*14}-{'-'*10}-+-{'-'*14}-{'-'*14}")

        a_sweep = {sp["concurrency"]: sp for sp in atlas_data["concurrency_sweep"]}
        v_sweep = {sp["concurrency"]: sp for sp in vllm_data["concurrency_sweep"]}

        for conc in CONCURRENCY_LEVELS:
            a = a_sweep.get(conc)
            v = v_sweep.get(conc)
            if not a or not v:
                continue
            if a.get("status") != "ok" or v.get("status") != "ok":
                print(f"  {conc:>4} | {'FAIL':>11} {'FAIL':>11} {'—':>7} | "
                      f"{'—':>14} {'—':>14} {'—':>10} | {'—':>14} {'—':>14}")
                continue

            a_tput = a["aggregate_throughput_tok_s"]
            v_tput = v["aggregate_throughput_tok_s"]
            ratio = a_tput / v_tput if v_tput > 0 else 0
            a_tpot = a["tpot_ms"]["p50"]
            v_tpot = v["tpot_ms"]["p50"]
            lat_ratio = v_tpot / a_tpot if a_tpot > 0 else 0
            a_ttft = a["ttft_ms"]["p50"]
            v_ttft = v["ttft_ms"]["p50"]

            print(f"  {conc:>4} | {a_tput:>11.1f} {v_tput:>11.1f} {ratio:>6.2f}x | "
                  f"{a_tpot:>13.1f}ms {v_tpot:>13.1f}ms {lat_ratio:>9.2f}x | "
                  f"{a_ttft:>13.0f}ms {v_ttft:>13.0f}ms")
        print()

    # Overall wins summary
    print("  WINS SUMMARY:")
    atlas_wins = 0
    vllm_wins = 0
    for config_key, atlas_data in atlas["results"].items():
        vllm_data = vllm["results"].get(config_key)
        if not vllm_data:
            continue
        for sp_a in atlas_data["concurrency_sweep"]:
            if sp_a.get("status") != "ok":
                continue
            sp_v = next((s for s in vllm_data["concurrency_sweep"]
                         if s["concurrency"] == sp_a["concurrency"] and s.get("status") == "ok"), None)
            if not sp_v:
                continue
            if sp_a["aggregate_throughput_tok_s"] > sp_v["aggregate_throughput_tok_s"]:
                atlas_wins += 1
            else:
                vllm_wins += 1
    print(f"    Atlas: {atlas_wins} wins  |  vLLM: {vllm_wins} wins  (aggregate throughput)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atlas Spark Concurrency Sweep Benchmark")
    parser.add_argument("--compare", metavar="VLLM_JSON",
                        help="Compare Atlas results with vLLM results JSON file")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare)
    else:
        asyncio.run(run_benchmark())
