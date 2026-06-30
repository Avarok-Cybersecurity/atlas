#!/usr/bin/env python3
"""
bench_findings.py — Benchmark for FINDINGS-27B-ATLAS.md table.

Measures: tok/s, TTFT, total/thinking/output tokens, accept/step, accept_%, verify_mult, propose_ms.
Reads accept metrics from server log at /tmp/atlas_bench_server.log.

Usage:
  python bench_findings.py --label "MTP fp8"              # 1 warmup + 2 runs
  python bench_findings.py --label "DFlash cap=1" --runs 3
  python bench_findings.py --url http://localhost:8888

Output: one result dict per run + summary row for the FINDINGS table.
"""

import argparse, json, os, re, sys, time
from urllib.request import Request, urlopen

PROMPT = (
    "Write a Python implementation of a min-heap with insert, extract_min, "
    "heapify (O(n) build from list), and peek. "
    "Show time complexity for each operation. "
    "Include a complete working demo with at least 10 elements."
)

SERVER_LOG = "/tmp/atlas_bench_server.log"


def stream_one(url: str, model: str, osl: int, temperature: float = 0.0) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": osl,
        "stream": True,
        "temperature": temperature,
    }).encode()

    t_start = time.perf_counter()
    t_first = t_last = None
    content_chunks = []
    comp_tokens = 0
    prompt_tokens = 0
    server_ttft = server_tps = None

    req = Request(f"{url}/v1/chat/completions", data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8").rstrip()
            if not line.startswith("data: "):
                continue
            s = line[6:]
            if s == "[DONE]":
                break
            try:
                chunk = json.loads(s)
            except json.JSONDecodeError:
                continue

            usage = chunk.get("usage")
            if usage:
                server_ttft = usage.get("time_to_first_token_ms")
                server_tps = usage.get("response_token/s")
                sc = usage.get("completion_tokens")
                if sc:
                    comp_tokens = sc
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            text = delta.get("content", "")
            if text:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                content_chunks.append(text)
                comp_tokens += 1

    full_text = "".join(content_chunks)
    t_end = time.perf_counter()

    # Count thinking tokens: characters before </think> divided by avg chars/token
    thinking_chars = 0
    think_end = full_text.find("</think>")
    if think_end >= 0:
        thinking_chars = think_end + len("</think>")
    thinking_tok = round(thinking_chars / 3.7) if thinking_chars > 0 else 0

    tps = comp_tokens / (t_end - t_start) if comp_tokens > 0 else 0
    ttft = (t_first - t_start) * 1000 if t_first else None

    return {
        "tokens": comp_tokens,
        "prompt_tokens": prompt_tokens,
        "tok/s": tps,
        "ttft_ms": ttft,
        "e2e_ms": (t_end - t_start) * 1000,
        "server_ttft_ms": server_ttft,
        "server_tps": server_tps,
        "thinking_tok": thinking_tok,
        "output_tok": max(0, comp_tokens - thinking_tok),
        "full_text": full_text,
    }


def read_log_metrics(log_start_line: int = 0) -> dict:
    """Grep server log for acceptance and timing metrics from the current run window.

    log_start_line: line index (0-based) where the timed runs started. Metrics
    are collected from that line onward so warmup steps don't pollute results.
    """
    if not os.path.exists(SERVER_LOG):
        return {}

    with open(SERVER_LOG, errors="replace") as f:
        all_lines = f.readlines()

    run_lines = all_lines[log_start_line:]
    result = {}

    # MTP verify_multiplier: last occurrence in run window.
    for line in reversed(run_lines):
        m = re.search(r"verify_multiplier[=:]\s*([\d.]+)", line)
        if m:
            result["verify_mult"] = float(m.group(1))
            break

    # K2 acceptance rate: AVERAGE over all K2 summary/drift lines in run window.
    # Two log formats:
    #   INFO  "K2 summary: N accept / M reject in last T steps (X.X% accept)"
    #   WARN  "K2 drift gauge: accept rate X.X% < 30%..."
    # Taking the last line only is noisy (~100-step sample); averaging gives a
    # stable estimate of output-phase acceptance (DFlash is disabled during
    # thinking, so K2 gauge only fires on output tokens).
    k2_rates = []
    for line in run_lines:
        # INFO format: "(X.X% accept)"
        m = re.search(r"\(([\d.]+)%\s+accept\)", line)
        if m:
            k2_rates.append(float(m.group(1)))
            continue
        # WARN/drift format: "accept rate X.X%"
        m = re.search(r"accept.?rate[=:\s]+([\d.]+)%?", line, re.IGNORECASE)
        if m:
            k2_rates.append(float(m.group(1)))
    if k2_rates:
        result["accept_rate_pct"] = round(sum(k2_rates) / len(k2_rates), 1)
        result["accept_rate_pct_samples"] = len(k2_rates)

    # DFlash propose_ms: median of all propose timing lines in run window.
    propose_vals = []
    for line in run_lines:
        m = re.search(r"propose[=:\s]+([\d.]+)\s*ms", line, re.IGNORECASE)
        if m:
            propose_vals.append(float(m.group(1)))
    if propose_vals:
        propose_vals.sort()
        result["propose_ms"] = propose_vals[len(propose_vals) // 2]

    # DFlash accept/step from step logs: "accepted=N/16" or "accept_len=X.XX"
    accept_lens = []
    for line in run_lines:
        m = re.search(r"accepted=(\d+)/\d+", line)
        if m:
            accept_lens.append(int(m.group(1)))
        m2 = re.search(r"accept_len[=:\s]+([\d.]+)", line, re.IGNORECASE)
        if m2:
            accept_lens.append(float(m2.group(1)))

    if accept_lens:
        result["accept_per_step"] = round(sum(accept_lens) / len(accept_lens), 2)
    elif "accept_rate_pct" in result:
        # MTP K=2: accept/step = draft accept rate (v0 always emitted, v1 conditional)
        result["accept_per_step"] = round(result["accept_rate_pct"] / 100, 2)

    return result


def avg(results, key):
    vals = [r[key] for r in results if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(v, decimals=1, suffix=""):
    if v is None:
        return "—"
    return f"{v:.{decimals}f}{suffix}"


def main():
    ap = argparse.ArgumentParser(description="FINDINGS benchmark")
    ap.add_argument("--label", default="Config", help="Config label for table")
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--osl", type=int, default=700)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    # Auto-detect model
    try:
        r = urlopen(f"{args.url}/v1/models", timeout=5)
        data = json.loads(r.read())
        model = data["data"][0]["id"]
    except Exception:
        model = "default"

    print(f"\n{'='*60}")
    print(f"Config: {args.label}  |  Model: {model}  |  T={args.temperature}")
    print(f"URL: {args.url}  |  Runs: {args.runs}  |  OSL: {args.osl}")
    print(f"{'='*60}")
    print(f"\nPrompt: {PROMPT[:80]}...")
    print()

    warmup_runs = 0 if args.no_warmup else args.warmup
    for i in range(warmup_runs):
        sys.stdout.write(f"  Warmup {i+1}/{warmup_runs}... ")
        sys.stdout.flush()
        r = stream_one(args.url, model, args.osl, args.temperature)
        print(f"{r['tokens']} tok, {r['tok/s']:.1f} tok/s")

    # Mark log position before timed runs
    log_start_line = 0
    if os.path.exists(SERVER_LOG):
        with open(SERVER_LOG, errors="replace") as f:
            log_start_line = sum(1 for _ in f)

    results = []
    for i in range(args.runs):
        sys.stdout.write(f"  Run {i+1}/{args.runs}... ")
        sys.stdout.flush()
        r = stream_one(args.url, model, args.osl, args.temperature)
        results.append(r)
        ttft_str = f"{r['server_ttft_ms']:.0f}ms" if r.get('server_ttft_ms') else "—"
        stps = r.get('server_tps')
        stps_str = f"{stps:.1f}" if stps else "—"
        print(f"{r['tokens']} tok, {stps_str} tok/s server, TTFT {ttft_str}, "
              f"think={r['thinking_tok']} out={r['output_tok']}")

    # Read log metrics (from run window only — after warmup)
    log_metrics = read_log_metrics(log_start_line)

    print()
    print("── Summary ──")
    toks = avg(results, 'tokens')
    server_tps = avg(results, 'server_tps')
    client_tps = avg(results, 'tok/s')
    ttft = avg(results, 'server_ttft_ms')
    thinking = avg(results, 'thinking_tok')
    output = avg(results, 'output_tok')

    print(f"  tok/s (server): {fmt(server_tps)}")
    print(f"  tok/s (client): {fmt(client_tps)}")
    print(f"  TTFT          : {fmt(ttft, 0, 'ms')}")
    print(f"  total_tok     : {fmt(toks, 0)}")
    print(f"  thinking_tok  : {fmt(thinking, 0)}")
    print(f"  output_tok    : {fmt(output, 0)}")
    n_samples = log_metrics.get('accept_rate_pct_samples', 0)
    samples_note = f" (avg over {n_samples} K2 batches)" if n_samples else ""
    print(f"  accept/step   : {fmt(log_metrics.get('accept_per_step'), 2)}")
    print(f"  accept_%      : {fmt(log_metrics.get('accept_rate_pct'), 1, '%')}{samples_note}")
    print(f"  verify_mult   : {fmt(log_metrics.get('verify_mult'), 2)}")
    print(f"  propose_ms    : {fmt(log_metrics.get('propose_ms'), 0, 'ms')}")

    # FINDINGS table row
    print()
    print("── FINDINGS table row ──")
    row = (
        f"| {args.label} "
        f"| {fmt(server_tps)} "
        f"| {fmt(ttft, 0)} "
        f"| {fmt(toks, 0)} "
        f"| {fmt(thinking, 0)} "
        f"| {fmt(output, 0)} "
        f"| {fmt(log_metrics.get('accept_per_step'), 2)} "
        f"| {fmt(log_metrics.get('accept_rate_pct'), 1)} "
        f"| {fmt(log_metrics.get('verify_mult'), 2)} "
        f"| {fmt(log_metrics.get('propose_ms'), 0)} "
        f"|"
    )
    print(row)
    print()

    if results:
        print("── Output sample (first 300 chars) ──")
        print(results[-1].get("full_text", "")[:300])
        print()


if __name__ == "__main__":
    main()
