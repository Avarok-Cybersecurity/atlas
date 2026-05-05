#!/usr/bin/env python3
"""
quick_bench.py — Fast single-user Atlas benchmark.

1 warmup + 5 timed runs with a fixed prompt. Reports average tok/s, TTFT,
server TPS, TPOT, and E2E latency.

Usage:
  python quick_bench.py                          # short prompt (~60 tokens)
  python quick_bench.py --isl 4096               # 4096-token prompt (uses fixture)
  python quick_bench.py --isl 128                # 128-token prompt
  python quick_bench.py --url http://host:port   # custom endpoint
  python quick_bench.py --osl 256                # longer output
  python quick_bench.py --runs 10                # more runs

TTFT measurement notes (session-scoped SSM snapshots, 2026-03-27):

  Atlas uses Marconi prefix caching with per-session SSM snapshot isolation.
  SSM snapshots are tagged with a session_hash (hash of first 64 prompt tokens).
  Cross-session snapshot restore is rejected — the model recomputes SSM state
  from scratch, which means:

  - Run 1 (cold): Full prefill, no cache. TTFT = prefill time.
  - Run 2+ (same session/prompt): Prefix cache HIT + SSM snapshot HIT.
    TTFT ≈ 50-100ms (near-instant, just LM head + first token).
  - Run 1 of NEW session (different first user message): Prefix cache HIT
    for shared system prompt, but SSM snapshot MISS (different session hash).
    TTFT = SSM recompute time for the shared prefix. Slower than intra-session
    cache hit, faster than fully cold start.

  To benchmark cross-session TTFT: vary the first user message between runs.
  To benchmark intra-session TTFT: keep the same prompt across runs (default).
  The warmup run primes the prefix cache; subsequent runs benefit from it.
"""

import argparse, json, os, sys, time
from urllib.request import Request, urlopen

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(SCRIPT_DIR, "tests", "fixtures")

DEFAULT_PROMPT = (
    "The quick brown fox jumped over the lazy dog near a river bank. "
    "Mountains rise above the clouds while birds sing their morning songs. "
    "Science explores the universe through careful observation and experiment. "
    "Ancient civilizations built remarkable structures that still stand today. "
    "Count from 1 upward, one number per line, until told to stop."
)

# Varied filler for ISLs without a fixture file (~6 chars per token)
_FILLER = (
    "The quick brown fox jumped over the lazy dog near a river bank. "
    "Mountains rise above the clouds while birds sing their morning songs. "
    "Science explores the universe through careful observation and experiment. "
    "Ancient civilizations built remarkable structures that still stand today. "
    "Music fills the air with rhythm and harmony across every culture. "
    "Technology advances rapidly changing how people communicate and work. "
    "Forests provide shelter for countless species of plants and animals. "
    "Ocean waves crash upon the shore under the light of the moon. "
)


def load_prompt(isl: int | None) -> str:
    if isl is None:
        return DEFAULT_PROMPT

    # Try fixture file first
    fixture = os.path.join(FIXTURES_DIR, f"bench_prompt_{isl}.txt")
    if os.path.exists(fixture):
        with open(fixture) as f:
            text = f.read().strip()
        return text + "\nCount from 1 upward, one number per line, until told to stop."

    # Fallback: pad with filler words (~6 chars per token)
    needed = max(1, isl - 12)
    reps = (needed * 6) // len(_FILLER) + 2
    raw = _FILLER * reps
    words = raw.split()
    filler = " ".join(words[:needed])
    return filler + " Count from 1 upward, one number per line, until told to stop."


def stream_one(url: str, model: str, prompt: str, osl: int) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": osl,
        "stream": True,
        "temperature": 0.0,
        "enable_thinking": False,
    }).encode()

    t_start = time.perf_counter()
    t_first = t_last = None
    comp_tokens = 0
    prompt_tokens = 0
    server_ttft = server_tps = None

    req = Request(f"{url}/v1/chat/completions", data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=300) as resp:
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
            if "content" in delta:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                comp_tokens += 1

    t_end = time.perf_counter()
    e2e = (t_end - t_start) * 1000
    ttft = (t_first - t_start) * 1000 if t_first else None
    tpot = ((t_last - t_first) / (comp_tokens - 1) * 1000
            if t_first and t_last and comp_tokens >= 2 else None)
    tps = comp_tokens / (t_end - t_start) if comp_tokens > 0 else 0

    return {
        "tokens": comp_tokens, "prompt_tokens": prompt_tokens,
        "tok/s": tps, "ttft_ms": ttft,
        "tpot_ms": tpot, "e2e_ms": e2e,
        "server_ttft_ms": server_ttft, "server_tps": server_tps,
    }


def main():
    ap = argparse.ArgumentParser(description="Quick Atlas benchmark")
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--model", default=None)
    ap.add_argument("--isl", type=int, default=None,
                    help="Input sequence length (loads fixture or pads prompt)")
    ap.add_argument("--osl", type=int, default=128)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    # Auto-detect model
    model = args.model
    if not model:
        try:
            r = urlopen(f"{args.url}/v1/models", timeout=5)
            data = json.loads(r.read())
            model = data["data"][0]["id"]
        except Exception:
            model = "default"

    prompt = load_prompt(args.isl)
    isl_label = f"ISL: ~{args.isl}" if args.isl else "ISL: ~60 (default)"

    print(f"Atlas Quick Bench — {model}")
    print(f"  URL: {args.url}  {isl_label}  OSL: {args.osl}  Runs: {args.runs}")
    print()

    # Warmup
    for i in range(args.warmup):
        sys.stdout.write(f"  Warmup {i+1}/{args.warmup}...")
        sys.stdout.flush()
        r = stream_one(args.url, model, prompt, args.osl)
        print(f" done ({r.get('prompt_tokens', '?')} prompt tokens)")

    # Timed runs
    results = []
    for i in range(args.runs):
        sys.stdout.write(f"  Run {i+1}/{args.runs}...")
        sys.stdout.flush()
        r = stream_one(args.url, model, prompt, args.osl)
        results.append(r)
        s_ttft = r.get('server_ttft_ms', 0)
        s_tps = r.get('server_tps', 0)
        print(f" {r['tokens']} tok, {s_tps:.1f} tok/s, TTFT {s_ttft:.0f}ms")

    # Averages
    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    print()
    print(f"  ── Results ({args.runs} runs) ──")
    print(f"  Prompt tok : {avg('prompt_tokens'):.0f}")
    print(f"  Output tok : {avg('tokens'):.0f}")
    print(f"  tok/s      : {avg('tok/s'):.1f} (client)")
    s_tps = avg('server_tps')
    if s_tps is not None:
        print(f"  tok/s      : {s_tps:.1f} (server)")
    s_ttft = avg('server_ttft_ms')
    if s_ttft is not None:
        print(f"  TTFT       : {s_ttft:.1f} ms (server prefill)")
    print(f"  TPOT       : {avg('tpot_ms'):.1f} ms")
    print(f"  E2E        : {avg('e2e_ms'):.0f} ms")


if __name__ == "__main__":
    main()
