#!/usr/bin/env python3
"""
bench_concurrency.py — Atlas Spark concurrency + latency benchmark.

Fires concurrent streaming requests and measures per-request and aggregate
metrics across ISL / concurrency sweeps.

Per-request metrics (client-measured via SSE):
  TTFT   Time To First Token (ms)   — request_start → first content token
  TPOT   Time Per Output Token (ms) — decode inter-token latency
           = (t_last_token − t_first_token) / (completion_tokens − 1)
  E2E    End-to-End latency (ms)    — request_start → last token received

Server-reported metrics (from SSE usage chunk):
  sTTFT  Server TTFT (ms)           — server-side prefill, excludes network RTT
  sTPS   Server decode tok/s        — server-side decode throughput

Aggregate metric (across concurrent batch):
  Tput   total_output_tokens / batch_wall_time  (tok/s)

All latency metrics reported as p50 / p90 / p99 across concurrent requests.

TTFT measurement notes (session-scoped SSM snapshots, 2026-03-27):

  Warmup runs prime the Marconi prefix cache. Subsequent runs at the same ISL
  reuse cached KV blocks + SSM snapshots (same session hash since same prompt).
  This means sTTFT after warmup reflects intra-session cache hit performance.

  Cross-session TTFT (new conversation, shared system prompt) is slower because
  SSM snapshots are session-gated: the cached KV blocks are reusable but the
  SSM state must be recomputed from scratch. To measure this, vary the prompt
  text between warmup and timed runs.

  See quick_bench.py docstring for full explanation of the session isolation
  mechanism and its impact on TTFT.

Usage:
  python bench_concurrency.py [--url URL] [--model MODEL] [--osl N] [--warmup N]
  python bench_concurrency.py --quick     # fast sweep: 3 ISLs × 2 concurrencies
"""

import argparse, json, sys, time, threading, statistics
from urllib.request import Request, urlopen
from urllib.error import HTTPError

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_URL   = "http://localhost:8888"
DEFAULT_MODEL = "Kbenkhaled/Qwen3.5-35B-A3B-NVFP4"
DEFAULT_OSL   = 128
DEFAULT_WARMUP = 1

ISLS  = [128, 512, 1024, 4096, 16384, 65536]
CONCS = [1, 4, 16]

# ── Result types ─────────────────────────────────────────────────────────────

class RequestResult:
    """Per-request timing collected from a single streaming call."""
    __slots__ = (
        "t_start", "t_first_token", "t_last_token", "t_end",
        "completion_tokens", "prompt_tokens",
        "server_ttft_ms", "server_tps",
        "error",
    )

    def __init__(self):
        self.t_start = self.t_first_token = self.t_last_token = self.t_end = None
        self.completion_tokens = self.prompt_tokens = 0
        self.server_ttft_ms: float | None = None
        self.server_tps: float | None = None
        self.error: str | None = None

    @property
    def ttft_ms(self) -> float | None:
        """Client-measured TTFT: request_start → first content token."""
        if self.t_first_token is not None:
            return (self.t_first_token - self.t_start) * 1000
        return None

    @property
    def tpot_ms(self) -> float | None:
        """Client-measured TPOT: decode phase duration / (tokens − 1)."""
        if (self.completion_tokens >= 2
                and self.t_first_token is not None
                and self.t_last_token is not None
                and self.t_last_token > self.t_first_token):
            return (self.t_last_token - self.t_first_token) / (self.completion_tokens - 1) * 1000
        return None

    @property
    def e2e_ms(self) -> float | None:
        """Client-measured E2E: request_start → last byte received."""
        if self.t_end is not None:
            return (self.t_end - self.t_start) * 1000
        return None

# ── HTTP / SSE ────────────────────────────────────────────────────────────────

def stream_request(url: str, model: str, isl: int, osl: int, result: RequestResult,
                   count_prompt: bool = False) -> None:
    """
    Send a streaming chat request and populate *result* with timing data.

    SSE event classification:
      role chunk   — delta has 'role' key, no 'content' key  → skip
      content chunk — delta has 'content' key (may be empty string for
                      mid-byte tokens)                        → record timing
      done chunk   — choices[0].finish_reason is set, usage present → extract server stats
      [DONE]       — sentinel, stop reading
    """
    prompt = make_prompt(isl, count_prompt)
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": osl,
        "stream": True,
        "temperature": 0.0,
    }).encode()

    result.t_start = time.perf_counter()
    try:
        req = Request(
            f"{url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").rstrip()
                if not line.startswith("data: "):
                    continue
                s = line[6:]
                if s == "[DONE]":
                    break
                try:
                    chunk = json.loads(s)
                except json.JSONDecodeError:
                    continue

                # Done chunk: usage present → extract server-side stats
                usage = chunk.get("usage")
                if usage:
                    result.server_ttft_ms = usage.get("time_to_first_token_ms")
                    result.server_tps     = usage.get("response_token/s")
                    result.prompt_tokens  = usage.get("prompt_tokens", result.prompt_tokens)
                    # Prefer server completion count (accurate); local count is fallback
                    sc = usage.get("completion_tokens")
                    if sc:
                        result.completion_tokens = sc
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                # Content chunk: 'content' key present (even if value is empty string)
                # Empty string = mid-byte token; still marks decode progress.
                if "content" in delta:
                    now = time.perf_counter()
                    if result.t_first_token is None:
                        result.t_first_token = now
                    result.t_last_token = now
                    result.completion_tokens += 1

    except Exception as exc:
        result.error = str(exc)
    finally:
        result.t_end = time.perf_counter()

# ── Benchmark runner ──────────────────────────────────────────────────────────

def make_prompt(isl_tokens: int, count_prompt: bool = False) -> str:
    """Build a prompt targeting ~isl_tokens total (chat template adds ~12 overhead).

    count_prompt=True appends a counting instruction so the model generates
    close to osl tokens instead of hitting EOS after a short greeting.

    Uses varied filler text instead of repeating "hello " — pure-attention
    models (e.g. VL-30B) produce degenerate EOS on uniform repetitive input.
    """
    # Varied filler: diverse vocabulary prevents attention collapse on pure-attn models
    _WORDS = (
        "The quick brown fox jumped over the lazy dog near a river bank. "
        "Mountains rise above the clouds while birds sing their morning songs. "
        "Science explores the universe through careful observation and experiment. "
        "Ancient civilizations built remarkable structures that still stand today. "
        "Music fills the air with rhythm and harmony across every culture. "
        "Technology advances rapidly changing how people communicate and work. "
        "Forests provide shelter for countless species of plants and animals. "
        "Ocean waves crash upon the shore under the light of the moon. "
    )
    needed = max(1, isl_tokens - 12)
    # Repeat the varied block enough times, then truncate to word boundary
    reps = (needed * 6) // len(_WORDS) + 2  # ~6 chars per token heuristic
    raw = (_WORDS * reps)
    words = raw.split()
    filler = " ".join(words[:needed])
    if count_prompt:
        return filler + " Count from 1 upward, one number per line, until told to stop."
    return filler


def pct(values: list[float], p: int) -> float:
    """p-th percentile (0–100) of a non-empty list."""
    if not values:
        return float("nan")
    sv = sorted(values)
    idx = min(int(len(sv) * p / 100 + 0.5), len(sv) - 1)
    return sv[idx]


def concurrent_bench(
    url: str, model: str, isl: int, osl: int, concurrency: int, warmup: int,
    count_prompt: bool = False,
) -> tuple[list[RequestResult], float]:
    """
    Run *concurrency* streaming requests simultaneously.

    All threads are synchronized at a barrier so they fire at the same instant,
    ensuring the scheduler sees a full batch and contention is realistic.

    Returns: (list of RequestResult, batch_wall_time_seconds)
    """
    # Warmup (sequential, not measured)
    for _ in range(warmup):
        r = RequestResult()
        stream_request(url, model, isl, osl, r, count_prompt)

    # Timed batch
    results  = [RequestResult() for _ in range(concurrency)]
    barrier  = threading.Barrier(concurrency)

    def _run(i: int) -> None:
        barrier.wait()  # synchronized start
        stream_request(url, model, isl, osl, results[i], count_prompt)

    threads = [threading.Thread(target=_run, args=(i,), daemon=True) for i in range(concurrency)]
    t_wall  = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_wall

    return results, wall

# ── Formatting ────────────────────────────────────────────────────────────────

def _p(values: list[float], p: int, fmt: str = ".0f") -> str:
    v = pct(values, p) if values else float("nan")
    return format(v, fmt) if v == v else "n/a"  # nan check


def fmt_lat(values: list[float], fmt: str = ".1f") -> str:
    """'p50 / p90 / p99' string or 'n/a'."""
    if not values:
        return "    n/a     "
    p50 = pct(values, 50)
    p90 = pct(values, 90)
    p99 = pct(values, 99)
    return f"{p50:{fmt}} / {p90:{fmt}} / {p99:{fmt}}"

# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(
    url: str, model: str, osl: int,
    isls: list[int], concs: list[int], warmup: int,
    count_prompt: bool = False,
) -> None:
    # Column widths chosen so the table is ~120 chars
    print(
        f"{'ISL':>6}  {'OSL':>5}  {'Conc':>4}  {'PTok':>5}  "
        f"{'Tput':>8}  "
        f"{'TTFT p50/p90/p99 ms':^22}  "
        f"{'TPOT p50/p90/p99 ms':^22}  "
        f"{'E2E p50 ms':>10}  "
        f"{'sTTFT p50 ms':>12}  "
        f"{'sTPS p50':>8}"
    )
    print("-" * 120)

    for isl in isls:
        for conc in concs:
            try:
                results, wall = concurrent_bench(url, model, isl, osl, conc, warmup, count_prompt)
            except Exception as exc:
                print(f"{isl:>6}  {osl:>5}  {conc:>4}  ERROR: {exc}")
                continue

            errors = [r for r in results if r.error]
            ok     = [r for r in results if not r.error]

            if errors:
                msg = errors[0].error or "unknown"
                print(f"{isl:>6}  {osl:>5}  {conc:>4}  "
                      f"{len(errors)} error(s): {msg[:60]}")
                continue

            total_out = sum(r.completion_tokens for r in ok)
            tput      = total_out / wall if wall > 0 else 0.0
            ptok      = ok[0].prompt_tokens if ok else 0

            ttfts  = [r.ttft_ms  for r in ok if r.ttft_ms  is not None]
            tpots  = [r.tpot_ms  for r in ok if r.tpot_ms  is not None]
            e2es   = [r.e2e_ms   for r in ok if r.e2e_ms   is not None]
            sttfts = [r.server_ttft_ms for r in ok if r.server_ttft_ms is not None]
            stpss  = [r.server_tps     for r in ok if r.server_tps     is not None]

            e2e_p50  = _p(e2es,   50, ".0f")
            sttft_p50 = _p(sttfts, 50, ".1f")
            stps_p50  = _p(stpss,  50, ".1f")

            print(
                f"{isl:>6}  {osl:>5}  {conc:>4}  {ptok:>5}  "
                f"{tput:>7.1f}t  "
                f"{fmt_lat(ttfts):^22}  "
                f"{fmt_lat(tpots, '.2f'):^22}  "
                f"{e2e_p50:>10}  "
                f"{sttft_p50:>12}  "
                f"{stps_p50:>8}"
            )
            sys.stdout.flush()
            time.sleep(1)  # brief pause between configs so GPU can settle

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url",    default=DEFAULT_URL,   help="Server base URL")
    ap.add_argument("--model",  default=DEFAULT_MODEL, help="Model name/ID")
    ap.add_argument("--osl",    type=int, default=DEFAULT_OSL,
                    help="Max output tokens per request (default: 128)")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                    help="Warmup requests per config (default: 1)")
    ap.add_argument("--isls",   nargs="+", type=int, default=ISLS,
                    help="ISL values to sweep")
    ap.add_argument("--concs",  nargs="+", type=int, default=CONCS,
                    help="Concurrency levels to sweep")
    ap.add_argument("--quick",  action="store_true",
                    help="Fast sweep: ISL=[128,1024,4096], Conc=[1,4]")
    ap.add_argument("--no-count-prompt", action="store_true",
                    help="Use natural prompts (model may EOS early). "
                         "Default is --count-prompt which forces full OSL output "
                         "for steady-state decode throughput measurement.")
    args = ap.parse_args()

    isls  = [128, 1024, 4096] if args.quick else args.isls
    concs = [1, 4]            if args.quick else args.concs

    # Health check
    try:
        urlopen(f"{args.url}/health", timeout=5).read()
    except Exception as exc:
        print(f"ERROR: server not reachable at {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Atlas Spark — Concurrency Benchmark")
    print(f"  Model  : {args.model}")
    print(f"  URL    : {args.url}")
    print(f"  OSL    : {args.osl} max output tokens per request")
    print(f"  Warmup : {args.warmup} request(s) per configuration")
    count_prompt = not args.no_count_prompt
    print(f"  Prompt : {'count (forces full OSL)' if count_prompt else 'hello (natural EOS)'}")
    print()
    print("  TTFT  = client Time To First Token  (prefill latency)")
    print("  TPOT  = client Time Per Output Token (decode inter-token latency)")
    print("  E2E   = client end-to-end latency    (start → last token)")
    print("  sTTFT = server TTFT (server-side, excludes network RTT)")
    print("  sTPS  = server decode tok/s")
    print("  Tput  = aggregate output tok/s across concurrent batch")
    print()
    run_sweep(args.url, args.model, args.osl, isls, concs, args.warmup, count_prompt)


if __name__ == "__main__":
    main()
