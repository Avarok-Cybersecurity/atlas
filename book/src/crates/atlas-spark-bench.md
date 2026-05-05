# atlas-spark-bench

**Role:** the benchmark harness that produces every throughput number in this book. HTTP client that targets a running Atlas Spark server and measures token rate, TTFT, and concurrency behaviour.
**Key file:** `src/lib.rs`.

## Design

This is a **client-side** harness. It does not link against `spark-runtime` or `atlas-kernels` — it just speaks OpenAI-compatible HTTP to a server on `localhost:8888` (or wherever `ATLAS_BENCH_URL` points).

That shape is deliberate:

1. **No GPU pollution.** Running benches from the same process that serves the model would compete for GPU memory and distort results.
2. **Same surface as real clients.** If the HTTP stack has a latency bug, `atlas-spark-bench` sees it. If the tokenizer is slow on some prompt shape, it shows up.
3. **Same harness tests correctness *and* perf.** The integration tests in `tests/` reuse the client to drive coherence checks against every model.

## What it measures

| Metric | How |
|---|---|
| Decode throughput (tok/s) | Time between first token and last token, divided by `n_output_tokens` |
| Time-to-first-token (TTFT) | Time from request send to first SSE chunk |
| Sustained concurrency | N parallel streams, each measured independently; aggregated into p50/p95 |
| Prefix-cache hit rate | Inferred by comparing TTFT for a request that shares a prefix with an earlier one |

The harness uses `ureq` for blocking requests and `std::thread::Barrier` for synchronising concurrent launches. No async — simpler, more reproducible.

## The bench shapes

The canonical shapes live in `bench/` at the repo root (the harness loads JSON fixtures) and include:

- **Short prompt, short output** — "What is the capital of France?", `max_tokens ≤ 30`. This is the number the READMEs quote: it emphasizes the decode hot loop, not prefill.
- **Long output, single request** — "Explain the theory of relativity", `max_tokens = 200`. Shows CUDA-graph sustained throughput.
- **Concurrency sweep** — the same prompt, 1 / 2 / 4 / 8 / 16 parallel streams. Reveals scheduler + KV-allocation behaviour.
- **Prefix warmup** — preflights the system prompt, then measures cold vs warm TTFT.
- **Tool-calling** — a single-tool request with a well-known function signature; measures tool-emit latency and token-streaming behaviour during tool blocks.

## Running a bench

```bash
# Start a server somewhere
docker run -d ... avarok/atlas-gb10:latest serve <model>

# In another terminal
export ATLAS_BENCH_URL=http://localhost:8888
cargo bench -p atlas-spark-bench
```

Criterion stores results under `target/criterion/`. The repo's `bench/` directory retains the stable JSON snapshots that the README tables are derived from; ephemeral bench runs are gitignored.

## Scripts that drive it

- `scripts/sweep_all_models.sh` — boots each model in turn, runs the short-prompt bench, and writes the `README.md` throughput table.
- `scripts/bench-concurrency.sh` — runs the N-stream sweep.
- `scripts/test-minimax-ep2.sh` — doubles as perf + correctness for EP=2 MiniMax.

All three live in the top-level `scripts/` directory, not in the crate.

## `require_server()` — the safety rail

```rust
pub fn require_server() -> String {
    let url = server_url();
    match ureq::get(&format!("{url}/health")).call() {
        Ok(resp) if resp.status() == 200 => url,
        _ => panic!("Server not reachable at {url}. Start Atlas Spark first."),
    }
}
```

Every bench starts with this. Failing fast if the server isn't up beats a confusing timeout minutes into a run.

## Where results live

- **Hand-vetted snapshots:** `bench/*.json` (gitignored at the file level but some stable ones are tracked). These feed the [Benchmarks](../operations/benchmarks.md) chapter.
- **Criterion outputs:** `target/criterion/` (gitignored).
- **Concurrency-sweep logs:** `CONCURRENCY_SWEEP_RESULTS.md` in the repo root (tracked, updated manually when a significant run completes).

## What this crate is not

- Not a generic LLM benchmark tool. It knows about Atlas's server and its SSE streaming format.
- Not a load tester. For that you want `vegeta` or `locust` pointed at the same server.
- Not a kernel micro-benchmark. Those live in `atlas-spark-bench`'s sister tests under each primitive crate's `benches/`, Criterion-driven, no HTTP.

The job of `atlas-spark-bench` is to produce **end-to-end**, **apples-to-apples** numbers that survive the tokenizer, the scheduler, the HTTP layer, and the kernel set. When the README says "131 tok/s on Qwen3.5-35B", that number came from here.
