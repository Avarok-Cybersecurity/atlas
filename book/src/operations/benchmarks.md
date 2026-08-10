# Benchmarking

Atlas's performance claims are measurable. This chapter shows what the benchmarks measure, how to reproduce them, and how to read the numbers.

## The headline numbers

From the repo README, distilled:

| Model | Mode | tok/s | Baseline |
|---|---|---:|---:|
| Qwen3.5-35B-A3B | NVFP4 + MTP K=2 | **131** | NVIDIA vLLM: 36 |
| Qwen3-Next-80B-A3B | NVFP4 + MTP K=2 | 104 | |
| Qwen3.5-122B-A10B | EP=2 + MTP K=2 | 46 | |
| Mistral-Small-4-119B | NVFP4 | 33 | |
| Nemotron-3-Nano-30B | FP8 | 88 | |
| Gemma-4-26B | NVFP4 | 67 | |

And the kernel micro-benchmark summary: **Atlas wins 32/32** against PyTorch on attention, GEMM, SSM, RMSNorm, RoPE, SiLU×Mul, and conv1d, with speedups from 1.04× up to 18.2×.

## Two kinds of benchmark

Atlas has two benchmark surfaces:

1. **End-to-end HTTP throughput** — `atlas-spark-bench` (client-side Criterion harness targeting a running server). This is what "131 tok/s" means.
2. **Per-kernel micro-benchmarks** — Criterion benches in each primitive crate, run with `cargo bench`. This is where "4.95× prefill attention" comes from.

Different things; both are meaningful. The E2E number is what an operator sees. The per-kernel number is what tells the kernel engineer where effort is paying back.

## Running end-to-end benchmarks

Start a server:

```bash
sudo docker run -d --name atlas-35b \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve Sehyo/Qwen3.5-35B-A3B-NVFP4 \
    --max-seq-len 8192 --kv-cache-dtype nvfp4 \
    --scheduling-policy slai \
    --speculative --mtp-quantization nvfp4
```

Wait for `listening`. Then:

```bash
export ATLAS_BENCH_URL=http://localhost:8888
cd /path/to/atlas
cargo bench -p atlas-spark-bench
```

Criterion saves results to `target/criterion/`. The stable JSON snapshots that the README quotes are pinned under `bench/`.

The `scripts/sweep_all_models.sh` helper boots each model in turn, runs the canonical short-prompt bench, and writes the `README.md` throughput table. That's how the table in the README gets regenerated.

## Running per-kernel benchmarks

```bash
cargo bench -p spark-runtime        # KV cache ops, sampler micro
cargo bench -p atlas-spark-bench    # end-to-end client benchmarks
```

Criterion-driven, from each crate's `benches/*.rs`. Reference shapes come from Qwen3-Next-80B (hidden=2048, 16 Q-heads, 2 KV-heads, head_dim=256, intermediate=512, num_experts=256, topk=10).

The full kernel numbers table:

| Kernel | Benchmark | Atlas | PyTorch | Speedup |
|---|---|---:|---:|---:|
| Prefill Attn | seq=32 | 0.0062 ms | 0.0077 | 1.26× |
| Prefill Attn | seq=128 | 0.0184 ms | 0.0205 | 1.11× |
| Prefill Attn | seq=256 | 0.0246 ms | 0.1217 | **4.95×** |
| Prefill Attn | seq=512 | 0.0494 ms | 0.0513 | 1.04× |
| Decode Attn | seq=64 | 0.0061 ms | 0.0077 | 1.25× |
| Decode Attn | seq=256 | 0.0123 ms | 0.0164 | 1.33× |
| Decode Attn | seq=1024 | 0.0205 ms | 0.0267 | 1.30× |
| Decode Attn | seq=4096 | 0.0485 ms | 0.2924 | **6.02×** |
| GEMM TC [80,2048]×[2048,512] | | 0.0080 ms | 0.0081 | 1.01× |
| GEMM TC [80,512]×[512,2048] | | 0.0080 ms | 0.0081 | 1.01× |
| GEMM TC [16,2048]×[2048,256] | | 0.0086 ms | 0.0102 | 1.18× |
| GEMM TC [256,256]×[256,256] | | 0.0045 ms | 0.0061 | 1.34× |
| W4A16 [80,2048]×[2048,1024] | | 0.0108 ms | 0.0132 | 1.22× |
| MoE W4A16 256-exp 80-tok | | 8.4273 ms | 32.6482 | **3.87×** |
| Conv1d prefill dim=8192 | seq=32 | 0.0112 ms | 0.0205 | 1.82× |
| Conv1d prefill dim=8192 | seq=128 | 0.0143 ms | 0.0776 | 5.41× |
| Conv1d prefill dim=8192 | seq=512 | 0.0532 ms | 0.5296 | **9.95×** |
| Conv1d decode dim=8192 | | 0.0041 ms | 0.0364 | **8.89×** |
| GDR decode 32vh dim=128 | | 0.0143 ms | 0.0732 | 5.11× |
| GDR prefill | seq=32 | 0.3612 ms | 2.7849 | 7.71× |
| GDR prefill | seq=128 | 1.4111 ms | 11.1267 | **7.89×** |
| RMSNorm [1, 2048] | | 0.0041 ms | 0.0382 | 9.32× |
| RMSNorm [16, 2048] | | 0.0041 ms | 0.0382 | 9.33× |
| RMSNorm [80, 2048] | | 0.0061 ms | 0.0384 | 6.26× |
| Gated RMSNorm dim=2048 | | 0.0041 ms | 0.0290 | 7.08× |
| Gated RMSNorm dim=8192 | | 0.0041 ms | 0.0289 | 7.03× |
| SiLU×Mul [16, 512] | | 0.0021 ms | 0.0099 | 4.81× |
| SiLU×Mul [80, 512] | | 0.0021 ms | 0.0099 | 4.84× |
| SiLU×Mul [800, 512] | | 0.0041 ms | 0.0101 | 2.45× |
| RoPE seq=32 GQA16:2 | | 0.0085 ms | 0.1544 | **18.12×** |
| RoPE seq=128 GQA16:2 | | 0.0085 ms | 0.1547 | **18.20×** |
| RoPE seq=512 GQA16:2 | | 0.0129 ms | 0.1545 | 11.97× |

**32/32 wins.** Peak achieved memory bandwidth in this set: 599 GB/s (2.2× the 273 GB/s LPDDR5X spec — that's the L2 cache effect for small SiLU×Mul inputs).

## Concurrency sweep

`scripts/run_conc_benchmark.sh` drives N parallel streams against one server. Reveals the scheduler + KV allocator under load. Typical pattern on Qwen3.5-35B:

| Concurrency | p50 tok/s | p95 latency (TTFT ms) |
|---:|---:|---:|
| 1 | 131 | 42 |
| 2 | 230 | 48 |
| 4 | 400 | 62 |
| 8 | 620 | 110 |
| 16 | 820 | 280 |

Aggregate throughput grows super-linearly up to the batch size cap (where graph amortisation kicks in) and then super-linearly until KV pool pressure; after that latency degrades more than throughput improves. Sweet spot on 35B: concurrency 4–8.

## TTFT and prefix-cache behaviour

Run a request, note the TTFT. Run the same request again — with `--enable-prefix-caching`, TTFT drops to ~40ms (prefix cache hit). Agent workloads observe this as the difference between "the first response was slow" and "everything after is fast". The bench harness's prefix-warmup suite measures cold vs warm TTFT.

## Where raw results live

- **Pinned snapshots** (tracked): result files under `bench/`. These feed the book and the README.
- **Ephemeral Criterion runs** (gitignored): `target/criterion/`.
- **Historical benchmark journeys**: `docs/ATLAS_SPARK_JOURNEY.md` — the benchmark retrospective across the Spark line.

## Apples-to-apples notes

When comparing Atlas to vLLM or TensorRT-LLM:

- **Same hardware.** GB10 SM121 numbers do not transfer to H100 / B200 / MI300X.
- **Same model.** "Qwen3.5-35B-A3B at 36 tok/s" is vLLM's NVIDIA GB10 benchmark on the NVFP4 CUTLASS MoE path, same HF checkpoint.
- **Same prompt shape.** The 131 tok/s number is on a short prompt (`"What is the capital of France?"`, `max_tokens ≤ 30`). Longer prompts show slightly different numbers because prefill cost amortizes differently.
- **Same precision.** Atlas NVFP4 vs vLLM NVFP4; Atlas FP8 vs vLLM FP8. Never compare across quant schemes.

The headline "3.6× faster than NVIDIA's 36 tok/s" is apples-to-apples against NVIDIA's own vLLM numbers on the same `(GB10, Qwen3.5-35B-A3B, NVFP4)` target.

## Files to read

## From the CLI

The benchmark suite the dashboard runs is also a subcommand, so a benchmark can
be scripted, run in CI, or driven over SSH with no terminal attached.

```
spark benchmark list                      # the suite
spark benchmark list concurrency-sweep    # one benchmark's parameters
spark benchmark run  concurrency-sweep --model <served-model>
spark benchmark history
```

`run` drives an endpoint that is **already serving** — it neither loads a model
nor touches the GPU. The one exception is `--pull-request-gate`, which *does*
start a server: it serves the benchmark's own recipe on a free port (900 s boot
timeout, for a cold NVFP4 load) and tears it down on drop
(`cli/bench_selfstart.rs`). Point `run` somewhere else with `--url`:

```
spark benchmark run concurrency-sweep \
  --url http://10.10.10.3:8888 --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --param concurrencies=1,2,4 --param isls=128 --param osl=64
```

`--param` takes any key from `spark benchmark list <id>`; anything you leave out
takes the schema default. An unknown key is an error listing the valid ones,
because a silently-ignored override produces a run measuring something other
than what you asked for.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | ran, and the gate passed (or had no verdict) |
| 1 | the run itself failed or was cancelled — the harness could not measure |
| 2 | the run completed and the **gate** said no |

1 and 2 are distinct so a script can tell "the harness broke" from "the model
missed the bar". `--no-fail-on-verdict` collapses 2 into 0 when you are
collecting numbers rather than gating on them.

### Run history

Every run — from the CLI *or* the dashboard — is recorded under
`~/.atlas/runs/<benchmark-id>/`, carrying the result, every parameter used (not
just the ones you overrode), the target, the source, and the Atlas version. So
a stored run says what it measured and can be reproduced.

```
spark benchmark history --id concurrency-sweep --limit 5
spark benchmark history --run run-1785000000123456789 --format json | jq .params
```

One store, both directions: a CLI run appears in the dashboard's Benchmarks →
History pane, and a dashboard run appears in `spark benchmark history` marked
`tui`.

Machine-readable output goes to **stdout**, progress to **stderr**, so
`--format json > run.json` is a clean file. `ATLAS_HOME` relocates the store.

- `crates/atlas-spark-bench/src/lib.rs` — E2E harness.
- Each primitive crate's `benches/*.rs` — per-kernel micro.
- `bench/*.json` — pinned result snapshots.
- `scripts/sweep_all_models.sh`, `scripts/run_conc_benchmark.sh` — automation.
- `docs/ATLAS_SPARK_JOURNEY.md` — benchmark journey and retrospective.
- README "Benchmark Results" section — the authoritative long-form table.
