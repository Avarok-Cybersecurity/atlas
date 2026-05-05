# OpenAI-Compatible Server

Atlas serves via `spark-server` — an OpenAI and Anthropic compatible HTTP API over axum. This chapter is the operator's reference for CLI flags, protocols supported, and the knobs that matter in production. The authoritative flag list is always `spark serve --help`; the headings below match the groupings in the CLI so cross-referencing is easy.

## CLI structure

```
spark serve <MODEL> [--flags...]
spark serve --model-from-path <PATH> [--flags...]
spark --version
spark --help
```

Every runtime configuration flag has a long-form name. Most are documented inline with `#[arg]` doc-strings in `crates/spark-server/src/cli.rs`.

## Model selection and I/O

| Flag | Default | Notes |
|---|---|---|
| `MODEL` (positional) | — | HF id (e.g. `Sehyo/Qwen3.5-35B-A3B-NVFP4`); resolves against `~/.cache/huggingface/hub` |
| `--model-from-path` | — | Local path; skips HF resolution entirely |
| `--model-name` (alias `--served-model-name`) | config `_name_or_path` or `MODEL` | Override what `/v1/models` reports |
| `--cache-dir` | `$HF_HUB_CACHE`, `$HF_HOME/hub`, `~/.cache/huggingface/hub` | HF cache root |
| `--port` | `8888` | HTTP listen port |
| `--no-fast-load` | off (fast on) | Revert to mmap loader — the O_DIRECT + pipelined fast path is default |

## Memory / budget

| Flag | Default | Notes |
|---|---|---|
| `--gpu-memory-utilization` | `0.90` | Fraction of GPU memory Atlas will claim |
| `--max-seq-len` | `4096` | Maximum sequence length in tokens; sizes KV pool |
| `--max-batch-size` | `8` | Max concurrent sequences per decode step |
| `--max-prefill-tokens` | `8192` | Chunked-prefill budget per iteration; sizes scratch |
| `--max-num-seqs` | `128` | Maximum queued sequences |
| `--oom-guard-mb` | `4096` | Runtime safety reserve held back from the KV pool |

Production rule of thumb for tight single-GPU deployments of 100B+ models: drop `--max-prefill-tokens` to `2048` and `--max-batch-size` to `1`. The default 8192 sizes the scratch arena, not the KV pool; tuning down frees hundreds of MB.

## KV cache precision

| Flag | Default | Notes |
|---|---|---|
| `--kv-cache-dtype` | `fp8` | One of `bf16`, `fp8`, `nvfp4`, `turbo3`, `turbo4` |
| `--kv-high-precision-layers` | `0` | Keep first/last N attention layers at BF16 (coherence protection) |
| `--fp8-kv-calibration-tokens` | `0` | Online max-‖K‖/‖V‖ calibration for first N tokens (FP8 only) |

See [FP8](../deep-dives/fp8.md) and [NVFP4](../deep-dives/nvfp4.md) for the trade-offs. Atlas's recommendation per model family:

- Qwen3.5 family → `nvfp4` KV.
- Qwen3.6 / Nemotron-H → `fp8` with calibration.
- 122B-class (memory-constrained) → `nvfp4` + `--kv-high-precision-layers 2`.
- Everything else → `fp8` (safe default).

## Speculative decoding

| Flag | Default | Notes |
|---|---|---|
| `--speculative` | off | Enable MTP — requires MTP weights in checkpoint |
| `--num-drafts` | `1` | Draft tokens per verify (K = num_drafts + 1); default per-model from `MODEL.toml` |
| `--mtp-quantization` | `nvfp4` | Must match main-model checkpoint (`nvfp4`, `fp8`, `bf16`) |
| `--mtp-vocab` | `0` | Limit MTP LM head to top-N tokens (0 = full vocab) |
| `--self-speculative` | off | Layer-skipping drafter (no MTP weights required) |
| `--ngram-speculative` | off | CPU-side n-gram matching |

See the [MTP deep dive](../deep-dives/mtp.md). Only one of `--speculative`, `--self-speculative`, `--ngram-speculative` at a time.

## Scheduling / caching

| Flag | Default | Notes |
|---|---|---|
| `--enable-prefix-caching` | off | RadixAttention + SSM snapshot cache (Marconi) |
| `--ssm-cache-slots` | `16` | Concurrent SSM snapshot slots |
| `--ssm-checkpoint-interval` | `256` | Blocks between SSM checkpoints |
| `--scheduling-policy` | `fifo` | `fifo` or `slai` (SLO-aware) |
| `--tbt-deadline-ms` | `100` | SLAI decode deadline |
| `--auto-compact` | off | Active context compression threshold (e.g. 0.75 = 75% of max-seq-len) |
| `--warmup-prompt` | — | File path; pre-filled at startup, its KV enters the prefix cache |

Agent workloads (Claude Code, OpenCode): always enable `--enable-prefix-caching` and `--scheduling-policy slai`. The prefix cache dominates wall-clock for system prompts + tool schemas; SLAI keeps streaming smooth under concurrent load.

## Multi-GPU (Expert Parallelism)

| Flag | Default | Notes |
|---|---|---|
| `--rank` | `0` | 0 = head (HTTP + scheduler); N > 0 = worker |
| `--world-size` | `1` | Total ranks; `2` enables EP=2 |
| `--master-addr` | — | Rendezvous host (e.g. head's IB IP) |
| `--master-port` | `29500` | NCCL rendezvous port |

See [Multi-GPU & EP=2](./multi-gpu.md) for the full setup, including the NCCL env vars that matter on GB10.

## Reasoning / tools

| Flag | Default | Notes |
|---|---|---|
| `--disable-thinking` | off | Kill-switch for `<think>` blocks |
| `--max-thinking-budget` | from `MODEL.toml` | Per-request `<think>` token ceiling |
| `--tool-call-parser` | auto from `model_type` | `hermes`, `qwen3_coder`, `mistral` |
| `--tool-max-tokens` | `512` | Soft cap on tool-call arg generation |

## Observability / experimental

| Flag | Default | Notes |
|---|---|---|
| `--profile` | off | Per-kernel sync + timing (disables CUDA graphs, +10% overhead) |
| `--adaptive-sampling` | off | Entropy-gated greedy path |
| `--default-top-n-sigma` | `1.0` | Default σ for top-n-sigma sampler |
| `--default-min-p` | `0.0` | Default min-p |
| `--swap-space-gb` | `3` | Disk-backed KV swap at `/tmp/atlas-swap/` |
| `--request-timeout` | `300` | Per-request seconds, 0 disables |

## Endpoints

| Route | Protocol | Notes |
|---|---|---|
| `GET /v1/models` | OpenAI | Returns one `ModelInfo` (Atlas serves one model per process) |
| `POST /v1/chat/completions` | OpenAI | Chat; streaming via SSE when `stream: true` |
| `POST /v1/completions` | OpenAI (legacy) | Plain completion |
| `POST /v1/responses` | OpenAI Responses | Stateful; supports `conversation_id` |
| `POST /v1/messages` | Anthropic | Full Messages API with streaming |
| `POST /tokenize`, `/detokenize` | helpers | Tokenizer round-trip; gated under `ATLAS_REQUIRE_AUTH` |
| `GET /health` | internal | 200; used by benchmarks |

## Rate limiting and auth

- `ATLAS_REQUIRE_AUTH=1` — requires an `Authorization: Bearer <key>` header on write endpoints. Accepts any non-empty key in the default build; plug in a real validator by customising the `auth::validator` hook.
- Token-bucket rate limiter per key (`crates/spark-server/src/rate_limiter.rs`). Defaults: 60 requests/min per key with a MAX_KEYS DoS guard.
- Body-size limit env-configurable via `ATLAS_MAX_BODY_BYTES` (default 8 MiB).

## Chat templating

Tokenization uses the HF `tokenizers` crate plus `minijinja` for chat templates. Atlas ships its own template overrides for a handful of models in `jinja-templates/<family>.j2` when the upstream template has known issues (e.g. template-forced `<think>` seeding). Naming convention: filename matches the HF repo.

## Observability

Prometheus-style metrics are exposed on `/metrics` (optional — gated behind a build feature):

- `spark_requests_total{status, model}`
- `spark_tokens_generated_total`
- `spark_ttft_seconds{model}`
- `spark_decode_throughput_tok_per_sec{model}`
- `spark_kv_pool_utilization_ratio`
- `spark_active_sequences`

Structured logs go to stderr; they're the primary operational signal. Atlas logs a brief line per completed request (model, prompt tokens, generated tokens, TTFT, elapsed, tools used). Per-token DECODE spam is deliberately not logged — it's useless.

## A safe production config (Qwen3.5-35B, agents)

```
serve Sehyo/Qwen3.5-35B-A3B-NVFP4 \
  --port 8888 \
  --max-seq-len 16384 \
  --max-batch-size 4 \
  --kv-cache-dtype nvfp4 \
  --gpu-memory-utilization 0.88 \
  --scheduling-policy slai \
  --tbt-deadline-ms 100 \
  --enable-prefix-caching \
  --speculative --mtp-quantization nvfp4 \
  --auto-compact 0.85 \
  --adaptive-sampling
```

Claude Code uses 16k+ context for tool use; running with `--max-seq-len 4096` will make agents fail mid-session. Always size up when running an agent workload.
