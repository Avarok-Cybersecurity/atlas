# OpenAI-Compatible Server

Atlas serves via `spark-server` — an OpenAI and Anthropic compatible HTTP API over axum. This chapter is the operator's reference for CLI flags, protocols supported, and the knobs that matter in production. The authoritative flag list is always `spark serve --help`; the headings below match the groupings in the CLI so cross-referencing is easy.

## CLI structure

```
spark serve <MODEL> [--flags...]
spark serve --model-from-path <PATH> [--flags...]
spark --version
spark --help
```

Every runtime configuration flag has a long-form name. Most are documented inline with `#[arg]` doc-strings in `crates/spark-server/src/cli/serve_args.rs`.

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
| `--max-seq-len` | `32768` | Maximum sequence length in tokens; sizes KV pool |
| `--max-batch-size` | `8` | Max concurrent sequences per decode step |
| `--max-prefill-tokens` | `8192` | Chunked-prefill budget per iteration; sizes scratch |
| `--max-num-seqs` | `128` | Maximum queued sequences |
| `--oom-guard-mb` | `4096` | Runtime safety reserve held back from the KV pool |

Production rule of thumb for tight single-GPU deployments of 100B+ models: drop `--max-prefill-tokens` to `2048` and `--max-batch-size` to `1`. The default 8192 sizes the scratch arena, not the KV pool; tuning down frees hundreds of MB.

## KV cache precision

| Flag | Default | Notes |
|---|---|---|
| `--kv-cache-dtype` | `fp8` | `bf16`, `fp8`, `nvfp4`, `turbo2`, `turbo3`, `turbo4`, `turbo8`, plus nine asymmetric K/V pairings — `KvCacheDtype`'s `FromStr` (`crates/spark-runtime/src/kv_cache.rs`) is the authority on the accepted set |
| `--kv-high-precision-layers` | `0` | Keep first/last N attention layers at BF16 (coherence protection). **`0` does not mean "none" for every dtype** — see below |
| `--fp8-kv-calibration-tokens` | `0` | Online max-‖K‖/‖V‖ calibration for first N tokens (FP8 only) |

The flag takes a number or one of three words: `auto` (a fixed alias for **2**,
not a heuristic) and `max`/`all` (every attention layer). Anything else that
fails to parse warns and falls back to `0`.

**`--kv-high-precision-layers 0` is not "no promotion" for the turbo\* family.**
`0` means *defer to the per-dtype automatic value*
(`main_modules/kv_dtypes.rs::auto_high_precision_layers`, applied at
`serve_phases/kv_cache.rs`). For `bf16` / `fp8` / `nvfp4` the automatic value is
`None`, so `0` really is zero. For every `turbo*` and asymmetric variant it is
not: `turbo2` and `bf16k_turbo3v` promote `max(4, ⌈4·L/5⌉)` layers, and all the
others promote `max(2, ⌈L/3⌉)` — roughly a third of attention layers forced to
BF16, which also shrinks the KV pool. Pass an explicit non-zero value if you want
to control it; there is no spelling of this flag that promotes nothing under a
turbo dtype.

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
| `--mtp-quantization` | `bf16` | Must match main-model checkpoint (`nvfp4`, `fp8`, `bf16`) |
| `--mtp-vocab` | `100000` | Limit MTP LM head to the first N token ids (`0` = full vocab). The default is **not** `0`: out of the box the draft head only scores ids `0..100000`, clamped to the model's real vocab size |
| `--self-speculative` | off | Layer-skipping drafter (no MTP weights required) |
| `--ngram-speculative` | off | CPU-side n-gram matching |

See the [MTP deep dive](../deep-dives/mtp.md). Use only one of `--speculative`,
`--self-speculative`, `--ngram-speculative` — but note this is **guidance, not an
enforced constraint**: none of the three carries a clap `conflicts_with` and
`cli/validate.rs` has no rule for the combination, so passing several parses and
serves. The scheduler then resolves them by silent precedence (ngram → self-spec
→ MTP) rather than rejecting the config. `--dflash` *is* enforced — it declares
`conflicts_with = "speculative"`.

`--num-drafts` is also not a plain constant: when it is still `1`, the model's
`MODEL.toml` `default_num_drafts` replaces it (`serve_phases/config.rs`). On
`qwen3.6-27b` that is `3`, i.e. K=4.

## Scheduling / caching

| Flag | Default | Notes |
|---|---|---|
| `--enable-prefix-caching` | off | RadixAttention + SSM snapshot cache (Marconi) |
| `--ssm-cache-slots` | `16` | Concurrent SSM snapshot slots |
| `--ssm-checkpoint-interval` | `256` | Blocks between SSM checkpoints |
| `--scheduling-policy` | `fifo` | `fifo` or `slai` (SLO-aware) |
| `--tbt-deadline-ms` | `100` | SLAI decode deadline |
| `--auto-compact` | off | Active context compression threshold (e.g. 0.75 = 75% of max-seq-len) |

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
| `--tool-call-parser` | auto from `model_type` | `hermes`, `qwen3_coder`, `qwen3_xml`, `gemma4`, `mistral`, `minimax_xml`, `bare_json` |
| `--tool-max-tokens` | `8192` | **Hard** cap on the whole completion whenever tools are present — `api/chat/sampling_setup.rs` takes `req.max_tokens.min(tool_max_tokens)`, covering prose and reasoning as well as tool arguments. Not a soft cap, and not scoped to arguments |

## Observability / experimental

| Flag | Default | Notes |
|---|---|---|
| `--profile` | off | Per-kernel sync + timing (disables CUDA graphs, +10% overhead) |
| `--adaptive-sampling` | off | Entropy-gated greedy path |
| `--default-top-n-sigma` | `1.0` | Default σ for top-n-sigma sampler |
| `--default-min-p` | `0.08` | Default min-p |
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
| `POST /tokenize`, `/detokenize` | helpers | Tokenizer round-trip; gated when `--require-auth` is set |
| `GET /health` | internal | 200; used by benchmarks |

## Rate limiting and auth

- `--require-auth` (with `--auth-token <key>` or `--auth-tokens-file <path>`) — requires an `Authorization: Bearer <key>` header on write endpoints. The presented token must match one of the loaded tokens (constant-time compare); there is no "accept any key" mode.
- Token-bucket rate limiter per key (`crates/spark-server/src/rate_limiter.rs`). Off by default; enable by setting `ATLAS_RATE_LIMIT_RPM` (requests/min) and/or `ATLAS_RATE_LIMIT_TPM` (tokens/min) > 0 (bursts via `ATLAS_RATE_LIMIT_BURST_RPM` / `ATLAS_RATE_LIMIT_BURST_TPM`, default = the cap). A MAX_KEYS DoS guard bounds the key table.
- Body-size limit env-configurable via `ATLAS_MAX_BODY_BYTES` (default **32 MiB** — `main_modules/serve_router.rs`).

## Changing the model without restarting

A running server can replace its model in place. Three routes reach it, and
they share one code path:

- **The dashboard.** `spark serve` with no MODEL starts the listener and opens
  the Library, where a model and one of its recipes can be picked. Selecting
  one loads it and returns to the Main view.
- **The Library, on a server that is already serving.** Same flow; the running
  model is released first.
- **A client request** naming a different model, Ollama-style — off unless
  `--auto-swap` is passed.

`--no-auto-swap` forbids request-triggered loading outright and **wins over
`--auto-swap`** regardless of order. For deployments where the served model is
part of the contract, pass it: no client can then change what the endpoint is
running, whatever else is on the command line.

Even with `--auto-swap`, a swap needs a request whose `model` resolves to a
DIFFERENT model with a known recipe. A name that is absent, unrecognised, or
already live is served by the current model, exactly as before the flag
existed.

**What a swap preserves.** Stored responses and conversations, rate-limiter
buckets, the API-key policy and `--dump` all belong to the process, not the
model, and survive unchanged — including while no model is loaded, when
`GET /v1/conversations/{id}` still answers rather than reporting the model
missing.

**What it cannot change.** The listening socket is bound once for the process
lifetime, so a recipe's `host`/`port` are ignored with a warning and the model
serves on the address already bound. Hot-swap is single-node: a recipe needing
more than one rank is refused rather than half-applied.

**While it runs.** In-flight requests finish on the model they started on. New
requests that need a model get `503 model_not_loaded` — retriable, and the same
shape `/health` already reports during startup. A load that fails restores the
previous model, and one this build has no kernels for is refused before
anything is released.

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
