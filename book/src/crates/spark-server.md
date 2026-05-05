# spark-server

**Role:** the binary. OpenAI- and Anthropic-compatible HTTP server, request scheduler, tokenizer, tool parsing, streaming, rate limiter, CLI.
**Key files:** `main.rs`, `cli.rs`, `api.rs`, `openai.rs`, `anthropic.rs`, `scheduler.rs`, `scheduling_policy.rs`, `tool_parser.rs`, `reasoning_parser.rs`, `tokenizer.rs`, `rate_limiter.rs`, `refusal.rs`, `metrics.rs`, `conversation_store.rs`, `response_store.rs`, `session_manager.rs`, `model_resolver.rs`, `grammar.rs`, `hint_injector.rs`, `citation.rs`, `adaptive_sampler.rs`, `ngram.rs`.

This crate is the only `bin` in the workspace — building `spark-server` produces the `spark` executable that the Docker image ships. Everything above is `lib` — `spark-server` ties it all together.

## Startup sequence (in `main.rs`)

1. **Parse CLI** (`cli::Cli::parse()`).
2. **Resolve model path** — HF id via `HF_HUB_CACHE` / `~/.cache/huggingface/hub` or explicit `--model-from-path`.
3. **Load `ModelConfig`** from `config.json`.
4. **Resolve `KernelTarget`** from the config's `model_type` + `--kv-cache-dtype`.
5. **Instantiate `GpuBackend`** — `AtlasCudaBackend::new(ordinal, &ptx_modules)`.
6. **Instantiate `CommBackend`** — `NcclBackend` if `--world-size > 1`, else `SingleGpuBackend`.
7. **`factory::build(config, gpu, comm)` → `Arc<dyn Model>`** — the model weights land on the GPU.
8. **Load the tokenizer** (`tokenizers` crate, optional chat template in `jinja-templates/<family>.j2`).
9. **Run `preflight::check(&model)`** — one synthetic decode, fails fast on shape / budget / kernel issues.
10. **Spawn the scheduler** — a dedicated tokio task with a bounded mpsc channel of `Request`s.
11. **Start axum** — `serve(&addr)` with the route table from `api.rs`.
12. **Capture CUDA graphs** in the background while the first request is in flight (warm-start).

The scheduler is driven by an inbound `mpsc::Receiver<Request>`; each HTTP handler enqueues a request and awaits a `mpsc::Sender<ResponseChunk>` handed back to it.

## HTTP routes (`api.rs` + `openai.rs` + `anthropic.rs`)

| Method | Path | Handler |
|---|---|---|
| GET | `/v1/models` | `api::list_models` — returns `ModelListResponse` containing one `ModelInfo` for the served model |
| POST | `/v1/chat/completions` | `api::chat_completions` (OpenAI) |
| POST | `/v1/completions` | `api::completions` (OpenAI legacy) |
| POST | `/v1/responses` | `api::responses` (OpenAI Responses API, stateful) |
| POST | `/v1/messages` | `anthropic::messages` (Anthropic) |
| GET | `/health` | simple 200 — used by the bench harness |
| POST | `/tokenize`, `/detokenize` | helpers, optionally gated behind `ATLAS_REQUIRE_AUTH` |

Streaming is the default for chat; non-streaming aggregates and returns `ChatCompletionResponse`. Tool-call chunks are emitted as `delta.tool_calls` in the SSE stream. Anthropic streaming populates `stop_sequence` on `message_delta` events (fixed in wave-12).

The server also implements the Responses API with a stateful backend (`response_store`, `conversation_store`) for multi-turn conversations, citation extraction, and a streaming refusal filter.

## The scheduler (`scheduler.rs` + `scheduling_policy.rs`)

Two policies:

- **FIFO** — first come, first served. Decode step picks up to `max_batch_size` active sequences and runs a batched forward.
- **SLAI** — SLO-aware. Each sequence has a time-between-tokens (TBT) deadline; the scheduler prioritizes the sequence closest to its deadline. Under mixed load, this is the difference between smooth streaming and bursty output.

Allocation discipline: KV pages are claimed on prefill start, released on completion. The scheduler tracks the KV budget and chunks prefills when memory is tight (`--max-prefill-tokens` caps the per-iteration tokens so scratch sizes stay bounded).

Active context compaction (the `compact_messages` function in `api.rs`) applies at the HTTP level before tokenization: if the tokenized prompt approaches `--max-seq-len`, the server progressively truncates middle tool responses (stage 2), replaces middle responses with pointers (stage 3), drops oldest middle pairs (stage 4), and finally trims the system prompt + keeps only the last 4 messages (stage 5). References `arXiv:2603.05344` (OpenDev).

## Tool-call parsing (`tool_parser.rs`)

The server supports three tool-call formats, auto-detected from the model's `MODEL.toml` with a `--tool-call-parser` override:

| Parser | Models | Format |
|---|---|---|
| `hermes` | Qwen3-VL, Qwen3-Next, MiniMax | JSON in `<tool_call>{...}</tool_call>` |
| `qwen3_coder` | Qwen3.5-27B / 35B / 122B, Nemotron-H, Qwen3.6 | XML-in-tool-call, nested `<function=...><parameter=...>` |
| `mistral` | Mistral-Small-4 | JSON block with explicit `[TOOL_CALLS]` prefix |

The Qwen3.5 coder parser had several robustness improvements in the bug sweeps (literal `</tool_call>` recovery, missing `</parameter>` recovery, empty `{}` tool-calls) — the parser is now tolerant of slightly-malformed model output.

## Reasoning / thinking (`reasoning_parser.rs`)

Models that emit `<think>...</think>` blocks (Qwen3.5, Nemotron-H, MiniMax) stream the thinking content to the client as a separate SSE channel keyed on `"reasoning"` (per OpenAI's `o1`-family convention). `--max-thinking-budget` caps the total thinking tokens. `--disable-thinking` is a kill-switch.

Several subtle fixes in this area were important:

- **Template-forced thinking** — some models emit `<think>` seeded by the chat template; Atlas's detector had to distinguish that from the model's own `<think>`. The pass-16 fix required the opening `<think>` to be *unclosed* to count as the model's own.
- **Closed empty thinking** — `<think>\n\n</think>\n\n` is a template no-op, not a reasoning block. Wave-4 fixed the false-positive.
- **Multi-block reasoning** — models occasionally emit multiple `<think>` blocks; the extractor concatenates them.

## Tokenizer (`tokenizer.rs`)

Wraps the HF `tokenizers` crate. Adds jinja chat-template expansion (via `minijinja`). Resolves special tokens (`<|im_start|>`, `<think>`, `<minimax:tool_call>`, etc.) from `tokenizer_config.json` so the raw token ids match what the model was trained on.

## Rate limiter (`rate_limiter.rs`)

Per-key token bucket. Wave-9 added a `MAX_KEYS` guard to prevent DoS via cardinality explosion. Body-size limits are env-configurable.

## What's explicitly not here

- **No GPU kernels.** Every GPU call delegates through `spark-runtime`.
- **No CUDA.** The crate's `Cargo.toml` does not depend on `cudarc`.
- **No model-specific weight code.** That's `spark-model`.

## Adding a new HTTP shape

- A new API endpoint (e.g. an Atlas-native `/v1/sessions/create`) — one handler in `api.rs`, one route in the router bindings in `main.rs`.
- A new tool-call format — one new parser module, one enum variant, one `--tool-call-parser` option.
- A new reasoning tag (`<scratchpad>`, etc.) — extend `reasoning_parser.rs`.
- A new chat template — a file in `jinja-templates/<model>.j2`, auto-picked up by the tokenizer layer if named after the HF repo.
