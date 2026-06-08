# Phase 01: Request-Entry Pipeline — vLLM vs Atlas

Scope: HTTP `POST /v1/chat/completions` arrives → routed → tokenized →
chat template applied → tool/grammar schema compiled → request added to
scheduler queue → batched → KV cache slot mapping computed → handed
off to the first GPU forward pass.

Source maps:

- vLLM: `vllm/entrypoints/openai/serving_chat.py`,
  `vllm/entrypoints/openai/serving_engine.py::_preprocess_chat`,
  `vllm/v1/engine/processor.py::process_inputs`,
  `vllm/v1/engine/async_llm.py::add_request`,
  `vllm/v1/structured_output/__init__.py::grammar_init`,
  `vllm/v1/core/sched/scheduler.py::schedule`,
  `vllm/v1/core/kv_cache_manager.py::{get_computed_blocks, allocate_slots}`,
  `vllm/v1/core/kv_cache_utils.py::{hash_block_tokens, get_request_block_hasher}`,
  `vllm/v1/core/block_pool.py`.
- Atlas: `crates/spark-server/src/main_modules/serve_router.rs`,
  `crates/spark-server/src/api/chat/mod.rs::chat_completions_inner`,
  `crates/spark-server/src/api/chat_phases.rs::validate_input`,
  `crates/spark-server/src/api/chat/msg_entry.rs`,
  `crates/spark-server/src/api/chat/thinking.rs`,
  `crates/spark-server/src/api/chat/loop_detect.rs`,
  `crates/spark-server/src/api/chat/template.rs`,
  `crates/spark-server/src/api/chat/sampling_setup.rs`,
  `crates/spark-server/src/api/chat_blocking.rs::run_blocking_path`,
  `crates/spark-server/src/scheduler/phase_start_prefills.rs`,
  `crates/spark-server/src/scheduler/prefill_a_step.rs`,
  `crates/spark-server/src/scheduler/emit_step.rs::compile_grammar_state`,
  `crates/spark-model/src/model/trait_impl/prefill_a.rs`,
  `crates/spark-runtime/src/prefix_cache.rs`,
  `crates/spark-runtime/src/radix_tree.rs`.

`Divergence flag` legend:
**E** = equivalent (functionally same op);
**A+** = Atlas does extra work vLLM doesn't;
**V+** = vLLM does extra work Atlas doesn't;
**!**  = different mechanics that could cause behavioural drift on the
same prompt.

| # | Step | vLLM action | Atlas action | Divergence |
|---|------|-------------|--------------|------------|
| 1 | TCP accept & HTTP framing | uvicorn/uvloop (Python asyncio) | tokio + axum (Rust) — `serve_router.rs:155` | E |
| 2 | Route to handler | FastAPI router → `OpenAIServingChat.create_chat_completion` | axum route `/v1/chat/completions` → `api::chat_completions` (`serve_router.rs:45`) | E |
| 3 | Body-size limit | uvicorn default (effectively unbounded; ad-hoc per-server) | Hard cap from `ATLAS_MAX_BODY_BYTES` (default 32 MB) via `DefaultBodyLimit::max` (`serve_router.rs:113`) | A+ |
| 4 | Auth | None by default (delegates to reverse-proxy) | `require_auth_middleware` (`serve_router.rs:123`) gates by bearer token | A+ |
| 5 | Rate limit | None in core (delegated) | Token-bucket `rate_limit_middleware` with reservation = `max_seq_len` (`serve_router.rs:119`); later true-up refund | A+ |
| 6 | Per-tenant observability | Generic FastAPI middleware | `openai_observability_middleware` (`serve_router.rs:127`) tags Prom labels | A+ |
| 7 | JSON parse | `pydantic` deserialises into `ChatCompletionRequest` model (auto-validation, coercion, error responses) | `serde_json::from_slice` into Rust struct in `chat/mod.rs:53`; on error returns 400 with the serde message verbatim | ! — vLLM coerces (e.g. string-int), Atlas strict. Custom field-name mismatches surface differently |
| 8 | Raw-request dump | None | When `--dump` set, writes verbatim request body + sequence number (`chat/mod.rs:64`) | A+ |
| 9 | Model name validation | `_check_model` walks `OpenAIServingModels` registry, returns 404 if unknown | No per-request check — single-model serving; the `model` field is echoed in the response unchanged | V+ |
| 10 | "Engine dead" / readiness check | `engine_client.errored` → raises `EngineDeadError` (503) | `model_ready` AtomicBool is global; never re-checked per request (set once in `serve_router.rs:133`) | V+ |
| 11 | OpenAI param schema validation | Pydantic enforces ranges/enums implicitly + `validate_request_params` for Mistral | Manual `chat_phases.rs::validate_input`: messages≥1, ≤2048; temperature 0–2; top_p 0–1 (exclusive 0); max_tokens≥1; tool_choice enum + required-with-no-tools | E |
| 12 | Lora-adapter resolution | `_maybe_get_adapters(request, supports_default_mm_loras=True)` resolves to a `LoRARequest` | No LoRA path | V+ |
| 13 | Tokenizer fetch | `await self.engine_client.get_tokenizer()` (async, optionally lora-tuned) | `state.tokenizer` is a single eagerly-loaded `Arc<Tokenizer>` (synchronous) | E |
| 14 | Mistral-tokenizer tool-call serialization | `maybe_serialize_tool_calls(request)` + `truncate_tool_call_ids(request)` rewrites tool-call ids to ≤9 chars for Mistral | Not done; Atlas tool-call ids are passed through as-is. Models requiring 9-char IDs (Mistral, Magistral) get untrimmed strings | V+ ! |
| 15 | Tool-choice + parser sanity (auto without parser) | Errors with explicit "auto tool choice requires `--enable-auto-tool-choice` and `--tool-call-parser`" | No equivalent error path — `tools_active = state.tool_call_parser.is_some() && !tools.empty()`; if parser missing, tools silently disabled (`chat/mod.rs:98`) | ! V+ — silently degrades vs vLLM's hard 400 |
| 16 | `tool_choice="none"` + `exclude_tools_when_tool_choice_none` | `tool_dicts = None` (suppresses tool defs entirely) | `tools_active` becomes false only if `tool_choice.is_none()` — `tool_choice="none"` still leaves the tool list visible to the Jinja template | ! |
| 17 | Chat-template-arg validation | `_validate_chat_template(request_chat_template, chat_template_kwargs, trust_request_chat_template)` — rejects client-supplied templates unless `--trust-request-chat-template` | No per-request `chat_template` override accepted; template is server-fixed at startup | V+ (smaller surface area on Atlas) |
| 18 | History-content normalization | `parse_chat_messages_futures` → walks content parts, schedules async multimodal-data future, returns `mm_data_future`, `mm_uuids` | `msg_entry::build_msg_entries` walks messages synchronously; image URIs preprocessed via `vision_preprocess::preprocess_image` inline (`msg_entry.rs:224`) | E (sync vs async, same net outputs) |
| 19 | Tool-response error injection | None | `hint_injector::looks_like_error` + `inject_hints` rewrites consecutive tool-error messages with hints to break failure loops (`msg_entry.rs:108-114`) | A+ ! — modifies the prompt the model sees |
| 20 | CWD / "working_directory" extraction | None | Atlas scans the leading system message for `working_directory:` / `cwd:` lines, captures the path, appends `<environment>…</environment>` to the system message (`msg_entry.rs:167-195`) | A+ ! |
| 21 | "Vacuous system prompt" neutralization | None | `is_vacuous_system_content` drops a leading system message that is empty or a bare `Label:` (Open WebUI residue) — model would otherwise produce terse output (`msg_entry.rs:207-216`) | A+ ! |
| 22 | Historical reasoning trace forwarding | None — vLLM does not roundtrip `reasoning_content` to the template | Atlas threads `reasoning_content` from prior assistant turns back into the Jinja template's `message.reasoning_content` so the qwen3.5/3.6 template can rehydrate the historical `<think>` block (`msg_entry.rs:148-156`, `template.rs:62-71`). Toggle: `ATLAS_STRIP_REASONING_HISTORY=1` | A+ ! — drift-relevant: corrupted historical reasoning re-enters the next prompt |
| 23 | Tool-call deserialization for the template | `tool_dicts = [t.model_dump() for t in request.tools]` is passed straight to `apply_hf_chat_template(tools=…)` | Two-tier: server may run **TSCG** (Tool-Signature-Compact-Grammar) which pre-injects compacted tool signatures into messages[0] and then passes `tools=None` to the template (`template.rs:81-90`). Historical assistant tool_calls are JSON-parsed from string → dict before template render (`msg_entry.rs:82-99`) | A+ ! — Atlas may give the template a different `tools` shape than vLLM |
| 24 | `chat_template_kwargs` default | None server-side | `state.default_chat_template_kwargs` merged into `req.chat_template_kwargs` when the client didn't explicitly request a thinking flag (`chat/mod.rs:153-157`) | A+ |
| 25 | Spontaneous-think / reasoning resolution | Implicit — controlled by sampling+reasoning parser at decode time | `thinking::resolve_thinking` runs PRE-template and returns `(enable_thinking, thinking_budget)` from request flags, server defaults, tool/thinking presets — feeds the Jinja `enable_thinking=` arg (`chat/mod.rs:160`) | A+ ! |
| 26 | Loop / spinning detection on incoming history | None | `loop_detect::check_loops` scans tool-call repetition across recent assistant turns; sets `suppress_tool_call` and `tool_call_repeat_count` for the sampler-side logit-bias decay (`chat/mod.rs:163`, `sampling_setup.rs:94-108`) | A+ ! — Atlas mutates effective sampling distribution based on **conversation history** |
| 27 | Auto-compact of long history | None | Trial-tokenises with `apply_chat_template_openai`; if > 70 % of `max_seq_len`, runs `compact_messages` to drop middle turns (gated; OFF by default per project_no_auto_compaction memory) (`template.rs:92-116`) | A+ |
| 28 | Chat template apply | `apply_hf_chat_template(tokenizer, conversation, model_config, **kwargs)` — HF tokenizer.apply_chat_template, returns a `str` prompt then tokenises | `tokenizer.apply_chat_template_openai(json_messages, jinja_tools, template_thinking, disable_tool_steering)` (`template.rs:118`) — a Rust port of Jinja that returns `Vec<u32>` directly (no intermediate string round-trip) | ! — same Jinja syntax but two independent engines; **byte-divergent** output for edge-case templates is possible |
| 29 | Image-pad expansion | Part of `mm_data_future` (vision preprocessor inserts the right placeholder count) | Post-template: `expand_image_pads(prompt_tokens, image_pad_counts)` (`template.rs:134-140`) expands single `<|image_pad|>` into N pads based on (grid_h/sms × grid_w/sms) | ! — Atlas runs expansion AFTER template; vLLM expands inside the template via `<|image|>` rendering |
| 30 | Template-forced-thinking detection | None | Atlas scans the tail-8 tokens for an unclosed `<think>` token; if present and the client didn't ask for thinking, force-enables it with the server's max thinking budget (`template.rs:142-163`) | A+ ! |
| 31 | Prompt length cap check | `_validate_model_input` rejects when `prompt_len > max_model_len` with multimodal/text-aware hint | Atlas returns 400 when `prompt_len >= state.max_seq_len` (no headroom for at least 1 output token) (`chat/mod.rs:193-201`) | E |
| 32 | Sampling-preset selection | `to_sampling_params(max_tokens, logits_processor_pattern, default_sampling_params)` merges request fields with server defaults; one flat `SamplingParams` | Atlas selects a **three-way preset** (`tools` / `thinking_text` / `non_thinking`) based on `tools_active` and `enable_thinking`, then overrides per-field with request values (`sampling_setup.rs:53-72`) | A+ ! — same prompt + same client params can produce different temperature/top_p across Atlas vs vLLM |
| 33 | OpenAI penalty-range validation | Pydantic field constraint | Manual range checks `-2.0..=2.0` for both `presence_penalty` and `frequency_penalty` (`sampling_setup.rs:73-85`) | E |
| 34 | Logit-bias coercion | `params.logit_bias` validated against vocab in `processor._validate_logit_bias` | Atlas parses string→u32, ignores unparseable keys (silent drop); no vocab-range check (`sampling_setup.rs:88-92`) | ! V+ — vLLM rejects invalid logit_bias; Atlas silently ignores |
| 35 | Tool-call-token logit-bias decay | None | When tools active + not suppressed, Atlas pushes `<tool_call>` token bias to `+3 / 0 / -5 / -10` based on `tool_call_repeat_count` (`sampling_setup.rs:96-108`) | A+ ! — drift-relevant |
| 36 | max_tokens cap when tools active | `get_max_tokens` enforces `max_model_len - prompt_len` only | Atlas additionally caps to `state.tool_max_tokens` when tools are active (`sampling_setup.rs:110-124`) | A+ |
| 37 | Stop-sequence tokenisation | `params.stop` / `stop_token_ids` pass through to scheduler (vLLM matches at token level) | `tokenize_stop_sequences(tokenizer, req.stop)` runs the tokenizer for each string stop; if tools active, also pushes `</tool_call>` as a stop token (`sampling_setup.rs:127-133`) | A+ ! |
| 38 | Tool-choice "required" hot wire | `params.tool_choice` and `tool_dicts` shape what arrives at the grammar backend | Atlas sets `tool_choice_required = true` when `tool_choice=required` OR parser is `minimax_xml` OR `bare_json` — bypasses the grammar trigger set (`sampling_setup.rs:135-149`) | A+ ! |
| 39 | response_format + tools coexistence | Pydantic + Processor allow both; the structured-output backend builds the grammar from `params.structured_outputs` | Atlas picks ONE: if `tool_choice=none`, enforces `response_format`; else enforces tool-call grammar and logs a "schema-shape compliance falls to the model" warning (`sampling_setup.rs:152-210`) | ! — different policy under the same request shape |
| 40 | Grammar backend selection | `_validate_structured_output` picks `xgrammar` / `guidance` / `outlines` / `lm-format-enforcer`; falls back xgrammar→guidance for unsupported features | Single backend: `crates/xgrammar` (vendored). `disable_tool_grammar` server flag short-circuits compilation entirely | V+ — vLLM has 4 backends and auto-fallback |
| 41 | Grammar compile timing | `StructuredOutputManager.grammar_init` submits `_async_create_grammar` to a `ThreadPoolExecutor` (max_workers = ceil(cpu/2)); request status flips to `WAITING_FOR_FSM` until grammar Future is ready, then `WAITING`. **Compile is OFF the request's hot path** (`scheduler.py:380-387`) | Grammar compile runs **inline on the scheduler's own thread** inside `compile_grammar_state` during `prefill_a_step.rs:62`, blocking the entire scheduler loop until done | ! ! — drift- and latency-relevant. Long tool schemas stall *all* requests, not just the one waiting for FSM |
| 42 | Grammar bitmask cache | Pre-allocated bitmask shared per batch + filled via parallel `ThreadPoolExecutor` ≥ N threshold | Per-request `GrammarState` allocated on the scheduler thread; bitmask filled inline | A+ overhead per request, no parallelisation |
| 43 | Tool-parser request adjustment | `tool_parser(tokenizer).adjust_request(request)` runs the parser's `adjust_request` hook (e.g. Hermes inserts schema into system prompt) | Atlas tool-parsers handle adjustment via `system_prompt()` injected by **TSCG path** or by the chat template's `tools` arg; no per-request `adjust_request` hook | V+ |
| 44 | Request-id generation | `f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"` — uses client-supplied `request_id` header if any | UUIDv4 generated at response finalize (`chat_blocking.rs:636`); no per-request id carried through the scheduler — sequences are identified by **session_hash** = FNV-1a of prompt_tokens (`chat/mod.rs:186`) | ! — Atlas cannot dedup retried client requests; same prompt = same session_hash = SAME prefix-cache slot |
| 45 | Timeout deadline | Cancelled by FastAPI's async cancellation if client disconnects | Atlas computes `timeout_at = Instant::now() + req.timeout` (default = `state.request_timeout`); checked inside the decode loop (`sampling_setup.rs:212-218`) | A+ ! |
| 46 | Top-logprobs cap | Pydantic field constraint (≤ 20) | Atlas caps `req.top_logprobs.map(|n| n.min(20))` (`sampling_setup.rs:220`) | E |
| 47 | Detokeniser handle | `output_processor.add_request(...)` registers the request with the streaming detokeniser BEFORE engine submission | Atlas does ad-hoc detokenisation in `decode_response_text` at finalize-time (blocking) or in streaming dispatch (per token) | ! — different latency profile; Atlas decodes once at the end for blocking, vLLM streams continuously |
| 48 | Multi-modal hash override | `_maybe_build_mm_uuids` builds `f"{request_id}-{modality}-{i}"` keys when prefix-caching disabled, else uses content-hash | Atlas image_pixels passed through verbatim; vision pad-count is NOT part of the prefix-cache key — and `tokens_have_vision_pad(tokens)` *disables prefix cache for that request* (`prefill_a.rs:115`) | ! — Atlas pessimistically skips the cache for any prompt with a vision-pad; vLLM caches with mm-hash extra keys |
| 49 | `cache_salt` plumbing | Optional per-request `cache_salt` is added to the first block's `extra_keys` so callers can force-segment prefix cache (`kv_cache_utils.py:507`) | No `cache_salt` field accepted; only `session_hash` (=FNV1a of prompt token ids) gates SSM-snapshot ownership | V+ — vLLM exposes a knob to defeat cross-tenant prefix bleed |
| 50 | Enqueue path | `engine_client.generate(...)` → `_add_request` → `engine_core.add_request_async(request)` over ZMQ to the **separate engine-core process** | `state.request_tx.send(InferenceRequest::…)` over a tokio mpsc channel **in-process**, same OS thread family (`chat_blocking.rs:147`) | ! — vLLM has process-boundary IPC; Atlas is single-process |
| 51 | Multi-completion (`n>1`) fan-out | `ParentRequest` + `n` child requests submitted in parallel; engine schedules them independently | Atlas serialises the `n` choices: one `oneshot::channel`, one `request_tx.send`, awaits, then loops (`chat_blocking.rs:108`) | ! — vLLM is throughput-parallel for n>1; Atlas is sequential |
| 52 | Scheduler queue | Priority or FCFS `RequestQueue` (`request_queue.py`); preemption with `RequestStatus.PREEMPTED` + KV-cache free | Single-queue FCFS; preemption path is `swap_remove(active.len() - 1)` returning a "503 Preempted" error to the victim instead of re-queueing (`phase_start_prefills.rs:108-125`) | ! — vLLM re-queues, Atlas surfaces a 503 to the *evicted* request |
| 53 | Encoder-cache budget | `EncoderCacheManager` reserves slots for ViT outputs to bound budget per step | No equivalent budget; vision encode runs synchronously inside prefill chunk 0 (`prefill_a_step.rs:124-127`) | V+ |
| 54 | Prefix-cache block hashing | `hash_block_tokens(parent_hash, tuple(curr_block_token_ids), extra_keys)` — chain-hash per BLOCK with `(parent, tokens, extra_keys)` tuple. `extra_keys` carries LoRA id, mm-hash, `cache_salt`. Block-size aligned (`kv_cache_utils.py:524`) | `RadixTree::lookup` walks the tree token-by-token at block granularity (`radix_tree.rs:62`); the helper `hash_token_prefix` is FNV-1a over the **raw token-id u32 stream** with no parent chain and no extra keys; used only to key SSM snapshots, NOT the block cache itself | ! — different cache identities: vLLM hashes (parent, block-tokens, extras); Atlas walks raw token edges in a radix trie. Equivalent in correctness on plain text; **divergent under image/LoRA** because Atlas has no `extra_keys` |
| 55 | Prefix-cache hit lookup | `kv_cache_manager.get_computed_blocks(request)` — `coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length = num_tokens - 1)` (`kv_cache_manager.py:176-217`). **Runs in the scheduler thread before allocate_slots** | `self.prefix_cache.lookup(tokens, block_size, seq.session_hash)` runs INSIDE `prefill_dispatch` (`prefill_a.rs:115-119`), AFTER `alloc_sequence` has reserved an SSM slot | ! — Atlas's hit/miss decision happens later in the pipeline, after committing one resource |
| 56 | SSM-snapshot lookup | None — vLLM v1 has no Mamba-state caching | After block-cache match, Atlas queries `SsmSnapshotIndex` for the *deepest* matched node and returns `(snap_id, snap_tokens)` (`radix_tree.rs:76-84`); subject to `session_hash` match | A+ ! — Marconi SSM caching; drift surface for hybrid models |
| 57 | Vision-pad cache-key behaviour | Encoded via `mm_hashes` in `extra_keys`; cached cleanly | Hard-coded skip: `tokens_have_vision_pad(tokens) ⇒ PrefixMatch::empty()` (`prefill_a.rs:115-118`) — vision prompts never hit the cache | ! V+ |
| 58 | Slot/block allocation | `allocate_slots(request, num_new_tokens, num_new_computed_tokens, new_computed_blocks, num_lookahead_tokens)` — `coordinator.allocate_new_blocks` from `BlockPool`; returns `None` if not enough free; **caller preempts lowest-priority running request** (`scheduler.py:262-300`). Caches blocks at `num_tokens_to_cache = min(num_computed_tokens + num_new_tokens, request.num_tokens)` | `model.alloc_sequence()` reserves SSM slot + lookup blocks via `ensure_blocks_through_prefill(seq, blocks_needed-1, …)` (`prefill_a.rs:156`). Eviction is internal to `PagedKvCache`; preemption is the 503 path above | ! — vLLM mixes preemption + retry; Atlas fails fast |
| 59 | Lookahead-block allocation (spec decode) | `num_lookahead_tokens = num_speculative_tokens` reserved with the slot | Atlas reserves spec slots up-front at preflight (`preflight.rs::ssm_multiplier`); per-request multiplier is fixed at startup | E |
| 60 | EP / TP broadcast | None at request-entry (workers are configured at engine init; runtime broadcast is per-step) | Atlas's prefill_a_step broadcasts `0xFFFFFFF0`, `chunk_len`, `chunk_start`, `prompt_len`, then bulk-broadcasts the prompt-token tensor to the EP worker via NCCL **before** prefill (`prefill_a_step.rs:134-138`) | A+ ! — EP=2 has explicit per-request NCCL ops on the request-entry hot path |
| 61 | Chunked-prefill chunk-0 dispatch | `Scheduler.schedule()` returns a `SchedulerOutput` with `num_scheduled_tokens` per request; first forward consumes chunk-0 of all queued prefills in a single mixed batch | Atlas's `start_chunked_prefill` runs **the chunk-0 forward inline** on the scheduler thread and returns `StartPrefillResult::{Active, InProgress, Finished}` — no batching across newly-arrived requests on the same scheduler tick (`prefill_a_step.rs:140-148`) | ! — vLLM continuously fuses fresh prefills; Atlas pays serial overhead for the first chunk |
| 62 | First-token sample (if chunk covers full prompt) | Falls out of the same forward as continuing decodes — the model runner samples on the unified output | Atlas samples synchronously inline (`prefill_a_step.rs:180-200`) and emits the first streaming token before returning | ! |
| 63 | TTFT clock | `arrival_time = time.time()` recorded in `processor.process_inputs` | `request_start = Instant::now()` recorded inside `start_chunked_prefill`, AFTER queue admission and AFTER `alloc_sequence`. **TTFT excludes scheduler queue wait** | ! — Atlas's published TTFT is structurally lower than vLLM's like-for-like measurement |

---

## Missing on Atlas (vLLM does, Atlas doesn't)

1. **Multi-backend grammar compile with fallback** — vLLM tries xgrammar → guidance/outlines/lm-format-enforcer; Atlas is xgrammar-only.
2. **Off-thread grammar compilation** — vLLM uses a `ThreadPoolExecutor` and a `WAITING_FOR_FSM` state. Atlas blocks the scheduler thread.
3. **Mistral tool-call ID truncation + re-serialization** (`maybe_serialize_tool_calls`, `truncate_tool_call_ids`).
4. **Hard error on `tool_choice="auto"` without a parser** — Atlas silently disables tool calling.
5. **Per-request `cache_salt`** for prefix-cache partitioning across tenants.
6. **`extra_keys` in prefix-cache hashing** (LoRA id, mm-hash, salt). Atlas's `hash_token_prefix` is a raw FNV-1a over u32 tokens.
7. **Mm-aware prefix caching** — vLLM hashes vision tokens via `mm_hashes`; Atlas refuses to cache any prompt containing a vision pad.
8. **Encoder-cache compute budget** for ViT outputs.
9. **Priority queue policy** (`SchedulingPolicy.PRIORITY`); preempt-and-requeue rather than 503-the-victim.
10. **LoRA-aware request routing**.
11. **Re-validation of logit-bias token IDs** against the vocab.
12. **Per-request `chat_template` override** (with the `--trust-request-chat-template` opt-in).
13. **Engine-process boundary** — vLLM's API server can't kill the engine by panicking; Atlas's scheduler crash is fatal to the server.
14. **Async detokenisation worker** — vLLM streams; Atlas decodes once at finalize for blocking calls.
15. **Beam search** path.
16. **Parallel `n>1` fan-out** at the engine layer — Atlas serialises choices.
17. **Per-request arrival-time → TTFT** — vLLM clocks at JSON parse; Atlas clocks at prefill start, hiding queue latency from TTFT.

## Missing on vLLM (Atlas does, vLLM doesn't)

1. **Body-size cap** (`ATLAS_MAX_BODY_BYTES`).
2. **Built-in bearer-token auth and rate-limiter with reservation/refund**.
3. **CWD extraction + `<environment>` injection** into the system message.
4. **Vacuous-system-prompt drop** (`is_vacuous_system_content`) — covers Open WebUI's empty RAG block.
5. **Hint injection on consecutive tool errors** (`hint_injector::inject_hints`).
6. **Historical-reasoning rehydration** — Atlas forwards `reasoning_content` of past turns back into the Jinja template (`ATLAS_STRIP_REASONING_HISTORY=1` to disable).
7. **Three-way sampling-preset selection** based on `(tools_active, enable_thinking)`.
8. **Exponential `<tool_call>` token logit-bias decay** based on the tool-call repeat count in the incoming history.
9. **Pre-template thinking resolution** + `thinking_budget`.
10. **Tail-token unclosed-`<think>` detection** that force-enables thinking server-side.
11. **TSCG (Tool-Signature-Compact-Grammar)** — compacted tool descriptors injected into system prompt instead of passing the schema to Jinja's `tools=` slot.
12. **Marconi SSM-state snapshot caching** (separate index from the radix tree, session-scoped).
13. **Per-request `timeout` deadline** propagated to the decode loop.
14. **EP-worker prompt broadcast** at request-entry time (NCCL bulk op).
15. **Server-fixed loop-detector** across multi-turn assistant tool calls.
16. **Auto-compact of long history** (disabled by default).
17. **`--dump` request/response capture** correlated by sequence number.
18. **`</tool_call>` auto-added as a stop sequence** when tools are active.

## Likely impact on the 10/10 gap (Atlas FP8 30 % vs vLLM BF16 100 %)

The request-entry phase is **not the primary driver** of the 70-point opencode quality gap — that is a hot kernel-precision issue (FP8 MMA accumulation + late-layer KV drift) already characterised in `project_fp8_ceiling_conclusive.md`. However, several request-entry divergences contribute multiplicatively and can move the floor:

**High-confidence contributors (load-bearing for drift, not just latency):**

- **#22 Historical-reasoning rehydration** — Atlas re-injects prior `<think>` traces into the next prompt. When earlier turns degenerated under FP8 (per Wave-3 strip experiments), this seeds loop-attractor patterns the next turn snowballs on. vLLM never feeds reasoning back. This is **the single most plausible non-kernel source of multi-turn FP8 collapse** that doesn't appear in vLLM.
- **#26 / #35 Loop-detector + tool-call logit-bias decay** — Atlas modifies the effective sampling distribution based on history. Two systems with the same temperature can sample different tokens before the model forward even runs.
- **#32 Three-way sampling-preset** — Atlas uses different `(temperature, top_p, top_k)` than vLLM under the same client request when `tools_active` or `enable_thinking` flips. A fair head-to-head requires forcing both engines to identical sampling params.
- **#41 Inline grammar compile** — Stalls the entire scheduler on tool-heavy opencode requests (large schema). Doesn't cause wrong tokens, but inflates TTFT for batched workloads — not a quality gap, a latency gap.
- **#54 Prefix-cache hash identity** — Different across the two engines. On a session_hash collision (FNV-1a of u32 tokens), Atlas may serve an SSM snapshot from an unrelated session if the raw token prefix happens to match. vLLM's `(parent, tokens, extra_keys)` is collision-resistant; Atlas's flat FNV is not. Plausible under adversarial token sequences but unlikely on natural prompts.
- **#48/#57 Vision-prompt cache skip** — Atlas pessimistically disables prefix caching for any vision-pad prompt. Hurts latency and forces a fresh prefill (more FP8 ops per request) — could amplify drift through more uncached forward passes.

**Likely zero-impact (latency/ergonomics only):**

- Body-size cap, auth, rate-limit, request-id, dump (#3-#8, #44).
- LoRA, beam search, multi-process IPC (#12, #50).
- TTFT clock origin (#63).

**Suggested next-phase probe:** rerun the 30 %/100 % opencode bench under
matrices `(reasoning_strip ∈ {on, off}) × (loop_detect ∈ {on, off}) × (sampling ∈ {atlas-preset, vllm-default})` to isolate how much of the 70-point gap is the **prompt the model sees** vs the **tokens the model emits under FP8**. The hypothesis from the existing memory (`project_fp8_ceiling_conclusive.md`) is that kernel precision dominates, but the controlled-input variant has not yet been run.
