# OpenAI API compatibility — remaining gaps (PR 4 / alpha-2.45)

Follow-up to the three-PR overhaul that landed in alpha-2.44 (`f4bcd11`,
`86ae8e7`). 2.44 closed the P0/P1 field and endpoint gaps; this PR closes
the P2 items so the server is a drop-in OpenAI-2026-stable replacement
for the chat / completions / responses surface.

## Scope

Purely additive. No inference hot-path edits, no behavior change for
existing clients when the new env-gated features are disabled.

In scope:
1. Stateful Responses API (`previous_response_id` resume).
2. Streaming `/v1/responses` (typed SSE event model).
3. Completion storage backend (`store: true` → `GET /v1/chat/completions/{id}`).
4. Heuristic refusal classifier populating `message.refusal`.
5. Real per-identity rate-limit enforcement (token bucket, 429).
6. URL annotation extractor quality improvements (markdown links, code-block exclusion).
7. 501 stubs on SDK-probed endpoints Atlas does not back (batches / files / audio / images / moderations).

Out of scope (genuine non-goals):
- Embedding / ASR / TTS / image-gen models — Atlas loads one chat/completion
  model at a time; the endpoints return 501 with OpenAI-shaped error bodies.
- Real safety classification — the refusal heuristic recognizes common
  refusal text; it is not a content filter.

## Design

### 1. Response/completion store
New module `crates/spark-server/src/response_store.rs`. In-memory LRU +
TTL map, bounded by count, `parking_lot::Mutex<HashMap<…>>` for
concurrent access. Storage is kind-typed (`Response` vs
`ChatCompletion`) so cross-kind probes return `None`.

Env:
- `ATLAS_STORE_MAX_ENTRIES` — default 10 000.
- `ATLAS_STORE_TTL_SECONDS` — default 86 400 (24 h).

Attached to `AppState` as `Arc<ResponseStore>`; accessed from
`chat_completions` (opt-in `store: true`) and `responses_endpoint`
(always stores when `store != Some(false)` — the Responses API default).

No filesystem persistence. Restart behavior: entries are lost. This matches
OpenAI's observable behavior for the 24-h store (they have durable storage
but the 24-h TTL dominates practical resumes) and avoids a PII / backup story.

### 2. Stateful Responses
`lower_responses_to_chat()` signature change:

```rust
pub fn lower_responses_to_chat(
    r: ResponsesRequest,
    resolve_prior: impl FnOnce(&str) -> Option<Vec<IncomingMessage>>,
) -> Result<ChatCompletionRequest, LowerResponsesError>;
```

When `previous_response_id` is set, the handler passes a closure that
consults `state.response_store`. The returned transcript is prepended to
the new turn's messages. Unknown ids map to
`LowerResponsesError::PriorNotFound → 400 code=response_not_found`.

After the turn completes, `translate_chat_response_to_responses()` stores
the full `{prior_transcript + new user input + assistant output}` under
`resp_<id>`. Next resume sees the complete history.

### 3. Streaming `/v1/responses`
New handler `responses_endpoint_stream()`. Strategy: reuse the existing
`chat_completions_stream` path (so thinking / tool parsing / logprobs /
prefix-cache all work unchanged), then transform the SSE byte stream via
a tokio task that parses `data: <json>` frames and emits typed events:

| Chat chunk                         | Responses event                              |
|------------------------------------|----------------------------------------------|
| (admission)                        | `response.created`                           |
| first `delta.content` token        | `response.output_item.added` (message)       |
|                                    | `response.content_part.added` (output_text)  |
| each `delta.content` frame         | `response.output_text.delta`                 |
| first `delta.tool_calls[].function.name` | close pending message; `response.output_item.added` (function_call) |
| each `delta.tool_calls[].function.arguments` | `response.function_call_arguments.delta`  |
| terminal `finish_reason`           | close open item with `response.*.done`       |
| (terminal)                         | `response.completed` (with final usage)      |

Event types live in `openai.rs::ResponsesStreamEvent`. Each carries a
monotonic `sequence_number` per the 2026-stable spec. SSE frames use
`event: <name>\ndata: <json>\n\n`.

Failure mode: when the inner chat handler returns a non-2xx response, we
forward the status/body unchanged — clients see the same error shape as
for blocking `/v1/responses`.

### 4. Refusal classifier
New module `crates/spark-server/src/refusal.rs`. Prefix-pattern matcher
against a curated list of refusal openers (`"I cannot "`, `"I'm sorry,
but I can't"`, `"As an AI,"`, …), case-insensitive, anchored at the start
of the stripped content. When matched, returns the first sentence
(terminated by `.`, `?`, or `!`). Applied to blocking chat-completions
at response assembly in `api.rs`: when triggered, `message.refusal =
Some(sentence)` and `message.content = None` (per OpenAI: mutually exclusive).

Env kill-switch: `ATLAS_DISABLE_REFUSAL_DETECTION=1` → always `None`,
preserves pre-PR-4 behavior byte-for-byte.

Streaming: not applied mid-stream. Clients that need refusal detection on
streams should either use blocking mode or re-classify accumulated output
client-side.

### 5. Rate-limit middleware
New module `rate_limiter.rs`. Per-identity token-bucket limiter with two
buckets (requests / tokens), refilled linearly from a monotonic clock.
Identity resolution: `Authorization: Bearer` (hashed to avoid leaking
secrets in map keys) → `X-Forwarded-For` → socket peer addr.

Env:
- `ATLAS_RATE_LIMIT_RPM` (default 0 = disabled)
- `ATLAS_RATE_LIMIT_TPM` (default 0 = disabled)
- `ATLAS_RATE_LIMIT_BURST_RPM` (default = RPM)
- `ATLAS_RATE_LIMIT_BURST_TPM` (default = TPM)

When both RPM and TPM are 0 the middleware is a pure passthrough (keeps
the static "unlimited" headers from `openai_observability_middleware`).
When enabled, it runs **before** the observability middleware so its
`x-ratelimit-{limit,remaining,reset}-{requests,tokens}` headers
overwrite the static stubs.

Denied requests return 429 + OpenAI error body + `retry-after` header.
Token cost is estimated conservatively at admission (max_seq_len). True-up
via `refund_tokens(key, reserved - actual)` is supported but not wired
into the streaming path in this PR — a follow-up can thread the final
usage back from `chat_completions_stream`.

### 6. URL annotation extractor
Rewritten `extract_url_annotations()` in `openai.rs`.

- Masking pass: replace bytes inside fenced ``` ``` ``` blocks and inline
  ` ` backtick spans with ASCII spaces (UTF-8 length preserved) so the
  URL scan skips illustrative code (`curl https://example.com`) without
  recomputing indices.
- First pass: markdown `[title](url)` links. The `[title]` text becomes
  the annotation `title`; bare URLs use the URL as title (unchanged).
- Second pass: bare URLs, respecting already-extracted markdown ranges
  (no duplicates).
- Smarter trailing-punctuation stripping: respect `(…)` pair counts so
  Wikipedia-style `Foo_(bar)` URLs survive; strip unmatched `)`, `*`,
  and `_` markdown emphasis markers.
- Sort annotations by start index (document order).

Tests: 11 unit tests in `openai::tests`.

### 7. 501 stubs
`batches_stub`, `files_stub`, `audio_stub`, `images_stub`,
`moderations_stub`, plus `batch_get_stub` / `batch_list_stub`. All return
`StatusCode::NOT_IMPLEMENTED` with an OpenAI-shaped error body. Routes
registered in `main.rs` for the full OpenAI surface so SDK auto-probes
see a clean 501 instead of a silent 404.

## Verification

**Compile-check + unit tests:**
```
cargo build  -p spark-server --release
cargo clippy -p spark-server --lib --no-deps
cargo test   -p spark-server          # 28 new tests across the four new modules
```

**E2E (`tests/test_openai_compat_v2.py`):**
1. `previous_response_id` round-trip ("remember 42" → recall).
2. Streaming `/v1/responses` emits `created` + `output_text.delta` + `completed`.
3. `store: true` + `GET /v1/chat/completions/{id}` returns the stored body.
4. Refusal path returns `refusal: "..."` with `content: null`.
5. Rate limit (requires `ATLAS_RATE_LIMIT_RPM=3` on server) triggers 429 with `retry-after`.
6. URL annotation — bare URL yields a `url_citation` entry.
7. 501 stubs on `/v1/batches`, `/v1/files`, `/v1/audio/speech`,
   `/v1/images/generations`, `/v1/moderations`.

Target sweep: 80B-NVFP4 and 35B-FP8+MTP (same coverage as 2.44).

## Follow-ups — closed in PR 5

All four deferred items shipped. Summary:

### Streaming refusal detection
- `ChunkDelta` gained `refusal: Option<String>` (serde-skipped when None).
- `ChatCompletionChunk::refusal_chunk()` constructor added.
- Chat streaming handler accumulates content into a separate
  `refusal_scan_buf` (bounded at 512 bytes) and, on `StreamEvent::Done`
  with no tool call, runs `refusal::detect` and emits a single
  `delta.refusal` chunk just before the terminal `done_chunk` /
  `usage_only_chunk`. Honest scope: post-hoc signal, not a streaming
  classifier — clients that branch on `delta.refusal != null` see a
  non-null value at end of stream.
- Responses streaming gained `response.refusal.delta` + `response.refusal.done`
  events; the Responses stream transformer recognizes the upstream chat
  `delta.refusal` and forwards it as `RefusalDelta`, emitting
  `RefusalDone` before `response.completed`.

### Rate-limit streaming true-up
- `rate_limiter::RequestContext { identity, reserved_tokens }` added.
- Middleware inserts it into `request.extensions_mut()` on successful
  admission.
- `chat_completions` signature gained `req_ctx: Option<Extension<...>>`.
- Blocking path refunds `reserved - actual` after building the response;
  streaming path refunds on `StreamEvent::Done` and refunds the full
  reservation on `StreamEvent::Error`.
- `responses_endpoint`'s internal re-entry passes `None` (reservation
  was accounted for on the outer request).

### Structured citation parser (`crate::citation`)
- New module recognizing three model-emitted citation patterns:
  - Markdown footnotes: `[^label]` + `[^label]: url title`
  - Numeric refs: `[1]` + `[1] url` (definition at line start)
  - "Sources:" / "References:" / "Citations:" sections with bulleted URLs
- `merged_annotations(content)` wraps `openai::extract_url_annotations`
  + `citation::extract` + `citation::merge_dedupe`. Handlers use this
  everywhere annotations are emitted (blocking chat, blocking responses,
  streaming responses).
- Honest scope: still post-hoc pattern matching — Atlas has no web-search
  tool. The shape clients receive matches what a real retrieval backend
  would emit.

### Persistent store backend
- `StoreBackend` trait with `NoopBackend` (default) and
  `FilesystemBackend` impls.
- Env `ATLAS_STORE_DIR=/path` activates filesystem persistence. Each
  entry writes `{dir}/{id}.json` via write-then-rename for crash-atomic
  updates. Evictions (capacity or TTL) delete the file.
- Startup replay: `FilesystemBackend::replay(ttl)` reads the dir,
  skips entries whose `persisted_at_unix + ttl` is in the past (and
  deletes the stale file), returns surviving entries for in-memory
  load.
- On-disk schema (`DiskEntry`) keeps wall-clock `persisted_at_unix`
  instead of the in-memory `Instant` so TTL survives restarts.
- `IncomingToolCall` / `IncomingFunction` gained `Serialize` derive so
  historical tool calls round-trip through disk cleanly.

## Verification (PR 5)

- `cargo build -p spark-server --bin spark` ✅
- `cargo test -p spark-server --bin spark` — 26 targeted tests pass
  (11 URL extractor + 7 response_store incl. 3 new filesystem tests +
  8 citation).
- `cargo test -p spark-server --lib` — 13 rate_limiter + refusal tests pass.
- Total: **39 unit tests across the two PRs, all green.**

## PR 6 — 2026-stable endpoint coverage

After a fresh pass through the current OpenAI API reference, PR 6 closes
the remaining surface gaps modern SDKs auto-probe:

### Responses CRUD
- `GET /v1/responses/{id}` — retrieve a stored Response. Pulls from
  `response_store` (kind-filtered to `Response`).
- `DELETE /v1/responses/{id}` — evict from memory + filesystem backend
  (if persistent). Returns `{id, object:"response.deleted", deleted:true}`.
- `GET /v1/responses/{id}/input_items` — list the caller's original
  input messages (assistant turn trimmed). Supports `limit` + `order`
  query params; cursor pagination returns `has_more:false` since the
  whole transcript fits in one page for any realistic conversation.
- `POST /v1/responses/{id}/cancel` — returns 400 `response_not_cancellable`
  with a clear message (Atlas completes responses synchronously; cancel
  only applies to `background: true` responses which we don't support).

### New accept-and-ignore fields
**ChatCompletionRequest:**
- `modalities` — `["text"]` default; `["text","audio"]` accepted, audio output silently skipped.
- `audio` — audio-output config, accepted + ignored.
- `prediction` — Predicted Outputs hint, accepted + ignored.
- `web_search_options` — accepted + ignored (no web-search tool).
- `reasoning_effort` — top-level shorthand for gpt-5.x SDKs.

**ResponsesRequest:**
- `background` — accepted + ignored (always synchronous).
- `include` — accepted + ignored.
- `truncation` — accepted + ignored (Atlas has its own `--auto-compact`).
- `conversation` — **implemented** (see below).
- `parallel_tool_calls`, `max_tool_calls` — accepted (Atlas enforces its own).
- `text` — advanced output config accepted + ignored.

### Built-in hosted tool rejection
The ResponsesRequest `tools` field is parsed as raw `Vec<Value>` and
classified in `lower_responses_to_chat`. Function tools pass through;
built-ins (`web_search` / `web_search_preview`, `file_search`,
`computer_use_preview`, `code_interpreter`, `image_generation`, `mcp`,
`local_shell`, `custom_tool`) return a per-tool 400 with a clear message
so clients can fall back.

### Conversations API
New module `crates/spark-server/src/conversation_store.rs` (~300 LOC,
6 unit tests). Same LRU+TTL shape as `response_store` but keyed on
`conv_<uuid>`. Env overrides:
- `ATLAS_CONVERSATION_MAX_ENTRIES` (default 10 000)
- `ATLAS_CONVERSATION_TTL_SECONDS` (default 86 400)

Handlers in `api.rs` cover the full OpenAI spec:
| Endpoint | Method | Behavior |
|----------|--------|----------|
| `/v1/conversations` | POST | Create with optional initial items (≤20) + metadata |
| `/v1/conversations/{id}` | GET | Return `{id, object, created_at, metadata}` |
| `/v1/conversations/{id}` | POST | Merge `metadata` patch |
| `/v1/conversations/{id}` | DELETE | Remove + return deletion envelope |
| `/v1/conversations/{id}/items` | POST | Append items (≤20/call) |
| `/v1/conversations/{id}/items` | GET | List with `limit`/`order` |
| `/v1/conversations/{id}/items/{item_id}` | GET | Retrieve single |
| `/v1/conversations/{id}/items/{item_id}` | DELETE | Remove single |

Responses API integration:
- `conversation: "<conv_id>"` (or `{"id": "..."}`) on a Responses request
  prepends the conversation's items to the turn's input messages.
- After completion, the new user message(s) AND the assistant output are
  appended back to the conversation (best-effort — silent on failure so
  the primary response isn't disrupted).
- Blocking + streaming paths both wired.
- Unknown `conversation` id → 404 with `code: "conversation_not_found"`.

### Files (PR 6 delta)
- NEW: `crates/spark-server/src/conversation_store.rs` (~320 LOC, 6 tests).
- MODIFIED: `openai.rs` — ChatCompletionRequest + ResponsesRequest new
  fields; `ResponsesRequest.tools` changed to `Vec<Value>`;
  `lower_responses_to_chat` classifies built-ins.
- MODIFIED: `api.rs` — 9 new handlers (4 responses CRUD + 8 conversations),
  `responses_endpoint` + `translate_chat_response_to_responses` wired for
  `conversation`, `responses_endpoint_stream` signature gained
  `conversation_id`.
- MODIFIED: `response_store.rs` — added `delete(id, kind) -> bool`.
- MODIFIED: `main.rs` — `AppState.conversation_store`, 13 new routes.

### Verification (PR 6)
- `cargo build -p spark-server --bin spark` ✅
- `cargo test -p spark-server --lib` — 13 passed
- `cargo test -p spark-server --bin spark -- conversation_store:: response_store:: openai:: citation::` — **32 passed**
- **Total: 45 unit tests across PRs 4/5/6, all green.**

## Future follow-ups (not blockers)

- Chunk-level streaming refusal classification (progressive pattern match
  as tokens arrive). Requires an incremental prefix-matching state machine.
- Atomic filesystem replay at runtime for multi-writer deployments.
- `/v1/moderations` backed by a real classifier (currently 501 stub).
- Conversation filesystem persistence (mirror `response_store`'s backend
  trait for durability across restarts).
- Proper cursor pagination (`after` / `before`) on list endpoints — today
  we return `has_more: false` after the limit slice.
