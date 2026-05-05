# Atlas Spark — Alpha 2.44

**Image:**

```bash
docker pull avarok/atlas-gb10:alpha-2.44        # pinned
docker pull avarok/atlas-gb10:2.11              # semver alias
docker pull avarok/atlas-gb10:latest             # floating
```

Digest: `sha256:1b3ce7b3c2750a8927602c0e5520571fea1578d4932a69315a4065d1e194c84c`

**Hardware:** NVIDIA DGX Spark GB10 (sm_121f, 120 GB LPDDR5X, 273 GB/s)
**License:** AGPL-3.0-only
**Commits:** `2b51bd8` → `f4bcd11` → `86ae8e7` on branch `minimax-ep2-complete`

---

## What's New (vs alpha-2.43)

### OpenAI API compatibility — three-PR overhaul

alpha-2.43 was missing several fields and endpoints that modern SDKs (OpenAI Python ≥1.50, LangChain `ChatOpenAI`, Vercel AI SDK, Langfuse/Helicone observability) assume. 2.44 closes the gap without touching inference hot paths.

**Request-field compatibility:**
- `max_completion_tokens` accepted as a `serde` alias for `max_tokens` — the OpenAI Python SDK switched to `max_completion_tokens` by default on reasoning models in ≥1.50.
- `stream_options.{include_usage, include_obfuscation}` parsed; when `include_usage=true` a dedicated terminal SSE chunk carries `usage` before `[DONE]`.
- Accept-and-ignore (prevents 4xx rejection from modern clients): `parallel_tool_calls`, `verbosity`, `service_tier`, `store`, `metadata`, `safety_identifier`, `prompt_cache_key`, `user` (deprecated).
- Response echoes `service_tier` and `metadata` when the request sets them.

**Usage detail structures:**
- `usage.prompt_tokens_details.cached_tokens` — threaded from `RadixTree.lookup()` → `SequenceState.cached_prefix_tokens` → `ActiveSeq.cached_prompt_tokens` → the per-request `Usage`. Matches the `atlas_prefix_cache_hit_tokens_total` Prometheus gauge at per-request granularity. Populated on the chunk-0 prefill of every sequence (`crates/spark-model/src/model.rs:2200`, `2657`, `3394`).
- `usage.completion_tokens_details.reasoning_tokens` — sourced from the existing `thinking_tokens` counter in `ActiveSeq`. o-series-style cost accounting works end-to-end (verified: 35B FP8 thinking path returns `reasoning_tokens: 512`).

**Error shape:**
- `param` populated on known validation sites (`messages`, `temperature`, `top_p`, `max_tokens`) via a new `openai_error_response_with_param()` helper. Matches OpenAI's `"param": "messages[0].role"` style.

**Optional auth gate:**
- `ATLAS_REQUIRE_AUTH=1` — tower middleware rejects `/v1/*` requests lacking `Authorization: Bearer ...` with 401 + JSON error body. Off by default.

**Observability headers** (new `openai_observability_middleware` in `crates/spark-server/src/main.rs`):
- `x-request-id` — echoes a client-supplied value, otherwise fresh UUID v4.
- `openai-processing-ms` — server wall-clock in milliseconds.
- `x-ratelimit-{limit,remaining,reset}-{requests,tokens}` — static "effectively unlimited" stubs so clients that parse them for backoff don't misinterpret absent headers as exhaustion.
- `openai-organization: atlas-local`, `openai-version: 2026-01-01` — parity with `api.openai.com`; some observability wrappers log them.

**New endpoints:**
- `GET /v1/chat/completions/{id}` — stub returning 404 "Completion storage is not enabled". Lets Helicone/Langfuse-style logging clients that auto-detect this round-trip fall back cleanly instead of hanging.
- `POST /v1/responses` — non-stateful adapter over chat completions. Accepts string-or-array `input`, `instructions`, `max_output_tokens`, and the common sampling fields, lowers into a `ChatCompletionRequest`, runs the existing pipeline, and re-emits the result in `ResponsesResponse` shape (`output` items: `message` with `output_text` parts, `function_call` entries). Streaming and `previous_response_id` (stateful conversations) are rejected with a 4xx + parameter-scoped error so clients fall back cleanly.

**Assistant-message extensions:**
- `message.annotations[]` — OpenAI `url_citation` entries auto-extracted from URLs in `content`. Omitted when empty; wire format unchanged for non-web-search responses.
- `message.refusal: Option<String>` — safety-aware clients that access `message.refusal` no longer panic on a missing key. Atlas does not currently classify refusals; the field stays `None` until a refusal detector lands.

### MiniMax M2.7-NVFP4 EP=2 — full-suite PASS

alpha-2.43 could run MiniMax M2/M2.7 single-GPU but EP=2 was fragile. 2.44 lands four critical fixes:

- `rms_norm` scale propagation across the EP boundary.
- `norm_topk_prob` — worker's expert-logit softmax matches head byte-for-byte (was drifting).
- FP8-free routing path on sm_121 (worker was falling into an FP8 path that doesn't exist on GB10).
- Template-forced thinking detection — MiniMax's chat template hardcodes `<think>\n`, so the EP=2 path needs to treat the opening `<think>` as seeded, not as a spontaneous emission.

**Observable effect:** M2.7-NVFP4 EP=2 suite goes from partial to Coh 3/3, Fib PASS, Tools 2/2, TPS 4/4.

### Suite hardening — 3 models recovered

- **Mistral-Small-4:** 4/13 → 13/13. Fixed the BF16-paged dispatch bug where the FP8 kernel was firing on a BF16 cache and returning NaN, plus the `reasoning_effort="none"` request field that had been silently dropped.
- **Gemma-4-31B:** 8/13 → 12/13. BF16 KV default + per-model `repetition_penalty` preset (new in this release — threaded as a `SamplingCategory` field through `atlas-kernels/build.rs`).
- **80B-MTP fib regression** healed through the scheduler rebuild (off-by-one in `seq_len += k-1` + MTP scheduler bootstrap precondition fix from commit `2b51bd8`).

### Scheduler / kernel fixes

- `SequenceState.cached_prefix_tokens` field added. All three prefix-cache call sites in the model write the match size so the scheduler can surface it at usage time.
- EP=2 KV budget / long-context OOM — bumped `--max-seq-len` default from 16384 to 32768 in `tests/run_all_models.py` so EP=2 runs don't exhaust the KV pool on large prompts.
- `ActiveSeq` / `SwappedSeq` gained `cached_prompt_tokens: u32`, populated right before struct construction from `seq.cached_prefix_tokens` (local binding pattern to avoid borrow-after-move).

---

## Benchmark — alpha-2.44 vs alpha-2.43

| Model | 2.44 | vs 2.43 | Coh | Fib | Tool | LC | Max tok/s |
|---|---|---|---|---|---|---|---|
| 35B-nvfp4-mtp | 9/9 | = | 3/3 | PASS | 2/2 | 3/3 | 113.8 |
| 35B-nvfp4 | 9/9 | = | 3/3 | PASS | 2/2 | 3/3 | 90.2 |
| 80B-nvfp4-mtp | 9/9 | = | 3/3 | PASS | 2/2 | 3/3 | 87.4 |
| 80B-nvfp4-ep2-mtp | 9/9 | = | 3/3 | PASS | 2/2 | 3/3 | 84.7 |
| 35B-fp8 | 9/9 | = | 3/3 | PASS | 2/2 | 3/3 | 70.8 |
| qwen3-vl-30B | 9/9 | = | 3/3 | PASS | 2/2 | 3/3 | 68.4 |
| 80B-nvfp4 | 9/9 | = | 3/3 | PASS | 2/2 | 3/3 | 64.9 |
| 122B-nvfp4-ep2-mtp | 9/9 | = | 3/3 | PASS | 2/2 | 3/3 | 38.3 |
| 122B-nvfp4 | 9/9 | = | 3/3 | PASS | 2/2 | 3/3 | 31.9 |
| mistral-small-4 | **13/13** | **+4** | 3/3 | PASS | 2/2 | 3/3 | 34.8 |
| coder-next-fp8 | 8/9 | = | 3/3 | PASS | 2/2 | 2/3 | 45.3 |
| gemma-4-31B | **12/13** | **+5** | 3/3 | PASS | 2/2 | 3/3 | 11.4 |
| gemma-4-26B | 7/9 | = | 3/3 | PASS | 1/2 W1 | 2/3 | 73.4 |
| minimax-m2.7-nvfp4-ep2 | **full PASS** | **new** | 3/3 | PASS | 2/2 | 4/4 | — |
| nemotron-nano-30B | 7/9 | = | 3/3 | FAIL | 1/2 W1 | 3/3 | 88.1 |
| nemotron-super-120B-ep2 | 4/9 | = | 2/3 | FAIL | 0/2 W2 | 2/3 | 27.2 |

Pass rate moved from **123/135 (91.1%)** → **pass-3 sweep at 97.4%, 14/19 perfect** across the expanded suite. Headline throughput numbers are unchanged (no hot-path edits this release).

### Known issues (not fixed in 2.44)

- **Nemotron-Super-120B tool calls** — still fails (0/2 W2). Model-level resistance to `qwen3_coder` XML tool format despite its own template specifying that shape; three different mitigations attempted (tool-steering prefix, `thinking_in_tools=true`, steering-prefix removal), each produced different degenerate outputs. Carried over to the next alpha.
- **gemma-4-26B LC16k** repetition loop past ~7k tokens on NVFP4 — model-quality issue, not Atlas.
- **gemma-4-26B Search** tool-call extraction still WARN — model emits unquoted prose under thinking-in-tools budget pressure.
- **nemotron-nano-30B fib** — model emits buggy code (`[0,1,1,1,2,…]` off-by-one). Model-quality issue.

---

## Verification

OpenAI SDK smoke test (`/tmp/test_openai_compat.py`, 9 checks) passes on both backends:

- **80B NVFP4** (blocking + streaming + thinking off): 9/9 PASS
- **35B FP8 + MTP** (blocking + streaming + thinking on): 9/9 PASS, `reasoning_tokens: 512` returned correctly

Coverage:
1. `max_completion_tokens` alias
2. `stream_options.include_usage` terminal chunk
3. `usage.{prompt,completion}_tokens_details` populated
4. `service_tier` + `metadata` echo
5. Observability headers (`x-request-id`, `openai-processing-ms`, rate-limit stubs, org/version)
6. `param` on validation errors
7. `/v1/responses` adapter (blocking, text in → text out)
8. `GET /v1/chat/completions/{id}` 404 stub
9. `message.annotations[]` URL citation extraction
