# Agentic Quality Research Synthesis — 16 Agents, Complete

## CRITICAL BUGS FOUND

### BUG 1: N-gram & self-speculative decode bypass grammar (CONFIRMED)
**Agent: MTP+Grammar** — `scheduler.rs` lines 723 and 728: N-gram and self-speculative decode paths have no `grammar_state.is_none()` guard. Grammar-constrained requests entering these paths produce unconstrained output. **Fix**: Add `&& active[0].grammar_state.is_none()` to both conditions.

### BUG 2: `flush()` drops buffered content after ANY tool call (CONFIRMED)
**Agent: Tool Parser Audit** — `tool_parser.rs` lines 988-994: When `emitted_tool_calls` is true, `flush()` silently discards remaining buffer. Second+ tool calls in streaming are lost. **Fix**: Remove the `emitted_tool_calls` early-return guard.

### BUG 3: Stop-string content leak (CONFIRMED)
**Agent: Streaming Audit** — `api.rs` lines 1052-1117: After `stop_string_triggered = true`, tokens continue flowing through the streaming decoder and are emitted to the client. **Fix**: Break out of content emission when stop triggered.

### BUG 4: Re-think transition corrupts content_decoder (CONFIRMED)
**Agent: Streaming Audit** — `api.rs` lines 1041-1043: When model re-opens `<think>`, `all_toks` has content-phase tokens but `emitted=0`, so ALL prior content is re-emitted as reasoning. **Fix**: Clear `all_toks` on re-think transition.

### BUG 5: Anthropic handler never decrements REQUESTS_ACTIVE (CONFIRMED)
**Agent: Streaming Audit** — `anthropic.rs` Done/Error branches never call `REQUESTS_ACTIVE.dec()`. The gauge increases monotonically. **Fix**: Add `dec()` calls.

### BUG 6: Anthropic handler ignores stop sequences (CONFIRMED)
**Agent: Streaming Audit** — `anthropic.rs` line 1078: Parameter named `_stop_sequences` (underscore = unused). **Fix**: Apply stop sequences to streaming output.

### BUG 7: Anthropic handler O(n²) decode (CONFIRMED)
**Agent: Streaming Audit** — `anthropic.rs` line 1176: Full `decode(&all_toks)` every token. O(n²) total. **Fix**: Use `StreamingDecoder` like the OpenAI path.

### BUG 8: `parse_bare_function_calls` only finds first call (CONFIRMED)
**Agent: Tool Parser Audit** — `tool_parser.rs` lines 744-773: Uses `text.find(...)` which returns first match only. Multiple bare function calls are dropped.

### BUG 9: SwappedSeq discards grammar_state (CONFIRMED)
**Agent: MTP+Grammar** — `scheduler.rs` line 3175: Sequences swapped to disk lose grammar enforcement permanently.

### BUG 10: Reasoning duplicate in streaming (CONFIRMED, ALREADY FIXED)
**Agent: Reasoning Bug** — The full-reasoning emission on `</think>` boundary was the bug. Code already emits only the residual delta (fixed in prior session).

## TOP TECHNIQUES TO IMPLEMENT

### 1. XGrammar `at_least_one` (Agent: XGrammar Research)
Requires forking xgrammar-rs to add `compile_structural_tag_json()` that passes raw JSON with `at_least_one` and `stop_after_first` fields. Maps to:
- `tool_choice="auto"`: `at_least_one=true, stop_after_first=false` (must produce ≥1 tool call)
- `tool_choice="required"`: `at_least_one=true, stop_after_first=true` (exactly one tool call)
**Important nuance**: `at_least_one=true` forces the first token into a tag begin — no free-text reasoning before the first tool call.

### 2. Adaptive sampling (Agent: Adaptive Sampling)
Zone-based temperature adjustment with 4 zones: FreeText, Thinking, ToolCall, StructuredOutput.
- ToolCall zone: clamp temp to ≤0.3
- Greedy-threshold gate (arXiv:2510.05987): if top-1 prob ≥ 0.9, use argmax regardless of temp
- Entropy-based diversity injection: boost temp when consecutive low-entropy > 8 tokens
- LZ compression ratio monitoring every 16 tokens
**Key constraint**: Disabled during MTP verify (argmax-only acceptance). Applied to bootstrap + non-MTP decode.
**Effort**: ~180 lines new `adaptive_sampler.rs` + ~35 lines scheduler changes.

### 3. DRY sampling from llama.cpp (Agent: llama.cpp Research)
Z-algorithm O(n) sequence matching with exponential penalty + sequence breakers.
- `penalty = multiplier * base^(match_length - allowed_length)`
- Sequence breakers (newlines, colons, quotes) reset tracking — critical for JSON/tool calls
- Params: `dry_multiplier=0.8, dry_base=1.75, dry_allowed_length=2`
**Higher value than LZ penalty** because of sequence breakers that prevent false positives in structured output.

### 4. Speculative edits / predicted output (Agent: Production Practices)
Cursor's killer feature: 1000+ tok/s for code edits. The existing file content is the draft (>90% acceptance rate for edits). Atlas already has MTP verify infrastructure — needs to accept an external draft sequence via API parameter `predicted_output`.
**Highest throughput win** for coding workloads: 5-13x speedup.

### 5. Template caching — precompiled Jinja env (IMPLEMENTED)
Already done in this session. Eliminates per-request environment creation + template compilation.

### 6. Prefix cache hit rate metric (IMPLEMENTED)
Already done in this session. Global atomic counters in `prefix_cache.rs`, exposed via `/metrics`.

### 7. Entropy monitoring (IMPLEMENTED)
Already done in this session. Shannon entropy computed from post-softmax distribution, exposed via `/metrics`.

### 8. Validation error logging (IMPLEMENTED)
Already done in this session. Errors logged server-side, not injected as assistant text.

### 9. Warm-up prefill at startup (Agent: Multi-turn KV Cache)
Pre-compute system prompt KV+SSM cache at server start via `--warmup-prompt` flag. Eliminates cold-start penalty (~196ms TTFT saved on first request). Low effort.

### 10. Cache decode output in radix tree (Agent: Multi-turn KV Cache)
After generation completes, insert prompt+output tokens into radix tree. Multi-turn sessions reuse prior assistant responses. 5-15ms TTFT per turn in deep conversations.

## QWEN3.5 SERVING GUIDE (Agent: Qwen3.5 Research)

### Critical Issues
- **Chat template breaks KV cache reuse** (QwenLM/Qwen3 #1826): Stock template renders historical assistant turns without `<think></think>` tags, causing token prefix mismatch. Atlas's override template in `jinja-templates/` may need this fix.
- **KV cache 7x overestimation in vLLM**: GDN layers get attention-sized allocations despite O(1) state. Atlas's custom allocator avoids this.
- **DeepGemm FP8 accuracy loss on Blackwell**: E8M0 scale format. Set `VLLM_USE_DEEP_GEMM=0`.

### Optimal Parameters (confirmed)
- Thinking mode: temp=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
- Non-thinking: temp=0.7, top_p=0.8, top_k=20, presence_penalty=1.5
- Tool calling: temp=0.6, top_p=0.95, top_k=20 (already in MODEL.toml)

## ARXIV LITERATURE (Agent: Arxiv Survey)

### Top papers for Atlas:
1. **LZ Penalty** (arXiv:2504.20131) — Already implemented in Atlas
2. **DCCD** (arXiv:2603.03305) — **NOT RECOMMENDED**: 4x throughput regression, marginal quality gain at 80B scale
3. **Grammar-Aligned Decoding** (arXiv:2405.21047) — Preserves model's ranking under grammar constraints
4. **Selective Sampling** (arXiv:2510.01218) — Dynamic greedy/temp switching per-token
5. **Don't Break the Cache** (arXiv:2601.06007) — Static tool schemas first, dynamic results last
6. **DeRep** (arXiv:2504.12608) — Post-processing code repetition fix, zero engine changes

## SGLang INSIGHTS (Agent: SGLang Research)

### Key techniques to adopt:
1. **Semantic radix cache eviction** (PR #20088): `semantic_event: "reset"` prunes stale branches. 25% TTFT improvement.
2. **Overlapped constrained decoding** (PR #15623): Grammar bitmask runs concurrent with GPU forward. 5-30% latency reduction.
3. **Grammar bitmask pooling**: SGLang leaks ~6-10 MiB/request. Atlas should pool/cache compiled grammars.
4. **`parallel_tool_calls` parameter**: `stop_after_first=true` for single tool call enforcement.

## PRIORITY IMPLEMENTATION ORDER

### Immediate (bug fixes, 1-2 days)
1. **Fix grammar bypass in n-gram/self-speculative** — Add grammar guards to scheduler lines 723, 728
2. **Fix flush() dropping buffered content** — Remove emitted_tool_calls guard
3. **Fix stop-string content leak** — Break content emission on stop trigger
4. **Fix Anthropic REQUESTS_ACTIVE leak** — Add dec() calls
5. **Fix Anthropic O(n²) decode** — Use StreamingDecoder

### Short-term (features, 1-2 weeks)
6. **Adaptive sampling** — Zone-based temperature + greedy threshold + entropy diversity
7. **DRY sampling** — Z-algorithm + sequence breakers (replaces/augments LZ penalty)
8. **Warm-up prefill** — Pre-compute system prompt cache at startup
9. **Cache decode output** — Insert prompt+output into radix tree after generation

### Medium-term (2-4 weeks)
10. **XGrammar `at_least_one`** — Fork xgrammar-rs, add structural tag JSON method
11. **Speculative edits** — Accept predicted_output from API, use as draft for verify
12. **Overlapped constrained decoding** — Grammar bitmask concurrent with GPU forward

### Long-term (1+ months)
13. **PEG autoparser** — Differential template analysis → auto-generated grammar
14. **Grammar-Aligned Decoding** — ASAp for provably correct constrained distribution
