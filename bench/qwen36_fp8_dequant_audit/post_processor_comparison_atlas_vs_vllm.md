# Atlas vs vLLM: Post-Processor Comparison

Scope: everything that runs *after* the model emits logits and *before* the SSE
chunk leaves the server. Pre-decode logit shaping (rep/presence/freq penalties)
and engine-mechanical stuff (KV/MTP/SSM) are out of scope.

vLLM references = local pip `0.17.0` at
`/workspace/.local/lib/python3.12/site-packages/vllm/` cross-checked against
`vllm-project/vllm@main` (HEAD as of 2026-05-24). Where main differs, both SHAs
are cited.

---

## 1. Repetition detection — vLLM is structurally better

**Atlas.** Multiple scan-anywhere detectors:
- `detect_token_loop` (helpers.rs:494) — period `p ∈ [period_min, period_max]`,
  scans last `scan_window` tokens, counts non-overlapping needle occurrences
  anywhere in the tail (content: 2..64, ≥2 reps, 280-win; thinking: 4..20, ≥3, 160-win).
- `detect_fuzzy_repetition` (Hamming-tolerant) for paraphrase loops
  (`decode_logits_step.rs:476`).
- Digit-normalized variant (`detect_content_token_loop_normalized`,
  helpers.rs:433) collapses numeric runs to sentinel.
- Streaming-side SimHash semantic guard (`handle_token.rs:388-404`) +
  `loop_scan_buf` line-level watchdog (sanitizer.rs:226-).

**vLLM.** Single anchored-at-end detector. `_has_repeating_pattern`
(`v1/core/sched/utils.py:10-25`) only checks whether the last `pattern_len`
tokens equal the `(min_count-1)` preceding `pattern_len`-blocks exactly:

```python
for n in range(1, pattern_len + 1):
    target_token = token_ids[-n]
    for m in range(1, repetition_min_count):
        if token_ids[-(pattern_len * m + n)] != target_token:
            return False
return True
```

It's `O(pattern_len × min_count)` per check vs Atlas's
`O(period × scan_window)` × periods, and runs every step in `check_stop`
(`utils.py:119`) — no stride throttle.

| Axis | Atlas | vLLM |
|---|---|---|
| Pattern anchor | Anywhere in tail | Tail-anchored to last `N` tokens |
| Detects A B C D A B C D `X` ? | Yes (with `min_repeats=2`) | No (last block has `X` not `D`) |
| Tolerates noise between repeats | Yes | No |
| Cost | O(period × window) per check | O(period × min_count) per check |
| Fuzzy/paraphrase | Yes (Hamming) | No |
| Digit-normalized | Yes | No |
| Configurable per-request | Boot-global | `RepetitionDetectionParams` per `SamplingParams` (`sampling_params.py:111-144`) |

**Verdict.** Atlas's algorithm is more powerful for known failure modes
(Qwen3.5 fence-narration, Qwen3.6 numbered-list loops, MoE-routing
sentence collapse) the strict anchored detector misses. vLLM's design
is cheaper and per-request `RepetitionDetectionParams` is a config win.

**Improvement opportunity (minor).** Expose
`RepetitionDetectionParams`-style per-request overrides through
`SamplingParams`. Boot-global `watchdog_params()` (helpers.rs:300) is
wrong granularity when a model serves both chat and structured-JSON.

---

## 2. Streaming stop-string handling — vLLM is meaningfully better

**Atlas.** `handle_token.rs:275-302` accumulates content into
`state.accumulated_content`, calls `find()` for each stop string, on hit
truncates `delta` and sets `state.stop_string_triggered = true`. There is no
"hold back N chars" buffer — a stop string that lands at end-of-delta is
emitted to the client before the next delta arrives and reveals the match.
Atlas also tokenizes stop strings server-side
(`inference_impl::tokenize_stop_sequences`) and matches on token IDs at
the scheduler level — but the *streaming* layer above doesn't know about
that.

**vLLM.** `IncrementalDetokenizer.update`
(`v1/engine/detokenizer.py:97-144`) maintains a `stop_buffer_length =
max(len(stop_str)) - 1` retention window
(`detokenizer.py:88-91`) so the last `len-1` characters of a partial stop
match are held back from the client until either the match completes (text
truncated to `stop_index`, `detokenizer.py:140-142`) or the buffer slides
past. `check_stop_strings` (`detokenizer.py:306-341`) searches only the new
chars by starting at `1 - new_char_count - stop_string_len`
(`detokenizer.py:327`) — an O(stop_len) seek per new char, not O(full
output).

**Verdict.** vLLM wins on three counts:
1. **Hold-back buffer.** Stop strings that span the chunk boundary won't
   leak partial matches to the client (`"<|im_st"` then `"art|>"`).
   Atlas's `chat_stream` has no equivalent — `<|im_start|>` text would
   stream as `<|im_st` then `art|>` if it weren't already a single token.
2. **Bounded search.** vLLM searches the new tail window; Atlas does
   `accumulated_content.find(stop_str)` which scans the full content every
   delta (`handle_token.rs:278`).
3. **Truncation precision.** vLLM returns
   `(stop_string, truncate_to_offset)` so the engine knows exactly where
   to cut. Atlas truncates the *delta* but never trims
   `accumulated_content`, leaving a drift between the accumulator and
   what was streamed.

**Improvement opportunity (worth doing).** Port `stop_buffer_length` to
Atlas's `chat_stream::handle_token` — hold back `max_stop_len - 1` chars
per delta until the next delta arrives. Combine with bounded-window
search using a sliding offset. ~30 LoC change, fixes a real bug class.

---

## 3. Force-end-thinking / `</think>` machinery — Atlas is more sophisticated

**Atlas.** Multi-layer:
- **Budget arming** (`decode_logits_step.rs:201-211`).
- **F2 confidence early-stop** (`decode_logits_seq.rs:62-87`): arms after
  60 consecutive ≥0.95 top-1 tokens.
- **Thinking-loop watchdog** (`decode_logits_step.rs:216-232`).
- **Boundary defer** (`confidence.rs:100-116`, `should_inject_think_end`):
  defers injection until (a) sentence boundary, (b) `\`\`\`` fence closes,
  or (c) hard override (`MAX_SENTENCE_DEFER_TOKENS=64`, 3× budget,
  ceiling 2048).
- **Mid-word `</think>` suppression** (`decode_logits_seq.rs:107-118`):
  masks `</think>` when prev token ends mid-word — catches FP8 drift bias.
- **Post-close masking** (`decode_logits_seq.rs:122-148`): masks both
  `</think>` and `<think>` after close (re-entry guard with decayed
  budget on watchdog re-fires).
- **Post-think min-content gate** (`decode_logits_step.rs:359-368`):
  suppresses EOS for 16 tokens after `</think>` so tool call can open.

**vLLM.** `ThinkingBudgetStateHolder`
(`v1/sample/thinking_budget_state.py`, present on main, NOT in 0.17.0).
Token-budget tracker only:
- Per-request via `params.thinking_token_budget`.
- On budget exhaustion, `_apply_forcing_to_logits` index-puts `1e9`
  into `think_end_token_ids` row.
- Handles multi-token end markers via `end_count` iteration.
- **No sentence-boundary defer. No code-fence defer. No mid-word guard.
  No confidence early-stop. No re-entry mask.**

Reasoning-parser layer (`qwen3_reasoning_parser.py`) only does post-hoc
text extraction; it doesn't influence sampling.

**Verdict.** Atlas's force-end logic is genuinely better than vLLM's
across the dimensions Atlas already debugged (mid-word cuts, fence
splits, re-entry loops, confidence-driven early stop, post-close
content-budget). vLLM's only structural advantage is **per-request**
budget (`thinking_token_budget` in SamplingParams) vs Atlas's
boot-global `max_thinking_budget`.

**Improvement opportunity (small).** Add `thinking_token_budget` as a
per-request `SamplingParams` field in Atlas (overrides
`max_thinking_budget`), then keep all of Atlas's deferral logic on top.
Lets clients hint "this is a quick lookup, cap reasoning at 256" without
restarting the server.

---

## 4. Tool-call open/close + grammar interaction — different design, neither dominates

**Atlas.** Scheduler-side xgrammar (`grammar_state` per `ActiveSeq`),
advanced via `gs.accept_token` (`decode_logits_step.rs:171`), mask via
`gs.apply_bitmask_to_logits` (`decode_logits_seq.rs:330`). Has a
**forced-token fast-path** (`decode_logits_seq.rs:307-317`) emitting
`gs.forced_token()` directly when grammar admits one legal token —
skips sampling AND mask fill. Phase-gated penalty scoping
(`decode_logits_seq.rs:395-421`) zeros DRY/presence/freq inside tool body.

**vLLM.** `XgrammarBackend` with batched-GPU `apply_grammar_bitmask`
(`v1/structured_output/utils.py:48-`). No forced-token fast-path in OSS
(comment at `backend_xgrammar.py:137` references `find_jump_forward_string`
but matcher isn't wired to short-circuit sampling).

**Verdict.** Atlas's forced-token fast-path and penalty-scoping are
unique wins. vLLM wins on **batched GPU bitmask** vs Atlas's per-seq CPU
mask — `decode_logits_step.rs:33-41` forces the whole batch onto host
when any seq has grammar.

**Improvement opportunity (real perf win).** Keep bitmask GPU-side when
batch has grammar; fall back to host only when seq also has
`inside_thinking || think_ended || top_logprobs`. Reference:
`v1/structured_output/utils.py:48-`.

---

## 5. Per-token sampler post-processing pipeline shape

**Atlas.** Monolithic `process_seq_logits` (~430 lines, per-sequence,
host-side): F2 confidence → mid-word mask → post-close mask →
tool-during-thinking mask → forced `</think>` → pin-to-tool-call mask
→ forced-token fast-path → grammar bitmask → adaptive sampling → sample.

**vLLM.** Pipelined. `LogitsProcessors` registry
(`v1/sample/logits_processor/state.py:148`) stacks `LogitsProcessor`
instances (`MinP`, `LogitBias`, `MinTokens`, `ThinkingBudgetStateHolder`,
user-registered). Each `.apply(logits)` runs on the **batched** GPU
tensor. `is_argmax_invariant()` lets engine skip greedy-invariant
processors. Cleaner, supports plugins.

**Verdict.** vLLM's architecture is cleaner and faster. Atlas has more
capability per processor but it's crammed into one host-side function.

**Improvement opportunity (medium effort).** Refactor into a
`LogitProcessor` trait pipeline — enables per-request disable, easier
testing, GPU batching, plugins. 1-2 week refactor.

---

## 6. Sanitizer / output cleanup — Atlas wins outright

**Atlas.** `sanitize_content_chunk` (`api/sanitizer.rs:51-224`) is a
state machine: orphan-open tag suppression, envelope tracking
(`<minimax:tool_call>`), boundary-straddle buffering, parser-driven
`LeakMarkers`. Plus role-literal stripping (`handle_token.rs:262`),
`strip_all_preserving_boundary` for `</parameter>` / `userassistant`
leaks, SimHash semantic guard, and `tool_salvage::salvage` as last-
resort recovery on watchdog fire (`handle_token.rs:425-456`).

**vLLM.** Nothing equivalent. `ToolParser.extract_tool_calls`
(`qwen3coder_tool_parser.py:291-341`) does post-hoc regex on full
output; if model emits leaked `<tool_call>` without close, vLLM returns
`tools_called=False` and the broken text reaches the client. No
orphan suppression, no envelope state, no salvage.

**Verdict.** Atlas is doing real work here that vLLM doesn't. Keep it.

---

## Summary of actionable items

Ranked by ROI:

1. **Stop-string hold-back buffer** (Section 2) — 30 LoC, fixes real
   client-visible leak bug. Port `stop_buffer_length` from
   `v1/engine/detokenizer.py:88-91`.
2. **GPU-resident grammar bitmask** (Section 4) — Real throughput win
   when a batch has any constrained sequence. Reference:
   `v1/structured_output/utils.py:48-`.
3. **Per-request `thinking_token_budget` field** (Section 3) — Small
   API change, lets clients hint reasoning length. Reference:
   `v1/sample/thinking_budget_state.py:89-94`.
4. **Per-request `RepetitionDetectionParams`** (Section 1) — Same
   shape, expose Atlas's much richer watchdog params per-request.
5. **`LogitsProcessor` trait pipeline** (Section 5) — Long-term
   architecture cleanup, not urgent.

Atlas keeps the win on: mid-word/fence/boundary-aware `</think>`
forcing, forced-token fast-path, output sanitizer, sampler phase
scoping inside tool body, fuzzy + digit-normalized repetition, SimHash
semantic guard, tool-call salvage.

---

## Source URLs

Atlas:
- `/workspace/atlas-mtp/crates/spark-server/src/scheduler/decode_logits_seq.rs`
- `/workspace/atlas-mtp/crates/spark-server/src/scheduler/decode_logits_content.rs`
- `/workspace/atlas-mtp/crates/spark-server/src/scheduler/decode_logits_step.rs`
- `/workspace/atlas-mtp/crates/spark-server/src/scheduler/emit_step.rs`
- `/workspace/atlas-mtp/crates/spark-server/src/scheduler/confidence.rs`
- `/workspace/atlas-mtp/crates/spark-server/src/scheduler/helpers.rs`
- `/workspace/atlas-mtp/crates/spark-server/src/api/sanitizer.rs`
- `/workspace/atlas-mtp/crates/spark-server/src/api/chat_stream/handle_token.rs`

vLLM (local pip 0.17.0):
- `/workspace/.local/lib/python3.12/site-packages/vllm/sampling_params.py`
- `/workspace/.local/lib/python3.12/site-packages/vllm/v1/core/sched/utils.py`
- `/workspace/.local/lib/python3.12/site-packages/vllm/v1/engine/output_processor.py`
- `/workspace/.local/lib/python3.12/site-packages/vllm/v1/engine/detokenizer.py`
- `/workspace/.local/lib/python3.12/site-packages/vllm/v1/sample/logits_processor/builtin.py`
- `/workspace/.local/lib/python3.12/site-packages/vllm/v1/structured_output/backend_xgrammar.py`
- `/workspace/.local/lib/python3.12/site-packages/vllm/v1/structured_output/utils.py`
- `/workspace/.local/lib/python3.12/site-packages/vllm/reasoning/qwen3_reasoning_parser.py`
- `/workspace/.local/lib/python3.12/site-packages/vllm/reasoning/abs_reasoning_parsers.py`
- `/workspace/.local/lib/python3.12/site-packages/vllm/tool_parsers/qwen3coder_tool_parser.py`

vLLM (GitHub main, SHAs as of 2026-05-24):
- `vllm/v1/core/sched/utils.py` SHA `c7cb6b94367e7ed331947c7ad27196165054c8ff` (identical to 0.17.0)
- `vllm/v1/engine/detokenizer.py` SHA `4700eecb59a7686a150ba6db934b076300c98088` (identical)
- `vllm/v1/sample/thinking_budget_state.py` (NEW on main, NOT in 0.17.0) —
  https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/thinking_budget_state.py
- `vllm/reasoning/qwen3_reasoning_parser.py` SHA `e38b0de3d822751d88156dd354aecb0b4d65cc7f`
- PR #20859 (merged) — Thinking Token Budget feature:
  https://github.com/vllm-project/vllm/pull/20859
- PR #37112 (closed before merge) — alternative ReasoningBudgetLogitsProcessor:
  https://github.com/vllm-project/vllm/pull/37112
