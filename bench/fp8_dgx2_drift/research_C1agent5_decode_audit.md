# Decode Logits Pipeline Audit (C1 Agent 5)

**Date**: 2026-05-26 | **File**: `/workspace/atlas-mtp/crates/spark-server/src/scheduler/decode_logits_seq.rs`  
**Scope**: Complete execution order of logit transformations; identification of 6 shipped fixes; feasibility for margin-ratio detector hook.

---

## 1. Pipeline Execution Order (Canonical Sequence)

Entry point: `process_seq_logits()` (line 9). Token flows:

| Step | Operation | What It Does | Gate |
|------|-----------|--------------|------|
| 1 | **Dequant** | BF16→FP32 or direct FP32 per `elem_bytes` | Always |
| 2 | **F2 Confidence Early Stop** | Arm `force_end_thinking` if top-1 prob ≥0.95 for 30 consecutive tokens after 400 thinking tokens | `!disable_watchdogs && inside_thinking && !force_end_thinking && thinking_tokens ≥ 400 && confidence_early_stop` |
| 3 | **Mid-Word `</think>` Defer** | Suppress `</think>` if prev token ends mid-word (e.g. "thep" suffix in mid-word mask) | `inside_thinking && think_end_token && prev_tok in mid_word_mask` |
| 4 | **Post-Close Think Mask** | After `think_ended=true`: mask `</think>` and `<think>` to prevent re-entry | `think_ended && (think_end_token OR think_start_token)` |
| 5 | **Tool During Thinking** | Hard-mask `<tool_call>` inside thinking (unparsable); soft-bias (−12.0) if tool-loop detected | `inside_thinking` → `-inf`; `!inside_thinking && suppress_tool_call` → `−12.0` |
| 6 | **Forced `</think>` Injection** | Blanket-mask to `</think>` when budget exhausted OR confidence fired OR sentence boundary reached (deferred in code fence) | `inside_thinking && should_inject_think_end(force_end_thinking, in_code_fence, at_sentence_boundary, defer_hard_override)` |
| 7 | **Pin to `<tool_call>` Post-Think** | One-shot: mask all logits except `tool_call_start_token` after `</think>` when `require_tool_call=true` | `think_just_ended && require_tool_call && !tool_call_opened` |
| 8 | **Tier-1 Forced-Token Fast-Path** | Return sole grammar-legal token directly; skip mask+sample. **SKIP if**: inside thinking, logprobs requested, or inside empty param body (Tier-1 gate) | `!inside_thinking && top_logprobs.is_none() && !tier1_active && grammar admits exactly 1 token` |
| 9 | **Grammar Bitmask Apply** | Apply xgrammar's next-token bitmask to logits | `!inside_thinking && grammar_state.fill_bitmask()` |
| 10 | **Adaptive Sampling Zone Update** | Observe entropy, check greedy gate, decide effective temperature | `adaptive_sampling=true` (flag disabled by default) |
| 11 | **Logit Bias Merge (Local)** | Build `logit_bias_local` by cloning `a.logit_bias` and adding Tier-1 parameter-body masks + POST_THINK floor | Always (may be empty or extended) |
| 12 | **Penalty Parameters Build** | Pass `repetition_penalty`, `presence_penalty`, `frequency_penalty` (full strength now), `dry_multiplier` (zeroed in tool body), `lz_penalty` (zeroed if grammar active), `temperature` | Always |
| 13 | **Sample with Params History** | Apply all penalties + logit_bias → temperature decision (greedy or stochastic) → argmax or multinomial → `sampled` token ID | Always |
| 14 | **Top-K Logprobs Extraction** | Extract top-K logprobs from post-pipeline f32_logits if `top_logprobs` requested | `a.top_logprobs.is_some()` |
| 15 | **Return** | `(sampled: u32, logprobs: Option<TokenLogprobs>)` to `decode_logits_step` | Always |

---

## 2. Six Shipped Fixes: Location & Description

### **A1 (2026-05-26): Full Penalties Inside Tool Body**

**File:Line** `/workspace/atlas-mtp/crates/spark-server/src/scheduler/decode_logits_seq.rs:406–428`

**Description**: Phase-3.1 (2026-04-25) zeroed ALL penalties inside tool-call body to avoid penalizing legitimate JSON (`":"`, `","`). This was the root cause of the worst attractor patterns: runaway bash commands with mismatched parens, `lean://` prefix loops, same-tool-call repetition. **Fix**: Restore `repetition_penalty`, `presence_penalty`, `frequency_penalty` at full strength inside tool body. DRY stays zeroed (n-gram heuristic legitimately fights short JSON repetitions; the trade-off risk is higher). The 9% soft downweight (rep_penalty=1.10, window=256) does not flip JSON tokens due to strong margins from xgrammar + training.

### **A4 (2026-05-26): POST_THINK_MIN_REASONING Floor**

**File:Line** `/workspace/atlas-mtp/crates/spark-server/src/scheduler/decode_logits_seq.rs:464–484`

**Description**: Suppress `</think>` token with bias −8.0 until at least 16 thinking tokens emitted. Closes the reasoning-collapse cascade documented in `research2_probe_forensics.md` (reasoning_content length decays 233→0 chars over 14 turns). When the model emits vanishingly short `<think>`, the downstream tool emission lacks planning context and drifts to phantom paths / leaked control characters. Floor applies only if `thinking_budget ≥ 16` (respects explicit opt-out).

### **A5: FP8 Path Default min_p = 0.08**

**Location**: Not explicitly in `decode_logits_seq.rs`. **Candidate**: Model-config or API-layer default. Search found no hardcoded FP8-conditional min_p in scheduler. **Status**: LIKELY APPLIED AT API/REQUEST PARSING LAYER (not visible in this file). When margin-ratio detector added, verify this is already active.

### **F8 (Repetition-Penalty Inversion) — NOT FOUND**

**Status**: No mention of F8 or rep_penalty inversion in this file. **Hypothesis**: F8 may refer to an older fix (pre-2026-05-26) applied in a different module (e.g., sampler or historical logit_processors). Or it is superseded by A1. Recommend: grep broader scheduler for "invert\|F8" or check historical commit logs.

### **Tier-1 Whitespace Mask at Parameter-Body Position 0**

**File:Line** `/workspace/atlas-mtp/crates/spark-server/src/scheduler/decode_logits_seq.rs:430–463`

**Description**: When inside `<parameter=KEY>…</parameter>` and zero content tokens emitted, apply bias −8.0 to:
- Token 510: `</` (close-tag opener)
- Token 220: ` ` (space)
- Token 198: `\n` (newline)
- Token 197: `\t` (tab)
- Token 256: `  ` (double space)
- Token 271: `\n\n` (double newline)

Epoch-2 v54 showed the model bypassed the close-only mask by emitting whitespace first (parser's `.trim()` at `tool_parser/parse_single_b.rs:105` strips it, leaving empty args). The full cluster forces the first body token to be non-whitespace content. Not bulletproof (Qwen3.6 has many multi-byte whitespace tokens), but covers empirically-most-likely tokens under FP8 drift.

### **Close-Tag Opener `</` Bias (part of Tier-1)**

**File:Line** `/workspace/atlas-mtp/crates/spark-server/src/scheduler/decode_logits_seq.rs:455–456`

**Description**: Hard bias −8.0 on token 510 (`</`). This is the first token of the `</parameter>` close. Masking it at empty-body position forces the model to emit content before closing, not skip the parameter. Paired with whitespace mask above.

---

## 3. Final Token Selection Point (Output Tokens Push)

**Primary Push Location**: `/workspace/atlas-mtp/crates/spark-server/src/scheduler/decode_logits_step.rs:395`

**Flow**:
1. `process_seq_logits()` returns `(sampled: u32, logprobs: Option<TokenLogprobs>)` (line 22 in decode_logits_seq.rs)
2. Caller `process_decode_logits()` in `decode_logits_step.rs` collects these into `new_tokens: Vec<(u32, Option<TokenLogprobs>)>` (line 40)
3. Loop at line 110 iterates over `new_tokens`; each tuple is destructured: `let (tok, logprobs) = ...`
4. Token proceeds through thinking state machine (lines 115–244), tool-call tracking (lines 246–268), logprobs accumulation (lines 270–273)
5. **FINAL PUSH** at line 395: `a.output_tokens.push(tok)` (for non-EOS, non-suppressed content tokens)
6. Alternative pushes: line 279 (tool_call_end), line 388 (EOS but not suppressed)

**Trace**: `sampled (u32)` → `tok (u32)` → `a.output_tokens.push(tok)`

---

## 4. Top-2 Extraction Feasibility

**Current State**: 
- Logprobs extraction (`extract_logprobs_from_f32()` in `logprobs.rs:11–46`) already computes top-K by partial sort (line 36: `select_nth_unstable_by`). When `k=2`, this yields both top-1 and top-2.
- **Extract cost**: O(V log V) for partial sort; negligible compared to sampling cost.
- **Data available**: The sampled token ID + the full f32_logits vector exist at line 522 in decode_logits_seq.rs, BEFORE logprobs extraction.

**Recommendation: CHEAP PATH — Extend logprobs logic**
- When a margin-ratio detector is added, use the **already-requested logprobs path** if `top_logprobs ≥ 2`.
- For margin detection **without** logprobs request: add a lightweight "top-2 only" extract that skips top-3..K:
  ```
  let top2 = extract_top_2_logits(&f32_logits); // O(V) single scan, no sort
  let margin_ratio = (top2[0] - top2[1]) / (top2[0].abs() + 1e-8);
  ```
- **No D2H copy required**: f32_logits already on host (post-grammar-bitmask, line 340). No extra GPU→CPU traffic.
- **No argpartition**: Single linear scan beats partial sort for K=2.

**Status**: ✅ **FEASIBLE — CHEAP**

---

## 5. Forced-Token Fast-Path Analysis

**Definition** (lines 280–327): When grammar admits exactly one legal next token, return it directly without mask fill or sampling. Short-circuits O(vocab) bitmask + O(vocab) CPU scan.

**Guards** (all must hold):
1. `!a.inside_thinking` — thinking is unconstrained
2. `a.top_logprobs.is_none()` — logprobs not requested
3. `!tier1_active` — **NOT inside empty parameter body** (Tier-1 gate, line 315)
4. `crate::scheduler::helpers::forced_token_fastpath_enabled()` — kill-switch on (default)
5. `a.grammar_state.is_some() && a.grammar_state.forced_token() == Some(id)` — grammar returns forced token

**Tier-1 Gate** (line 315): `let tier1_active = a.inside_parameter_body && a.param_body_chars_emitted == 0;`

**Issue Raised (A7 in research_synthesis.md)**: The fast-path returns the sole grammar-legal token **WITHOUT applying logit_bias**. This bypasses the anti-empty-parameter mask (token 510 `</`, whitespace cluster). **At position 0 of a parameter body, the grammar may legally emit the close token or whitespace, but logit_bias should still suppress them.**

**Fastpath Bypass of logit_bias**: 
- Line 326 returns `forced as u32` directly.
- No call to `sample_with_params_history()` → no logit_bias applied.
- Result: forced token goes straight to output, ignoring the −8.0 bias added at lines 455–462.

**Will margin-gate MISS this fastpath?**
- **YES, if margin gate is post-sampler inside process_seq_logits()**: the fastpath returns at line 326, before any post-sampler gate.
- **NO, if margin gate is in decode_logits_step (post-push) or as a pre-force-token sampler stage**: it would catch every token regardless of fast-path.

**Recommendation**: Place margin-gate at:
- **Option A (safer)**: Inside `process_seq_logits()` AFTER forced-token fast-path (post-line 326 return), as a pipeline stage in logit_processors (executed by both normal path + verify_pick_with_pipeline).
- **Option B (post-emit)**: In `decode_logits_step` after token push (line 395+), gating token acceptance at the sequence-state level.

---

## 6. MTP (Verify-Pipeline) Path Relationship

**MTP Architecture**: Speculative decode generates K draft tokens; verify path validates each one against the model's logits.

**Key Point** (verify_pipeline_helper.rs:3–26): The verify path **NOW** (post-C-2 fix) runs the same dequant + 8-stage pipeline on the verify logits, replacing the raw GPU `argmax_bf16`. This means:

1. **verify_pick_all_with_pipeline()** (line 131) copies `[K, vocab]` BF16 logits to host.
2. For each position i=0..K-1: calls `verify_pick_with_pipeline()` (line 72).
3. Inside that call: **runs `run_pipeline()` on the dequantized f32_logits** (line 104 in verify_pipeline_helper.rs).
4. The pipeline includes all 8 stages: F2 confidence, mid-word mask, post-close mask, tool-during-think, forced-think-end, pin-tool-call, **forced-token fast-path**, grammar-bitmask.
5. Speculative advance: after picking position i, the grammar matcher is advanced by `accept_token(pick[i])` (line 182) so position i+1's bitmask reflects post-emit state.
6. Rollback: after all K picks, the matcher is rolled back (line 202) so real emit_token re-advances normally.

**If a margin-ratio detector is added to the pipeline (as a 9th stage after grammar-bitmask)**:
- ✅ **It AUTOMATICALLY applies to MTP-verified tokens too**.
- ✅ **It runs before the forced-token fast-path returns** (if placed before stage 7) **or after** (if placed after stage 8).
- ⚠️ **Speculative advance timing**: Margin gate should NOT mutate grammar state (see invariants below).

**Confirmation**: The same `run_pipeline()` is called in both:
- `decode_logits_seq.rs` line 485: `sample_with_params_history()` context (normal decode)
- `verify_pipeline_helper.rs` line 104: `verify_pick_with_pipeline()` context (MTP verify)

Both see identical pipeline state → identical margin gate behavior. ✅ **MTP commitment preserved**.

---

## 7. Don't Break These Invariants

Any margin-ratio gate hook **MUST NOT**:

1. **Mutate grammar state**: Do not call `gs.accept_token()` or `gs.rollback()` inside the gate. The speculative-advance machinery in verify_pick_all_with_pipeline (line 182) handles grammar state. A margin gate that advances the matcher will desync the emit path.

2. **Skip or defer MTP commit**: If the gate rejects a token (returns a fallback), the fallback must itself pass the gate on the next step. Do not create a loop where margin rejection prevents emit and blocks the sequence.

3. **Re-trigger logit_bias rebuild**: The logit_bias_local is built once per token (lines 453–463). If the margin gate modifies a token and that token later feeds back into the pipeline (e.g. via a retry), ensure logit_bias is not re-applied to the token in a second pass.

4. **Feed back to grammar accept**: If the margin gate downgrades token selection (e.g. from top-1 to top-2), the downgraded token must be legal in the current grammar bitmask. Do not return a token that the grammar's next position would reject. (The verify path speculatively advances; a grammar-illegal fallback will trip the kill-switch at verify_pipeline_helper.rs:182.)

5. **Consume logprobs data intended for the API**: The extracted top-K logprobs (line 525) are packed into the response for client consumption. The margin gate logic must not steal or re-sort these tuples. If the gate's fallback affects the sampled_token, the logprobs extraction must run AFTER margin rejection so it reflects the final token, not the original top-1.

6. **Fire unconditionally during thinking or forced-think injection**: Inside thinking (`inside_thinking=true`) and during forced `</think>` injection (all-masked case at line 228), the margin gate should be disabled. These are high-stakes structural boundaries; low-margin fallback selection could emit spurious tokens mid-reasoning or defer the critical `</think>` close.

---

**Document End** | Total Lines: ≤1500 ✓

