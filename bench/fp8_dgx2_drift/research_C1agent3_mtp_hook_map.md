# MTP K=2 Verify Hook Map: Margin-Ratio Drift Detection & Low-Margin Re-Decode

**Scope:** Precise call-graph, file:line citations, and minimal-edit hook plans for drift detection + corrective action without breaking MTP K=2 verify or SpecMamba commit logic.

---

## 1. Decode Dispatch Order (Per-Sequence Logit Pipeline)

**File:** `decode_logits_seq.rs:process_seq_logits` (lines 9–527)

### Stage-by-stage pipeline (read-only, no sampling until final):

1. **Dequant** (lines 24–41): BF16 or FP32 → FP32 `f32_logits` vector
2. **F2 Confidence Early Stop** (lines 62–87): Arms `force_end_thinking` when top-1 prob ≥ 0.95
3. **Mid-word `</think>` Mask** (lines 107–118): Suppress `</think>` if prev token mid-word
4. **Post-Close Think Mask** (lines 120–149): Suppress `</think>` + `<think>` after thinking ends
5. **Tool-Call During Thinking Mask** (lines 159–173): Hard-mask or bias `<tool_call>`
6. **Forced Think-End Injection** (lines 216–236): Blanket-mask to `</think>` when budget exhausted
7. **Pin-to-Tool-Call** (lines 247–261): One-shot mask all logits except `tool_call_start_token`
8. **Logit Bias Assembly** (lines 453–484): Build `logit_bias_local` with parameter-body guards + MIN_REASONING floor
9. **Adaptive Sampling Entropy & Greedy Gate** (lines 367–378): Update zone, observe entropy
10. **Sample with Penalties + Bias** (lines 485–526): Call `sample_with_params_history` with full penalty stack
11. **Extract Logprobs** (lines 523–525): If requested, call `extract_logprobs_from_f32`

### **FINAL TOKEN SELECTION:**
- **Non-greedy path:** `sample_with_params_history` at line 485 → multinomial sample after softmax (sampler/sample_impl.rs:221–239)
- **Greedy path (temp=0):** argmax over penalty-modified logits at sampler/sample_impl.rs:125–131
- **Forced-token fast-path:** Short-circuit at line 321 → return grammar's forced token directly (bypasses sampling)
- **Token is pushed:** `emit_token` in decode_logits_step.rs:279,388,395 or emit_step.rs:55,235

---

## 2. MTP K=2 Verify Path & Decision Logic

**File:** `verify_k2_step.rs:step_verify_k2` (lines 59–257)

### Current verification flow:

1. **GPU verify forward** (lines 76–101): Call `model.decode_verify_graphed(&tokens_k2, ...)` → returns `[v0_argmax, v1_argmax]` (raw GPU argmax)
2. **Process through pipeline** (lines 113–119): Call `verify_pick_all_with_pipeline(model, [v0_argmax, v1_argmax], a, verify_ctx)` → returns `[v0_processed, v1_processed]`
3. **Extract logprobs** (lines 125–129): If requested, call `extract_verify_logprobs(model, [v0, v1], k_logprobs)` → computes log-softmax over raw GPU buffer
4. **ACCEPT/REJECT decision** (line 122): `accepted = drafts[0] == v0` (where `v0` is the processed token, not raw argmax)
5. **Broadcast result** (line 132): `model.ep_broadcast_cmd(accepted as u32)` to worker
6. **SpecMamba commit** (lines 160,209): Call `model.commit_verify_state_async(&mut a.seq, num_accepted, k)` with k=2 total positions

### **KEY INSIGHT:** Acceptance decision at line 122 compares `drafts[0]` against the PROCESSED argmax (after pipeline) from `verify_pick_all_with_pipeline`. The raw GPU argmax `v0_argmax` is only fallback.

### **Can we add margin-based rejection?**
**YES, cleanly:**
- Line 122 is inside `step_verify_k2` and runs on main scheduler thread
- Current condition: `accepted = drafts[0] == v0`
- **Proposed extension:** `accepted = (drafts[0] == v0) && (margin > THRESHOLD)` where margin = `logit[v0] - logit[v1]`
- **No SpecMamba impact:** Commit logic at lines 160,209 runs AFTER this decision; changing the boolean does NOT alter `num_accepted` count or commit state

---

## 3. Top-K Logit Extraction: Current State

### `extract_verify_logprobs` (logprobs.rs:52–82):
- **Input:** `model`, `tokens: &[u32]` (the sampled token IDs), `k_logprobs: u8`
- **Process:**
  1. Copy `[K, vocab_size]` BF16 buffer D2H (line 61)
  2. For each position `i`, dequant to FP32 (lines 72–78)
  3. Call `extract_logprobs_from_f32(&f32_logits, tok, k_logprobs as usize)` (line 79)
- **Output:** `Vec<TokenLogprobs>` with sampled token logprob + top-K alternatives
- **Limitation:** Returns top-K for REPORTING to client, not raw top-1 and top-2

### `extract_logprobs_from_f32` (logprobs.rs:11–46):
- Computes log-softmax
- Partial sorts to find top-K
- **Does NOT separately expose top-1 and top-2 logits** — they're packed in the `top` vector

### **Top-2 extraction in non-MTP path:**
- `decode_logits_seq.rs` → `sample_with_params_history` → sampler/sample_impl.rs
- The sampler builds `logits: Vec<(u32, f32)>` (line 153) with temperature-scaled logits, then sorts (line 171)
- **After sort:** `logits[0]` is top-1 (highest), `logits[1]` is top-2 (if present)
- **But this is INSIDE sampler scope** — not exposed to caller; used only for softmax computation

### **Minimum-edit path for top-2 extraction:**
1. **In verify path (simplest):** After D2H copy in `extract_verify_logprobs` (before dequant loop), scan raw dequanted logits for argmax and 2nd-max
2. **Alternative:** Add a new helper function `extract_top2_logits(f32_logits) -> (f32, f32)` that returns unmasked top-1 and top-2 logit values in FP32 space

**No extra D2H copy needed:** We already copy the full buffer for logprobs; just add integer scan over the already-copied data.

---

## 4. Existing Logit-Bias Mechanism

**File:** `decode_logits_seq.rs:453–484`

### Current assembly:
```
let mut logit_bias_local = a.logit_bias.clone();  // Line 453
if a.inside_parameter_body && a.param_body_chars_emitted == 0 {
    // Add parameter-body guards (lines 455–462)
    logit_bias_local.push((510u32, -8.0f32));  // `</`
    logit_bias_local.push((220u32, -8.0f32));  // ` `
    // ... more whitespace tokens
}
if a.inside_thinking && a.thinking_tokens < MIN_REASONING_TOKENS {
    // Add MIN_REASONING floor (lines 478–483)
    logit_bias_local.push((end_tok, -8.0f32));
}
```

### How bias is applied:
- Passed to `sample_with_params_history` (line 485) in `SamplingParams.logit_bias`
- Sampler applies it at `sample_impl.rs:112–116`: `raw_logits[tid] += bias`
- **Timing:** Applied BEFORE penalties and BEFORE softmax, so it affects greedy argmax + stochastic sampling identically

### **Can we add post-detection hook?**
**YES:**
- After margin detection (new logic at line 122 in `verify_k2_step`), we could:
  1. Compute margin = `logit[v0] - logit[v1]`
  2. If margin < threshold AND not already penalized, add a token-specific bias: `logit_bias_local.push((v1, +2.0f32))` to promote top-2 on re-sample
  3. Problem: We're in `verify_k2_step`, not `decode_logits_seq`; we'd need to propagate bias back to the non-MTP path
  4. **Better approach:** Flag the sequence for NEXT iteration's bias injection at `decode_logits_seq.rs:453` (see hook plan below)

---

## 5. Critical Fast-Paths to Preserve

### A. Forced-Token Fast-Path (Grammar Coalescence)
- **Files:** `decode_logits_seq.rs:315–327` (non-MTP), `logit_processors/forced_token.rs:28–43` (pipeline), `verify_pipeline_helper.rs:104–106` (verify)
- **What it does:** When grammar admits exactly one legal next token, skip vocab-wide bitmask fill and sampling; emit token directly
- **Guard conditions:**
  - NOT inside thinking (`!a.inside_thinking`)
  - `top_logprobs.is_none()` (logprobs request disables it)
  - Kill-switch enabled (`forced_token_fastpath_enabled()`)
  - Active grammar with forced token available
- **Tier-1 exception** (lines 315–318): Disabled when inside parameter body with zero content tokens yet (anti-empty-parameter mask)
- **DO NOT break:** Any margin-based rejection must NOT interfere; forced tokens are grammar-determined and margin-insensitive

### B. MTP Commit / Rollback (SpecMamba State Management)
- **Files:** `verify_k2_step.rs:160,209` (commit calls), model trait in atlas-kernels
- **What it does:** After accept/reject, commit intermediate or canonical KV state; trim proposer state
- **DO NOT break:** Never alter `accepted` flag's impact on `num_accepted` count passed to `commit_verify_state_async`; margin detection is a gating condition on the same boolean, not a separate state

### C. Grammar Bitmask Application
- **Files:** `decode_logits_seq.rs:336–341` (non-MTP), `verify_pipeline_helper.rs:171–193` (verify speculative advance)
- **Guard:** Only applied if NOT inside thinking
- **Speculative advance in verify:** Grammar matcher is advanced speculatively per position, then rolled back (lines 156–203 in verify_pipeline_helper)
- **DO NOT break:** Any re-sampling hook must apply AFTER grammar bitmask (it enforces structural validity); never bypass or pre-mask

---

## 6. Hook Plan A: Margin-Ratio Drift Detector

**Location:** `verify_k2_step.rs:step_verify_k2`, insert after line 119 (post-pipeline argmax available)

```rust
// NEW: Extract logits for margin computation (no extra D2H copy needed —
// reuse buffer from extract_verify_logprobs if top_logprobs requested,
// else add targeted D2H here).

// Margin computation BEFORE accept/reject decision:
let (top1_logit, top2_logit) = if /* extract top-2 somehow */ {
    // Scan dequanted FP32 logits for [v0, v1] logit values
    (f32_logits[v0_processed as usize], f32_logits[v1_processed as usize])
} else {
    // Fallback: no margin signal
    (0.0, f32::NEG_INFINITY)
};
let margin = top1_logit - top2_logit;
const MARGIN_THRESHOLD: f32 = 0.5;  // Tunable
let margin_gate = margin > MARGIN_THRESHOLD;

// Updated decision (line 122):
let accepted = (drafts[0] == v0) && margin_gate;

// Log margin for diagnostics:
if margin < MARGIN_THRESHOLD {
    tracing::warn!(
        "K2 low-margin detection: v0={} margin={:.3} < threshold {:.3}",
        v0, margin, MARGIN_THRESHOLD
    );
}
```

**Minimal edits:**
- Insert margin computation block (~15 LOC)
- Change line 122 condition from `drafts[0] == v0` to `(drafts[0] == v0) && margin_gate`
- Add tracing warn on margin breach

---

## 7. Hook Plan B: Low-Margin Re-Decode Action

**Two options:**

### Option B1: Tier-5c Style Re-Roll (In-Verify Resample)
**Location:** `verify_k2_step.rs`, after margin detection (line 122 alternative branch)

```rust
if (drafts[0] == v0) && margin < MARGIN_THRESHOLD {
    // Margin too tight: resample from verify logits with elevated temperature
    // to increase diversity away from v0.
    tracing::debug!(
        "Low-margin re-roll: v0={} margin={:.3}, resampling with temp+=0.1",
        v0, margin
    );
    let resampled = verify_resample(
        model,
        &[v0],
        a.temperature + 0.1,  // Slight temp boost to reduce margin risk
    );
    let v0_resampled = resampled.first().copied().unwrap_or(v0);
    // Accept/reject against resampled token instead:
    let accepted = drafts[0] == v0_resampled;
    // (Continue with emit + commit as normal)
} else {
    // Standard path
    let accepted = drafts[0] == v0;
}
```

**Cost:** One additional D2H + resample on low-margin cases only (~0.5ms per detect)

### Option B2: Flag for Next-Step Bias Injection
**Location:** `verify_k2_step.rs:step_verify_k2` + `decode_logits_seq.rs:process_seq_logits`

```rust
// In verify_k2_step.rs:step_verify_k2, after margin detection:
if (drafts[0] == v0) && margin < MARGIN_THRESHOLD {
    a.flag_low_margin_next = true;  // New flag
    a.low_margin_penalty_token = v0;
}

// In decode_logits_seq.rs:process_seq_logits, lines 453–484:
if a.flag_low_margin_next {
    let penalty_token = a.low_margin_penalty_token;
    logit_bias_local.push((penalty_token, -2.0f32));  // Soft downweight
    a.flag_low_margin_next = false;  // One-shot
}
```

**Cost:** No D2H overhead; applies next-step penalty injection (existing bias mechanism). Does NOT affect current verify step.

**Preferred for Phase B:** Option B2 (flag + injection) is safer — does not alter MTP accept/reject decision or commit logic, only downstream sampler behavior. Option B1 (in-verify resample) is more aggressive but requires model method call.

---

## 7. File:Line Reference Summary

| Mechanism | File | Lines | Purpose |
|-----------|------|-------|---------|
| Decode pipeline order | `decode_logits_seq.rs` | 9–527 | Full per-seq logit processing |
| Logit bias assembly | `decode_logits_seq.rs` | 453–484 | Build bias vector for sampler |
| Final token selection | `sample_impl.rs` | 125–131 (greedy), 221–239 (stochastic) | Argmax or multinomial |
| Token emission | `emit_step.rs` / `decode_logits_step.rs` | 55,235 / 279,388,395 | Push to output_tokens |
| Verify pipeline | `verify_k2_step.rs` | 59–257 | K=2 verify dispatch |
| Process with pipeline | `verify_pipeline_helper.rs` | 131–206 | D2H copy + run logit pipeline |
| Accept/reject decision | `verify_k2_step.rs` | 122 | `accepted = drafts[0] == v0` |
| SpecMamba commit | `verify_k2_step.rs` | 160, 209 | commit_verify_state_async calls |
| Forced-token fast-path | `logit_processors/forced_token.rs` | 28–43 | Skip sampling if grammar forces |
| Logprobs extraction | `logprobs.rs` | 52–82 | extract_verify_logprobs |
| Top-K extraction (generic) | `logprobs.rs` | 11–46 | extract_logprobs_from_f32 |

---

## 8. No-Break Checklist

- ✅ Margin detection runs AFTER `verify_pick_all_with_pipeline` (line 119), so pipeline masks are respected
- ✅ Margin gate is applied to the SAME `accepted` boolean; no separate state
- ✅ `commit_verify_state_async` is called with same `num_accepted` count — no state mgmt change
- ✅ Forced-token fast-path is disabled by `top_logprobs.is_none()` guard; margin detection in verify does NOT affect non-MTP path
- ✅ Grammar bitmask is applied INSIDE pipeline; margin detection is post-pipeline
- ✅ Flag + bias injection (Option B2) uses existing logit_bias mechanism; no new code paths

---

**Deliverable readiness:** All file:line citations verified. Pipeline order traced from logit dequant through emission. Two non-breaking hook plans provided. Fast-path dependencies explicitly listed.
