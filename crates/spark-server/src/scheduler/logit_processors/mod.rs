// SPDX-License-Identifier: AGPL-3.0-only

//! Composable pre-sample logits pipeline.
//!
//! Pre-sample logit transformations used to live as a ~200-line inline
//! block inside `decode_logits_seq::process_seq_logits`. The blob
//! hard-coded ordering, made per-stage opt-out impossible, and
//! prevented moving individual stages to GPU. This module decomposes
//! it into eight per-stage [`LogitsProcessor`] impls plus a thin
//! pipeline driver. Stage order matches the pre-refactor monolith;
//! semantics are byte-identical (verified via the integration tests
//! in [`pipeline_tests`]).
//!
//! ## Stage order
//!
//! 1. [`f2_confidence::F2ConfidenceEarlyStop`] — arms `force_end_thinking`
//!    when top-1 probability stays ≥ 0.95 for the configured run.
//! 2. [`mid_word::MidWordThinkEndMask`] — suppresses `</think>` when
//!    the previous token decoded to mid-word text.
//! 3. [`post_close::PostCloseThinkMask`] — after `</think>` fires,
//!    masks `</think>` + `<think>` so the model can't re-enter.
//! 4. [`tool_during_think::ToolCallDuringThinkingMask`] — masks
//!    `<tool_call>` during thinking; biases it down on tool-loop.
//! 5. [`forced_think_end::ForcedThinkEndInjector`] — when budget
//!    + sentence-boundary policy says inject, blanket-mask to `</think>`.
//! 6. [`pin_tool_call::PinToToolCallStart`] — one-shot pin to
//!    `<tool_call>` immediately after `</think>` when require_tool_call.
//! 7. [`forced_token::ForcedTokenFastPath`] — when grammar admits
//!    exactly one next token, short-circuit pipeline + sampling.
//! 8. [`grammar_bitmask::GrammarBitmaskApply`] — apply grammar's
//!    next-token bitmask.
//!
//! ## Out of scope
//!
//! Adaptive-sampling entropy observation runs after this pipeline; it
//! decides sampling policy (greedy gate, effective temperature), not
//! logit transforms. The final `sample_with_params_history` call is
//! also downstream.

use crate::scheduler::ActiveSeq;
use crate::scheduler::decode_logits_seq::force_temp_zero_enabled;
use crate::scheduler::sample_step::PositionKind;
use spark_runtime::sampler::{SamplingParams, apply_penalties_and_bias};

pub mod adadec_diag;
mod b1_margin;
pub mod f2_confidence;
pub mod forced_think_end;
pub mod forced_token;
pub mod grammar_bitmask;
pub mod mid_word;
pub mod pin_tool_call;
pub mod post_close;
pub mod tool_during_think;

#[cfg(test)]
mod pipeline_tests;

/// Per-step environment passed to every processor. Holds tokenizer-
/// special tokens the pipeline cares about. `Copy` so it threads
/// through cheaply.
#[derive(Debug, Clone, Copy)]
pub struct LogitsContext {
    pub think_end_token: Option<u32>,
    pub think_start_token: Option<u32>,
    pub tool_call_start_token: Option<u32>,
    pub tool_call_end_token: Option<u32>,
}

/// Outcome of one [`LogitsProcessor::apply`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessorOutcome {
    /// Logits modified in place (or no change). Continue pipeline.
    Continue,
    /// Pipeline short-circuit: emit this token directly with no
    /// further masking, grammar advance, or sampling.
    EmitToken(u32),
}

/// One stage of the pre-sample pipeline. Implementations are pure-CPU
/// today; a future GPU-resident bitmask kernel can implement this
/// trait too without changing the driver.
pub trait LogitsProcessor: Send + Sync {
    /// Apply this stage's transform to `logits`. May read+mutate
    /// `seq` state (e.g. F2 sets `seq.force_end_thinking`; grammar
    /// stages mutate `seq.grammar_state`).
    fn apply(
        &self,
        logits: &mut [f32],
        seq: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome;

    /// Stable identifier for tracing + future per-request enable/
    /// disable. Static, no allocation.
    fn name(&self) -> &'static str;

    /// vLLM convention: `true` if this stage never alters which token
    /// wins under argmax (e.g. additive bias preserving ordering).
    /// Currently advisory — reserved for future GPU-batched skip paths.
    fn is_argmax_invariant(&self) -> bool {
        false
    }
}

/// Run the canonical pipeline. Returns `Some(token)` when any stage
/// short-circuited via [`ProcessorOutcome::EmitToken`]; `None`
/// otherwise (caller proceeds to sampling).
///
/// The post-pipeline AdaDec diagnostic is logged with the `"verify"`
/// path label (this is the MTP/verify entry point). The non-MTP decode
/// path uses [`run_pipeline_with_path`] with `"decode"` so its
/// `ATLAS_ADADEC_DIAGNOSTIC` records keep their pre-unification label.
pub fn run_pipeline(
    logits: &mut [f32],
    seq: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> Option<u32> {
    run_pipeline_with_path(logits, seq, ctx, "verify")
}

/// Canonical pipeline driver with an explicit AdaDec diagnostic path
/// label. SSOT for the per-stage transform order: both the non-MTP
/// decode path (`decode_logits_seq::process_seq_logits`, label
/// `"decode"`) and the MTP verify path (`run_pipeline`, label
/// `"verify"`) route through this one function. `path` only tags the
/// env-gated `ATLAS_ADADEC_DIAGNOSTIC` JSONL record — it never alters
/// any logit transform.
pub fn run_pipeline_with_path(
    logits: &mut [f32],
    seq: &mut ActiveSeq,
    ctx: &LogitsContext,
    path: &'static str,
) -> Option<u32> {
    let stages: [&dyn LogitsProcessor; 8] = [
        &f2_confidence::F2ConfidenceEarlyStop,
        &mid_word::MidWordThinkEndMask,
        &post_close::PostCloseThinkMask,
        &tool_during_think::ToolCallDuringThinkingMask,
        &forced_think_end::ForcedThinkEndInjector,
        &pin_tool_call::PinToToolCallStart,
        &forced_token::ForcedTokenFastPath,
        &grammar_bitmask::GrammarBitmaskApply,
    ];
    for stage in stages.iter() {
        match stage.apply(logits, seq, ctx) {
            ProcessorOutcome::Continue => {}
            ProcessorOutcome::EmitToken(tok) => return Some(tok),
        }
    }
    // AdaDec Phase 1 diagnostic — observes the post-grammar-bitmask
    // distribution, never mutates. No-op when ATLAS_ADADEC_DIAGNOSTIC is
    // unset. Called directly (not as a pipeline stage) so the caller's
    // path label is preserved byte-identically across both decode paths.
    adadec_diag::log_step(logits, seq, path);
    None
}

/// Unified per-position logit post-processing — the SINGLE function both
/// the non-MTP final-decode path (`decode_logits_seq::process_seq_logits`)
/// and the MTP verify path (`verify_pipeline_helper::verify_pick_with_pipeline`)
/// call. Replaces the two divergent inline blocks.
///
/// Stages, in order:
///  1. **ATLAS_FORCE_TEMP_ZERO bypass** (eligible on BOTH kinds): when the
///     diagnostic flag is set, return the raw-logit argmax with no pipeline,
///     no penalties, no bias — matching vLLM at temperature 0 for
///     apples-to-apples layer-cosine comparison. Returned as the emitted
///     token (`Some`).
///  2. **`run_pipeline`** (the 8 masking stages + AdaDec diagnostic). A
///     `Some(tok)` return is the forced-token fast-path short-circuit; the
///     caller emits it directly.
///  3. **B1 margin observer** — `FinalDecode` only (risk R6): observes the
///     post-mask top-1/top-2 gap. Pure observability, never mutates.
///  4. **`apply_penalties_and_bias`** — the repetition / presence /
///     frequency / LZ / DRY penalties + logit_bias (incl. the A4 floor)
///     carried in `penalties`, applied with the seq's token history.
///
/// Returns `Some(token)` ONLY for the force-temp-zero bypass or the
/// forced-token fast-path (caller emits directly, no argmax/sample). On
/// `None` the caller samples / argmaxes the now-masked-and-penalised
/// `logits`.
///
/// **R1 (matcher ownership):** this fn NEVER calls `gs.accept_token` /
/// `gs.rollback`. Matcher advancement stays caller-owned — the K-loop in
/// the verify path and `decode_logits_step` after sampling for the non-MTP
/// path. The idempotent `gs.fill_bitmask()` inside `GrammarBitmaskApply`
/// is the only grammar mutation and is safe to repeat.
pub fn process_position_logits(
    logits: &mut [f32],
    seq: &mut ActiveSeq,
    ctx: &LogitsContext,
    penalties: &SamplingParams,
    kind: PositionKind,
) -> Option<u32> {
    // 1. ATLAS_FORCE_TEMP_ZERO: pure argmax on raw logits — no pipeline, no
    //    penalties, no bias. Eligible on both kinds (the diagnostic's point
    //    is an identical bypass everywhere).
    if force_temp_zero_enabled() {
        let mut best_idx: u32 = 0;
        let mut best_val: f32 = f32::NEG_INFINITY;
        for (j, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = j as u32;
            }
        }
        return Some(best_idx);
    }

    // 2. Canonical pre-sample pipeline (+ AdaDec diagnostic under this
    //    position's path label). Short-circuit returns the forced token.
    if let Some(forced) = run_pipeline_with_path(logits, seq, ctx, kind.adadec_label()) {
        return Some(forced);
    }

    // 3. B1 margin observer — FINAL decode position only (risk R6). Reads
    //    the post-mask distribution; never mutates.
    if kind == PositionKind::FinalDecode {
        b1_margin::observe(logits, seq);
    }

    // 4. Penalties + bias (incl. A4) on the now-masked logits, using the
    //    seq's output-token history — the SSOT stage shared by both paths.
    apply_penalties_and_bias(logits, penalties, &seq.output_tokens);

    None
}
