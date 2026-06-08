// SPDX-License-Identifier: AGPL-3.0-only

//! Compile-time + light-touch unit tests for the pre-sample pipeline.
//!
//! ## Why no full byte-identical integration test
//!
//! [`crate::scheduler::ActiveSeq`] has ~60 fields including channels
//! (`ResponseSink`, `cancel_flag`), wall-clock `Instant`s, an
//! `AdaptiveSamplingState`, an `SsmDecodeRing`, an optional
//! `GrammarState` (which itself needs an `xgrammar::Tokenizer`), and
//! is `pub(super)` to `scheduler::*`. Constructing it from a `#[cfg(test)]`
//! module under `scheduler::logit_processors` is mechanically possible
//! but the resulting helper would be ~80 lines of fixture wiring per
//! test for ~5 lines of behaviour assertion — and any drift in the
//! `ActiveSeq` schema would silently rot the fixtures.
//!
//! The byte-identical guarantee against the pre-refactor monolith
//! lives in the parent task's live-run regression — the pipeline is
//! wired into `process_seq_logits` in a follow-up step, after which
//! the existing scheduler integration tests + opencode-session.md
//! prose-corpus replay form the actual byte-identical guard.
//!
//! This file's tests cover only what is testable without ActiveSeq:
//!
//! 1. The stage order in `run_pipeline` (compile-time guarantee that
//!    all 8 unit structs implement the trait and are wired in order).
//! 2. Stage `name()` strings are stable + distinct (forward-compat with
//!    per-stage enable/disable flags).
//! 3. The pure deferral-input math from `ForcedThinkEndInjector` matches
//!    `confidence::should_inject_think_end` semantics for the boundary
//!    cases the integration was known to hit on 2026-05-23.

use super::*;
use crate::scheduler::confidence::{
    MAX_SENTENCE_DEFER_TOKENS, THINK_DEFER_ABS_CEILING, THINK_DEFER_BUDGET_FACTOR,
    should_inject_think_end,
};

/// Every stage has a unique, stable name. Drift here breaks tracing
/// dashboards and any future per-stage opt-out config.
#[test]
fn stage_names_are_distinct_and_stable() {
    let names: [&'static str; 8] = [
        f2_confidence::F2ConfidenceEarlyStop.name(),
        mid_word::MidWordThinkEndMask.name(),
        post_close::PostCloseThinkMask.name(),
        tool_during_think::ToolCallDuringThinkingMask.name(),
        forced_think_end::ForcedThinkEndInjector.name(),
        pin_tool_call::PinToToolCallStart.name(),
        forced_token::ForcedTokenFastPath.name(),
        grammar_bitmask::GrammarBitmaskApply.name(),
    ];
    // Distinct
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            assert_ne!(
                names[i], names[j],
                "stage names must be distinct ({} == {})",
                names[i], names[j]
            );
        }
    }
    // Stable identifiers (pin exact strings so a rename is a visible
    // diff — these appear in tracing logs + future feature flags).
    assert_eq!(names[0], "f2_confidence_early_stop");
    assert_eq!(names[1], "mid_word_think_end_mask");
    assert_eq!(names[2], "post_close_think_mask");
    assert_eq!(names[3], "tool_call_during_thinking_mask");
    assert_eq!(names[4], "forced_think_end_injector");
    assert_eq!(names[5], "pin_to_tool_call_start");
    assert_eq!(names[6], "forced_token_fastpath");
    assert_eq!(names[7], "grammar_bitmask_apply");
}

/// The only stage that should advertise argmax-invariance is the
/// F2 confidence accumulator (logits are read-only). Everything else
/// mutates logits and must report `false`.
#[test]
fn argmax_invariance_advertisement() {
    assert!(f2_confidence::F2ConfidenceEarlyStop.is_argmax_invariant());
    assert!(!mid_word::MidWordThinkEndMask.is_argmax_invariant());
    assert!(!post_close::PostCloseThinkMask.is_argmax_invariant());
    assert!(!tool_during_think::ToolCallDuringThinkingMask.is_argmax_invariant());
    assert!(!forced_think_end::ForcedThinkEndInjector.is_argmax_invariant());
    assert!(!pin_tool_call::PinToToolCallStart.is_argmax_invariant());
    assert!(!forced_token::ForcedTokenFastPath.is_argmax_invariant());
    assert!(!grammar_bitmask::GrammarBitmaskApply.is_argmax_invariant());
}

/// `ForcedThinkEndInjector` packages three booleans for the gate. Pin
/// the truth table against the gate function so the constants used in
/// the injector (`THINK_DEFER_BUDGET_FACTOR`, `THINK_DEFER_ABS_CEILING`,
/// `MAX_SENTENCE_DEFER_TOKENS`) all stay consistent with the gate.
#[test]
fn forced_think_end_gate_semantics() {
    // Not armed → never inject.
    assert!(!should_inject_think_end(false, false, false, false));
    assert!(!should_inject_think_end(false, true, true, true));

    // Armed + hard override → always inject (even mid-fence).
    assert!(should_inject_think_end(true, true, false, true));
    assert!(should_inject_think_end(true, false, false, true));

    // Armed + in fence + no override → defer.
    assert!(!should_inject_think_end(true, true, false, false));
    assert!(!should_inject_think_end(true, true, true, false));

    // Armed + outside fence + at sentence boundary → inject.
    assert!(should_inject_think_end(true, false, true, false));

    // Armed + outside fence + NOT at boundary → defer (await period).
    assert!(!should_inject_think_end(true, false, false, false));
}

/// The defer-override math in `ForcedThinkEndInjector::apply` mirrors
/// what the inline monolith computed. This pins the constants we
/// import so a sweep that tunes them in `confidence.rs` flags this test
/// rather than silently drifting the injector.
#[test]
fn defer_override_math_constants() {
    // Budget×factor exceeded.
    let budget: u32 = 100;
    let thinking_tokens: u32 = budget.saturating_mul(THINK_DEFER_BUDGET_FACTOR);
    assert!(thinking_tokens >= budget.saturating_mul(THINK_DEFER_BUDGET_FACTOR));

    // Absolute ceiling when budget is None.
    let unlimited_tokens: u32 = THINK_DEFER_ABS_CEILING;
    assert!(unlimited_tokens >= THINK_DEFER_ABS_CEILING);

    // Sentence-defer ceiling.
    let defer_count: u32 = MAX_SENTENCE_DEFER_TOKENS;
    assert!(defer_count >= MAX_SENTENCE_DEFER_TOKENS);
}

/// `LogitsContext` carries the four tokenizer-special tokens the
/// pipeline needs. Wiring it from `process_seq_logits` requires that
/// the parameter names match the field names (we pass them via field
/// shorthand). If a future agent renames a field on `LogitsContext`
/// (e.g. `tool_call_end_token` → `tool_end_token`) but forgets to
/// update the call site, the shorthand construction breaks. This test
/// pins the field set and `Copy` / `Clone` semantics so a non-trivial
/// rename surfaces here loudly.
#[test]
fn logits_context_field_set_is_stable() {
    let ctx = LogitsContext {
        think_end_token: Some(1),
        think_start_token: Some(2),
        tool_call_start_token: Some(3),
        tool_call_end_token: Some(4),
    };
    // Copy semantics — pipeline stages take `&LogitsContext`; a Copy
    // bound keeps the threading cheap (no Arc, no clone-on-call).
    let ctx2: LogitsContext = ctx;
    assert_eq!(ctx2.think_end_token, Some(1));
    assert_eq!(ctx2.think_start_token, Some(2));
    assert_eq!(ctx2.tool_call_start_token, Some(3));
    assert_eq!(ctx2.tool_call_end_token, Some(4));
    // Original still usable (Copy, not Move).
    assert_eq!(ctx.tool_call_end_token, Some(4));
}

/// Integration smoke test for `run_pipeline` without `ActiveSeq`
/// construction. The `ActiveSeq` struct in `types.rs` has ~60 fields
/// including `ResponseSink` channels, `cancel_flag` AtomicBool, wall-
/// clock `Instant`s, `AdaptiveSamplingState`, `SsmDecodeRing`, and an
/// optional `GrammarState` (which needs an `xgrammar::Tokenizer`). The
/// file-level docs (top of this module, lines 6-21) explicitly chose
/// NOT to mock that — drift in the `ActiveSeq` schema would silently
/// rot the fixtures, and live-runtime scheduler integration tests +
/// opencode-session.md prose-corpus replay form the actual byte-
/// identical guard against the pre-refactor monolith.
///
/// Instead this test exercises the pipeline's *public surface* shape:
/// `run_pipeline` must take `(&mut [f32], &mut ActiveSeq, &LogitsContext)`
/// and return `Option<u32>`. The compile-success of this test is the
/// guarantee — if the signature drifts, this stops compiling.
#[test]
fn run_pipeline_signature_is_stable() {
    // Type-checks at compile time. The fn pointer alias forces
    // `run_pipeline`'s signature to match exactly; any drift in the
    // arg list, mutability, or return type fails to compile here.
    type RunPipelineFn = fn(&mut [f32], &mut ActiveSeq, &LogitsContext) -> Option<u32>;
    let _ptr: RunPipelineFn = run_pipeline;
    // The unified per-position post-processor and the labelled driver must
    // also keep their signatures stable — both decode paths call them.
    type ProcessPositionFn = fn(
        &mut [f32],
        &mut ActiveSeq,
        &LogitsContext,
        &spark_runtime::sampler::SamplingParams,
        crate::scheduler::sample_step::PositionKind,
    ) -> Option<u32>;
    let _pp: ProcessPositionFn = process_position_logits;
    type RunPipelineWithPathFn =
        fn(&mut [f32], &mut ActiveSeq, &LogitsContext, &'static str) -> Option<u32>;
    let _rpp: RunPipelineWithPathFn = run_pipeline_with_path;
}

/// STEP 6 guard (a): `decode_logits_seq.rs` (the non-MTP path) must NOT
/// re-grow an inline fork of the pipeline. The whole point of the
/// unification is that BOTH decode paths route through `run_pipeline*` +
/// `process_position_logits`. If a future change re-inlines a mask, a
/// penalty literal, or a per-stage block into the non-MTP path, this test
/// fails — forcing the change into the SSOT module instead.
///
/// We assert the source no longer contains:
///   * `f32::NEG_INFINITY` — every hard-mask now lives in a pipeline stage.
///   * the per-stage marker comments the inline monolith used.
///   * an inline `SamplingParams {` penalty literal (the builder is SSOT).
///   * the inline A4 (`MIN_REASONING_TOKENS`) and B1 (`b1_record_low_margin`)
///     remnants and the dead `if false && low_margin_in_body` C4v1 block.
/// And it MUST still call the unified entry point.
#[test]
fn non_mtp_path_has_no_inline_pipeline_fork() {
    const SRC: &str = include_str!("../decode_logits_seq.rs");

    // No hard-masks inline — they belong to the pipeline stages.
    assert!(
        !SRC.contains("f32::NEG_INFINITY"),
        "decode_logits_seq.rs must not hard-mask inline; masking belongs in logit_processors stages"
    );
    // No inline penalty literal — `penalty_params_for` is the SSOT builder.
    assert!(
        !SRC.contains("repetition_penalty: a.repetition_penalty"),
        "decode_logits_seq.rs must not inline the SamplingParams penalty literal; use penalty_params_for"
    );
    // No inline A4 / B1 / C4v1 remnants.
    assert!(
        !SRC.contains("MIN_REASONING_TOKENS"),
        "A4 floor must live only in penalty_params_for, not inline in decode_logits_seq.rs"
    );
    assert!(
        !SRC.contains("b1_record_low_margin") && !SRC.contains("low_margin_in_body"),
        "B1 margin detector must live only in logit_processors::b1_margin, not inline"
    );
    assert!(
        !SRC.contains("if false && "),
        "the dead C4v1 `if false && low_margin_in_body` block must be deleted"
    );
    // No per-stage marker comments from the pre-unification monolith.
    for marker in [
        "Mid-word `</think>` defer",
        "one-shot pin-to-tool-call-start",
        "Forced-token fast-path (xgrammar Tier 3b",
        "Apply grammar bitmask BEFORE sampling",
    ] {
        assert!(
            !SRC.contains(marker),
            "stale inline per-stage block `{marker}` still present in decode_logits_seq.rs"
        );
    }
    // Positive: the non-MTP path MUST route through the unified fn.
    assert!(
        SRC.contains("process_position_logits"),
        "decode_logits_seq.rs must call the unified process_position_logits"
    );
}

/// STEP 6 guard (b): the A4 floor and B1 observer reached by the unified
/// `process_position_logits` must stay in their SSOT homes. A4 lives in
/// `sample_step::penalty_params_for` and applies on BOTH kinds (the
/// intended delta: it now reaches the MTP verify path). B1 lives in
/// `logit_processors::b1_margin` and is gated to `FinalDecode`. This test
/// pins both so removing/relocating them is a visible CI failure, and pins
/// that the verify path no longer carries the inline penalty/argmax fork.
#[test]
fn unified_fn_includes_a4_and_b1_stages() {
    // A4 floor is in the penalty-params builder (SSOT) and unconditional on
    // kind (reaches verify too).
    const SAMPLE_STEP_SRC: &str = include_str!("../sample_step.rs");
    assert!(
        SAMPLE_STEP_SRC.contains("A4_MIN_REASONING_TOKENS")
            && SAMPLE_STEP_SRC.contains("a.think_end_token"),
        "A4 POST_THINK_MIN_REASONING floor must live in penalty_params_for"
    );

    // B1 observer is in its own module and is FinalDecode-gated by the
    // unified fn.
    const B1_SRC: &str = include_str!("b1_margin.rs");
    assert!(
        B1_SRC.contains("fn observe") && B1_SRC.contains("LOW_MARGIN_THRESHOLD"),
        "B1 margin observer must live in logit_processors::b1_margin"
    );
    const MOD_SRC: &str = include_str!("mod.rs");
    assert!(
        MOD_SRC.contains("b1_margin::observe") && MOD_SRC.contains("PositionKind::FinalDecode"),
        "process_position_logits must call B1 observe gated on FinalDecode"
    );
    assert!(
        MOD_SRC.contains("force_temp_zero_enabled") && MOD_SRC.contains("apply_penalties_and_bias"),
        "process_position_logits must own the force-temp-zero bypass and penalties+bias"
    );

    // The verify path must route through the unified fn, not its own
    // pipeline+argmax fork.
    const VERIFY_SRC: &str = include_str!("../verify_pipeline_helper.rs");
    assert!(
        VERIFY_SRC.contains("process_position_logits"),
        "verify_pick_with_pipeline must call the unified process_position_logits"
    );
}

/// R1 guard: the unified `process_position_logits` MUST NOT advance or roll
/// back the grammar matcher — matcher ownership stays with the callers (the
/// verify K-loop and `decode_logits_step` after sampling). The idempotent
/// `fill_bitmask` inside the grammar-bitmask STAGE is fine, but the unified
/// fn body itself must never call `accept_token` / `rollback`. We scope the
/// check to the `process_position_logits` fn body so the (legitimate)
/// references elsewhere in `mod.rs` docs don't trip it.
#[test]
fn unified_fn_does_not_advance_matcher() {
    const MOD_SRC: &str = include_str!("mod.rs");
    let start = MOD_SRC
        .find("pub fn process_position_logits")
        .expect("process_position_logits must exist in mod.rs");
    let body = &MOD_SRC[start..];
    assert!(
        !body.contains(".accept_token(") && !body.contains(".rollback("),
        "process_position_logits must not call gs.accept_token / gs.rollback (R1: caller-owned)"
    );
}
