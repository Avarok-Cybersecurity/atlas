// SPDX-License-Identifier: AGPL-3.0-only

//! Chat fast path for masked verify picks (2026-07-08):
//! masked-greedy == raw-argmax guard.
//!
//! The DFlash MASKED_VERIFY fix routes verify picks through
//! `verify_pick_all_with_pipeline` so structural specials can never leak
//! unmasked (the T=0 wrong-universe derails). But the grammar fast path
//! there requires an active grammar AND `!inside_thinking`, so a plain
//! chat request — no tools, no grammar — paid the slow path on EVERY
//! position: `[K, vocab]` D2H + 248k BF16→F32 dequant + 8-stage pipeline
//! + host argmax, ×K rows/step (measured 2026-07-08: 11.5 tok/s vs 15.8
//! NOSPEC on the prose probe at γ17 — the masking was correct but the
//! detour ate the spec win).
//!
//! For a grammarless request the pipeline can change the emitted pick
//! ONLY when one of these holds:
//!  (a) the raw argmax IS a maskable structural id: think_end
//!      (MidWordThinkEndMask / PostCloseThinkMask), think_start
//!      (PostCloseThinkMask), or tool_call_start
//!      (ToolCallDuringThinkingMask — both branches: its -12.0 bias only
//!      LOWERS tool_call_start, which cannot promote another token above
//!      the raw max);
//!  (b) a forced/stateful stage is armed on the seq: F2 confidence
//!      early-stop (reads the row's softmax + mutates
//!      consecutive_confident), ForcedThinkEndInjector (blanket-mask
//!      injection — or its armed-deferring branch, which must tick
//!      sentence_defer_count), PinToToolCallStart;
//!  (c) penalties are not provably argmax-preserving (same SSOT gate as
//!      the grammar fast path: Neutral, or ReduceOnly + per-pick immune;
//!      the A4-floor/logit-bias case classifies as Blocked);
//!  (d) the AdaDec diagnostic is recording (must observe the post-mask
//!      distribution).
//! (b)-(d) are step-level flag/env checks with no logits access; (a) is
//! three integer compares per position. When none hold, masking only
//! removes non-argmax candidates, so masked-greedy == `argmax_ids[i]` by
//! definition and the call does no D2H at all.
//!
//! State-staleness parity: the slow path evaluates `a.*` flags and
//! `a.output_tokens` FIXED across the K loop (they only advance in
//! `emit_token`, after the helper), and the gates here read exactly that
//! same fixed state — eligibility is uniform across positions and the
//! equivalence is per-call exact. Any ineligible position falls through
//! to the unmodified slow path for the whole call.
//! Kill-switch: `ATLAS_DISABLE_FAST_MASKED=1`.

use crate::scheduler::ActiveSeq;
use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::mtp_timing::Phase;
use spark_model::traits::Model;

/// Returns `Some(picks)` when the fast path proves masked-greedy ==
/// raw-argmax for every position (picks are the raw `argmax_ids`);
/// `None` when any gate fails and the caller must run the slow path.
///
/// `row_base`: first logits row of this sequence's verify span in the
/// shared `[R, vocab]` buffer (0 on all single-sequence paths; `i*4` for
/// sequence i on the batched K=4 verify — see `verify_pick_all_with_pipeline`).
pub(super) fn try_chat_fast_path(
    model: &dyn Model,
    argmax_ids: &[u32],
    a: &ActiveSeq,
    ctx: &LogitsContext,
    row_base: usize,
) -> Option<Vec<u32>> {
    // DFlash masked-verify mode ONLY. The fast path exists to make
    // ATLAS_DFLASH_MASKED_VERIFY affordable; it must never run for MTP:
    // returning the GPU argmax where the slow path computes a host-side
    // argmax over dequantized F32 logits changes tie-breaking on
    // near-tie tokens — measured 2026-07-11 as temp-0 MTP output drift
    // vs an unpatched binary (think block identical, answer flips at
    // low-margin tokens). MTP keeps the slow path unconditionally so
    // its behavior is byte-invariant by construction.
    if !ctx.sampling.dflash_masked_verify {
        return None;
    }
    let fast_masked_enabled = ctx.sampling.fast_masked;
    let adadec_recording = ctx.sampling.adadec_diagnostic;
    if !fast_masked_enabled || a.grammar_state.is_some() || adadec_recording {
        return None;
    }
    // P1-3 parity (2026-08-14): at temperature > 0 the slow path SAMPLES
    // each verify position (the `mtp_verify_sample` branch in
    // `verify_pick_with_pipeline`), so returning the raw argmax here would
    // silently disable temperature/top_p/min_p/seed for every spec-accepted
    // chat token. The grammar sibling has carried this exact load-bearing
    // gate since P1-3 (2026-07-09, see `verify_pick_all_with_pipeline`);
    // this path landed one day earlier and missed it. The condition mirrors
    // the slow path's sampling branch exactly: when that branch cannot fire
    // (`ATLAS_NO_MTP_VERIFY_SAMPLE` or forced temp-0), the slow path pins
    // the argmax too and the fast path stays equivalent — and available.
    if ctx.sampling.mtp_verify_sample && a.temperature > 0.0 && !ctx.sampling.force_temp_zero {
        return None;
    }
    use crate::scheduler::confidence::{
        MAX_SENTENCE_DEFER_TOKENS, THINK_DEFER_ABS_CEILING, THINK_DEFER_BUDGET_FACTOR,
    };
    // (b) forced/stateful stage preconditions — mirrored exactly from
    // f2_confidence.rs / forced_think_end.rs / pin_tool_call.rs.
    let f2_active = !ctx.sampling.disable_watchdogs
        && a.inside_thinking
        && !a.force_end_thinking
        && a.thinking_tokens >= 400
        && ctx.watchdog.confidence_early_stop;
    let defer_hard_override = match a.thinking_budget {
        Some(b) => a.thinking_tokens >= b.saturating_mul(THINK_DEFER_BUDGET_FACTOR),
        None => a.thinking_tokens >= THINK_DEFER_ABS_CEILING,
    } || a.sentence_defer_count >= MAX_SENTENCE_DEFER_TOKENS;
    let think_end_inject_armed = a.inside_thinking && (a.force_end_thinking || defer_hard_override);
    let pin_tool_armed =
        a.think_just_ended && a.require_tool_call && !a.tool_call_opened && !a.inside_thinking;
    // (c) penalty gate — same construction the slow path uses per
    // position (penalty_params_for is position-independent here).
    let penalty_gate = crate::scheduler::fast_greedy::classify_penalties(
        &crate::scheduler::sample_step::penalty_params_for(
            a,
            crate::scheduler::sample_step::PositionKind::Verify,
            0.0,
            None,
            Vec::new(),
        ),
    );
    if f2_active
        || think_end_inject_armed
        || pin_tool_armed
        || penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::Blocked
    {
        return None;
    }
    let t_fast = std::time::Instant::now();
    let scoped_history: Vec<u32> =
        if penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly {
            crate::scheduler::sample_step::penalty_history_scope(
                &a.output_tokens,
                ctx.tool_call_end_token,
            )
            .to_vec()
        } else {
            Vec::new()
        };
    let vocab = model.vocab_size();
    let logits_base = model.logits_buffer_ptr();
    let mut all_clear = true;
    for (i, &tok) in argmax_ids.iter().enumerate() {
        // (a) maskable structural ids → slow path for the call.
        if Some(tok) == ctx.think_end_token
            || Some(tok) == a.think_start_token
            || Some(tok) == ctx.tool_call_start_token
        {
            all_clear = false;
            break;
        }
        if penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly
            && !crate::scheduler::fast_greedy::argmax_immune(tok, &scoped_history, || {
                crate::scheduler::fast_greedy::logit_is_positive(
                    model,
                    logits_base,
                    row_base + i,
                    vocab,
                    tok,
                )
            })
        {
            all_clear = false;
            break;
        }
    }
    ctx.timing.record(Phase::FastGreedy, t_fast);
    if all_clear {
        if ctx.stats.once("log:verify_chat_fast_path") {
            tracing::info!(
                "verify chat fast path ACTIVE: masked-greedy == raw argmax, no D2H \
                 (kill-switch: ATLAS_DISABLE_FAST_MASKED=1)"
            );
        }
        return Some(argmax_ids.to_vec());
    }
    // Fall through — grammar fast path can't fire (grammar_state is
    // None), so the slow path handles the call.
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::lifecycle_tests::StubModel;
    use crate::scheduler::logit_processors::SamplingLevers;
    use crate::scheduler::test_support::test_seq;

    /// Drive `try_chat_fast_path` in the eligible dflash masked-verify
    /// regime (no grammar, no armed stage, argmax-preserving penalties)
    /// with only the sampling knobs varying.
    fn run(temperature: f32, force_temp_zero: bool) -> Option<Vec<u32>> {
        let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
        let dumps = crate::scheduler::dumps::RunDumps::default();
        let ctx = LogitsContext {
            scratch: &scratch,
            dumps: &dumps,
            stats: std::sync::Arc::new(crate::scheduler::spec_stats::SpecStats::new()),
            watchdog: crate::scheduler::helpers::WatchdogParams::default(),
            boundary_mask: None,
            mid_word_mask: None,
            sampling: SamplingLevers {
                dflash_masked_verify: true,
                fast_masked: true,
                mtp_verify_sample: true,
                force_temp_zero,
                ..Default::default()
            },
            timing: std::sync::Arc::default(),
            think_end_token: None,
            think_start_token: None,
            tool_call_start_token: None,
            tool_call_end_token: None,
        };
        let (mut a, _rx) = test_seq(Vec::new(), 5, None, 0);
        a.temperature = temperature;
        // The fixture ships the production LZ default; the penalty gate is
        // not under test here, so pin it argmax-preserving.
        a.lz_penalty = 0.0;
        try_chat_fast_path(&StubModel, &[42], &a, &ctx, 0)
    }

    /// Batch4 leftover: at temperature > 0 the slow path SAMPLES each
    /// verify position (the `mtp_verify_sample` branch), so the fast path
    /// returning the raw argmax silently disabled temperature/top_p/min_p/
    /// seed for every spec-accepted chat token. P1-3 put this load-bearing
    /// gate on the grammar sibling; this path (landed one day earlier)
    /// was missing it.
    #[test]
    fn nonzero_temperature_defers_to_the_sampling_slow_path() {
        assert_eq!(run(1.2, false), None);
    }

    /// Greedy chat is exactly the regime the fast path exists for — the
    /// new gate must not eat it.
    #[test]
    fn greedy_chat_still_takes_the_fast_path() {
        assert_eq!(run(0.0, false), Some(vec![42]));
    }

    /// Under ATLAS_FORCE_TEMP_ZERO the slow path pins the argmax at any
    /// request temperature, so the fast path stays equivalent — and taken.
    #[test]
    fn force_temp_zero_keeps_the_fast_path_at_nonzero_temperature() {
        assert_eq!(run(1.2, true), Some(vec![42]));
    }
}
