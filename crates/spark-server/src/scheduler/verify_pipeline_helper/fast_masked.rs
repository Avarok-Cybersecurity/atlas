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
use crate::scheduler::mtp_timing::{self, Phase};
use spark_model::traits::Model;

/// Returns `Some(picks)` when the fast path proves masked-greedy ==
/// raw-argmax for every position (picks are the raw `argmax_ids`);
/// `None` when any gate fails and the caller must run the slow path.
pub(super) fn try_chat_fast_path(
    model: &dyn Model,
    argmax_ids: &[u32],
    a: &ActiveSeq,
    ctx: &LogitsContext,
) -> Option<Vec<u32>> {
    // DFlash masked-verify mode ONLY. The fast path exists to make
    // ATLAS_DFLASH_MASKED_VERIFY affordable; it must never run for MTP:
    // returning the GPU argmax where the slow path computes a host-side
    // argmax over dequantized F32 logits changes tie-breaking on
    // near-tie tokens — measured 2026-07-11 as temp-0 MTP output drift
    // vs an unpatched binary (think block identical, answer flips at
    // low-margin tokens). MTP keeps the slow path unconditionally so
    // its behavior is byte-invariant by construction.
    if !super::dflash_masked_verify_enabled() {
        return None;
    }
    let fast_masked_enabled = {
        static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *CACHED
            .get_or_init(|| std::env::var("ATLAS_DISABLE_FAST_MASKED").ok().as_deref() != Some("1"))
    };
    let adadec_recording = {
        static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *CACHED.get_or_init(|| std::env::var("ATLAS_ADADEC_DIAGNOSTIC").is_ok())
    };
    if !fast_masked_enabled || a.grammar_state.is_some() || adadec_recording {
        return None;
    }
    use crate::scheduler::confidence::{
        MAX_SENTENCE_DEFER_TOKENS, THINK_DEFER_ABS_CEILING, THINK_DEFER_BUDGET_FACTOR,
    };
    // (b) forced/stateful stage preconditions — mirrored exactly from
    // f2_confidence.rs / forced_think_end.rs / pin_tool_call.rs.
    let f2_active = !crate::scheduler::helpers::disable_watchdogs()
        && a.inside_thinking
        && !a.force_end_thinking
        && a.thinking_tokens >= 400
        && crate::scheduler::helpers::watchdog_params().confidence_early_stop;
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
                crate::scheduler::fast_greedy::logit_is_positive(model, logits_base, i, vocab, tok)
            })
        {
            all_clear = false;
            break;
        }
    }
    mtp_timing::record(Phase::FastGreedy, t_fast);
    if all_clear {
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "verify chat fast path ACTIVE: masked-greedy == raw argmax, no D2H \
                 (kill-switch: ATLAS_DISABLE_FAST_MASKED=1)"
            );
        });
        return Some(argmax_ids.to_vec());
    }
    // Fall through — grammar fast path can't fire (grammar_state is
    // None), so the slow path handles the call.
    None
}
