// SPDX-License-Identifier: AGPL-3.0-only

//! Spec-in-think parity (A146 / AVAROK_DFLASH_SPEC_THINK): the per-committed
//! thinking-token transition, shared by all three places that advance it.
//!
//! GLM-5.3 serving has two hand-duplicated token-commit paths — spec-off
//! decode (`decode_logits_step::process_decode_logits`) and the MTP/DFlash
//! verify-accept path (`emit_step::emit_token`) — plus the speculative
//! per-position advance inside the K-position verify window
//! (`verify_pipeline_helper::pick_positions::pick_positions_from_host`). The
//! thinking branch had drifted: `emit_token` never toggled `in_code_fence`
//! and never ran the THINK_LOOP watchdog, and the pick window never advanced
//! `thinking_tokens` at all. With speculation inside `<think>` every one of
//! those gaps changes which token a later position picks, so a T=0 spec-on
//! run diverged from spec-off. This is the one body all three sites call.

use crate::scheduler::confidence::{MAX_SENTENCE_DEFER_TOKENS, toggle_code_fence};
use crate::scheduler::helpers::{
    THINK_LOOP_CHECK_STRIDE, THINK_LOOP_MIN_TOKENS, THINK_LOOP_PERIOD_MAX, THINK_LOOP_PERIOD_MIN,
    WatchdogParams, detect_thinking_token_loop_with,
};
use crate::scheduler::types::ActiveSeq;

/// Run-level inputs of [`advance_thinking_token`], identical at every site.
#[derive(Clone, Copy)]
pub(crate) struct ThinkTokenEnv {
    /// The tokenizer's atomic ``` token (`None` = fence tracking inert).
    pub code_fence_token: Option<u32>,
    /// `!disable_watchdogs && watchdog.enable_think_loop_watchdog`.
    pub think_loop_enabled: bool,
    pub watchdog: WatchdogParams,
}

/// Commit-time transition for ONE token committed inside `<think>` that is
/// NOT `</think>` (an EOS sampled inside think included — decode advances the
/// thinking state for it before discarding it).
///
/// `history_len`: how many `a.output_tokens` entries precede `tok` — the
/// THINK_LOOP scan must see exactly the history decode sees (decode runs this
/// BEFORE pushing `tok`; `emit_token` pushes first and passes `len - 1`).
/// `log`: false on the speculative pick window, which only predicts.
pub(crate) fn advance_thinking_token(
    a: &mut ActiveSeq,
    tok: u32,
    history_len: usize,
    env: ThinkTokenEnv,
    log: bool,
) {
    a.thinking_tokens += 1;
    // Track ``` code-fence parity within the thinking block: each fence
    // token flips in/out of a fenced code span. The forced `</think>`
    // injection defers while `in_code_fence` (`should_inject_think_end`).
    a.in_code_fence = toggle_code_fence(a.in_code_fence, tok, env.code_fence_token);
    // Set force_end_thinking when budget exhausted (picked up next position).
    if let Some(budget) = a.thinking_budget
        && a.thinking_tokens >= budget
        && !a.force_end_thinking
    {
        a.force_end_thinking = true;
        a.sentence_defer_count = 0;
        if log {
            // Name the budget's SOURCE: a 256-class cut with a large
            // --max-thinking-budget in force means the CLIENT sent the
            // budget (explicit tokens or a reasoning_effort rung).
            tracing::info!(
                source = if a.enable_thinking {
                    "request (client budget/effort; scaled by --max-thinking-budget)"
                } else {
                    "spontaneous <think> (--max-thinking-budget / MODEL.toml)"
                },
                "Thinking budget exhausted ({budget} tokens), arming </think>; \
                 deferring up to {MAX_SENTENCE_DEFER_TOKENS} tokens for sentence boundary"
            );
        }
    }
    // Token-level fence-loop detection (THINK_LOOP). Catches the phrase
    // attractor within ~24-60 tokens of the loop starting instead of waiting
    // for the thinking budget.
    if env.think_loop_enabled
        && !a.force_end_thinking
        && a.thinking_tokens >= THINK_LOOP_MIN_TOKENS
        && a.thinking_tokens.is_multiple_of(THINK_LOOP_CHECK_STRIDE)
        && detect_thinking_token_loop_with(
            &a.output_tokens[..history_len.min(a.output_tokens.len())],
            a.repetition_detection,
            env.watchdog,
        )
    {
        a.force_end_thinking = true;
        a.sentence_defer_count = 0;
        a.think_watchdog_fires = a.think_watchdog_fires.saturating_add(1);
        if log {
            tracing::warn!(
                thinking_tokens = a.thinking_tokens,
                watchdog_fires = a.think_watchdog_fires,
                "Thinking-loop watchdog fired (period-{}…{} repeat in tail); forcing </think> early",
                THINK_LOOP_PERIOD_MIN,
                THINK_LOOP_PERIOD_MAX,
            );
        }
    }
}

/// Budget for a spontaneous `<think>` re-entry: decays `>> fires.min(4)` per
/// prior THINK_LOOP fire, floored at 8. SSOT for decode and `emit_token`
/// (the emit twin used to reset to the undecayed budget).
pub(crate) fn spontaneous_think_budget(a: &ActiveSeq) -> u32 {
    let decay_shift = a.think_watchdog_fires.min(4);
    (a.spontaneous_think_budget >> decay_shift).max(8)
}

/// Post-pipeline accumulator state recorded per verify position by the pick
/// window, consumed by `emit_token` when that position commits.
///
/// The logits pipeline mutates three per-sequence accumulators inside
/// `<think>` (F2 `consecutive_confident` / arming `force_end_thinking`, the
/// forced-`</think>` injector's `sentence_defer_count` tick). Spec-off decode
/// runs the pipeline then commits, once per token. The verify window runs the
/// pipeline for ALL K positions before knowing how many commit, so the
/// sequence used to keep the K-th position's accumulators even when fewer
/// positions were accepted — the next window then double-counted the
/// rejected tail (defer ticks reach the 64-token hard override early, F2
/// streaks over-count). The window now restores them and leaves this trail;
/// each committing `emit_token` re-applies its own position's entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpecThinkTrail {
    /// The pick at this position (the token `emit_token` must be committing).
    pub tok: u32,
    /// `output_tokens.len()` just before this position commits.
    pub out_len: usize,
    pub consecutive_confident: u32,
    pub sentence_defer_count: u32,
    pub force_end_thinking: bool,
}

impl SpecThinkTrail {
    pub(crate) fn capture(a: &ActiveSeq, tok: u32) -> Self {
        Self {
            tok,
            out_len: a.output_tokens.len(),
            consecutive_confident: a.consecutive_confident,
            sentence_defer_count: a.sentence_defer_count,
            force_end_thinking: a.force_end_thinking,
        }
    }
}

/// `emit_token` entry hook: if `tok` is the next position of the last pick
/// window, apply that position's post-pipeline accumulators (what decode's
/// pipeline would have left before committing `tok`). Any mismatch means the
/// trail is stale (rejected tail, or a non-window emit) — drop it.
pub(crate) fn apply_spec_think_trail(a: &mut ActiveSeq, tok: u32) {
    let Some(t) = a.spec_think_trail.pop_front() else {
        return;
    };
    if t.tok == tok && t.out_len == a.output_tokens.len() {
        a.consecutive_confident = t.consecutive_confident;
        a.sentence_defer_count = t.sentence_defer_count;
        a.force_end_thinking = t.force_end_thinking;
    } else {
        a.spec_think_trail.clear();
    }
}
