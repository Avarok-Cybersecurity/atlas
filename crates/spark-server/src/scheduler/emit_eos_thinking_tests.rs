// SPDX-License-Identifier: AGPL-3.0-only

//! Behavioural tests for thinking-mode EOS suppression on the MTP /
//! spec-verify path (`emit_step::emit_token`).
//!
//! A model-sampled EOS inside a `<think>` span is normally spurious —
//! `</think>` is the only legal exit — so `emit_token` must discard it and
//! keep generating (Nemotron-3.5-Lightning's greedy reasoning argmaxes
//! `<|im_end|>` ~30 tokens in; honoring it ended every speculative thinking
//! turn there, 2026-08-25). The suppression has exactly one escape: at a
//! HARD ceiling (completion budget exhausted / served max_seq_len reached)
//! the EOS MUST be honored so generation cannot overrun its declared limits.
//!
//! These tests drive the real `emit_token` over a real `ActiveSeq` — no
//! re-implementation of the `eos_suppressed_by_thinking` predicate (its pure
//! core is pinned in `helpers.rs::hard_limit_tests`) — and pin all three
//! sides of the decision as a caller observes them:
//!  * inside `<think>`, EOS must NOT finish the sequence;
//!  * outside `<think>`, the same EOS must finish it;
//!  * inside `<think>` AT the hard ceiling, EOS must finish it (without the
//!    escape the suppressed-EOS branch returns BEFORE the trailing
//!    `remaining == 0` length stop, so the sequence would run on past its
//!    budget — the overrun this lane exists to prevent).

use super::emit_step::emit_token;
use super::sched_ctx::SchedCtx;
use super::test_support::{EOS, test_seq};
use super::types::ActiveSeq;

/// A sequence mid-`<think>`: seven content tokens already emitted so the
/// fixture's `min_tokens` (7) can never be the suppressor — after the EOS
/// push `output_tokens.len()` is 8, past the floor. The budget (5000) is far
/// from exhausted and `SchedCtx::for_test()` serves `max_seq_len == 0`
/// (unset), so no hard ceiling is in play: `finished` can only move on the
/// EOS decision itself, never on a length stop.
fn thinking_phase_seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq((1000..1007).collect(), 5000, None, 10);
    a.finished = false;
    a.inside_thinking = true;
    // `_rx` is dropped: the fixture's sink is Blocking, which `emit_token`
    // never sends on (only `finish_sequence` does), so no send can fail.
    a
}

#[test]
fn eos_inside_thinking_does_not_finish_the_sequence() {
    // RED if the emit path loses its `thinking_suppresses_eos` term (the gap
    // it historically had vs `decode_logits_step`): EOS would end the turn
    // ~30 tokens into the reasoning span.
    let sched = SchedCtx::for_test();
    let mut a = thinking_phase_seq();
    emit_token(&mut a, EOS[0], None, &sched);
    assert!(
        !a.finished,
        "a spurious EOS inside <think> ended the turn — only </think> may \
         exit a thinking span while no hard ceiling is hit"
    );
    assert!(
        a.inside_thinking,
        "the discarded EOS must not exit thinking mode"
    );
    // The sequence is still live: a subsequent thinking token is processed
    // normally, not dropped by a half-finished state.
    emit_token(&mut a, 2000, None, &sched);
    assert!(
        !a.finished,
        "the token after the discarded EOS ended the turn"
    );
}

#[test]
fn same_eos_outside_thinking_finishes_the_sequence() {
    // The positive twin: identical state, identical token, thinking off —
    // proves the suppression above is keyed on `inside_thinking` and not on
    // some other term (min_tokens, grammar, budget) accidentally masking EOS.
    let sched = SchedCtx::for_test();
    let mut a = thinking_phase_seq();
    a.inside_thinking = false;
    emit_token(&mut a, EOS[0], None, &sched);
    assert!(
        a.finished,
        "EOS outside <think> with min_tokens met and no grammar must finish \
         the sequence"
    );
}

#[test]
fn eos_at_hard_ceiling_finishes_even_inside_thinking() {
    // The escape (§C-2): `remaining` is 1, so processing this EOS token
    // consumes the last of the completion budget and the hard ceiling is hit
    // at the decision point. Without the `!hard_ceiling` term the suppressed
    // branch RETURNS before the trailing length stop — `finished` stays
    // false and the think block runs past max_tokens (the R1X overrun).
    let sched = SchedCtx::for_test();
    let mut a = thinking_phase_seq();
    a.remaining = 1;
    emit_token(&mut a, EOS[0], None, &sched);
    assert!(
        a.finished,
        "a model-sampled EOS at the exhausted completion budget must be \
         honored even inside <think> — the escape that stops budget overrun"
    );
}
