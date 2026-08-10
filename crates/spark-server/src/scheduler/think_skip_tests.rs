// SPDX-License-Identifier: AGPL-3.0-only

//! Behavioural tests for the stray-`</think>` watchdog on the MTP /
//! spec-verify path (`emit_step::emit_token`).
//!
//! The watchdog exists for one documented degeneration: at long context
//! the model repeats `</think>` forever. That failure is CONSECUTIVE by
//! definition, and the non-MTP twin (`decode_logits_step.rs`) counts it
//! that way — it zeroes `think_skip_count` on every real content token.
//! `emit_token` had the increment and the `>= 50` threshold but no
//! reset, so on the MTP path the counter was CUMULATIVE: 50 SCATTERED
//! strays spread across otherwise healthy content force-stopped the
//! response, while the identical token stream on the non-MTP path ran
//! to completion.
//!
//! These tests drive the real `emit_token` over a real `ActiveSeq` — no
//! re-implementation of the predicate — and pin both halves:
//!  * scattered strays must NOT stop the turn, however many there are;
//!  * a consecutive run of 50 must STILL stop it, so the bug cannot be
//!    "fixed" by defanging the watchdog.

use super::emit_step::emit_token;
use super::lifecycle::derive_finish_reason;
use super::sched_ctx::SchedCtx;
use super::test_support::{EOS, test_seq};
use super::types::{ActiveSeq, GUARD_STOP_THINK_SKIP};

/// Qwen3's `</think>`. Distinct from the fixture's EOS (151645) and
/// tool-call close (151658) so neither of those paths is entered.
const THINK_END: u32 = 151668;

/// The force-stop point in both `emit_step.rs` and `decode_logits_step.rs`.
const SKIP_LIMIT: u32 = 50;

/// A post-`</think>` sequence: thinking is over (`think_ended`), we are
/// outside a thinking span, and `</think>` is a live stray token. The
/// budget is deliberately far larger than any test emits so a
/// `finished` flag can only have come from the watchdog and never from
/// the `remaining == 0` length stop.
fn content_phase_seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq(Vec::new(), 5000, None, 10);
    a.finished = false;
    a.inside_thinking = false;
    a.think_ended = true;
    a.think_end_token = Some(THINK_END);
    // `_rx` is dropped: the fixture's sink is Blocking, which `emit_token`
    // never sends on (only `finish_sequence` does), so no send can fail.
    a
}

/// Distinct, strictly increasing content tokens. Using distinct ids keeps
/// the content-loop watchdog (period-N repeat detector, arms at 48 content
/// tokens) out of the experiment, so `finished` isolates the skip counter.
fn content_token(i: u32) -> u32 {
    1000 + i
}

#[test]
fn scattered_think_strays_do_not_stop_the_turn() {
    // POSITIVE. 60 strays — well past the 50 threshold — each separated
    // by one real content token. On the non-MTP path this generation
    // runs to completion; the MTP path must agree.
    //
    // RED without the `if a.think_ended { a.think_skip_count = 0; }`
    // reset in `emit_step.rs`: the cumulative counter reaches 50 on the
    // 50th stray and force-stops the sequence.
    let sched = SchedCtx::for_test();
    let mut a = content_phase_seq();
    let strays = 60;
    for i in 0..strays {
        emit_token(&mut a, THINK_END, None, &sched);
        assert!(
            !a.finished,
            "stray #{} of {strays} force-stopped the turn: scattered `</think>` \
             must not accumulate — the counter is reset by intervening content \
             on the non-MTP path, and `emit_token` must match it",
            i + 1
        );
        emit_token(&mut a, content_token(i), None, &sched);
        assert!(
            !a.finished,
            "content token after stray #{} ended the turn",
            i + 1
        );
    }
    // Asserted AFTER the loop, not inside it, so the loop's only failure
    // mode is the BEHAVIOUR (`finished`) rather than this white-box peek
    // at the counter — an unfixed `emit_token` must go red on "the turn
    // was force-stopped", which is what a user actually observes.
    assert_eq!(
        a.think_skip_count, 0,
        "the last content token must have zeroed the stray counter"
    );
    // The content actually reached the response — the strays were skipped,
    // not emitted, exactly as on the non-MTP path.
    assert_eq!(a.output_tokens.len(), strays as usize);
    assert!(!a.finished);
}

#[test]
fn fifty_consecutive_think_strays_still_stop_the_turn() {
    // NEGATIVE / boundary control. This must hold with AND without the
    // fix: it is the guard against "fixing" the asymmetry by deleting
    // the watchdog. 49 consecutive strays are survivable; the 50th is not.
    let sched = SchedCtx::for_test();
    let mut a = content_phase_seq();
    for i in 1..SKIP_LIMIT {
        emit_token(&mut a, THINK_END, None, &sched);
        assert!(
            !a.finished,
            "stray #{i} fired the watchdog early — the threshold is {SKIP_LIMIT}"
        );
        assert_eq!(a.think_skip_count, i, "stray #{i} did not increment");
    }
    emit_token(&mut a, THINK_END, None, &sched);
    assert_eq!(a.think_skip_count, SKIP_LIMIT);
    assert!(
        a.finished,
        "{SKIP_LIMIT} CONSECUTIVE `</think>` strays are the degeneration the \
         watchdog exists for; it must still force-stop the sequence"
    );
    // And the strays never reached the client.
    assert!(a.output_tokens.is_empty());
}

#[test]
fn the_watchdog_rearms_after_content_defuses_it() {
    // The reset must not be a one-way disarm: once content has zeroed the
    // counter, a FRESH consecutive run of 50 must still fire. Without this
    // the positive test above could be satisfied by never counting again.
    let sched = SchedCtx::for_test();
    let mut a = content_phase_seq();
    // Nearly trip it, then defuse with one content token.
    for _ in 0..SKIP_LIMIT - 1 {
        emit_token(&mut a, THINK_END, None, &sched);
    }
    emit_token(&mut a, content_token(0), None, &sched);
    assert!(!a.finished, "content must defuse a nearly-tripped counter");
    assert_eq!(a.think_skip_count, 0);
    // Now a full consecutive run, from zero.
    for _ in 0..SKIP_LIMIT {
        emit_token(&mut a, THINK_END, None, &sched);
    }
    assert!(
        a.finished,
        "the watchdog must re-arm: a fresh run of {SKIP_LIMIT} consecutive \
         strays after a reset must still stop the turn"
    );
}

/// The exact wiring `finish_sequence` performs over the sequence — kept as
/// one helper so both wire-reason tests below derive the reason the way
/// production does, not a re-implementation.
fn wire_reason(a: &ActiveSeq) -> &'static str {
    derive_finish_reason(
        a.guard_stop,
        a.output_tokens.last().copied(),
        &a.eos_tokens,
        a.tool_call_end_token,
        a.remaining,
        a.seq.seq_len,
        0, // max_seq_len unlimited — the ceiling rung stays out of the way
    )
}

#[test]
fn the_watchdog_cut_names_its_guard_and_wires_length() {
    // POSITIVE for the guard-naming fix. The watchdog SKIPS the stray
    // tokens (they are never pushed), so `last_tok` is not `</think>` and
    // `</think>` is not eos-registered anyway (Qwen3.6 eos = {248046,
    // 248044}, `</think>` = 248069). Unnamed, `derive_finish_reason` falls
    // through every rung and wires "stop" — and the agentic harness's
    // `was_cut_off()` (atlas-plugin agent.rs) grants a recovery turn ONLY
    // on "length", so a "stop" with no tool calls ends the whole run.
    let sched = SchedCtx::for_test();
    let mut a = content_phase_seq();
    for i in 1..SKIP_LIMIT {
        emit_token(&mut a, THINK_END, None, &sched);
        assert!(
            a.guard_stop.is_none(),
            "guard named before the threshold (stray #{i}) — the name must \
             mark the CUT, not the counting"
        );
    }
    emit_token(&mut a, THINK_END, None, &sched);
    assert!(a.finished);
    assert_eq!(
        a.guard_stop,
        Some(GUARD_STOP_THINK_SKIP),
        "the watchdog cut must name its guard at the call site"
    );
    assert_eq!(
        wire_reason(&a),
        "length",
        "a server-side watchdog cut with budget left is a truncation; \
         \"length\" is what lets an agentic client recover the turn"
    );
}

#[test]
fn a_genuine_eos_stop_is_not_converted_to_length() {
    // NEGATIVE — guards over-application. The fix names ONE cut; a model
    // that finishes naturally must still wire "stop", even after a
    // below-threshold burst of strays on the same turn.
    let sched = SchedCtx::for_test();
    let mut a = content_phase_seq();
    a.min_tokens = 0; // the fixture's 7 would suppress a bare EOS
    for i in 0..5 {
        emit_token(&mut a, content_token(i), None, &sched);
    }
    for _ in 0..3 {
        emit_token(&mut a, THINK_END, None, &sched);
    }
    emit_token(&mut a, EOS[0], None, &sched);
    assert!(a.finished, "EOS must finish the turn");
    assert!(
        a.guard_stop.is_none(),
        "a natural EOS finish must NOT name a guard — that would relabel a \
         real model stop as a server truncation and grant phantom recovery \
         turns"
    );
    assert_eq!(
        wire_reason(&a),
        "stop",
        "the model finished; the wire must say so"
    );
}

#[test]
fn strays_inside_thinking_are_not_counted_as_strays() {
    // Scope check on the increment's `!inside_thinking` gate: a `</think>`
    // that legitimately CLOSES a thinking span is not a stray. It exits
    // thinking and leaves the counter alone, so a request that thinks
    // repeatedly does not accrue watchdog credit.
    let sched = SchedCtx::for_test();
    let mut a = content_phase_seq();
    a.inside_thinking = true;
    a.think_ended = false;
    emit_token(&mut a, THINK_END, None, &sched);
    assert!(
        !a.inside_thinking,
        "`</think>` must close the thinking span"
    );
    assert!(a.think_ended);
    assert_eq!(a.think_skip_count, 0, "a legitimate close is not a stray");
    assert!(!a.finished);
}
