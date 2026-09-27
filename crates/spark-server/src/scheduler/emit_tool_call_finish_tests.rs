// SPDX-License-Identifier: AGPL-3.0-only

//! Behavioural tests for the plain-chat `</tool_call>` hard stop on the MTP /
//! spec-verify path (`emit_step::emit_token`).
//!
//! A143 part B (2026-09-24): `decode_logits_step::process_decode_logits`
//! (the serial decode path) ends the turn when a request with NO tools
//! declared and no active grammar emits `</tool_call>` — a request with no
//! tools has no legitimate reason to emit tool-call syntax at all. The
//! speculative emit path (`emit_step::emit_token`, driven by MTP K2/K3 and
//! DFlash verify-accept) never had the twin guard, so a spec-on turn kept
//! decoding past the first (and only legal) call while spec-off correctly
//! stopped — a spec-on/spec-off behavioural divergence on the identical
//! prompt. These tests drive the real `emit_token` (no re-implementation of
//! the predicate) and pin both sides of the #192 distinction:
//!  * no tools declared, no grammar ⇒ `</tool_call>` finishes the turn;
//!  * tools declared (`tools_present`) ⇒ the turn survives so the model can
//!    emit PARALLEL calls, vLLM parity (#192).
//!
//! A third test proves the fix at the shape every verify-accept caller
//! (`mtp_step`, `verify_k3_step`, `verify_dflash_step`,
//! `verify_dflash_batch_step`) actually uses: `emit_token(...)` then
//! `if a.finished { break }` inside a loop over one verify window's accepted
//! tokens. A full step-function-level integration test (driving
//! `step_mtp` / `step_verify_k3` / `step_verify_dflash` /
//! `step_verify_dflash_batched` end-to-end with a scripted stub `Model`) was
//! judged impractical for this fix: each of those needs its own
//! draft-proposal / CUDA-graph-verify / DFlash ctx-commit plumbing on the
//! stub well beyond the base `Model` trait `test_support.rs` covers — the
//! same reason `test_support`'s existing fixtures stop at the `emit_token`
//! entry point rather than the step functions that call it.

use super::emit_step::emit_token;
use super::sched_ctx::SchedCtx;
use super::test_support::test_seq;
use super::types::ActiveSeq;

/// A live (not-yet-finished) sequence with no grammar and no tools declared
/// — the "plain chat" shape the A143 part B guard targets. Seven content
/// tokens already emitted so the fixture's `min_tokens` (7) is never in play.
fn plain_chat_seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq((1000..1007).collect(), 50, None, 10);
    a.finished = false;
    a.inside_thinking = false;
    debug_assert!(a.grammar_state.is_none());
    debug_assert!(!a.tools_present);
    a
}

#[test]
fn plain_chat_tool_call_end_finishes_the_sequence() {
    // RED if the emit path is missing the decode_logits_step twin: a
    // no-tools turn would keep decoding past `</tool_call>` on the
    // speculative path while the serial path (process_decode_logits) stops —
    // exactly the spec-on/spec-off divergence A143 part B reports.
    let sched = SchedCtx::for_test();
    let mut a = plain_chat_seq();
    let end_tok = a
        .tool_call_end_token
        .expect("fixture sets tool_call_end_token");
    emit_token(&mut a, end_tok, None, &sched);
    assert!(
        a.finished,
        "a no-tools, no-grammar </tool_call> must hard-stop the turn on the \
         speculative emit path, mirroring decode_logits_step"
    );
    assert!(
        a.tool_call_completed,
        "the completion flag must still be set (Fix A, unrelated to the stop)"
    );
    // "the </tool_call> itself IS emitted, nothing after it": emit_token
    // falls through to its normal push, it does not early-return.
    assert_eq!(
        a.output_tokens.last(),
        Some(&end_tok),
        "</tool_call> must still be pushed to output_tokens before the turn ends"
    );
}

#[test]
fn tools_present_tool_call_end_does_not_finish_the_sequence() {
    // #192 twin: a request that DID declare tools must survive a closed call
    // so the model can emit PARALLEL calls; the turn ends at natural EOS or
    // a watchdog, never at this guard.
    let sched = SchedCtx::for_test();
    let mut a = plain_chat_seq();
    a.tools_present = true;
    let end_tok = a
        .tool_call_end_token
        .expect("fixture sets tool_call_end_token");
    emit_token(&mut a, end_tok, None, &sched);
    assert!(
        !a.finished,
        "tools_present must NOT hard-stop at </tool_call> — #192 parallel calls"
    );
    assert!(a.tool_call_completed);
    assert_eq!(a.output_tokens.last(), Some(&end_tok));
}

#[test]
fn finished_stops_the_accepted_token_loop_mid_window() {
    // Shape-level proof at the actual call-site pattern used by every
    // verify-accept caller (mtp_step.rs / verify_k3_step.rs /
    // verify_dflash_step.rs / verify_dflash_batch_step.rs):
    //   for tok in accepted_window { emit_token(a, tok, ..); if a.finished { break; } }
    // A verify window that accepted 3 drafts, where the FIRST accepted
    // token happens to be `</tool_call>` on a no-tools turn, must emit only
    // that one token and never reach the remaining two.
    let sched = SchedCtx::for_test();
    let mut a = plain_chat_seq();
    let end_tok = a
        .tool_call_end_token
        .expect("fixture sets tool_call_end_token");
    let window = [end_tok, 2000, 2001];
    let mut calls = 0usize;
    for tok in window {
        calls += 1;
        emit_token(&mut a, tok, None, &sched);
        if a.finished {
            break;
        }
    }
    assert_eq!(
        calls, 1,
        "the loop must stop after the </tool_call> token — the remaining \
         accepted drafts of this verify window must never reach emit_token"
    );
    assert_eq!(
        a.output_tokens.last(),
        Some(&end_tok),
        "output_tokens must end at </tool_call>, not run on into 2000/2001"
    );
    assert!(a.finished);
}
