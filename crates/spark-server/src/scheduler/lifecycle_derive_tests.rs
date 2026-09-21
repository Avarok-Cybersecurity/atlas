// SPDX-License-Identifier: AGPL-3.0-only

//! The pure precedence matrix over `derive_finish_reason`.
//!
//! Split out of `lifecycle_tests.rs` for the <=500-line cap, along the seam that file
//! already documents: this half never builds an `ActiveSeq`, never touches the stub
//! `Model`, and asserts only the pure function. The call-site proof stays next door.

use super::super::lifecycle::derive_finish_reason;
use super::super::types::GUARD_STOP_REQUEST_TIMEOUT;
use super::{EOS, MAX_SEQ_LEN, TOOL_END};
use crate::ir::FINISH_REASON_TIMEOUT;

/// Common-case shorthand: mid-context position (no seqlen ceiling), so
/// the budget dimension under test is the `remaining` countdown.
fn derive(guard: Option<&'static str>, last: Option<u32>, remaining: usize) -> &'static str {
    derive_finish_reason(guard, last, EOS, TOOL_END, remaining, 10, MAX_SEQ_LEN)
}

#[test]
fn length_means_budget_exhausted_only() {
    // max_tokens countdown exhausted on an ordinary content token.
    assert_eq!(derive(None, Some(42), 0), "length");
    // Served context ceiling reached with budget left — the other half
    // of the hard-ceiling stop predicate. OpenAI reports "length" for
    // the context limit too.
    assert_eq!(
        derive_finish_reason(
            None,
            Some(42),
            EOS,
            TOOL_END,
            500,
            MAX_SEQ_LEN - 1,
            MAX_SEQ_LEN
        ),
        "length"
    );
}

#[test]
fn early_stop_one_token_short_of_budget_is_not_length() {
    // The bug this wave fixes: "length" was the catch-all, so ANY stop
    // whose last token was not EOS/tool-end claimed the budget was hit
    // (observed live: `Done: 573 tokens (length)` under
    // max_new_tokens=1024). One token of budget left ⇒ not "length".
    let r = derive(None, Some(42), 1);
    assert_ne!(r, "length");
    // Early finalize with no guard = client cancel / dropped receiver /
    // shutdown drain ⇒ "stop" (generation stopped; budget not hit).
    assert_eq!(r, "stop");
}

#[test]
fn normal_stops_are_unchanged() {
    assert_eq!(derive(None, Some(151645), 100), "stop");
    assert_eq!(derive(None, Some(151658), 100), "tool_calls");
}

#[test]
fn eos_on_the_last_budgeted_token_is_stop_not_length() {
    // The model finished naturally ON the final budgeted token: the
    // sampled EOS outranks the exhausted budget (OpenAI parity).
    assert_eq!(derive(None, Some(151645), 0), "stop");
}

#[test]
fn timeout_unchanged_and_still_outranks_everything() {
    // Shipped contract (2026-08 deadline wave): a deadline cut must be
    // distinguishable from both a natural stop and a max_tokens stop,
    // even when the deadline lands on an EOS / tool-close / budget-end
    // step.
    for last in [Some(42), Some(151645), Some(151658)] {
        assert_eq!(
            derive(Some(GUARD_STOP_REQUEST_TIMEOUT), last, 100),
            FINISH_REASON_TIMEOUT
        );
    }
    assert_eq!(
        derive(Some(GUARD_STOP_REQUEST_TIMEOUT), Some(42), 0),
        FINISH_REASON_TIMEOUT
    );
}

#[test]
fn guard_cuts_report_length_because_the_model_did_not_finish() {
    // POSITIVE case. A guard cut is a server-side truncation: the model
    // was still mid-output. `"length"` is the OpenAI-spec slot for
    // "forcibly truncated" and is what every client's truncation handling
    // keys on (openai-python `LengthFinishReasonError`, aider's
    // continuation, Instructor, pydantic-ai).
    //
    // ★ This assertion was briefly INVERTED to `"stop"`, and that shipped
    // a measured regression: the agentic gate fell to 8/10 then 4/10
    // followed_directions because its `was_cut_off()` stopped firing and
    // runs ended at 3-10 turns instead of the 12-22 a recovery needs.
    // `"stop"` claims the model finished; for a mid-sentence repetition
    // cut that is false, and every client action keyed on it (accept,
    // validate, commit, end the run) is then wrong. Do not re-invert.
    for guard in [
        "fuzzy_repetition",
        "inter_tool_prose_budget",
        "tool_envelope_stuck",
        "simhash_semantic_loop",
        "token_loop_watchdog",
    ] {
        assert_eq!(
            derive(Some(guard), Some(42), 100),
            "length",
            "guard={guard}"
        );
        // A guard trip on the exact step the budget ran out is still a
        // truncation, and both paths agree — precedence is deterministic.
        assert_eq!(derive(Some(guard), Some(42), 0), "length", "guard={guard}");
    }
}

#[test]
fn non_truncating_stops_are_not_relabelled_as_length() {
    // NEGATIVE case, and the whole point of the original fix: `"length"`
    // must NOT become a catch-all again. The bug this replaced derived it
    // from "the last token wasn't EOS", sweeping in early finalizes and
    // client cancels that are not truncations at all.
    //
    // No guard, budget left (client cancel / dropped receiver / drain):
    // generation stopped, nothing was truncated ⇒ "stop", never "length".
    assert_eq!(derive(None, Some(42), 100), "stop");
    // And the timeout guard keeps its own distinct reason rather than
    // collapsing into the truncation bucket.
    assert_eq!(
        derive(Some(GUARD_STOP_REQUEST_TIMEOUT), Some(42), 100),
        FINISH_REASON_TIMEOUT
    );
}

#[test]
fn token_derived_stops_outrank_non_timeout_guards() {
    // Preserved from the old test: a guard that fires on the same step
    // the model sampled EOS / the tool-call close reports what the
    // model actually did.
    assert_eq!(
        derive(Some("tool_envelope_stuck"), Some(151645), 100),
        "stop"
    );
    assert_eq!(
        derive(Some("fuzzy_repetition"), Some(151658), 100),
        "tool_calls"
    );
}

#[test]
fn empty_output_edges() {
    // max_tokens==0 scoring path: empty output, remaining==0 ⇒ "length"
    // (the budget of zero was exhausted before the first token).
    assert_eq!(derive(None, None, 0), "length");
    // Empty output on a model with NO tool-call end token configured
    // must not satisfy `None == None` and misreport "tool_calls".
    assert_eq!(
        derive_finish_reason(None, None, EOS, None, 5, 10, MAX_SEQ_LEN),
        "stop"
    );
}
