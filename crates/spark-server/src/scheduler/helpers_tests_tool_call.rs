// SPDX-License-Identifier: AGPL-3.0-only

//! `tool_call_open_in_tail` unit tests. Split out of `helpers_tests.rs`
//! to keep that file ≤500 LoC (CI file-size-cap); attached to
//! `helpers.rs` with `#[path]` exactly like its sibling, so `use super::*`
//! still resolves to the helpers under test.

use super::*;

// ── tool_call_open_in_tail: "is the model mid-<tool_call>?" ────────
// SSOT for the content-phase fuzzy detector's tool exclusion and for
// the thinking-phase `</think>` force-close deferral.

const TC_START: u32 = 151657; // <tool_call>
const TC_END: u32 = 151658; // </tool_call>

#[test]
fn tool_call_open_after_unmatched_opener() {
    let tokens = [1, 2, TC_START, 3, 4];
    assert!(tool_call_open_in_tail(
        &tokens,
        Some(TC_START),
        Some(TC_END)
    ));
}

#[test]
fn tool_call_closed_after_matching_close() {
    let tokens = [1, TC_START, 3, TC_END, 9];
    assert!(!tool_call_open_in_tail(
        &tokens,
        Some(TC_START),
        Some(TC_END)
    ));
}

#[test]
fn tool_call_reopened_after_a_completed_call() {
    // A completed call must not mask a second, still-open one.
    let tokens = [TC_START, 3, TC_END, 7, TC_START, 8];
    assert!(tool_call_open_in_tail(
        &tokens,
        Some(TC_START),
        Some(TC_END)
    ));
}

#[test]
fn tool_call_open_is_false_without_marker_ids() {
    // Tokenizer did not encode `<tool_call>` atomically: every caller
    // must fall back to its pre-existing behaviour, never guess.
    let tokens = [1, TC_START, 3];
    assert!(!tool_call_open_in_tail(&tokens, None, Some(TC_END)));
    assert!(!tool_call_open_in_tail(&[1, 2, 3], Some(TC_START), None));
}

#[test]
fn tool_call_open_with_no_close_id_registered() {
    // `</tool_call>` unresolved but `<tool_call>` resolved: an opener in
    // the tail still means "open" (nothing can close it).
    let tokens = [1, TC_START, 3];
    assert!(tool_call_open_in_tail(&tokens, Some(TC_START), None));
}
