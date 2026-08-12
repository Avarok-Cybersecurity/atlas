// SPDX-License-Identifier: AGPL-3.0-only

//! Per-round comparison, tested without a server.

use super::{RoundVerdict, compare_round, first_divergence};
use crate::benchmarks::transcript::Transcript;

fn t(text: &str, tokens: usize) -> Transcript {
    Transcript {
        text: text.into(),
        finish_reason: Some("stop".into()),
        completion_tokens: tokens,
        ..Default::default()
    }
}

#[test]
fn identical_rounds_are_invariant() {
    let reference = vec![t("one", 10), t("two", 20)];
    let replay = vec![t("one", 10), t("two", 20)];
    assert_eq!(compare_round(&reference, &replay), RoundVerdict::Invariant);
}

#[test]
fn a_text_difference_is_diverged_and_names_the_turn() {
    let reference = vec![t("one", 10), t("two", 20), t("three", 30)];
    let replay = vec![t("one", 10), t("TWO", 20), t("three", 30)];
    assert_eq!(
        compare_round(&reference, &replay),
        RoundVerdict::Diverged { turns: vec![2] }
    );
}

#[test]
fn finish_reason_difference_counts() {
    // A length cut changes the finish_reason but can leave the text equal up
    // to the cut — the canonical form must catch it anyway.
    let mut reference = vec![t("abc", 3)];
    reference[0].finish_reason = Some("stop".into());
    let mut replay = vec![t("abc", 2)];
    replay[0].finish_reason = Some("length".into());
    assert_eq!(
        compare_round(&reference, &replay),
        RoundVerdict::Diverged { turns: vec![1] }
    );
}

#[test]
fn reasoning_difference_counts() {
    let mut reference = vec![t("same", 10)];
    reference[0].reasoning = "thought A".into();
    let mut replay = vec![t("same", 10)];
    replay[0].reasoning = "thought B".into();
    assert_eq!(
        compare_round(&reference, &replay),
        RoundVerdict::Diverged { turns: vec![1] }
    );
}

#[test]
fn two_empty_replies_are_unmeasured_not_invariant() {
    // Two empty replies are "equal" and prove nothing — the contamination
    // detector's same rule.
    let reference = vec![t("", 0)];
    let replay = vec![t("", 0)];
    match compare_round(&reference, &replay) {
        RoundVerdict::Unmeasured { reason } => assert!(reason.contains("no tokens")),
        other => panic!("expected Unmeasured, got {other:?}"),
    }
}

#[test]
fn a_short_replay_is_unmeasured() {
    let reference = vec![t("one", 10), t("two", 20)];
    let replay = vec![t("one", 10)];
    match compare_round(&reference, &replay) {
        RoundVerdict::Unmeasured { reason } => assert!(reason.contains("turn(s)")),
        other => panic!("expected Unmeasured, got {other:?}"),
    }
}

#[test]
fn an_empty_reference_is_unmeasured() {
    let reference: Vec<Transcript> = vec![];
    let replay: Vec<Transcript> = vec![];
    assert!(matches!(
        compare_round(&reference, &replay),
        RoundVerdict::Unmeasured { .. }
    ));
}

#[test]
fn first_divergence_reports_char_offset_and_excerpts() {
    let reference = t("the quick brown fox", 10);
    let replay = t("the quick green fox", 10);
    let (offset, ref_ex, rep_ex) = first_divergence(&reference, &replay);
    // canonical() = reasoning + '\u{1}' + text + ..., so the common prefix is
    // '\u{1}' + "the quick " = 11 chars.
    assert_eq!(offset, 11);
    assert!(ref_ex.starts_with("brown"), "{ref_ex}");
    assert!(rep_ex.starts_with("green"), "{rep_ex}");
}
