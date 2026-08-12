// SPDX-License-Identifier: AGPL-3.0-only

//! Per-round comparison, tested without a server.

use super::{
    COLLAPSE_RATIO_CEIL, COLLAPSE_RATIO_FLOOR, RoundVerdict, TurnDelta, compare_round,
    first_divergence,
};
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
fn a_healthy_length_jitter_is_jittered_not_collapsed() {
    // The clean-main finding: same finish reason, a few percent of length
    // change. This must be Jittered (recorded, passed), not a failure.
    let reference = vec![t("one", 100), t("two", 200)];
    let replay = vec![t("one-a", 98), t("two-b", 206)];
    match compare_round(&reference, &replay) {
        RoundVerdict::Jittered { turns } => {
            assert_eq!(turns.len(), 2);
            assert_eq!(turns[0].turn, 1);
            assert_eq!(turns[1].turn, 2);
        }
        other => panic!("expected Jittered, got {other:?}"),
    }
}

#[test]
fn an_early_eos_collapse_is_collapsed() {
    // The batch4 signature: the reference answered fully, the replay hit
    // EOS immediately — drastically shorter.
    let reference = vec![t("one", 100), t("two", 200)];
    let replay = vec![t("one", 98), t("", 3)];
    match compare_round(&reference, &replay) {
        RoundVerdict::Collapsed { turns } => {
            assert_eq!(turns.len(), 1);
            assert_eq!(turns[0].turn, 2);
        }
        other => panic!("expected Collapsed, got {other:?}"),
    }
}

#[test]
fn a_runaway_generation_is_collapsed() {
    // Poisoning can also manifest as runaway output that hits the budget.
    let reference = vec![t("one", 100)];
    let replay = vec![t(&format!("one{}", "x".repeat(400)), 260)];
    match compare_round(&reference, &replay) {
        RoundVerdict::Collapsed { turns } => assert_eq!(turns[0].turn, 1),
        other => panic!("expected Collapsed, got {other:?}"),
    }
}

#[test]
fn a_different_finish_reason_is_collapsed() {
    // Same length but a different finish reason means the generation ended
    // differently — collapse.
    let mut reference = vec![t("abc", 100)];
    reference[0].finish_reason = Some("stop".into());
    let mut replay = vec![t("abd", 100)];
    replay[0].finish_reason = Some("length".into());
    match compare_round(&reference, &replay) {
        RoundVerdict::Collapsed { turns } => assert_eq!(turns[0].turn, 1),
        other => panic!("expected Collapsed, got {other:?}"),
    }
}

#[test]
fn reasoning_difference_is_jitter_when_shape_is_healthy() {
    let mut reference = vec![t("same", 100)];
    reference[0].reasoning = "thought A".into();
    let mut replay = vec![t("same", 101)];
    replay[0].reasoning = "thought B".into();
    assert!(matches!(
        compare_round(&reference, &replay),
        RoundVerdict::Jittered { .. }
    ));
}

#[test]
fn two_empty_replies_are_unmeasured_not_invariant() {
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
fn collapse_bounds_are_the_documented_window() {
    let floor = TurnDelta {
        turn: 1,
        ref_tokens: 100,
        replay_tokens: 49,
        ref_finish: Some("stop".into()),
        replay_finish: Some("stop".into()),
    };
    let healthy = TurnDelta {
        replay_tokens: 51,
        ..floor.clone()
    };
    assert!(floor.is_collapse());
    assert!(!healthy.is_collapse());
    assert!((COLLAPSE_RATIO_FLOOR - 0.5).abs() < 1e-9);
    assert!((COLLAPSE_RATIO_CEIL - 2.0).abs() < 1e-9);
}

#[test]
fn first_divergence_reports_char_offset_and_excerpts() {
    let reference = t("the quick brown fox", 10);
    let replay = t("the quick green fox", 10);
    let (offset, ref_ex, rep_ex) = first_divergence(&reference, &replay);
    assert_eq!(offset, 11); // '\u{1}' + "the quick "
    assert!(ref_ex.starts_with("brown"), "{ref_ex}");
    assert!(rep_ex.starts_with("green"), "{rep_ex}");
}
