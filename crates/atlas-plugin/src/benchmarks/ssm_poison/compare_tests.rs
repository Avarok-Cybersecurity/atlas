// SPDX-License-Identifier: AGPL-3.0-only

//! Per-round comparison, tested without a server.

use super::{COLLAPSE_RATIO_CEIL, COLLAPSE_RATIO_FLOOR, RoundVerdict, TurnDelta, compare_round};
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
    let mut replay = vec![t("one", 10), t("two", 20)];
    replay[1].cached_prompt_tokens = 99;
    assert_eq!(compare_round(&reference, &replay), RoundVerdict::Invariant);
}

#[test]
fn equal_text_still_compares_token_count_shape() {
    let reference = vec![t("same emitted text", 100)];
    let collapsed = vec![t("same emitted text", 10)];
    assert_eq!(
        compare_round(&reference, &collapsed),
        RoundVerdict::Collapsed {
            turns: vec![TurnDelta {
                turn: 1,
                ref_tokens: 100,
                replay_tokens: 10,
                ref_finish: Some("stop".into()),
                replay_finish: Some("stop".into()),
            }],
        }
    );

    let jittered = vec![t("same emitted text", 101)];
    assert_eq!(
        compare_round(&reference, &jittered),
        RoundVerdict::Jittered {
            turns: vec![TurnDelta {
                turn: 1,
                ref_tokens: 100,
                replay_tokens: 101,
                ref_finish: Some("stop".into()),
                replay_finish: Some("stop".into()),
            }],
        }
    );
}

#[test]
fn a_healthy_length_jitter_is_jittered_not_collapsed() {
    // The clean-main finding: same finish reason, a few percent of length
    // change. This must be Jittered (recorded, passed), not a failure.
    let reference = vec![t("one", 100), t("two", 200)];
    let replay = vec![t("one-a", 98), t("two-b", 206)];
    assert_eq!(
        compare_round(&reference, &replay),
        RoundVerdict::Jittered {
            turns: vec![
                TurnDelta {
                    turn: 1,
                    ref_tokens: 100,
                    replay_tokens: 98,
                    ref_finish: Some("stop".into()),
                    replay_finish: Some("stop".into()),
                },
                TurnDelta {
                    turn: 2,
                    ref_tokens: 200,
                    replay_tokens: 206,
                    ref_finish: Some("stop".into()),
                    replay_finish: Some("stop".into()),
                },
            ],
        }
    );
}

#[test]
fn an_early_eos_collapse_is_collapsed() {
    // The batch4 signature: the reference answered fully, the replay hit
    // EOS immediately — drastically shorter.
    let reference = vec![t("one", 100), t("two", 200)];
    let replay = vec![t("one", 98), t("", 3)];
    assert_eq!(
        compare_round(&reference, &replay),
        RoundVerdict::Collapsed {
            turns: vec![TurnDelta {
                turn: 2,
                ref_tokens: 200,
                replay_tokens: 3,
                ref_finish: Some("stop".into()),
                replay_finish: Some("stop".into()),
            }],
        }
    );
}

#[test]
fn a_runaway_generation_is_collapsed() {
    // Poisoning can also manifest as runaway output that hits the budget.
    let reference = vec![t("one", 100)];
    let replay = vec![t(&format!("one{}", "x".repeat(400)), 260)];
    assert_eq!(
        compare_round(&reference, &replay),
        RoundVerdict::Collapsed {
            turns: vec![TurnDelta {
                turn: 1,
                ref_tokens: 100,
                replay_tokens: 260,
                ref_finish: Some("stop".into()),
                replay_finish: Some("stop".into()),
            }],
        }
    );
}

#[test]
fn a_different_finish_reason_is_collapsed() {
    // Same length but a different finish reason means the generation ended
    // differently — collapse.
    let mut reference = vec![t("abc", 100)];
    reference[0].finish_reason = Some("stop".into());
    let mut replay = vec![t("abd", 100)];
    replay[0].finish_reason = Some("length".into());
    assert_eq!(
        compare_round(&reference, &replay),
        RoundVerdict::Collapsed {
            turns: vec![TurnDelta {
                turn: 1,
                ref_tokens: 100,
                replay_tokens: 100,
                ref_finish: Some("stop".into()),
                replay_finish: Some("length".into()),
            }],
        }
    );
}

#[test]
fn reasoning_difference_is_jitter_when_shape_is_healthy() {
    let mut reference = vec![t("same", 100)];
    reference[0].reasoning = "thought A".into();
    let mut replay = vec![t("same", 101)];
    replay[0].reasoning = "thought B".into();
    assert_eq!(
        compare_round(&reference, &replay),
        RoundVerdict::Jittered {
            turns: vec![TurnDelta {
                turn: 1,
                ref_tokens: 100,
                replay_tokens: 101,
                ref_finish: Some("stop".into()),
                replay_finish: Some("stop".into()),
            }],
        }
    );
}

#[test]
fn two_empty_replies_are_unmeasured_not_invariant() {
    let reference = vec![t("", 0)];
    let replay = vec![t("", 0)];
    assert_eq!(
        compare_round(&reference, &replay),
        RoundVerdict::Unmeasured {
            reason: "turn 1 returned no tokens".into(),
        }
    );
}

#[test]
fn different_turn_counts_are_unmeasured() {
    let reference = vec![t("one", 10), t("two", 20)];
    assert_eq!(
        compare_round(&reference, &[t("one", 10)]),
        RoundVerdict::Unmeasured {
            reason: "replay produced 1 turn(s), reference has 2".into(),
        }
    );
    assert_eq!(
        compare_round(&reference, &[t("one", 10), t("two", 20), t("three", 30)]),
        RoundVerdict::Unmeasured {
            reason: "replay produced 3 turn(s), reference has 2".into(),
        }
    );
}

#[test]
fn an_empty_reference_round_is_unmeasured() {
    assert_eq!(
        compare_round(&[], &[]),
        RoundVerdict::Unmeasured {
            reason: "reference round has no turns".into(),
        }
    );
}

#[test]
fn collapse_bounds_are_the_documented_window() {
    let delta = |replay_tokens| TurnDelta {
        turn: 1,
        ref_tokens: 100,
        replay_tokens,
        ref_finish: Some("stop".into()),
        replay_finish: Some("stop".into()),
    };
    assert!(delta(49).is_collapse());
    assert!(!delta(50).is_collapse());
    assert!(!delta(51).is_collapse());
    assert!(!delta(200).is_collapse());
    assert!(delta(201).is_collapse());
    assert_eq!(COLLAPSE_RATIO_FLOOR, 0.5);
    assert_eq!(COLLAPSE_RATIO_CEIL, 2.0);
}

#[test]
fn a_zero_token_reference_with_a_nonzero_replay_is_collapsed() {
    // B1: ref=0, replay>0 is an infinite length ratio — above any ceiling.
    // The old zero-ref branch returned false here, so an unbounded blowup
    // relative to an empty reference passed as Jittered.
    let delta = TurnDelta {
        turn: 1,
        ref_tokens: 0,
        replay_tokens: 80,
        ref_finish: Some("stop".into()),
        replay_finish: Some("stop".into()),
    };
    assert!(delta.is_collapse());

    let reference = vec![t("", 0)];
    let replay = vec![t("suddenly a full reply", 80)];
    assert_eq!(
        compare_round(&reference, &replay),
        RoundVerdict::Collapsed { turns: vec![delta] }
    );
}

#[test]
fn a_zero_token_reference_with_a_zero_replay_is_not_a_collapse() {
    // The both-empty case stays with the upstream Unmeasured rule; the
    // collapse predicate must not claim it.
    let delta = TurnDelta {
        turn: 1,
        ref_tokens: 0,
        replay_tokens: 0,
        ref_finish: Some("stop".into()),
        replay_finish: Some("stop".into()),
    };
    assert!(!delta.is_collapse());
}

#[test]
fn an_unmeasured_turn_is_not_masked_by_a_jittered_turn() {
    // B2: the gate's stated rule is that ANY unmeasured round fails. A round
    // with one empty-pair turn and one jittered turn used to return Jittered
    // (a pass), letting the unmeasured turn hide behind tolerated jitter.
    let reference = vec![t("", 0), t("two", 200)];
    let replay = vec![t("", 0), t("two-b", 206)];
    assert_eq!(
        compare_round(&reference, &replay),
        RoundVerdict::Unmeasured {
            reason: "turn 1 returned no tokens".into(),
        }
    );
}
