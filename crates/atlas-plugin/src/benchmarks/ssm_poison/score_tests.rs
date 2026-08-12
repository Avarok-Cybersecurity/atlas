// SPDX-License-Identifier: AGPL-3.0-only

//! The decision rule, tested as pure functions. No server.

use super::super::compare::{RoundVerdict, TurnDelta};
use super::{score, verdict};
use crate::result::VerdictKind;

fn inv(n: usize) -> (usize, RoundVerdict) {
    (n, RoundVerdict::Invariant)
}
fn jit(n: usize) -> (usize, RoundVerdict) {
    (
        n,
        RoundVerdict::Jittered {
            turns: vec![TurnDelta {
                turn: 2,
                ref_tokens: 200,
                replay_tokens: 206,
                ref_finish: Some("stop".into()),
                replay_finish: Some("stop".into()),
            }],
        },
    )
}
fn col(n: usize) -> (usize, RoundVerdict) {
    (
        n,
        RoundVerdict::Collapsed {
            turns: vec![TurnDelta {
                turn: 2,
                ref_tokens: 200,
                replay_tokens: 3,
                ref_finish: Some("stop".into()),
                replay_finish: Some("stop".into()),
            }],
        },
    )
}
fn unm(n: usize) -> (usize, RoundVerdict) {
    (
        n,
        RoundVerdict::Unmeasured {
            reason: "reset".into(),
        },
    )
}

#[test]
fn all_invariant_is_pass() {
    let replays: Vec<_> = (1..=12).map(inv).collect();
    let v = verdict(&score(&replays), 12);
    assert_eq!(v.kind, VerdictKind::Pass);
}

#[test]
fn jitter_is_recorded_but_passes() {
    // The clean-main reality: every replay jitters a little (restore anchor
    // selection varies). That is a healthy engine, not a failure.
    let replays: Vec<_> = (1..=12).map(jit).collect();
    let s = score(&replays);
    assert_eq!(s.jittered, 12);
    assert_eq!(s.collapsed, 0);
    let v = verdict(&s, 12);
    assert_eq!(v.kind, VerdictKind::Pass);
    assert!(v.reason.contains("jittered"), "{}", v.reason);
}

#[test]
fn a_single_collapse_is_fail_and_names_the_round() {
    // The batch4 shape: most rounds fine, one round collapses.
    let mut replays: Vec<_> = (1..=7).map(inv).collect();
    replays.push(col(8));
    replays.push(inv(9));
    replays.push(jit(10));
    let s = score(&replays);
    assert_eq!(s.collapsed, 1);
    assert_eq!(s.collapsed_rounds[0].0, 8);
    let v = verdict(&s, 10);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.contains("COLLAPSED"), "{}", v.reason);
    assert!(v.reason.contains("round 8"), "{}", v.reason);
}

#[test]
fn an_unmeasured_round_fails_the_gate() {
    let replays = vec![inv(1), unm(2), inv(3)];
    let v = verdict(&score(&replays), 3);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.contains("unmeasurable"), "{}", v.reason);
}

#[test]
fn a_short_run_cannot_pass_by_running_fewer_replays() {
    let replays = vec![inv(1), inv(2)];
    let v = verdict(&score(&replays), 12);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.contains("2 of 12"), "{}", v.reason);
}

#[test]
fn collapse_wins_over_jitter_in_the_reason() {
    // A run with both a collapse and jitter must fail on the collapse, not
    // pass because jitter is tolerated.
    let replays = vec![jit(1), col(2)];
    let v = verdict(&score(&replays), 2);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.contains("COLLAPSED"), "{}", v.reason);
}
