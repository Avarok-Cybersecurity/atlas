// SPDX-License-Identifier: AGPL-3.0-only

//! The zero-tolerance decision rule, tested as pure functions. No server.

use super::super::compare::RoundVerdict;
use super::{score, verdict};
use crate::result::VerdictKind;

fn inv(n: usize) -> (usize, RoundVerdict) {
    (n, RoundVerdict::Invariant)
}
fn div(n: usize, turns: &[usize]) -> (usize, RoundVerdict) {
    (
        n,
        RoundVerdict::Diverged {
            turns: turns.to_vec(),
        },
    )
}
fn unm(n: usize, why: &str) -> (usize, RoundVerdict) {
    (
        n,
        RoundVerdict::Unmeasured {
            reason: why.to_string(),
        },
    )
}

#[test]
fn all_invariant_is_pass() {
    let replays: Vec<_> = (1..=12).map(inv).collect();
    let s = score(&replays);
    assert_eq!(s.rounds, 12);
    assert_eq!(s.invariant, 12);
    assert_eq!(s.diverged, 0);
    let v = verdict(&s, 12);
    assert_eq!(v.kind, VerdictKind::Pass);
}

#[test]
fn a_single_divergence_is_fail_and_names_the_round() {
    // The incident's shape: rounds 1-7 clean, round 8 turns 3-4 diverged.
    let mut replays: Vec<_> = (1..=7).map(inv).collect();
    replays.push(div(8, &[3, 4]));
    replays.push(inv(9));
    replays.push(inv(10));
    let s = score(&replays);
    assert_eq!(s.diverged, 1);
    assert_eq!(s.diverged_rounds, vec![(8, vec![3, 4])]);
    let v = verdict(&s, 10);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.contains("round 8"), "{}", v.reason);
    assert!(v.reason.contains("1 of 10"), "{}", v.reason);
}

#[test]
fn multiple_divergences_are_all_named() {
    let replays = vec![inv(1), div(2, &[1]), div(3, &[2, 4]), inv(4)];
    let s = score(&replays);
    assert_eq!(s.diverged, 2);
    let v = verdict(&s, 4);
    assert!(v.reason.contains("round 2"));
    assert!(v.reason.contains("round 3"));
    assert!(v.reason.contains("2 of 4"), "{}", v.reason);
}

#[test]
fn an_unmeasured_round_fails_the_gate() {
    // A transport error means the invariant was NOT proven for that round —
    // failing is correct: a gate that cannot prove its invariant must not
    // pass.
    let replays = vec![inv(1), unm(2, "connection reset"), inv(3)];
    let s = score(&replays);
    assert_eq!(s.unmeasured, 1);
    let v = verdict(&s, 3);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.contains("unmeasurable"), "{}", v.reason);
}

#[test]
fn a_short_run_cannot_pass_by_running_fewer_replays() {
    // Only 2 replays completed when 12 were configured. Even though both
    // were invariant, the gate must not pass on partial evidence.
    let replays = vec![inv(1), inv(2)];
    let s = score(&replays);
    let v = verdict(&s, 12);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.contains("2 of 12"), "{}", v.reason);
}

#[test]
fn divergence_wins_over_unmeasured_in_the_reason() {
    // If both a divergence and an unmeasured round exist, the divergence is
    // the corruption signal and is named; unmeasured is still counted.
    let replays = vec![div(1, &[2]), unm(2, "timeout")];
    let s = score(&replays);
    let v = verdict(&s, 2);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.contains("diverged"), "{}", v.reason);
}
