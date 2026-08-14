// SPDX-License-Identifier: AGPL-3.0-only

//! The decision rule, tested as pure functions. No server.

use super::super::compare::{RoundVerdict, TurnDelta};
use super::{RoundRecord, score, verdict};
use crate::result::VerdictKind;

/// A healthy turn-1 cache attestation. 992 is the smallest restore anchor
/// the 2026-08-12 run observed; any nonzero value exercises the same path.
const WARM: Option<usize> = Some(992);

fn record(round: usize, verdict: RoundVerdict, turn1_cached: Option<usize>) -> RoundRecord {
    RoundRecord {
        round,
        verdict,
        turn1_cached,
    }
}

fn inv(n: usize) -> RoundRecord {
    record(n, RoundVerdict::Invariant, WARM)
}
fn jit(n: usize) -> RoundRecord {
    record(
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
        WARM,
    )
}
fn col(n: usize) -> RoundRecord {
    record(
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
        WARM,
    )
}
fn unm(n: usize) -> RoundRecord {
    record(
        n,
        RoundVerdict::Unmeasured {
            reason: "reset".into(),
        },
        None,
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

#[test]
fn a_zero_cache_replay_fails_even_when_every_transcript_matched() {
    // The vacuity finding: with prefix caching off, every replay reproduces
    // the reference byte-for-byte (nothing was restored, so nothing could
    // be poisoned) and the gate returned a green PASS proving nothing.
    let mut replays: Vec<_> = (1..=2).map(inv).collect();
    replays.push(record(3, RoundVerdict::Invariant, Some(0)));
    let s = score(&replays);
    assert_eq!(s.vacuous_rounds, vec![3]);
    assert_eq!(s.min_turn1_cached, Some(0));
    let v = verdict(&s, 3);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.contains("0 cached prompt tokens"), "{}", v.reason);
    assert!(v.reason.contains("[3]"), "{}", v.reason);
}

#[test]
fn an_all_cold_run_fails_on_every_round() {
    // Prefix caching disabled: all rounds attest zero. This is the exact
    // configuration under which the pre-fix gate passed.
    let replays: Vec<_> = (1..=12)
        .map(|n| record(n, RoundVerdict::Invariant, Some(0)))
        .collect();
    let s = score(&replays);
    assert_eq!(s.vacuous_rounds.len(), 12);
    let v = verdict(&s, 12);
    assert_eq!(v.kind, VerdictKind::Fail);
}

#[test]
fn collapse_outranks_vacuity_in_the_reason() {
    // A poisoned AND cache-less round must fail on the poisoning signature,
    // the more specific finding.
    let replays = vec![record(
        1,
        RoundVerdict::Collapsed {
            turns: vec![TurnDelta {
                turn: 1,
                ref_tokens: 200,
                replay_tokens: 3,
                ref_finish: Some("stop".into()),
                replay_finish: Some("stop".into()),
            }],
        },
        Some(0),
    )];
    let v = verdict(&score(&replays), 1);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.contains("COLLAPSED"), "{}", v.reason);
}

#[test]
fn unmeasured_rounds_carry_their_number_and_reason() {
    // The report reads unmeasured attribution from here; losing the round
    // number was how the table misattributed transport failures.
    let replays = vec![inv(1), unm(2), inv(3)];
    let s = score(&replays);
    assert_eq!(s.unmeasured_rounds.len(), 1);
    assert_eq!(s.unmeasured_rounds[0].0, 2);
    assert_eq!(s.unmeasured_rounds[0].1, "reset");
    // A never-completed turn 1 (None) is not a vacuous round: it is already
    // an unmeasured failure, and claiming zero cache for it would be data
    // the server never attested.
    assert!(s.vacuous_rounds.is_empty());
    assert_eq!(s.min_turn1_cached, Some(992));
}
