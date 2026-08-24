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
    assert_eq!(v.reason, "12 of 12 replays byte-identical to the reference");
}

#[test]
fn jitter_is_recorded_but_passes() {
    // The clean-main reality: every replay jitters a little (restore anchor
    // selection varies). That is a healthy engine, not a failure.
    let replays: Vec<_> = (1..=12).map(jit).collect();
    let s = score(&replays);
    assert_eq!(s.rounds, 12);
    assert_eq!(s.invariant, 0);
    assert_eq!(s.jittered, 12);
    assert_eq!(s.collapsed, 0);
    assert_eq!(s.unmeasured, 0);
    assert_eq!(s.jittered_rounds.len(), 12);
    assert!(s.collapsed_rounds.is_empty());
    assert!(s.unmeasured_rounds.is_empty());
    assert!(s.vacuous_rounds.is_empty());
    assert_eq!(s.min_turn1_cached, WARM);
    let v = verdict(&s, 12);
    assert_eq!(v.kind, VerdictKind::Pass);
    assert_eq!(
        v.reason,
        "0 of 12 replays byte-identical, 12 jittered within bounds (restore anchor selection varies between rounds on a healthy engine), 0 collapsed"
    );
}

#[test]
fn a_single_collapse_is_fail_and_names_the_round() {
    // The batch4 shape: most rounds fine, one round collapses.
    let mut replays: Vec<_> = (1..=7).map(inv).collect();
    replays.push(col(8));
    replays.push(inv(9));
    replays.push(jit(10));
    let s = score(&replays);
    assert_eq!(
        (s.rounds, s.invariant, s.jittered, s.collapsed, s.unmeasured),
        (10, 8, 1, 1, 0)
    );
    assert_eq!(
        s.collapsed_rounds,
        [(
            8,
            vec![TurnDelta {
                turn: 2,
                ref_tokens: 200,
                replay_tokens: 3,
                ref_finish: Some("stop".into()),
                replay_finish: Some("stop".into()),
            }],
        )]
    );
    let v = verdict(&s, 10);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert_eq!(
        v.reason,
        "1 of 10 replays COLLAPSED against the reference: round 8: turn 2 (200 -> 3 tokens, finish Some(\"stop\") -> Some(\"stop\")) — a restored prefix produced degenerate output (early-EOS or runaway), the SSM state poisoning signature"
    );
}

#[test]
fn an_unmeasured_round_fails_the_gate() {
    let replays = vec![
        inv(1),
        unm(2),
        inv(3),
        record(
            4,
            RoundVerdict::Unmeasured {
                reason: "timeout".into(),
            },
            None,
        ),
    ];
    let v = verdict(&score(&replays), 4);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert_eq!(
        v.reason,
        "2 of 4 replays were unmeasurable (transport errors): round 2: reset | round 4: timeout — the replay invariant is unproven for those rounds"
    );
}

#[test]
fn a_short_run_cannot_pass_by_running_fewer_replays() {
    let replays = vec![inv(1), inv(2)];
    let v = verdict(&score(&replays), 12);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert_eq!(v.reason, "2 of 12 replay rounds completed");
}

#[test]
fn collapse_wins_over_jitter_in_the_reason() {
    // A run with both a collapse and jitter must fail on the collapse, not
    // pass because jitter is tolerated.
    let replays = vec![jit(1), col(2)];
    let v = verdict(&score(&replays), 2);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert_eq!(
        v.reason,
        "1 of 2 replays COLLAPSED against the reference: round 2: turn 2 (200 -> 3 tokens, finish Some(\"stop\") -> Some(\"stop\")) — a restored prefix produced degenerate output (early-EOS or runaway), the SSM state poisoning signature"
    );
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
    assert_eq!(
        v.reason,
        "replay round(s) [3] attested 0 cached prompt tokens on turn 1 — the prefix restore path this gate polices was never exercised, so their transcripts prove nothing about the poisoning class (is prefix caching enabled on the served recipe?)"
    );
}

#[test]
fn an_all_cold_run_fails_on_every_round() {
    // Prefix caching disabled: all rounds attest zero. This is the exact
    // configuration under which the pre-fix gate passed.
    let replays: Vec<_> = (1..=12)
        .map(|n| record(n, RoundVerdict::Invariant, Some(0)))
        .collect();
    let s = score(&replays);
    assert_eq!(s.vacuous_rounds, (1..=12).collect::<Vec<_>>());
    assert_eq!(s.min_turn1_cached, Some(0));
    let v = verdict(&s, 12);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert_eq!(
        v.reason,
        "replay round(s) [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] attested 0 cached prompt tokens on turn 1 — the prefix restore path this gate polices was never exercised, so their transcripts prove nothing about the poisoning class (is prefix caching enabled on the served recipe?)"
    );
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
    assert_eq!(
        v.reason,
        "1 of 1 replays COLLAPSED against the reference: round 1: turn 1 (200 -> 3 tokens, finish Some(\"stop\") -> Some(\"stop\")) — a restored prefix produced degenerate output (early-EOS or runaway), the SSM state poisoning signature"
    );
}

#[test]
fn unmeasured_rounds_carry_their_number_and_reason() {
    // The report reads unmeasured attribution from here; losing the round
    // number was how the table misattributed transport failures.
    let replays = vec![inv(1), unm(2), inv(3)];
    let s = score(&replays);
    assert_eq!(s.unmeasured_rounds, [(2, "reset".into())]);
    // A never-completed turn 1 (None) is not a vacuous round: it is already
    // an unmeasured failure, and claiming zero cache for it would be data
    // the server never attested.
    assert!(s.vacuous_rounds.is_empty());
    assert_eq!(s.min_turn1_cached, Some(992));
}
