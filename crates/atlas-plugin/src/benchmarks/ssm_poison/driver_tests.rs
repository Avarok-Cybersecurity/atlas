// SPDX-License-Identifier: AGPL-3.0-only

//! The driver's non-network surface: descriptor wiring and parameter
//! validation. The decision logic itself is covered by `score_tests`; here we
//! pin the registration contract and the configure-time guards.

use super::{DEFAULT_ROUNDS, DESCRIPTOR, RoundRecord, SsmPoison};
use crate::benchmark::Benchmark;
use crate::params::{ParamValue, ParamValues};
use crate::result::VerdictKind;

fn configured() -> SsmPoison {
    let mut b = SsmPoison::default();
    let v = ParamValues::defaults(&b.parameters());
    b.configure(&v).unwrap();
    b
}

#[test]
fn descriptor_id_is_stable_and_filename_safe() {
    assert_eq!(DESCRIPTOR.id, "ssm-state-poisoning-gate");
    assert!(
        DESCRIPTOR
            .id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-')
    );
    assert!(!DESCRIPTOR.detail.is_empty());
    assert!(!DESCRIPTOR.summary.is_empty());
}

#[test]
fn defaults_validate_and_pin_twelve_rounds() {
    let b = configured();
    assert_eq!(b.rounds, 12);
    assert_eq!(DEFAULT_ROUNDS, 12);
    // 1024, not the old 256: the runaway ceiling (2.0x the reference) must
    // be reachable before the budget clamps the replay. At 256 any turn past
    // 128 reference tokens could never ratio out to a collapse.
    assert_eq!(b.max_tokens, 1024);
}

#[test]
fn rounds_below_three_are_rejected_at_configure() {
    let mut b = SsmPoison::default();
    let specs = b.parameters();
    let mut v = ParamValues::defaults(&specs);
    v.0.insert("rounds".to_string(), ParamValue::Int(2));
    // rounds min is 3, so validate_against rejects before configure body runs.
    assert!(b.configure(&v).is_err());
}

#[test]
fn scored_fails_on_collapse_via_the_driver_seam() {
    // Exercise scored() through the driver's own replays field: build a
    // poisoned shape (early-EOS collapse) and confirm the verdict fails.
    let mut b = configured();
    b.replays = vec![
        RoundRecord {
            round: 1,
            verdict: super::compare::RoundVerdict::Invariant,
            turn1_cached: Some(992),
        },
        RoundRecord {
            round: 2,
            verdict: super::compare::RoundVerdict::Collapsed {
                turns: vec![super::compare::TurnDelta {
                    turn: 2,
                    ref_tokens: 200,
                    replay_tokens: 3,
                    ref_finish: Some("stop".into()),
                    replay_finish: Some("stop".into()),
                }],
            },
            turn1_cached: Some(992),
        },
    ];
    b.rounds = 2; // match the number collected so only the collapse fails
    let (s, v) = b.scored();
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.contains("round 2"));
    assert_eq!(s.collapsed, 1);
}

#[test]
fn scored_passes_on_jitter_via_the_driver_seam() {
    // Jitter (healthy restore-anchor variance) must not fail the gate.
    let mut b = configured();
    b.replays = vec![
        RoundRecord {
            round: 1,
            verdict: super::compare::RoundVerdict::Invariant,
            turn1_cached: Some(992),
        },
        RoundRecord {
            round: 2,
            verdict: super::compare::RoundVerdict::Jittered {
                turns: vec![super::compare::TurnDelta {
                    turn: 2,
                    ref_tokens: 200,
                    replay_tokens: 206,
                    ref_finish: Some("stop".into()),
                    replay_finish: Some("stop".into()),
                }],
            },
            turn1_cached: Some(992),
        },
    ];
    b.rounds = 2;
    let (s, v) = b.scored();
    assert_eq!(v.kind, VerdictKind::Pass);
    assert_eq!(s.jittered, 1);
    assert_eq!(s.collapsed, 0);
}

#[test]
fn scored_fails_on_a_cold_replay_via_the_driver_seam() {
    // The vacuity finding, exercised through the driver's own state: two
    // byte-identical replays, one of which attests zero cached tokens on
    // turn 1. Before the fix this was a green PASS with caching off.
    let mut b = configured();
    b.replays = vec![
        RoundRecord {
            round: 1,
            verdict: super::compare::RoundVerdict::Invariant,
            turn1_cached: Some(992),
        },
        RoundRecord {
            round: 2,
            verdict: super::compare::RoundVerdict::Invariant,
            turn1_cached: Some(0),
        },
    ];
    b.rounds = 2;
    let (s, v) = b.scored();
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.contains("0 cached prompt tokens"), "{}", v.reason);
    assert_eq!(s.vacuous_rounds, vec![2]);
}
