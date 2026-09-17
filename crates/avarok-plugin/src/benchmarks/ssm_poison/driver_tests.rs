// SPDX-License-Identifier: AGPL-3.0-only

//! The driver's non-network surface: descriptor wiring and parameter
//! validation. The decision logic itself is covered by `score_tests`; here we
//! pin the registration contract and the configure-time guards.

use super::{DEFAULT_ROUNDS, DESCRIPTOR, Phase, RoundRecord, SsmPoison};
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
    assert_eq!(b.timeout, std::time::Duration::from_secs(300));
    assert_eq!(b.phase, Phase::Baseline);
    assert!(!b.probed);
    assert!(b.reference.is_empty());
    assert!(b.replays.is_empty());
}

#[test]
fn rounds_below_three_are_rejected_at_configure() {
    let mut b = SsmPoison::default();
    let specs = b.parameters();
    let mut v = ParamValues::defaults(&specs);
    v.0.insert("rounds".to_string(), ParamValue::Int(2));
    // rounds min is 3, so validate_against rejects before configure body runs.
    let err = b.configure(&v).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Replay rounds: must be between 3 and 30, got 2"
    );

    v.set("rounds", ParamValue::Int(3));
    b.configure(&v).unwrap();
    assert_eq!(b.rounds, 3);
}

#[test]
fn reconfiguring_restarts_the_probe_and_clears_collected_state() {
    let mut b = configured();
    b.probed = true;
    b.phase = Phase::Done;
    b.reference.push(Default::default());
    b.replays.push(RoundRecord {
        round: 1,
        verdict: super::compare::RoundVerdict::Invariant,
        turn1_cached: Some(992),
    });

    let mut values = ParamValues::defaults(&b.parameters());
    values.set("rounds", ParamValue::Int(3));
    values.set("max_tokens", ParamValue::Int(32));
    values.set("request_timeout_s", ParamValue::Int(10));
    b.configure(&values).unwrap();

    assert_eq!(b.rounds, 3);
    assert_eq!(b.max_tokens, 32);
    assert_eq!(b.timeout, std::time::Duration::from_secs(10));
    assert_eq!(b.phase, Phase::Baseline);
    assert!(!b.probed);
    assert!(b.reference.is_empty());
    assert!(b.replays.is_empty());
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
