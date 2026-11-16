// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: unit tests of the copy-draft estimators: Cinference's
//! decayed per-window estimate (prior, decay, clamp, long-match floor and
//! limit, the 0.96 lookup threshold) and the KVMem yield gate (cold start,
//! the advantage test, the refusal streak and its probe, per-request reset).

use super::estimator::*;

const EPS: f64 = 1e-12;

#[test]
fn the_prior_is_one_hit_in_four() {
    let e = WindowEstimates::new();
    for w in 0..4 {
        assert!((e.probability(w, 0) - 0.25).abs() < EPS);
        assert!(!e.confident(w, 0));
    }
}

#[test]
fn observe_decays_old_evidence_and_counts_a_miss_only_on_a_break() {
    let mut e = WindowEstimates::new();
    // Full match of a 4-token proposal: 4 hits, no miss.
    e.observe(1, &[1, 2, 3, 4], &[1, 2, 3, 4, 9]);
    let p1 = (4.0 + PRIOR_HITS) / (4.0 + PRIOR_HITS + PRIOR_MISSES);
    assert!((e.probability(1, 0) - p1).abs() < EPS);
    // Broke after 1 of 4: hits 0.9*4 + 1, misses 0.9*0 + 1.
    e.observe(1, &[1, 2, 3, 4], &[1, 7]);
    let (h, m) = (DECAY * 4.0 + 1.0, 1.0);
    let p2 = (h + PRIOR_HITS) / (h + m + PRIOR_HITS + PRIOR_MISSES);
    assert!((e.probability(1, 0) - p2).abs() < EPS);
    // A committed stream shorter than the proposal that agrees is no miss.
    e.observe(2, &[5, 6, 7], &[5]);
    let p3 = (1.0 + PRIOR_HITS) / (1.0 + PRIOR_HITS + PRIOR_MISSES);
    assert!((e.probability(2, 0) - p3).abs() < EPS);
    // Windows are independent.
    assert!((e.probability(0, 0) - 0.25).abs() < EPS);
}

#[test]
fn the_estimate_is_clamped() {
    let mut e = WindowEstimates::new();
    for _ in 0..200 {
        e.observe(0, &[1; 15], &[1; 15]);
    }
    assert!((e.probability(0, 0) - P_MAX).abs() < EPS);
    let mut f = WindowEstimates::new();
    for _ in 0..200 {
        f.observe(0, &[1; 15], &[2; 15]);
    }
    // Hits 0, misses -> 10: 1 / (1 + 10 + 3) is above the floor; the floor
    // itself is reached only by the clamp.
    assert!(f.probability(0, 0) >= P_MIN);
    assert!(f.probability(0, 0) < 0.1);
}

#[test]
fn a_long_match_is_trusted_from_its_first_round() {
    let e = WindowEstimates::new();
    // Up to the floor a match adds nothing.
    assert!((e.probability(0, MATCH_FLOOR) - 0.25).abs() < EPS);
    // 128 matched tokens: 120 prior hits -> (1 + 120) / (4 + 120) = 0.9758,
    // clamped to 0.97: above the 0.96 lookup threshold.
    let p = e.probability(0, 128);
    assert!((p - P_MAX).abs() < EPS);
    assert!(e.confident(0, 128));
    // 80 matched: (1 + 72) / (4 + 72) = 0.9605: just confident.
    assert!(e.confident(0, 80));
    // 70 matched: (1 + 62) / (4 + 62) = 0.9545: not confident.
    assert!(!e.confident(0, 70));
}

#[test]
fn the_gate_takes_its_first_two_rounds_unconditionally() {
    let mut g = YieldGate::new();
    assert_eq!(g.decide(3, 1.3), GateVerdict::Take);
    assert_eq!(g.decide(3, 1.3), GateVerdict::Take);
    // No copy observed yet: expected copy yield 0 < 1.3 * prior.
    assert_eq!(g.decide(3, 1.3), GateVerdict::Skip);
}

#[test]
fn the_gate_compares_copy_yield_with_mtp_by_the_advantage() {
    let mut g = YieldGate::new();
    g.decide(3, 1.3);
    g.decide(3, 1.3);
    g.observe_copy(3, 3); // copy_avg 1.0
    g.observe_mtp(2); // mtp_avg 2.0
    // 3 * 1.0 = 3.0 > 1.3 * 2.0 = 2.6: take.
    assert_eq!(g.decide(3, 1.3), GateVerdict::Take);
    // Width 2: 2.0 < 2.6: skip.
    assert_eq!(g.decide(2, 1.3), GateVerdict::Skip);
    assert_eq!(g.skip_streak(), 1);
    // A taken round resets the streak.
    assert_eq!(g.decide(3, 1.3), GateVerdict::Take);
    assert_eq!(g.skip_streak(), 0);
}

#[test]
fn averages_are_ewma_with_alpha_a_quarter() {
    let mut g = YieldGate::new();
    g.observe_copy(1, 2);
    assert_eq!(g.copy_avg(), Some(0.5));
    g.observe_copy(2, 2);
    assert_eq!(g.copy_avg(), Some(0.5 + GATE_ALPHA * 0.5));
    g.observe_copy(5, 0); // nothing drafted: ignored
    assert_eq!(g.copy_avg(), Some(0.625));
    g.observe_mtp(3);
    g.observe_mtp(1);
    assert_eq!(g.mtp_avg(), Some(3.0 + GATE_ALPHA * (1.0 - 3.0)));
}

#[test]
fn eight_refusals_earn_one_probe() {
    let mut g = YieldGate::new();
    g.decide(3, 1.3);
    g.decide(3, 1.3);
    g.observe_copy(0, 3);
    g.observe_mtp(3);
    for i in 0..GATE_PROBE_AFTER {
        assert_eq!(g.decide(3, 1.3), GateVerdict::Skip, "refusal {i}");
    }
    assert_eq!(g.decide(3, 1.3), GateVerdict::Probe);
    assert!(GateVerdict::Probe.takes());
    // The streak restarts after the probe.
    assert_eq!(g.decide(3, 1.3), GateVerdict::Skip);
    assert_eq!(g.skip_streak(), 1);
}

#[test]
fn a_new_request_starts_cold() {
    let mut g = YieldGate::new();
    for _ in 0..20 {
        g.decide(3, 1.3);
        g.observe_copy(0, 3);
    }
    // Per-sequence state: a new request gets a default gate.
    let fresh = YieldGate::default();
    assert_eq!(fresh.copy_avg(), None);
    assert_eq!(fresh.mtp_avg(), None);
    assert_eq!(fresh.skip_streak(), 0);
    let mut fresh = fresh;
    assert_eq!(fresh.decide(3, 1.3), GateVerdict::Take);
}
