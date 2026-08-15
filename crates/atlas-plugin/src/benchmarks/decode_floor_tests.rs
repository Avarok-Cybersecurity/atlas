// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the decode-floor pins. `evaluate` is pure, so every verdict path
//! is provable without an endpoint — which is the point: the vacuity pins are
//! the gate's honesty, and each one gets a test that fails if it is removed.

use super::*;

/// A healthy run at the measured basis: full-ish budget, server rate present,
/// accept depth ~2 (code-prompt regime).
fn healthy(tps: f64) -> RunObs {
    RunObs {
        completion_tokens: 1450,
        server_tps: Some(tps),
        accepted_prediction_tokens: Some(700),
        e2e_ms: 50_000.0,
    }
}

// ── Path A: the success path ────────────────────────────────────────────────

#[test]
fn three_healthy_runs_measure_the_median() {
    let samples = [healthy(31.5), healthy(29.6), healthy(30.5)];
    match evaluate(&samples) {
        Evaluation::Measured {
            median_decode_tok_s,
            min_output_tokens,
            accept_len_mean,
        } => {
            // ★ 30.5 — the MIDDLE run. stats::percentile(_, 50) would have
            // returned 31.5 (nearest-rank p50 of n=3 is the max), silently
            // reporting the best run as the floor's evidence.
            assert_eq!(median_decode_tok_s, 30.5);
            assert_eq!(min_output_tokens, 1450);
            // 1450 / (1450 - 700) = 1.9333…
            assert!((accept_len_mean - 1450.0 / 750.0).abs() < 1e-9);
        }
        other => panic!("expected Measured, got {other:?}"),
    }
}

#[test]
fn min_output_tokens_is_the_worst_run_not_the_mean() {
    let mut samples = [healthy(30.0), healthy(30.0), healthy(30.0)];
    samples[1].completion_tokens = 1200; // exactly at the floor: still valid
    match evaluate(&samples) {
        Evaluation::Measured {
            min_output_tokens, ..
        } => assert_eq!(min_output_tokens, 1200),
        other => panic!("expected Measured, got {other:?}"),
    }
}

// ── Path B: the boundaries where the bugs live ──────────────────────────────

#[test]
fn output_floor_is_inclusive_and_one_below_is_inconclusive() {
    let mut samples = [healthy(30.0), healthy(30.0), healthy(30.0)];
    samples[2].completion_tokens = MIN_OUTPUT_TOKENS - 1; // 1199
    match evaluate(&samples) {
        Evaluation::Inconclusive(why) => {
            assert!(why.contains("run 3"), "{why}");
            assert!(why.contains("1199"), "{why}");
        }
        other => panic!("a 1199-token run must be inconclusive, got {other:?}"),
    }
}

#[test]
fn accept_len_floor_is_inclusive() {
    // completion 1500, accepted 500 → 1500/1000 = exactly 1.5. `>=` passes.
    let run = RunObs {
        completion_tokens: 1500,
        server_tps: Some(25.0),
        accepted_prediction_tokens: Some(500),
        e2e_ms: 60_000.0,
    };
    let samples = [run.clone(), run.clone(), run];
    match evaluate(&samples) {
        Evaluation::Measured {
            accept_len_mean, ..
        } => assert!((accept_len_mean - 1.5).abs() < 1e-9),
        other => panic!("accept_len exactly 1.5 must measure, got {other:?}"),
    }
}

#[test]
fn a_disengaged_speculation_run_is_inconclusive_not_a_floor() {
    // accepted 100 of 1400 → 1400/1300 ≈ 1.077: speculation nominally on but
    // not at gate depth. This is the serial-floor trap (thinking-on, prompt
    // regression) and must never be recorded as the decode floor.
    let run = RunObs {
        completion_tokens: 1400,
        server_tps: Some(15.0),
        accepted_prediction_tokens: Some(100),
        e2e_ms: 90_000.0,
    };
    let samples = [run.clone(), run.clone(), run];
    match evaluate(&samples) {
        Evaluation::Inconclusive(why) => {
            assert!(why.contains("not"), "{why}");
            assert!(why.contains("serial floor"), "{why}");
        }
        other => panic!("expected Inconclusive, got {other:?}"),
    }
}

#[test]
fn corrupt_accounting_is_inconclusive() {
    let mut samples = [healthy(30.0), healthy(30.0), healthy(30.0)];
    samples[0].accepted_prediction_tokens = Some(1450); // == completion_tokens
    match evaluate(&samples) {
        Evaluation::Inconclusive(why) => assert!(why.contains("corrupt"), "{why}"),
        other => panic!("expected Inconclusive, got {other:?}"),
    }
}

#[test]
fn fewer_than_the_pinned_runs_cannot_measure() {
    let samples = [healthy(30.0), healthy(30.0)];
    match evaluate(&samples) {
        Evaluation::Inconclusive(why) => assert!(why.contains("pinned count is 3"), "{why}"),
        other => panic!("expected Inconclusive, got {other:?}"),
    }
}

// ── Path C: the dependency on the accept-stats instrumentation ──────────────

#[test]
fn an_absent_accept_field_names_the_instrumentation_dependency() {
    let mut samples = [healthy(30.0), healthy(30.0), healthy(30.0)];
    samples[1].accepted_prediction_tokens = None;
    match evaluate(&samples) {
        Evaluation::Inconclusive(why) => {
            assert!(why.contains("accepted_prediction_tokens"), "{why}");
            assert!(why.contains("accept-stats instrumentation"), "{why}");
        }
        other => panic!("expected Inconclusive, got {other:?}"),
    }
}

#[test]
fn a_zero_accept_count_is_inconclusive_never_pass() {
    let mut samples = [healthy(30.0), healthy(30.0), healthy(30.0)];
    samples[0].accepted_prediction_tokens = Some(0);
    match evaluate(&samples) {
        Evaluation::Inconclusive(why) => {
            assert!(why.contains("accepted 0 draft tokens"), "{why}");
        }
        other => panic!("expected Inconclusive, got {other:?}"),
    }
}

#[test]
fn a_missing_server_rate_is_inconclusive() {
    let mut samples = [healthy(30.0), healthy(30.0), healthy(30.0)];
    samples[2].server_tps = None;
    match evaluate(&samples) {
        Evaluation::Inconclusive(why) => {
            assert!(why.contains("response_token/s"), "{why}");
        }
        other => panic!("expected Inconclusive, got {other:?}"),
    }
}

// ── The pinned request and the plumbing around it ───────────────────────────

#[test]
fn the_pins_are_the_documented_fingerprint() {
    assert_eq!(RUNS, 3);
    assert_eq!(MAX_TOKENS, 1500);
    assert_eq!(MIN_OUTPUT_TOKENS, 1200);
    assert_eq!(MIN_ACCEPT_LEN, 1.5);
    assert!(MINHEAP_PROMPT.contains("MinHeap"));
}

#[test]
fn accept_len_derivation_matches_its_definition() {
    let r = RunObs {
        completion_tokens: 1200,
        server_tps: Some(30.0),
        accepted_prediction_tokens: Some(600),
        e2e_ms: 0.0,
    };
    // 1200 tokens over 600 steps = 2.0 tokens per decode step.
    assert_eq!(r.accept_len(), Some(2.0));
    let none = RunObs {
        accepted_prediction_tokens: None,
        ..r.clone()
    };
    assert_eq!(none.accept_len(), None);
}

#[test]
fn the_descriptor_is_registered_and_defaults_configure() {
    assert_eq!(
        crate::registry::find("decode-floor")
            .expect("registered")
            .name,
        "Decode Floor Gate"
    );
    let mut b = DecodeFloor::default();
    let v = ParamValues::defaults(&b.parameters());
    b.configure(&v).expect("defaults configure");
    assert_eq!(b.timeout, Duration::from_secs(300));
}

#[test]
fn reconfiguring_clears_collected_samples() {
    let mut b = DecodeFloor::default();
    let v = ParamValues::defaults(&b.parameters());
    b.configure(&v).unwrap();
    b.samples.push(healthy(30.0));
    b.probed = true;
    b.configure(&v).unwrap();
    assert!(b.samples.is_empty());
    assert!(!b.probed);
}
