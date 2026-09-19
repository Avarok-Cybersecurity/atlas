// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the decode-floor pins. `evaluate` is pure, so every verdict path
//! is provable without an endpoint — which is the point: the vacuity pins are
//! the gate's honesty, and each one gets a test that fails if it is removed.

use super::score::*;
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

#[test]
fn run_observation_preserves_wire_evidence() {
    let outcome = crate::http::ChatOutcome {
        completion_tokens: 941,
        server_tps: Some(28.125),
        accepted_prediction_tokens: Some(417),
        e2e_ms: 33_456.75,
        ..Default::default()
    };

    assert_eq!(
        RunObs::from_outcome(&outcome),
        RunObs {
            completion_tokens: 941,
            server_tps: Some(28.125),
            accepted_prediction_tokens: Some(417),
            e2e_ms: 33_456.75,
        }
    );
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
    samples[1].completion_tokens = MIN_OUTPUT_TOKENS; // exactly at the floor: still valid
    match evaluate(&samples) {
        Evaluation::Measured {
            min_output_tokens, ..
        } => assert_eq!(min_output_tokens, MIN_OUTPUT_TOKENS),
        other => panic!("expected Measured, got {other:?}"),
    }
}

// ── Path B: the boundaries where the bugs live ──────────────────────────────

#[test]
fn one_below_the_output_floor_is_inconclusive() {
    let mut samples = [healthy(30.0), healthy(30.0), healthy(30.0)];
    samples[2].completion_tokens = MIN_OUTPUT_TOKENS - 1; // 749
    match evaluate(&samples) {
        Evaluation::Inconclusive(why) => {
            assert!(why.contains("run 3"), "{why}");
            assert!(why.contains(&(MIN_OUTPUT_TOKENS - 1).to_string()), "{why}");
        }
        other => panic!("a one-below-the-floor run must be inconclusive, got {other:?}"),
    }
}

/// ★ The calibrated instrument itself must MEASURE, never INCONCLUSIVE: the
/// 12-run promotion basis completes at a deterministic 915 tokens (natural
/// stop of the MinHeap task at temp 0 / seed 0), and the pre-calibration
/// 1200 floor failed exactly that. A vacuity pin that deterministically
/// rejects the gate's own reference behaviour gates nothing.
#[test]
fn the_calibration_instruments_915_token_stop_is_a_measurement() {
    let run = RunObs {
        completion_tokens: 915,
        server_tps: Some(28.0),
        accepted_prediction_tokens: Some(569),
        e2e_ms: 33_000.0,
    };
    let samples = [run.clone(), run.clone(), run];
    match evaluate(&samples) {
        Evaluation::Measured {
            median_decode_tok_s,
            min_output_tokens,
            ..
        } => {
            assert_eq!(median_decode_tok_s, 28.0);
            assert_eq!(min_output_tokens, 915);
        }
        other => panic!("the calibration fingerprint must measure, got {other:?}"),
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
fn a_disengaged_speculation_mean_is_inconclusive_not_a_floor() {
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

#[test]
fn a_nonpositive_or_nonfinite_server_rate_is_inconclusive() {
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut samples = [healthy(30.0), healthy(30.0), healthy(30.0)];
        samples[2].server_tps = Some(invalid);
        match evaluate(&samples) {
            Evaluation::Inconclusive(why) => {
                assert!(why.contains("run 3"), "{why}");
                assert!(why.contains("positive"), "{why}");
            }
            other => panic!("server rate {invalid} must be inconclusive, got {other:?}"),
        }
    }
}

// ── The run verdict vs the baseline floor (review C1) ───────────────────────

fn measured(median: f64) -> Evaluation {
    Evaluation::Measured {
        median_decode_tok_s: median,
        min_output_tokens: 1450,
        accept_len_mean: 1.93,
    }
}

/// With the gate-filled floor set, a Measured run self-verdicts PASS/FAIL —
/// the PASS the gate machinery requires the day this gate is promoted to
/// REQUIRED. The comparison is the raw median >= min, deliberately stricter
/// than gate scoring's value + noise >= min (see `verdict_for`).
#[test]
fn a_measured_run_self_verdicts_against_the_floor_param() {
    use crate::result::VerdictKind;
    let v = verdict_for(&measured(30.5), 29.0);
    assert_eq!(v.kind, VerdictKind::Pass, "{}", v.reason);
    assert!(
        v.reason.contains("30.5") && v.reason.contains("29.0"),
        "{}",
        v.reason
    );

    let v = verdict_for(&measured(28.5), 29.0);
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("BELOW THE DECODE FLOOR"), "{}", v.reason);
    assert!(
        v.reason.contains("28.5") && v.reason.contains("29.0"),
        "{}",
        v.reason
    );

    // Exactly on the floor passes — inclusive, like the BENCH.toml bound.
    assert_eq!(verdict_for(&measured(29.0), 29.0).kind, VerdictKind::Pass);
}

/// Floor 0 (the schema default) keeps today's info verdict: a standalone run
/// has no committed floor to be judged against.
#[test]
fn no_floor_param_keeps_the_info_verdict() {
    use crate::result::VerdictKind;
    let v = verdict_for(&measured(30.5), 0.0);
    assert_eq!(v.kind, VerdictKind::Info, "{}", v.reason);
    assert!(v.reason.contains("--pull-request-gate"), "{}", v.reason);
}

/// Vacuity stays INCONCLUSIVE (a failing verdict) no matter what the floor
/// param says — a run that measured nothing must never PASS, even one whose
/// pins failed on a healthy-looking median.
#[test]
fn vacuous_runs_stay_inconclusive_regardless_of_the_floor_param() {
    use crate::result::VerdictKind;
    let eval = Evaluation::Inconclusive("accept_len_mean 1.10 < 1.5".to_string());
    for floor in [0.0, 29.0] {
        let v = verdict_for(&eval, floor);
        assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
        assert!(v.reason.contains("INCONCLUSIVE"), "{}", v.reason);
    }
}

/// The descriptor couples the floor param to the metric the BENCH.toml bound
/// is written on, and the schema default is the documented OFF state.
#[test]
fn the_floor_param_is_wired_to_the_gate() {
    assert_eq!(
        DESCRIPTOR.threshold_params,
        [("min_tok_s", "server_decode_tok_s")]
    );
    let b = DecodeFloor::default();
    let v = ParamValues::defaults(&b.parameters());
    assert_eq!(v.float("min_tok_s").unwrap(), 0.0);
}

// ── The pinned request and the plumbing around it ───────────────────────────

#[test]
fn the_pins_are_the_documented_fingerprint() {
    assert_eq!(RUNS, 3);
    assert_eq!(MAX_TOKENS, 1500);
    // 750 since the 2026-08-15 promotion calibration: the instrument's
    // deterministic natural stop is 915, and the pin must sit under it.
    assert_eq!(MIN_OUTPUT_TOKENS, 750);
    assert_eq!(MIN_ACCEPT_LEN, 1.5);
    assert_eq!(
        DecodeFloor::request_body("fixture-model"),
        serde_json::json!({
            "model": "fixture-model",
            "stream": true,
            "temperature": 0.0,
            "seed": 0,
            "max_tokens": 1500,
            "reasoning_effort": "none",
            "messages": [{
                "role": "user",
                "content": "Implement a complete, production-quality MinHeap class in Python. Include the methods insert, extract_min, peek, heapify (bottom-up from an arbitrary list), decrease_key, delete_at_index, merge (with another MinHeap), __len__ and __iter__. Every method needs a full docstring with time-complexity analysis. Then write a comprehensive pytest test suite covering the empty heap, a single element, duplicate keys, and long interleaved insert/extract sequences. Finish with a line-by-line explanation of the sift_up and sift_down invariants. Be exhaustive and do not stop early."
            }],
        })
    );
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
