// SPDX-License-Identifier: AGPL-3.0-only

//! The fidelity metric.
//!
//! This is the instrument every expert-pruning decision will be judged with,
//! so its failure modes matter more than its happy path. The two that would
//! do real damage: reporting 0 divergence for a model that diverged, and
//! averaging over positions that are not the same positions.

use super::*;

fn reference(lp: &[f32], argmax: &[&str]) -> Reference {
    Reference {
        id: "py-001".into(),
        category: "code-python".into(),
        prompt: "p".into(),
        continuation: "c".into(),
        logprobs: lp.to_vec(),
        argmax: argmax.iter().map(|s| s.to_string()).collect(),
    }
}

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

// ---------------------------------------------------------------- Path A

#[test]
fn an_identical_model_scores_zero_divergence() {
    // The control. A restricted serve that changed nothing must read exactly
    // 0.0 nats and 1.0 agreement — if the floor drifted, every comparison
    // built on this metric would inherit the offset.
    let r = reference(&[-0.5, -1.5, -0.25], &["def", " reverse", "("]);
    let s = score_one(&r, &[-0.5, -1.5, -0.25], &strs(&["def", " reverse", "("])).unwrap();
    assert_eq!(s.delta_ce, 0.0);
    assert_eq!(s.top1_agreement, 1.0);
    assert_eq!(s.positions, 3);
}

#[test]
fn extra_surprise_is_reported_in_nats_per_token() {
    // The restricted model finds each token 1 nat less likely.
    let r = reference(&[-1.0, -2.0], &["a", "b"]);
    let s = score_one(&r, &[-2.0, -3.0], &strs(&["a", "b"])).unwrap();
    assert!((s.delta_ce - 1.0).abs() < 1e-9);
    assert_eq!(
        s.top1_agreement, 1.0,
        "surprise rose without changing argmax"
    );
}

#[test]
fn a_more_confident_restricted_model_scores_negative() {
    // Not clamped: pruning CAN make a continuation more likely, and hiding
    // that behind a floor of 0 would misreport the direction of the effect.
    let r = reference(&[-2.0], &["a"]);
    let s = score_one(&r, &[-1.0], &strs(&["a"])).unwrap();
    assert!((s.delta_ce + 1.0).abs() < 1e-9, "got {}", s.delta_ce);
}

// ---------------------------------------------------------------- Path B

#[test]
fn agreement_and_surprise_move_independently() {
    // The case that justifies reporting both. Every argmax still matches, so
    // greedy output is unchanged and byte-identity would call this perfect —
    // but the model is far less confident, which predicts fragility the
    // moment anyone samples.
    let r = reference(&[-0.1, -0.1, -0.1], &["a", "b", "c"]);
    let s = score_one(&r, &[-2.1, -2.1, -2.1], &strs(&["a", "b", "c"])).unwrap();
    assert_eq!(s.top1_agreement, 1.0);
    // 1e-6, not 1e-9: log-probabilities arrive as f32 from the wire, and
    // neither -0.1 nor -2.1 is exactly representable there, so their
    // difference lands ~1e-7 from 2.0 before this function sees it. The
    // tolerance has to match the precision of the input, not of f64.
    assert!((s.delta_ce - 2.0).abs() < 1e-6, "got {}", s.delta_ce);
}

#[test]
fn a_mismatched_position_count_is_refused_not_averaged() {
    // Different tokenization of the same text. Averaging across misaligned
    // positions produces a plausible number that means nothing, so this
    // returns None and the caller drops the prompt loudly.
    let r = reference(&[-1.0, -1.0, -1.0], &["a", "b", "c"]);
    assert!(score_one(&r, &[-1.0, -1.0], &strs(&["a", "b"])).is_none());
}

#[test]
fn an_empty_continuation_is_refused() {
    let r = reference(&[], &[]);
    assert!(score_one(&r, &[], &[]).is_none());
}

#[test]
fn aggregation_weights_by_position_not_by_prompt() {
    // A 100-token continuation carries more evidence than a 1-token one.
    // Prompt-weighting would let a trivial prompt outvote a long one.
    let long = Scored {
        id: "a".into(),
        category: "c".into(),
        delta_ce: 0.0,
        top1_agreement: 1.0,
        positions: 99,
    };
    let short = Scored {
        id: "b".into(),
        category: "c".into(),
        delta_ce: 10.0,
        top1_agreement: 0.0,
        positions: 1,
    };
    let f = aggregate(&[long, short]).unwrap();
    assert!(
        (f.delta_ce - 0.1).abs() < 1e-9,
        "position-weighted mean is 10/100 = 0.1, got {}",
        f.delta_ce
    );
    assert_eq!(f.positions, 100);
}

#[test]
fn per_category_splits_the_serve_it_was_built_for_from_the_rest() {
    // A category-restricted serve must be judged on BOTH: the category it
    // holds experts for, and the traffic it does not. One aggregate number
    // hides exactly the trade being made.
    let mk = |cat: &str, ce: f64| Scored {
        id: "x".into(),
        category: cat.into(),
        delta_ce: ce,
        top1_agreement: 1.0,
        positions: 10,
    };
    let f = aggregate(&[mk("code-python", 0.01), mk("translation", 3.0)]).unwrap();
    assert!((f.per_category["code-python"].0 - 0.01).abs() < 1e-9);
    assert!((f.per_category["translation"].0 - 3.0).abs() < 1e-9);
    assert_eq!(f.worst[0].category, "translation", "worst is listed first");
}

// ---------------------------------------------------------------- Path C

#[test]
fn a_response_without_logprobs_names_the_flags_that_produce_them() {
    // The likeliest operator error, and the message has to carry the fix.
    let v = serde_json::json!({"choices": [{"text": "hi"}]});
    let err = extract(&v, 0).unwrap_err().to_string();
    assert!(err.contains("echo=true"), "got: {err}");
    assert!(err.contains("logprobs"), "got: {err}");
}

#[test]
fn only_continuation_positions_are_scored() {
    // The echoed prompt is context, not the thing under test. Including it
    // would dilute the measurement with positions every configuration agrees
    // on, making a broken serve look close to the control.
    let v = serde_json::json!({"choices": [{"logprobs": {
        "text_offset": [0, 5, 10, 15],
        "token_logprobs": [null, -1.0, -2.0, -3.0],
        "top_logprobs": [null, {"a": -1.0}, {"b": -2.0}, {"c": -3.0}]
    }}]});
    // Prompt occupies bytes 0..10, so only offsets 10 and 15 are scored.
    let (lp, am) = extract(&v, 10).unwrap();
    assert_eq!(lp, vec![-2.0, -3.0]);
    assert_eq!(am, strs(&["b", "c"]));
}

#[test]
fn the_first_position_has_no_predecessor_and_is_skipped() {
    // token_logprobs[0] is null: nothing precedes it, so there is no
    // conditional to score. Reading it as 0.0 would silently credit every
    // configuration with one free perfect position.
    let v = serde_json::json!({"choices": [{"logprobs": {
        "text_offset": [0, 3],
        "token_logprobs": [null, -1.5],
        "top_logprobs": [null, {"x": -1.5}]
    }}]});
    let (lp, _) = extract(&v, 0).unwrap();
    assert_eq!(lp, vec![-1.5]);
}
