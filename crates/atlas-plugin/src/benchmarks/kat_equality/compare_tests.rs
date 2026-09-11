// SPDX-License-Identifier: AGPL-3.0-only

//! The whole verdict is a pure function of what the orders produced, so every
//! failure this gate can report is constructed here without a server.

use super::*;
use crate::benchmarks::transcript::{RequestOutcome, Transcript};
use crate::result::VerdictKind;

fn ok(text: &str, tokens: usize) -> RequestOutcome {
    RequestOutcome::Ok(Box::new(Transcript {
        text: text.to_string(),
        completion_tokens: tokens,
        ..Default::default()
    }))
}

fn run(label: &str, samples: &[(&str, RequestOutcome)]) -> OrderRun {
    OrderRun {
        label: label.to_string(),
        observations: samples
            .iter()
            .map(|(id, o)| Observation {
                sample_id: (*id).to_string(),
                outcome: o.clone(),
            })
            .collect(),
    }
}

#[test]
fn order_zero_is_the_canonical_order_and_order_one_reverses_it() {
    assert_eq!(permutation(0, 4), vec![0, 1, 2, 3]);
    assert_eq!(permutation(1, 4), vec![3, 2, 1, 0]);
}

#[test]
fn every_order_is_a_real_permutation_and_none_degenerates_to_the_first() {
    // A "rotation" that returned the identity would make an extra order cost
    // GPU time and prove nothing — it would compare order 0 against itself.
    for len in [1usize, 2, 5, 17, 995] {
        let identity: Vec<usize> = (0..len).collect();
        for k in 0..6 {
            let p = permutation(k, len);
            assert_eq!(p.len(), len, "order {k} len {len} dropped samples");
            let mut sorted = p.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, identity, "order {k} len {len} is not a permutation");
            if k > 0 && len > 1 {
                assert_ne!(p, identity, "order {k} len {len} degenerated to order 0");
            }
        }
    }
}

#[test]
fn identical_replies_are_equal() {
    let a = run("canonical", &[("s1", ok("hello", 5))]);
    let b = run("reversed", &[("s1", ok("hello", 5))]);
    assert_eq!(verdict_for("s1", &a, &[b]), SampleVerdict::Equal);
}

#[test]
fn a_different_reply_is_a_divergence_that_names_the_order_and_locates_itself() {
    let a = run("canonical", &[("s1", ok("hello world", 5))]);
    let b = run("reversed", &[("s1", ok("hello there", 5))]);
    match verdict_for("s1", &a, &[b]) {
        SampleVerdict::Diverged {
            other_order,
            common_prefix,
        } => {
            assert_eq!(other_order, "reversed");
            // "\u{1}hello " — the reasoning separator plus the shared word.
            assert_eq!(common_prefix, 7, "should localise at the first difference");
        }
        other => panic!("expected a divergence, got {other:?}"),
    }
}

/// THE SUBTLE ONE. Identical text, different token count: the server disagrees
/// with itself about what it emitted, which is a real difference in what ran.
/// `Transcript::canonical()` deliberately excludes the count, so comparing only
/// `canonical()` would call this equal.
#[test]
fn identical_text_with_a_different_token_count_is_still_a_divergence() {
    let a = run("canonical", &[("s1", ok("same", 5))]);
    let b = run("reversed", &[("s1", ok("same", 6))]);
    assert!(
        matches!(verdict_for("s1", &a, &[b]), SampleVerdict::Diverged { .. }),
        "a token-count mismatch under identical text must not read as equal"
    );
}

#[test]
fn a_failed_request_is_unmeasured_never_equal() {
    let a = run("canonical", &[("s1", ok("hello", 5))]);
    let b = run(
        "reversed",
        &[("s1", RequestOutcome::Error("timeout".into()))],
    );
    assert!(matches!(
        verdict_for("s1", &a, &[b]),
        SampleVerdict::Unmeasured(_)
    ));
}

/// Two failures must not cancel out into agreement. `RequestOutcome::Error`
/// exists precisely so this cannot happen.
///
/// There are TWO paths that must refuse it — the reference order's own
/// failure, and a later order's — and they are different branches of
/// `verdict_for`. An earlier version of this test set BOTH to `Error`, which
/// meant it returned through the reference branch and never exercised the
/// other one; a control that neutered the later-order branch left it green.
/// Named and split so each branch has a test that actually reaches it.
#[test]
fn a_failure_in_the_reference_order_is_unmeasured() {
    let a = run(
        "canonical",
        &[("s1", RequestOutcome::Error("timeout".into()))],
    );
    let b = run("reversed", &[("s1", ok("hello", 5))]);
    assert!(matches!(
        verdict_for("s1", &a, &[b]),
        SampleVerdict::Unmeasured(_)
    ));
}

/// And with the reference healthy, so this reaches the LATER-order branch.
#[test]
fn a_failure_in_a_later_order_is_unmeasured_not_skipped() {
    let a = run("canonical", &[("s1", ok("hello", 5))]);
    let b = run(
        "reversed",
        &[("s1", RequestOutcome::Error("timeout".into()))],
    );
    let c = run("rotated", &[("s1", ok("hello", 5))]);
    assert!(
        matches!(verdict_for("s1", &a, &[b, c]), SampleVerdict::Unmeasured(_)),
        "skipping a failed order would let the remaining orders agree and certify"
    );
}

#[test]
fn a_sample_missing_from_a_later_order_is_unmeasured() {
    let a = run("canonical", &[("s1", ok("hello", 5))]);
    let b = run("reversed", &[("s2", ok("hello", 5))]);
    assert!(matches!(
        verdict_for("s1", &a, &[b]),
        SampleVerdict::Unmeasured(_)
    ));
}

#[test]
fn all_equal_passes() {
    let a = run("canonical", &[("s1", ok("x", 1)), ("s2", ok("y", 1))]);
    let b = run("reversed", &[("s2", ok("y", 1)), ("s1", ok("x", 1))]);
    let s = score(&[a, b]);
    assert_eq!((s.equal, s.diverged, s.unmeasured), (2, 0, 0));
    assert_eq!(verdict(&s).kind, VerdictKind::Pass);
}

#[test]
fn one_divergence_fails_and_the_reason_names_the_sample() {
    let a = run("canonical", &[("s1", ok("x", 1)), ("s2", ok("y", 1))]);
    let b = run(
        "reversed",
        &[("s2", ok("y", 1)), ("s1", ok("DIFFERENT", 1))],
    );
    let s = score(&[a, b]);
    assert_eq!(s.diverged, 1);
    let v = verdict(&s);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.contains("s1"), "reason must name it: {}", v.reason);
    assert!(v.reason.contains("ORDER-DEPENDENT"));
}

/// THE GUARD THAT KEEPS A GREEN HONEST. A server answering nothing is
/// byte-identical across every order. Without this the gate certifies a dead
/// endpoint as proof of order-independence.
#[test]
fn a_run_where_every_reply_was_empty_is_vacuous_not_a_pass() {
    let a = run("canonical", &[("s1", ok("", 0)), ("s2", ok("", 0))]);
    let b = run("reversed", &[("s2", ok("", 0)), ("s1", ok("", 0))]);
    let s = score(&[a, b]);
    assert_eq!(s.equal, 2, "they ARE equal — that is the trap");
    assert_eq!(s.empty_replies, 2);
    let v = verdict(&s);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(v.reason.contains("VACUOUS"), "{}", v.reason);
}

#[test]
fn some_empty_replies_are_not_vacuous_because_the_run_still_had_power() {
    let a = run("canonical", &[("s1", ok("", 0)), ("s2", ok("real", 2))]);
    let b = run("reversed", &[("s2", ok("real", 2)), ("s1", ok("", 0))]);
    let s = score(&[a, b]);
    assert_eq!(s.empty_replies, 1);
    assert_eq!(verdict(&s).kind, VerdictKind::Pass);
}

#[test]
fn a_single_order_cannot_prove_equality() {
    let a = run("canonical", &[("s1", ok("x", 1))]);
    let s = score(&[a]);
    let v = verdict(&s);
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(
        v.reason.contains("one order compares with nothing"),
        "{}",
        v.reason
    );
}

#[test]
fn an_unmeasured_sample_fails_rather_than_being_skipped() {
    let a = run("canonical", &[("s1", ok("x", 1))]);
    let b = run("reversed", &[("s1", RequestOutcome::Error("boom".into()))]);
    let s = score(&[a, b]);
    assert_eq!(s.unmeasured, 1);
    assert_eq!(verdict(&s).kind, VerdictKind::Fail);
}
