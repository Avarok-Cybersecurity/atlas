// SPDX-License-Identifier: AGPL-3.0-only
//! The signer rule is class-conditional. These pin both halves, because a rule
//! that only ever permits is indistinguishable from no rule at all.

use super::agreement::{AddedRecord, Disagreement, check, required_by_class, sensitivity_of};
use crate::hardware::policy::Sensitivity;

fn rec(gate: &str, sha: &str, signer: &str) -> AddedRecord {
    AddedRecord {
        path: format!(".benchmarks/{gate}/2026-09-06-{sha}.json"),
        benchmark_id: gate.into(),
        git_sha: sha.into(),
        signer: signer.into(),
    }
}

/// The line this whole module turns on. If a gate's class ever flips, the rule
/// silently changes meaning for it, so the mapping is pinned here as well as in
/// the descriptors.
#[test]
fn the_required_gates_split_the_way_the_rule_assumes() {
    let (speed, correctness) = required_by_class();
    for g in [
        "decode-floor",
        "ttft-warm-gate",
        "ttft-cold-gate",
        "agentic-webserver",
    ] {
        assert!(speed.contains(&g), "{g} must be speed-class, got {speed:?}");
    }
    for g in ["bfcl-subset", "bfcl-subset-echolp", "vision-fidelity"] {
        assert!(
            correctness.contains(&g),
            "{g} must be correctness-class, got {correctness:?}"
        );
    }
    assert!(
        !speed.is_empty() && !correctness.is_empty(),
        "a split with an empty side would make the rule vacuous"
    );
}

#[test]
fn an_empty_set_agrees_with_itself() {
    assert!(check(&[]).is_empty());
}

/// One commit, one signer, mixed classes — the ordinary passing shape.
#[test]
fn one_commit_one_signer_is_fine() {
    let v = check(&[
        rec("decode-floor", "abc123", "k1"),
        rec("bfcl-subset", "abc123", "k1"),
    ]);
    assert!(v.is_empty(), "{v:?}");
}

/// Two commits is fatal regardless of class — this half of the rule did NOT
/// relax, and a test that only exercised the relaxed half would hide that.
#[test]
fn two_commits_still_fail_even_for_correctness_gates() {
    let v = check(&[
        rec("bfcl-subset", "abc123", "k1"),
        rec("vision-fidelity", "def456", "k1"),
    ]);
    assert!(
        matches!(&v[..], [Disagreement::Commits(s)] if s.len() == 2),
        "{v:?}"
    );
}

/// THE RELAXATION. Correctness gates are box-independent — proven by measuring
/// the same BFCL gate on two boxes at one commit and getting identical scores —
/// so two signers is allowed, which is what makes a sharded gate parallelisable.
#[test]
fn correctness_gates_may_span_two_signers() {
    let v = check(&[
        rec("bfcl-subset", "abc123", "dgx1key"),
        rec("bfcl-subset-echolp", "abc123", "dgx2key"),
        rec("vision-fidelity", "abc123", "dgx3key"),
    ]);
    assert!(v.is_empty(), "correctness may span boxes, got {v:?}");
}

/// THE PART THAT MUST NOT RELAX. Throughput is box-dependent by 0.66 tok/s
/// against a 0.07 within-box sigma, so a speed record set spanning boxes is
/// comparing numbers that were never comparable.
#[test]
fn speed_gates_may_not_span_signers() {
    let v = check(&[
        rec("decode-floor", "abc123", "dgx1key"),
        rec("ttft-cold-gate", "abc123", "dgx2key"),
    ]);
    match &v[..] {
        [Disagreement::SpeedSigners { gates, signers }] => {
            assert_eq!(signers.len(), 2, "{signers:?}");
            assert!(gates.contains(&"decode-floor".to_string()), "{gates:?}");
        }
        other => panic!("expected a speed-signer disagreement, got {other:?}"),
    }
}

/// A mixed set where only the correctness half spans boxes must still pass —
/// otherwise the relaxation is unusable in the very campaign shape it exists for.
#[test]
fn a_mixed_set_is_judged_per_class_not_as_a_whole() {
    let v = check(&[
        rec("decode-floor", "abc123", "dgx1key"),
        rec("ttft-warm-gate", "abc123", "dgx1key"),
        rec("bfcl-subset", "abc123", "dgx2key"),
        rec("vision-fidelity", "abc123", "dgx3key"),
    ]);
    assert!(v.is_empty(), "{v:?}");
}

/// Fail closed. An id the registry does not know must not inherit the
/// permissive class — that would be a way to smuggle a speed record past the
/// rule by naming a gate that does not exist.
#[test]
fn an_unknown_benchmark_is_refused_rather_than_assumed_correctness() {
    let v = check(&[rec("not-a-real-gate", "abc123", "k1")]);
    assert!(
        matches!(&v[..], [Disagreement::UnknownBenchmark(id)] if id == "not-a-real-gate"),
        "{v:?}"
    );
}

#[test]
fn the_message_names_the_gates_that_must_be_redone() {
    let v = check(&[
        rec("decode-floor", "abc123", "k1"),
        rec("ttft-cold-gate", "abc123", "k2"),
    ]);
    let msg = v[0].to_string();
    assert!(msg.contains("decode-floor"), "{msg}");
    assert!(msg.contains("ttft-cold-gate"), "{msg}");
    assert!(msg.contains("ONE box"), "{msg}");
}

#[test]
fn sensitivity_comes_from_the_registry_not_the_record() {
    assert_eq!(sensitivity_of("decode-floor"), Some(Sensitivity::Speed));
    assert_eq!(
        sensitivity_of("bfcl-subset"),
        Some(Sensitivity::Correctness)
    );
    assert_eq!(sensitivity_of("nope"), None);
}
