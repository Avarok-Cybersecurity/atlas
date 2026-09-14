// SPDX-License-Identifier: AGPL-3.0-only
//! The signer rule is class-conditional. These pin both halves, because a rule
//! that only ever permits is indistinguishable from no rule at all.

use super::agreement::{AddedRecord, Disagreement, check, required_by_class, sensitivity_of};
use crate::hardware::equivalence::HardwareFingerprint;
use crate::hardware::policy::Sensitivity;

/// A record with NO hardware capture — the pre-`--with-nodes` shape, and
/// what a pre-schema record on an old branch still looks like.
fn rec(gate: &str, sha: &str, signer: &str) -> AddedRecord {
    AddedRecord {
        path: format!(".benchmarks/{gate}/2026-09-06-{sha}.json"),
        benchmark_id: gate.into(),
        git_sha: sha.into(),
        signer: signer.into(),
        hardware: None,
    }
}

/// A healthy GB10 capture at `chassis` °C.
fn gb10(chassis: f64) -> HardwareFingerprint {
    HardwareFingerprint {
        gpu: "NVIDIA GB10".into(),
        driver_major: Some(580),
        sm_clock_max_mhz: Some(3_003.0),
        mem_total_kb: Some(127_601_452),
        thermal_alert: Some(false),
        hottest_chassis_c: Some(chassis),
        postcheck_valid: Some(true),
    }
}

fn rec_on(gate: &str, signer: &str, fp: HardwareFingerprint) -> AddedRecord {
    AddedRecord {
        hardware: Some(fp),
        ..rec(gate, "abc123", signer)
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
        [
            Disagreement::SpeedSigners {
                gates,
                signers,
                mismatches,
            },
        ] => {
            assert_eq!(signers.len(), 2, "{signers:?}");
            assert!(gates.contains(&"decode-floor".to_string()), "{gates:?}");
            assert!(
                mismatches[0].contains("no hardware capture"),
                "{mismatches:?}"
            );
        }
        other => panic!("expected a speed-signer disagreement, got {other:?}"),
    }
}

/// THE SECOND RELAXATION, bounded by the records. Two signers on two GB10s
/// whose captures agree are one box; the same two with a 24 °C chassis gap
/// (the 2026-09-06 incident) are not, and the message says why.
#[test]
fn speed_gates_may_span_signers_only_when_the_records_prove_equivalence() {
    let ok = check(&[
        rec_on("decode-floor", "dgx2key", gb10(65.0)),
        rec_on("ttft-cold-gate", "dgx3key", gb10(70.0)),
        rec_on("ttft-warm-gate", "dgx2key", gb10(66.0)),
    ]);
    assert!(ok.is_empty(), "{ok:?}");
    // NEGATIVE CONTROL: the incident pair.
    let v = check(&[
        rec_on("decode-floor", "dgx2key", gb10(65.0)),
        rec_on("ttft-cold-gate", "dgx3key", gb10(89.0)),
    ]);
    match &v[..] {
        [Disagreement::SpeedSigners { mismatches, .. }] => {
            assert_eq!(mismatches.len(), 1, "{mismatches:?}");
            assert!(mismatches[0].contains("chassis 65 vs 89"), "{mismatches:?}");
            let msg = v[0].to_string();
            assert!(msg.contains("chassis 65 vs 89"), "{msg}");
        }
        other => panic!("{other:?}"),
    }
    // NEGATIVE CONTROL: equivalent captures but ONE record's postcheck was
    // invalid — the number it produced is not trusted across boxes.
    let mut bad = gb10(66.0);
    bad.postcheck_valid = Some(false);
    let v = check(&[
        rec_on("decode-floor", "dgx2key", gb10(65.0)),
        rec_on("ttft-cold-gate", "dgx3key", bad),
    ]);
    assert!(
        matches!(&v[..], [Disagreement::SpeedSigners { .. }]),
        "{v:?}"
    );
    // NEGATIVE CONTROL: one side has a capture, the other none.
    let v = check(&[
        rec_on("decode-floor", "dgx2key", gb10(65.0)),
        rec("ttft-cold-gate", "abc123", "dgx3key"),
    ]);
    assert!(
        matches!(&v[..], [Disagreement::SpeedSigners { .. }]),
        "{v:?}"
    );
    // Same signer, wildly different captures: not this rule's business —
    // one box drifting is the hardware policy's job, not agreement's.
    let v = check(&[
        rec_on("decode-floor", "dgx2key", gb10(65.0)),
        rec_on("ttft-cold-gate", "dgx2key", gb10(89.0)),
    ]);
    assert!(v.is_empty(), "{v:?}");
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
