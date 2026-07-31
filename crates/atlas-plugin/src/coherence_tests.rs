// SPDX-License-Identifier: AGPL-3.0-only

//! Pure-logic tests for the coherence probe. The socket-level behaviour is
//! covered end to end in `tests/coherence.rs`, against the mock endpoint.

use super::*;

/// Does the answer text satisfy the check? Mirrors `ask`'s decision so the
/// matching rule can be tested without a server.
fn accepts(check: &Check, answer: &str) -> bool {
    let lowered = answer.to_lowercase();
    check.accept.iter().any(|a| lowered.contains(a))
}

fn check(label: &str) -> &'static Check {
    CHECKS
        .iter()
        .find(|c| c.label == label)
        .expect("check exists")
}

#[test]
fn a_model_answering_correctly_passes_however_it_phrases_it() {
    let arith = check("arithmetic");
    for answer in ["4", "4.", " 4\n", "The answer is 4.", "Four", "FOUR"] {
        assert!(accepts(arith, answer), "should accept {answer:?}");
    }
    let recall = check("recall");
    for answer in ["Paris", "paris", "The capital of France is Paris."] {
        assert!(accepts(recall, answer), "should accept {answer:?}");
    }
}

#[test]
fn a_wrong_or_empty_answer_fails() {
    let arith = check("arithmetic");
    for answer in ["5", "", "I cannot help with that", "twenty-two"] {
        assert!(!accepts(arith, answer), "should reject {answer:?}");
    }
    // "22" contains no "4" — a garbled quantization producing digits is not a pass.
    assert!(!accepts(arith, "22"));
    assert!(!accepts(check("recall"), "London"));
}

#[test]
fn the_checks_cover_two_different_faculties() {
    // Arithmetic and recall fail independently; a probe made of two arithmetic
    // questions would be one signal counted twice.
    assert_eq!(CHECKS.len(), 2);
    assert!(CHECKS.iter().any(|c| c.label == "arithmetic"));
    assert!(CHECKS.iter().any(|c| c.label == "recall"));
}

#[test]
fn every_accept_pattern_is_lower_case() {
    // `accepts` lower-cases the answer, so an upper-case pattern could never
    // match and the check would silently always fail.
    for c in CHECKS {
        for pattern in c.accept {
            assert_eq!(
                *pattern,
                pattern.to_lowercase(),
                "{}: {pattern:?} must be lower-case",
                c.label
            );
        }
    }
}

#[test]
fn probing_is_the_default_but_it_only_ever_warns() {
    // On by default so a wrong --model is noticed; advisory so a benchmark
    // aimed at a different model is still allowed to run.
    assert_eq!(CoherencePolicy::default(), CoherencePolicy::Probe);
}

#[test]
fn an_empty_answer_reads_as_answered_nothing() {
    // A model that returns no text at all produced the useless message
    // `recall answered ""`. Say what actually happened.
    let report = Report {
        answers: vec![Answer {
            label: "recall",
            answer: String::new(),
            passed: false,
        }],
        transport_error: None,
        served_instead: None,
    };
    let target = TargetEndpoint::local(8888, "m");
    let concern = report.concern(&target).expect("a concern");
    assert!(concern.contains("answered nothing"), "{concern}");
    assert!(!concern.contains("\"\""), "no empty quotes: {concern}");
    assert!(!report.is_clean());
}

#[test]
fn the_concern_describes_rather_than_forbids() {
    let report = Report {
        answers: vec![Answer {
            label: "recall",
            answer: "London".into(),
            passed: false,
        }],
        transport_error: None,
        served_instead: None,
    };
    let concern = report
        .concern(&TargetEndpoint::local(8888, "m"))
        .expect("a concern");
    // The old wording called it a failure and told the user to pass a flag.
    assert!(!concern.contains("failed"), "not a verdict: {concern}");
    assert!(
        concern.contains("still valid"),
        "says the run may proceed: {concern}"
    );
    assert!(concern.contains("different model"), "{concern}");
}

#[test]
fn a_transport_error_is_worded_as_one() {
    let report = Report {
        answers: Vec::new(),
        transport_error: Some("connection refused".into()),
        served_instead: None,
    };
    let concern = report
        .concern(&TargetEndpoint::local(8888, "m"))
        .expect("a concern");
    assert!(concern.contains("did not answer"), "{concern}");
    assert!(
        !concern.contains("different model"),
        "a closed port is not a model problem: {concern}"
    );
}

#[test]
fn a_clean_report_has_nothing_to_say() {
    let report = Report {
        answers: vec![Answer {
            label: "recall",
            answer: "Paris".into(),
            passed: true,
        }],
        transport_error: None,
        served_instead: None,
    };
    assert!(report.is_clean());
    assert!(report.concern(&TargetEndpoint::local(8888, "m")).is_none());
}

#[test]
fn a_long_answer_is_truncated_for_the_error_message() {
    let long = "x".repeat(500);
    let out = truncate(&long, 80);
    assert_eq!(out.chars().count(), 81, "80 chars plus the ellipsis");
    assert!(out.ends_with('…'));
    // Short answers survive intact, trimmed.
    assert_eq!(truncate("  Paris\n", 80), "Paris");
}

#[test]
fn truncate_counts_characters_not_bytes() {
    // A byte-slicing implementation panics on a multi-byte boundary.
    let s = "é".repeat(200);
    let out = truncate(&s, 10);
    assert_eq!(out.chars().count(), 11);
}

#[test]
fn a_wrong_model_name_is_reported_ahead_of_the_answers() {
    // THE case this check exists for: Atlas answers a completion whatever
    // model name it is sent, so the questions cannot see the mistake. Only the
    // model list can — and it must lead, because a wrong name explains any
    // oddity downstream of it.
    let report = Report {
        answers: vec![Answer {
            label: "recall",
            answer: String::new(),
            passed: false,
        }],
        transport_error: None,
        served_instead: Some(vec!["nvidia/Qwen3.6-27B-NVFP4".into()]),
    };
    let target = TargetEndpoint::local(8888, "does/not-exist");
    let concern = report.concern(&target).expect("a concern");
    assert!(
        concern.contains("nvidia/Qwen3.6-27B-NVFP4"),
        "names what IS served: {concern}"
    );
    assert!(
        concern.contains("does/not-exist"),
        "and what was asked for: {concern}"
    );
    assert!(
        !concern.contains("answered nothing"),
        "the cause leads, not the symptom: {concern}"
    );
    assert!(!report.is_clean());
}

#[test]
fn a_server_serving_the_requested_model_is_clean() {
    let report = Report {
        answers: vec![Answer {
            label: "recall",
            answer: "Paris".into(),
            passed: true,
        }],
        transport_error: None,
        served_instead: None,
    };
    assert!(report.is_clean());
}
