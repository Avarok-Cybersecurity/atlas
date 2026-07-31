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
fn requiring_the_probe_is_the_default() {
    // A probe you have to remember to switch on does not prevent the 12-hour
    // failure it exists to prevent.
    assert_eq!(CoherencePolicy::default(), CoherencePolicy::Require);
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
