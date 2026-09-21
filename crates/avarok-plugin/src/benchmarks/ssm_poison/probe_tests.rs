// SPDX-License-Identifier: AGPL-3.0-only

//! The pinned script: the gate's falsifiability rests on it being frozen
//! byte-for-byte, so the tests pin the shape, not just the content.

use super::{LONG_PREFIX, TURNS, first_turn, request_body, validate_reference};
use crate::benchmarks::transcript::Transcript;
use sha2::{Digest, Sha256};

fn sha256(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

#[test]
fn the_first_turn_carries_the_prefix() {
    assert_eq!(first_turn(), format!("{LONG_PREFIX}\n\n{}", TURNS[0]));
}

#[test]
fn the_complete_script_bytes_are_pinned() {
    assert!(LONG_PREFIX.chars().count() > 4000);
    assert_eq!(
        sha256(LONG_PREFIX),
        "ed401b9b80fb43644e0349ebf1a16fdef6662769726979732b2c9369a906d853"
    );
    assert_eq!(
        TURNS.map(sha256),
        [
            "4720e48d963d43e9e7d780c0f29520f1482f6dffca19891cd36932c0aa06fe3b",
            "fdd0b1c5475654d1d531f869143fc0643b33a67a6988fb71b6ee833d249c021f",
            "fcd408a20bee1a9189721e626e1025d5fe0ebdc1de6e2b145a744e18b9006062",
            "647d6d2b92ce20ad171a672d49c1608e262e989c0467a4f98027ede0db3e5a9b",
        ]
    );
    assert_eq!(
        sha256(&first_turn()),
        "2f0b09e5c479bcc17b1aa2e1e60d6c7fab057a8b87eb3efb2fb4c16b6a58fdf0"
    );
}

#[test]
fn request_body_is_greedy_pinned_seed_stream() {
    let messages = vec![serde_json::json!({"role": "user", "content": "hello"})];
    let body = request_body("m", &messages, 256);
    // The OpenAI streaming contract only ships `usage` when asked. Without
    // this, `completion_tokens` and the `cached_tokens` vacuity attestation
    // depend on Atlas volunteering usage frames — correct against Atlas,
    // silently zero against any contract-faithful server.
    // Thinking is disabled via the serve configuration, not via a
    // per-request chat_template_kwargs field. The complete object below owns
    // that absence as well as every field that changes the instrument.
    assert_eq!(
        body,
        serde_json::json!({
            "model": "m",
            "stream": true,
            "stream_options": {"include_usage": true},
            "temperature": 0.0,
            "seed": 0,
            "max_tokens": 256,
            "messages": messages,
        })
    );
}

// ---- reference anchors (B4) ----------------------------------------------

fn turn(text: &str) -> Transcript {
    Transcript {
        text: text.into(),
        finish_reason: Some("stop".into()),
        completion_tokens: text.split_whitespace().count().max(1),
        ..Default::default()
    }
}

/// A reference round that satisfies every anchor.
fn healthy_reference() -> Vec<Transcript> {
    vec![
        turn("ACK 7741-C — 7 sections."),
        turn(
            "1. Monotonic sequence numbers, gaps are corruption.\n\
              2. Bounded clock drift under forty milliseconds.\n\
              3. Closed membership via signed admission and departure.",
        ),
        turn(
            "The envelope checksum covers batch id, sequence number, node id, timestamp, then \
              payload length in serialized order, and excludes the payload itself. The archive tier \
              recomputes it, refuses and quarantines a mismatch, with the recomputed value \
              attached.",
        ),
        turn(
            "The checksum covers batch id, sequence number, node id, timestamp, then payload \
              length; it excludes the payload itself; a mismatching recomputation quarantines \
              the record with the recomputed value attached.",
        ),
    ]
}

#[test]
fn a_healthy_reference_satisfies_the_anchors() {
    assert_eq!(
        validate_reference(&healthy_reference()),
        Vec::<String>::new()
    );
}

#[test]
fn a_spelled_out_section_count_also_anchors() {
    let mut reference = healthy_reference();
    reference[0] = turn("ACK 7741-C, the document lists seven sections.");
    assert_eq!(validate_reference(&reference), Vec::<String>::new());
}

#[test]
fn a_reference_missing_the_ack_is_rejected() {
    // Poisoning deterministic from round 0: the reference itself is garbage,
    // every replay matches the garbage, and the old gate said Invariant.
    let mut reference = healthy_reference();
    reference[0] = turn("The document has 7 sections.");
    let v = validate_reference(&reference);
    assert!(
        v.iter().any(|s| s.contains("ACK 7741-C")),
        "expected an ACK violation, got {v:?}"
    );
}

#[test]
fn a_reference_with_the_wrong_section_count_is_rejected() {
    // "7741-C" carries a 7 of its own; the count must appear OUTSIDE the
    // document id, or this anchor would be vacuous.
    let mut reference = healthy_reference();
    reference[0] = turn("ACK 7741-C — 5 sections.");
    let v = validate_reference(&reference);
    assert!(
        v.iter().any(|s| s.contains("section count")),
        "expected a section-count violation, got {v:?}"
    );
}

#[test]
fn the_section_count_must_be_a_whole_token() {
    let mut reference = healthy_reference();
    reference[0] = turn("ACK 7741-C — seventeen sections.");
    assert_eq!(
        validate_reference(&reference),
        ["turn 1: does not state the document's section count (7)"]
    );
}

#[test]
fn a_two_line_invariant_list_is_rejected() {
    // Turn 2 demands exactly three numbered lines; an early-EOS stub that
    // dropped one is the batch4 shape showing up in the REFERENCE.
    let mut reference = healthy_reference();
    reference[1] = turn("1. Monotonic sequence.\n2. Bounded drift.");
    let v = validate_reference(&reference);
    assert!(
        v.iter().any(|s| s.contains("exactly 3")),
        "expected a line-count violation, got {v:?}"
    );
}

#[test]
fn unnumbered_invariant_lines_are_rejected() {
    let mut reference = healthy_reference();
    reference[1] = turn("Monotonic sequence.\nBounded drift.\nClosed membership.");
    let v = validate_reference(&reference);
    assert!(
        v.iter()
            .any(|s| s.contains("does not start with its number")),
        "expected a numbering violation, got {v:?}"
    );
}

#[test]
fn larger_number_prefixes_do_not_count_as_one_through_three() {
    let mut reference = healthy_reference();
    reference[1] = turn("10. Monotonic sequence.\n20. Bounded drift.\n30. Closed membership.");
    assert_eq!(
        validate_reference(&reference),
        [
            "turn 2: line 1 does not start with its number: \"10. Monotonic sequence.\"",
            "turn 2: line 2 does not start with its number: \"20. Bounded drift.\"",
            "turn 2: line 3 does not start with its number: \"30. Closed membership.\"",
        ]
    );
}

#[test]
fn a_budget_truncated_reference_is_rejected() {
    // A reference turn that hit max_tokens caps every collapse ratio near
    // 1.0 and makes the runaway ceiling unreachable — the budget must let
    // the reference finish on its own terms.
    let mut reference = healthy_reference();
    reference[3].finish_reason = Some("length".into());
    let v = validate_reference(&reference);
    assert!(
        v.iter().any(|s| s.contains("token budget")),
        "expected a budget violation, got {v:?}"
    );
}

#[test]
fn every_reference_turn_must_finish_normally() {
    let mut reference = healthy_reference();
    reference[2].finish_reason = None;
    assert_eq!(
        validate_reference(&reference),
        ["turn 3: reference did not finish normally (finish_reason=None)"]
    );
}

#[test]
fn the_later_reference_turns_must_answer_their_prompts() {
    let mut reference = healthy_reference();
    reference[2] = turn("Nothing relevant. Still nothing.");
    reference[3] = turn("A checksum exists.");
    let violations = validate_reference(&reference);
    assert!(
        violations.iter().any(|v| v.starts_with("turn 3:")),
        "{violations:?}"
    );
    assert!(
        violations.iter().any(|v| v.starts_with("turn 4:")),
        "{violations:?}"
    );
}

#[test]
fn a_wrong_turn_count_is_rejected() {
    let reference = healthy_reference()[..2].to_vec();
    let v = validate_reference(&reference);
    assert!(!v.is_empty());
}
