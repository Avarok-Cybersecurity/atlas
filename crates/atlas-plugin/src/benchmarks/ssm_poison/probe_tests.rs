// SPDX-License-Identifier: AGPL-3.0-only

//! The pinned script: the gate's falsifiability rests on it being frozen
//! byte-for-byte, so the tests pin the shape, not just the content.

use super::{LONG_PREFIX, TURNS, first_turn, request_body, validate_reference};
use crate::benchmarks::transcript::Transcript;

#[test]
fn the_script_is_four_turns() {
    assert_eq!(TURNS.len(), 4);
}

#[test]
fn the_prefix_is_long_enough_to_force_a_prefill_restore() {
    // The probe exists to exercise the prefix-cache restore path. A prefix
    // too short to survive a chunk boundary would never populate a Marconi
    // checkpoint, and the gate would pass vacuously. ~1.5K tokens ≈ 6K+
    // chars; require a floor well above any chunk.
    assert!(
        LONG_PREFIX.chars().count() > 4000,
        "prefix is {} chars — too short to force prefix-cache state",
        LONG_PREFIX.chars().count()
    );
}

#[test]
fn the_first_turn_carries_the_prefix() {
    let t1 = first_turn();
    assert!(t1.starts_with(LONG_PREFIX));
    assert!(t1.contains(TURNS[0]));
}

#[test]
fn the_script_is_deterministic_by_construction() {
    // No run-id, no date, no randomness: two calls produce identical bytes.
    // (Trivially true for consts, but the test documents the invariant the
    // gate depends on.)
    assert_eq!(first_turn(), first_turn());
    for t in TURNS {
        assert!(!t.is_empty());
    }
}

#[test]
fn request_body_is_greedy_pinned_seed_stream() {
    let body = request_body("m", &[], 256);
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["seed"], 0);
    assert_eq!(body["stream"], true);
    // The OpenAI streaming contract only ships `usage` when asked. Without
    // this, `completion_tokens` and the `cached_tokens` vacuity attestation
    // depend on Atlas volunteering usage frames — correct against Atlas,
    // silently zero against any contract-faithful server.
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert_eq!(body["max_tokens"], 256);
    assert_eq!(body["model"], "m");
    // The replay invariant only holds when the sampler cannot vary: if the
    // temperature were > 0 the gate would measure sampling noise, not state.
    assert_eq!(body["temperature"].as_f64(), Some(0.0));
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
            "The envelope checksum covers the header fields in serialized order. It excludes \
              the payload, and the archive tier quarantines mismatches.",
        ),
        turn(
            "The checksum covers batch id, sequence number, node id, timestamp, then payload \
              length; the payload itself is excluded; a mismatching recomputation quarantines \
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
fn a_wrong_turn_count_is_rejected() {
    let reference = healthy_reference()[..2].to_vec();
    let v = validate_reference(&reference);
    assert!(!v.is_empty());
}
