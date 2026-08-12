// SPDX-License-Identifier: AGPL-3.0-only

//! The pinned script: the gate's falsifiability rests on it being frozen
//! byte-for-byte, so the tests pin the shape, not just the content.

use super::{LONG_PREFIX, TURNS, first_turn, request_body};

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
    assert_eq!(body["max_tokens"], 256);
    assert_eq!(body["model"], "m");
    // The replay invariant only holds when the sampler cannot vary: if the
    // temperature were > 0 the gate would measure sampling noise, not state.
    assert_eq!(body["temperature"].as_f64(), Some(0.0));
}
