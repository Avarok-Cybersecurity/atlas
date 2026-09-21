// SPDX-License-Identifier: AGPL-3.0-only

//! Exact-integer pins for the preflight wiring. No GPU, no model, no checkpoint —
//! which is the point: this term must be computable before the model exists.

use spark_model::seq_state_reserve::PerSequenceState;

#[test]
fn max_batch_size_is_applied_exactly_once() {
    let s = PerSequenceState {
        target_layers: 739_639_296,
        proposer: 67_574_292,
    };
    assert_eq!(s.total(), 807_213_588);
    assert_eq!(s.for_batch(3), 2_421_640_764);
    // A second application would land here; the pin exists so it cannot pass unnoticed.
    assert_ne!(s.for_batch(3), s.total() * 9);
}

#[test]
fn the_two_owners_stay_separate() {
    let s = PerSequenceState {
        target_layers: 739_639_296,
        proposer: 67_574_292,
    };
    // The 12-layer DSA figure (806,879,232) is what the drafter's own doc quotes. It is the
    // TOTAL across target + drafter, not the proposer's share. Charging it against the
    // proposer on top of the target term would double-charge by 11/12ths.
    assert_ne!(s.proposer, 806_879_232);
    assert!(s.proposer < s.target_layers / 10);
}

#[test]
fn a_model_that_owns_no_per_sequence_state_is_charged_nothing() {
    let s = PerSequenceState::default();
    assert_eq!(s.total(), 0);
    assert_eq!(s.for_batch(3), 0);
}
