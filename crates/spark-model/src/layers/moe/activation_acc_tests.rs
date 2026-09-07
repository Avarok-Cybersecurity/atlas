// SPDX-License-Identifier: AGPL-3.0-only

//! Fold semantics for the per-request expert accumulator.
//!
//! Every case here is one the benchmark downstream would misread as real
//! routing if the fold got it wrong: a padded staging row read as expert 0,
//! a LongCat zero-expert counted as an expert with weights, or a mass that
//! no longer matches the counts it was summed beside.

use super::ExpertActivationAcc;

fn acc() -> ExpertActivationAcc {
    ExpertActivationAcc::new(4, 8, 2)
}

// ---------------------------------------------------------------- Path A

#[test]
fn folds_counts_and_mass_per_layer() {
    let mut a = acc();
    a.fold_row(1, &[3, 5], &[0.7, 0.3]);
    a.fold_row(1, &[3, 6], &[0.6, 0.4]);
    a.fold_row(2, &[0, 7], &[0.5, 0.5]);

    let layers = a.to_layers();
    assert_eq!(layers.len(), 2, "only layers with routing appear");

    let (l1, experts1) = &layers[0];
    assert_eq!(*l1, 1);
    // Experts ascending; expert 3 chosen twice, 5 and 6 once each.
    assert_eq!(experts1[0].0, 3);
    assert_eq!(experts1[0].1, 2);
    assert!((experts1[0].2 - 1.3).abs() < 1e-6);
    assert_eq!(experts1[1], (5, 1, 0.3));
    assert_eq!(experts1[2], (6, 1, 0.4));

    let (l2, experts2) = &layers[1];
    assert_eq!(*l2, 2);
    assert_eq!(experts2.len(), 2);

    assert_eq!(a.tokens_routed(), 3);
}

#[test]
fn counts_conserve_against_tokens_routed() {
    // The benchmark's vacuity pin: Σcounts must equal tokens_routed × top_k
    // when every slot carried weight. A tap that recorded nothing, or folded
    // a row twice, breaks this equality and nothing else would show it.
    let mut a = acc();
    for _ in 0..5 {
        a.fold_row(0, &[1, 2], &[0.5, 0.5]);
    }
    let total: u32 = a
        .to_layers()
        .iter()
        .flat_map(|(_, e)| e.iter().map(|(_, c, _)| *c))
        .sum();
    assert_eq!(u64::from(total), a.tokens_routed() * u64::from(a.top_k()));
}

// ---------------------------------------------------------------- Path B

#[test]
fn zero_weight_slots_do_not_count_as_routing() {
    // Two ways a zero-weight slot arises, and neither is this request using
    // expert 0: a staging row the pass never wrote (memset to zero), and a
    // LongCat zero/identity expert folded out of the blend upstream.
    let mut a = acc();
    a.fold_row(0, &[0, 0], &[0.0, 0.0]);
    assert!(
        a.to_layers().is_empty(),
        "an all-zero row is not routing and must not appear"
    );
    assert_eq!(a.tokens_routed(), 0);
}

#[test]
fn partially_zero_row_folds_only_the_live_slot() {
    let mut a = acc();
    a.fold_row(0, &[4, 0], &[0.9, 0.0]);
    let layers = a.to_layers();
    assert_eq!(layers[0].1, vec![(4, 1, 0.9)]);
    // The row DID route, so it counts once toward tokens_routed even though
    // only one of its two slots was live.
    assert_eq!(a.tokens_routed(), 1);
}

#[test]
fn expert_id_beyond_the_expert_count_is_skipped() {
    // Zero-computation experts (LongCat) score in a wider id space than the
    // model has expert weights for. Folding one would index out of the
    // dense arrays, or — worse, if it happened to land in range — attribute
    // mass to an expert that was never run.
    let mut a = acc();
    a.fold_row(0, &[8, 2], &[0.4, 0.6]);
    assert_eq!(a.to_layers()[0].1, vec![(2, 1, 0.6)]);
}

#[test]
fn non_finite_weight_is_skipped() {
    // A NaN would poison the layer's mass total and every budget computed
    // from it; the comparison in budget_experts silently orders NaN last.
    let mut a = acc();
    a.fold_row(0, &[1, 2], &[f32::NAN, 0.5]);
    assert_eq!(a.to_layers()[0].1, vec![(2, 1, 0.5)]);
}

#[test]
fn out_of_range_layer_is_dropped_not_wrapped() {
    // A drain that mis-sized its layer stride must not fold layer 9 onto
    // layer 1 of a 4-layer model.
    let mut a = acc();
    a.fold_row(9, &[1, 2], &[0.5, 0.5]);
    assert!(a.to_layers().is_empty());
    assert_eq!(a.tokens_routed(), 0);
}

// ---------------------------------------------------------------- Path C

#[test]
fn unattributed_rows_are_recorded_not_hidden() {
    // A pass wider than the staging buffer stages a prefix. The shortfall
    // has to reach the response: silently reporting the prefix would read
    // as "these are all the experts the request used".
    let mut a = acc();
    a.fold_row(0, &[1, 2], &[0.5, 0.5]);
    a.note_unattributed_rows(7);
    assert_eq!(a.unattributed_rows(), 7);
    assert_eq!(a.tokens_routed(), 1);
}

#[test]
fn decode_rows_are_counted_separately_from_prefill() {
    // A consumer has to be able to tell which half of the request the routing
    // came from: a prefill-only report and a whole-request report describe
    // different things, and the wire `scope` is derived from these counters.
    let mut a = acc();
    a.fold_row(0, &[1, 2], &[0.5, 0.5]);
    a.fold_row(0, &[1, 3], &[0.5, 0.5]);
    a.fold_decode_row(0, &[4, 5], &[0.5, 0.5]);

    assert_eq!(a.tokens_routed(), 3, "every folded row counts once overall");
    assert_eq!(a.decode_tokens_routed(), 1, "one of them was a decode row");
}

#[test]
fn a_decode_row_that_folds_nothing_counts_nothing() {
    // The zero-weight case again, on the decode path: a staged row the pass
    // never wrote must not inflate the decode count and make a request look
    // like it decoded more than it did. This is the counter the drain's
    // per-step accounting is checked against end to end (prefill rows equal
    // prompt tokens, decode rows equal completion tokens minus one), which is
    // how a double-drain shows up as a number rather than as silence.
    let mut a = acc();
    a.fold_decode_row(0, &[0, 0], &[0.0, 0.0]);
    assert_eq!(a.decode_tokens_routed(), 0);
    assert_eq!(a.tokens_routed(), 0);
}

#[test]
fn unattributed_decode_rows_are_tracked_apart_from_prefill_ones() {
    // MTP verify rows are staged but not folded in v1. Counting them in the
    // prefill bucket would say the PROMPT was partly unmeasured, which is a
    // different defect with a different fix.
    let mut a = acc();
    a.note_unattributed_rows(3);
    a.note_decode_unattributed_rows(7);
    assert_eq!(a.unattributed_rows(), 3);
    assert_eq!(a.decode_unattributed_rows(), 7);
}

#[test]
fn empty_accumulator_reports_no_layers() {
    let a = acc();
    assert!(a.to_layers().is_empty());
    assert_eq!(a.tokens_routed(), 0);
    assert_eq!(a.unattributed_rows(), 0);
}
