// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for [`super`] (`mtp_step`). A sibling file via `#[path]` — the
//! `mtp_dcut.rs`/`mtp_dcut_tests.rs` idiom — so the allow-listed `step_mtp`
//! monolith doesn't grow; module position (child of `mtp_step`) is
//! unchanged, so `super::*` paths are untouched.
//!
//! Pins the batched DFlash verify dispatch order (`sort_batch_by_slot`, the
//! extracted core of the `by_slot` sort in `step_mtp`): the batch handed to
//! `step_verify_dflash_batched` must be in ssm-SLOT order, never arrival
//! order. The pool freelist is LIFO, so after the first fill arrival order
//! is reverse finish order and an arrival-ordered batch fails the
//! consecutive-slot layout check — the model then DECLINES to the
//! per-sequence loop (~47% of steps engaged in the two-round C=16 prose
//! bench, 2026-09-02) and the amortisation silently vanishes.

use super::sort_batch_by_slot;

#[test]
fn batched_verify_dispatch_is_slot_order_not_arrival_order() {
    // The LIFO-freelist shape: three verify-ready sequences whose ssm slots
    // are [3, 1, 2], arriving in exactly that order (batchable_idxs is
    // filled in verify_idxs order). RED under the old arrival-order
    // dispatch: the batch would be handed over as slots [3, 1, 2].
    let slots = [Some(3usize), Some(1), Some(2)];
    let mut by_slot = vec![0usize, 1, 2]; // arrival order
    sort_batch_by_slot(&mut by_slot, |i| slots[i]);
    let dispatched: Vec<usize> = by_slot.iter().map(|&i| slots[i].unwrap()).collect();
    assert_eq!(
        dispatched,
        vec![1, 2, 3],
        "the batch must be dispatched in ascending ssm-slot order — arrival \
         order fails the consecutive-slot pointer check and declines the \
         batched verify"
    );
}

#[test]
fn slotless_sequences_sort_last_and_index_breaks_ties() {
    // A sequence without an ssm slot cannot anchor the consecutive-slot
    // layout — it must sort after every slotted one (the model re-checks
    // the layout and declines safely if a gap remains). Two slotless
    // entries keep their active-index order, so the result is total and
    // deterministic across steps.
    let slots = [None, Some(5usize), None, Some(0)];
    let mut by_slot = vec![0usize, 1, 2, 3];
    sort_batch_by_slot(&mut by_slot, |i| slots[i]);
    assert_eq!(
        by_slot,
        vec![3, 1, 0, 2],
        "slotted ascending first (0 then 5), then slotless by active index"
    );
}
