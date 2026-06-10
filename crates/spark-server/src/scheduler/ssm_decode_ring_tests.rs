// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for [`SsmDecodeRing`] — insertion, eviction, boundary
//! selection, and the disabled-ring path. Pure data-structure tests:
//! no GPU, no model, no mocks needed (the ring tracks only slot
//! indices; the GPU D2D copies live behind the `Model` trait).

use super::SsmDecodeRing;

#[test]
fn disabled_ring_records_nothing() {
    let mut ring = SsmDecodeRing::new(0);
    assert!(!ring.is_enabled());
    assert_eq!(ring.record(10), None);
    assert_eq!(ring.slot_for_position(10), None);
    assert_eq!(ring.len(), 0);
}

#[test]
fn record_assigns_distinct_slots_until_full() {
    let mut ring = SsmDecodeRing::new(3);
    assert!(ring.is_enabled());
    assert_eq!(ring.record(5), Some(0));
    assert_eq!(ring.record(12), Some(1));
    assert_eq!(ring.record(20), Some(2));
    assert_eq!(ring.len(), 3);
    // All three positions are independently addressable.
    assert_eq!(ring.slot_for_position(5), Some(0));
    assert_eq!(ring.slot_for_position(12), Some(1));
    assert_eq!(ring.slot_for_position(20), Some(2));
}

#[test]
fn record_evicts_oldest_when_full_and_reuses_its_slot() {
    let mut ring = SsmDecodeRing::new(3);
    ring.record(5); // slot 0
    ring.record(12); // slot 1
    ring.record(20); // slot 2
    // Fourth record evicts position 5 (slot 0) and reuses slot 0.
    assert_eq!(ring.record(31), Some(0));
    assert_eq!(ring.len(), 3);
    // Position 5 is gone; 12/20/31 remain.
    assert_eq!(ring.slot_for_position(5), None);
    assert_eq!(ring.slot_for_position(12), Some(1));
    assert_eq!(ring.slot_for_position(20), Some(2));
    assert_eq!(ring.slot_for_position(31), Some(0));
}

#[test]
fn record_wraps_round_robin_over_capacity() {
    let mut ring = SsmDecodeRing::new(2);
    assert_eq!(ring.record(1), Some(0));
    assert_eq!(ring.record(2), Some(1));
    assert_eq!(ring.record(3), Some(0)); // evict pos 1
    assert_eq!(ring.record(4), Some(1)); // evict pos 2
    assert_eq!(ring.record(5), Some(0)); // evict pos 3
    // Only the two most recent positions survive.
    assert_eq!(ring.slot_for_position(4), Some(1));
    assert_eq!(ring.slot_for_position(5), Some(0));
    assert_eq!(ring.slot_for_position(3), None);
}

#[test]
fn slot_for_position_requires_exact_match() {
    let mut ring = SsmDecodeRing::new(3);
    ring.record(10);
    ring.record(20);
    // A rollback `keep_len` that lands between snapshots has no slot.
    assert_eq!(ring.slot_for_position(15), None);
    assert_eq!(ring.slot_for_position(10), Some(0));
    assert_eq!(ring.slot_for_position(20), Some(1));
}

#[test]
fn snapshot_positions_lists_live_entries_oldest_first() {
    let mut ring = SsmDecodeRing::new(3);
    ring.record(7);
    ring.record(14);
    ring.record(21);
    let positions: Vec<usize> = ring.snapshot_positions().collect();
    assert_eq!(positions, vec![7, 14, 21]);
    // After eviction the oldest is dropped.
    ring.record(28);
    let positions: Vec<usize> = ring.snapshot_positions().collect();
    assert_eq!(positions, vec![14, 21, 28]);
}

#[test]
fn truncate_after_drops_entries_past_keep_len() {
    let mut ring = SsmDecodeRing::new(3);
    ring.record(10);
    ring.record(20);
    ring.record(30);
    // Rollback keeps 20 tokens — the snapshot at 30 is in the discarded
    // tail and must be dropped; 10 and 20 stay.
    ring.truncate_after(20);
    assert_eq!(ring.len(), 2);
    assert_eq!(ring.slot_for_position(30), None);
    assert_eq!(ring.slot_for_position(20), Some(1));
    assert_eq!(ring.slot_for_position(10), Some(0));
}

#[test]
fn truncate_after_keeps_exact_boundary_snapshot() {
    let mut ring = SsmDecodeRing::new(3);
    ring.record(10);
    ring.record(25);
    // Rolling back to exactly 25 keeps the 25 snapshot (resume point).
    ring.truncate_after(25);
    assert_eq!(ring.slot_for_position(25), Some(1));
    assert_eq!(ring.len(), 2);
}

#[test]
fn boundary_with_snapshot_selection_picks_latest_eligible() {
    // Simulates the snapshot-aware boundary search: given a set of
    // candidate boundary token positions and the ring's live snapshots,
    // the rollback must pick the latest boundary that also has a
    // snapshot.
    let mut ring = SsmDecodeRing::new(3);
    ring.record(8);
    ring.record(16);
    ring.record(40); // a later boundary with a snapshot
    // Candidate boundaries the watchdog found (descending preference).
    let candidate_boundaries = [40usize, 32, 16, 8];
    let chosen = candidate_boundaries
        .iter()
        .copied()
        .find(|&b| ring.slot_for_position(b).is_some());
    assert_eq!(chosen, Some(40));
}

#[test]
fn decline_when_no_boundary_has_snapshot() {
    // No candidate boundary coincides with a live snapshot → the
    // rollback must be declined (caller falls back to hard stop).
    let mut ring = SsmDecodeRing::new(3);
    ring.record(5);
    ring.record(11);
    let candidate_boundaries = [48usize, 36, 24];
    let chosen = candidate_boundaries
        .iter()
        .copied()
        .find(|&b| ring.slot_for_position(b).is_some());
    assert_eq!(chosen, None);
}

#[test]
fn record_after_truncate_does_not_panic_or_share_slots() {
    // Regression for the live 2026-06-10 panic: rollback truncation
    // removes entries without the cursor knowing, so the next full-ring
    // record() collided with a non-oldest entry and the old
    // debug_assert_eq!(entries[0].snapshot_slot, slot) fired
    // (left: 1, right: 5 shape). Release builds instead let two live
    // entries share a slot — the save overwrote a snapshot another entry
    // still pointed at.
    let mut ring = SsmDecodeRing::new(5);
    for (i, pos) in [10, 20, 30, 40, 50].iter().enumerate() {
        assert_eq!(ring.record(*pos), Some(i));
    }
    // Rollback keeps only position 10 (slot 0).
    ring.truncate_after(10);
    assert_eq!(ring.len(), 1);
    // Refill past capacity twice over; every step must keep the
    // no-shared-slot invariant and never panic.
    let mut pos = 60;
    for _ in 0..12 {
        ring.record(pos);
        let mut slots: Vec<usize> = ring.snapshot_positions().collect();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(slots.len(), ring.len(), "duplicate token_position");
        pos += 10;
    }
    assert_eq!(ring.len(), 5);
}

#[test]
fn truncate_rewinds_cursor_to_preserve_survivors() {
    // After truncation the cursor resumes after the newest survivor, so
    // freed tail slots are reused before any surviving snapshot is
    // evicted — the just-restored boundary must stay restorable.
    let mut ring = SsmDecodeRing::new(3);
    ring.record(10); // slot 0
    ring.record(20); // slot 1
    ring.record(30); // slot 2
    ring.truncate_after(20); // survivors: 10/s0, 20/s1
    // Next record must take slot 2 (the freed tail), not evict 10 or 20.
    ring.record(40);
    assert_eq!(ring.slot_for_position(10), Some(0));
    assert_eq!(ring.slot_for_position(20), Some(1));
    assert_eq!(ring.slot_for_position(40), Some(2));
    // One more wraps to slot 0, evicting the oldest (10) as usual.
    ring.record(50);
    assert_eq!(ring.slot_for_position(10), None);
    assert_eq!(ring.slot_for_position(50), Some(0));
}

#[test]
fn truncate_to_empty_resets_cursor() {
    let mut ring = SsmDecodeRing::new(3);
    ring.record(10);
    ring.record(20);
    ring.truncate_after(0);
    assert_eq!(ring.len(), 0);
    assert_eq!(ring.record(30), Some(0));
}
