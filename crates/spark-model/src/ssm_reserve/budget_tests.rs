// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for [`super`] (`ssm_reserve::budget`). A sibling file via `#[path]`,
//! the `decode_ring_tests.rs` idiom.
//!
//! Two machines pin the arithmetic:
//!
//! * GB10, the default part: ~115 GiB free, a 17 GiB NVFP4 checkpoint of the
//!   27B, whose per-sequence SSM state blob is 151.5 MiB (the "158,859,264 B"
//!   every #915 log line quotes). Everything the budget does must leave its
//!   default configurations alone.
//! * An RTX 4060 Ti (16 GB, Ada) serving a 4B hybrid model in BF16: ~15.3 GiB
//!   free, ~8.3 GiB of weights, a 51 MiB blob. At the default batch of 8 the
//!   unbudgeted ring is 3,264 MiB (8 slots x 8 sequences x 51 MiB) and Marconi
//!   816 MiB (16 x 51 MiB).
use super::*;

const MIB: usize = 1024 * 1024;
const GIB: usize = 1024 * MIB;

/// 27B per-sequence SSM state blob (48 GDN layers x h + conv).
const BLOB_27B: usize = 158_859_264;
/// 4B per-sequence SSM state blob: 3,264 MiB / (8 slots x 8 seqs).
const BLOB_4B: usize = 51 * MIB;

fn gb10_beside() -> usize {
    memory_beside_weights(115 * GIB, 17 * GIB)
}

fn ada_beside() -> usize {
    memory_beside_weights(15 * GIB + 300 * MIB, 8 * GIB + 300 * MIB)
}

#[test]
fn beside_weights_subtracts_the_checkpoint() {
    assert_eq!(memory_beside_weights(100, 30), 70);
    assert_eq!(memory_beside_weights(100, 0), 100);
}

#[test]
fn a_checkpoint_larger_than_half_of_free_memory_is_clamped() {
    // A BF16 checkpoint quantized to NVFP4 while it loads is far bigger on
    // disk than it is resident: do not zero the budget of a serve that fits.
    assert_eq!(memory_beside_weights(100, 49), 51);
    assert_eq!(memory_beside_weights(100, 50), 50);
    assert_eq!(memory_beside_weights(100, 244), 50);
    assert_eq!(memory_beside_weights(100, usize::MAX), 50);
    assert_eq!(memory_beside_weights(0, 10), 0);
}

#[test]
fn shares_are_the_stated_fractions() {
    assert_eq!(ring_budget_bytes(900), 900 / RING_BUDGET_DIVISOR);
    assert_eq!(marconi_budget_bytes(900), 900 / MARCONI_BUDGET_DIVISOR);
    assert_eq!(RING_BUDGET_DIVISOR, 3);
    assert_eq!(MARCONI_BUDGET_DIVISOR, 8);
}

#[test]
fn gb10_keeps_depth_8_through_batch_27_on_the_27b() {
    let ring = avarok_kernels::DECODE_ROLLBACK_RING_SLOTS;
    for batch in [1usize, 4, 8, 16, 24, 27] {
        assert_eq!(
            cap_decode_ring_slots(ring, batch * BLOB_27B, gb10_beside()),
            ring,
            "batch {batch} must keep the default ring on GB10"
        );
    }
}

#[test]
fn gb10_gives_up_depth_not_batch_once_the_ring_would_swallow_the_card() {
    let ring = avarok_kernels::DECODE_ROLLBACK_RING_SLOTS;
    // 28 x 8 x 151.5 MiB = 33.1 GiB against a 32.7 GiB budget.
    assert_eq!(cap_decode_ring_slots(ring, 28 * BLOB_27B, gb10_beside()), 4);
    assert_eq!(cap_decode_ring_slots(ring, 32 * BLOB_27B, gb10_beside()), 4);
    assert_eq!(cap_decode_ring_slots(ring, 64 * BLOB_27B, gb10_beside()), 2);
    assert_eq!(
        cap_decode_ring_slots(ring, 128 * BLOB_27B, gb10_beside()),
        1
    );
}

#[test]
fn the_cap_never_raises_the_depth() {
    // An already-shrunk or skipped ring stays as it was asked.
    assert_eq!(cap_decode_ring_slots(0, BLOB_27B, gb10_beside()), 0);
    assert_eq!(cap_decode_ring_slots(2, BLOB_27B, gb10_beside()), 2);
    assert_eq!(cap_decode_ring_slots(2, 64 * BLOB_27B, gb10_beside()), 2);
}

#[test]
fn a_16_gb_ada_card_takes_depth_4_at_the_default_batch_and_8_at_batch_1() {
    let ring = avarok_kernels::DECODE_ROLLBACK_RING_SLOTS;
    // The 8.3 GiB checkpoint is over half of the 15.3 GiB free, so it is
    // clamped to 7,830 MiB and 7,830 MiB stay beside it. Default batch 8:
    // 8 x 8 x 51 MiB = 3,264 MiB against a 2,610 MiB share, so the ring
    // halves to 1,632 MiB.
    let batch8 = 8 * BLOB_4B;
    assert_eq!(batch8 * ring, 3_264 * MIB);
    assert_eq!(cap_decode_ring_slots(ring, batch8, ada_beside()), 4);
    assert!(4 * batch8 <= ring_budget_bytes(ada_beside()));
    // A single-stream serve pays 408 MiB for the full ring and keeps it.
    assert_eq!(cap_decode_ring_slots(ring, BLOB_4B, ada_beside()), ring);
    // Batch 16 doubles the cost of a depth unit: depth 2.
    assert_eq!(cap_decode_ring_slots(ring, 16 * BLOB_4B, ada_beside()), 2);
}

#[test]
fn a_ring_that_cannot_fit_at_all_is_dropped_not_forced() {
    assert_eq!(cap_decode_ring_slots(8, 4 * GIB, GIB), 0);
    assert_eq!(cap_decode_ring_slots(8, BLOB_4B, 0), 0);
}

#[test]
fn a_zero_cost_slot_is_not_capped() {
    // No SSM layers (or a zero batch): nothing to size, nothing to cap.
    assert_eq!(cap_decode_ring_slots(8, 0, 0), 8);
    assert_eq!(cap_marconi_slots(16, 0, 0), 16);
}

#[test]
fn marconi_keeps_the_default_16_slots_on_gb10_and_on_the_ada_card() {
    assert_eq!(cap_marconi_slots(16, BLOB_27B, gb10_beside()), 16);
    // 816 MiB of Marconi against 7,830 MiB / 8 = 978 MiB.
    assert_eq!(16 * BLOB_4B, 816 * MIB);
    assert_eq!(cap_marconi_slots(16, BLOB_4B, ada_beside()), 16);
}

#[test]
fn marconi_yields_slots_on_a_tighter_card() {
    // A 12 GB card with the same 8.3 GiB checkpoint: 11.2 GiB free, the
    // checkpoint is clamped to 5.6 GiB, so 5.6 GiB / 8 = 0.7 GiB = 14 slots.
    let beside = memory_beside_weights(11 * GIB + 200 * MIB, 8 * GIB + 300 * MIB);
    assert_eq!(cap_marconi_slots(16, BLOB_4B, beside), 14);
    // The cap never raises a smaller request, nor invents slots for 0.
    assert_eq!(cap_marconi_slots(4, BLOB_4B, beside), 4);
    assert_eq!(cap_marconi_slots(0, BLOB_4B, beside), 0);
    assert_eq!(cap_marconi_slots(16, BLOB_4B, 0), 0);
}

#[test]
fn a_published_cap_only_ever_lowers_the_slot_count() {
    assert_eq!(apply_marconi_cap(16, None), 16);
    assert_eq!(apply_marconi_cap(16, Some(14)), 14);
    assert_eq!(apply_marconi_cap(16, Some(64)), 16);
    assert_eq!(apply_marconi_cap(0, Some(14)), 0);
    assert_eq!(apply_marconi_cap(16, Some(0)), 0);
}
