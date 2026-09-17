// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for `verify_hc.rs`, in a sibling so the parent stays under the
//! 500-line cap. Declared there with `#[path]`, the repo's existing
//! pattern; the module keeps its name and its `super::` imports.

use super::{
    VERIFY_POS_STRIDE, VERIFY_SEQ_LEN_STRIDE, VERIFY_SLOT_STRIDE, verify_row_decode_seq_len,
    verify_row_seq_len_offset, verify_row_seq_len_value,
};

/// THE contract the decode-shaped attention replay rests on: row 0 of a
/// K-row verify re-processes a token a serial decode already committed,
/// so it must be handed EXACTLY the arguments that decode was handed.
/// Serial decode of the token at absolute position `p` passes host
/// `seq_len = p` (pre-append) and a device `seq_len = p + 1` (keys
/// visible after `write_kv_cache`). Off by one in either direction and
/// row 0 attends over the wrong prefix.
#[test]
fn row_zero_matches_a_serial_decode_of_the_committed_token() {
    for base in [1usize, 35, 367, 4096] {
        assert_eq!(verify_row_decode_seq_len(base, 0), base);
        assert_eq!(verify_row_seq_len_value(base, 0), base as i32 + 1);
    }
}

/// Each further row is one token later, and the device value stays
/// exactly one ahead of the host one.
#[test]
fn each_row_advances_by_exactly_one_token() {
    let base = 367usize;
    for k in 1..=8usize {
        for t in 0..k {
            assert_eq!(verify_row_decode_seq_len(base, t), base + t, "t={t}");
            assert_eq!(
                verify_row_seq_len_value(base, t),
                verify_row_decode_seq_len(base, t) as i32 + 1,
                "t={t}: device seq_len must count the row's own key"
            );
        }
    }
}

/// The three metadata streams are packed at DIFFERENT element widths.
/// This is the arithmetic the per-row pointer bumps use; the failure it
/// pins is the natural (and wrong) assumption that a row is `t * 4` in
/// all three.
#[test]
fn the_per_row_metadata_strides_are_not_uniform() {
    assert_eq!(VERIFY_POS_STRIDE, 4, "positions are [N] u32");
    assert_eq!(VERIFY_SLOT_STRIDE, 8, "slots are [N] i64");
    assert_eq!(VERIFY_SEQ_LEN_STRIDE, 4, "seq_lens are [N] i32");
    assert_ne!(
        VERIFY_SLOT_STRIDE, VERIFY_POS_STRIDE,
        "a row's slot is not at the same byte offset as its position"
    );
}

/// The per-row seq_len array must start past the LAST slot entry the
/// pack writes -- `fill_slots_from_block_table` fills `k` i64 slots at
/// `slot_offset` -- and must be 4-byte aligned for an i32 store.
#[test]
fn the_row_seq_len_array_clears_the_slot_table() {
    for slot_offset in [8usize, 16, 24, 4104] {
        for k in 1..=8usize {
            let at = verify_row_seq_len_offset(slot_offset, k);
            assert!(
                at >= slot_offset + k * VERIFY_SLOT_STRIDE,
                "slot_offset={slot_offset} k={k}: seq_len array at {at} overlaps \
                 the {k}-entry i64 slot table"
            );
            assert_eq!(at % VERIFY_SEQ_LEN_STRIDE, 0, "unaligned i32 store");
        }
    }
}

/// Distinct rows address distinct seq_len words -- a shared word would
/// give every row the same visible-key count, which is precisely the
/// prefill shape this replaces.
#[test]
fn every_row_gets_its_own_seq_len_word() {
    let base = verify_row_seq_len_offset(16, 4);
    let mut seen = std::collections::BTreeSet::new();
    for t in 0..4usize {
        assert!(seen.insert(base + t * VERIFY_SEQ_LEN_STRIDE));
    }
    assert_eq!(seen.len(), 4);
}
