// SPDX-License-Identifier: AGPL-3.0-only

//! Fixed-stride batched-decode metadata layout, derived from the serve
//! `max_batch_size` (SSOT — consumed by `sizes.rs` for the scratch envelope
//! and by `spark-model`'s `upload_batch_metadata_fixed`/`_at` for the
//! upload offsets, replacing the former hardcoded 0/128/256/512/768 gaps
//! that fit exactly 32 rows).
//!
//! Region shapes (byte offsets, `R = rows`):
//!   positions  u32  [0,       4R)
//!   seq_slot   i32  [4R,      8R)   (per-request LoRA routing)
//!   slots      i64  [8R,     16R)
//!   seq_lens   i32  [16R,    20R)
//!   (gap            [20R,    24R))  — legacy [640,768) pad, scaled
//!   block_tbl  i32  [24R,    24R + R·max_blocks·4)
//!
//! At `R = 32` this reproduces the legacy layout BYTE-FOR-BYTE
//! (0/128/256/512/768), so every boot with `max_batch_size <= 32` is
//! byte-identical in addresses, strides and upload sizes.

/// Layout floor: the legacy fixed layout was sized for exactly 32 rows;
/// deriving `rows = max(32, bs)` keeps every `bs <= 32` boot byte-identical.
pub const DECODE_META_MIN_ROWS: usize = 32;

/// Layout ceiling, checked at serve time (`serve.rs`). The metadata gaps
/// themselves derive cleanly to any width; the binding constraints on this
/// tip are downstream row consumers sized at 96+ rows:
/// * the logits arena (`sizes.rs`) — derived `max(96, rows+1)` rows, where
///   `rows+1` covers the run_standard mixed path parking prefill logits at
///   row `padded_n`;
/// * the scratch block-table envelope (`sizes.rs`) — derived
///   `max(verify 96-row overlay, decode `rows`-row layout)`.
///
/// Batched-decode kernels are row-count parametric (grid.y = n, smem per
/// CTA constant, split-K workspace derived from the pinned max batch), so
/// 128 is a policy cap for the widths validated by the enterprise-
/// concurrency campaign, not an smem wall.
pub const DECODE_META_MAX_ROWS: usize = 128;

/// Derived fixed-stride decode-metadata layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeMetaLayout {
    rows: usize,
}

impl DecodeMetaLayout {
    /// Derive the layout from the serve `max_batch_size`. Callers gate
    /// `max_batch_size <= DECODE_META_MAX_ROWS` at serve time; this
    /// constructor only applies the byte-identity floor.
    pub fn for_max_batch_size(max_batch_size: usize) -> Self {
        Self {
            rows: max_batch_size.max(DECODE_META_MIN_ROWS),
        }
    }

    /// Row capacity of the metadata block (== the widest `padded_n` the
    /// upload accepts).
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// `positions` u32 stream offset.
    pub fn positions_off(&self) -> usize {
        0
    }

    /// Per-request LoRA adapter-slot i32 stream offset.
    pub fn seq_slot_off(&self) -> usize {
        4 * self.rows
    }

    /// KV `slots` i64 stream offset (8-byte aligned: `8R`).
    pub fn slots_off(&self) -> usize {
        8 * self.rows
    }

    /// `seq_lens` i32 stream offset.
    pub fn seq_lens_off(&self) -> usize {
        16 * self.rows
    }

    /// Flattened block-table offset (row stride `max_blocks · 4` bytes).
    pub fn block_table_off(&self) -> usize {
        24 * self.rows
    }

    /// Total bytes of the metadata block for `max_blocks` blocks per row.
    pub fn meta_bytes(&self, max_blocks: usize) -> usize {
        self.block_table_off() + self.rows * max_blocks * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bs <= 32 must reproduce the legacy hardcoded layout byte-for-byte.
    #[test]
    fn legacy_layout_at_or_below_32() {
        for bs in [1usize, 31, 32] {
            let l = DecodeMetaLayout::for_max_batch_size(bs);
            assert_eq!(l.rows(), 32, "bs={bs}");
            assert_eq!(l.positions_off(), 0);
            assert_eq!(l.seq_slot_off(), 128);
            assert_eq!(l.slots_off(), 256);
            assert_eq!(l.seq_lens_off(), 512);
            assert_eq!(l.block_table_off(), 768);
            // Legacy total: 768 + 32·mb·4 (decode bt region in sizes.rs).
            assert_eq!(l.meta_bytes(257), 768 + 32 * 257 * 4);
        }
    }

    /// Widened layouts: regions must be contiguous-or-gapped exactly like
    /// the legacy shape scaled by R/32, non-overlapping, and 8-byte aligned
    /// where i64 lands.
    #[test]
    fn widened_layout_arithmetic() {
        for bs in [33usize, 64, 128] {
            let l = DecodeMetaLayout::for_max_batch_size(bs);
            let r = l.rows();
            assert_eq!(r, bs, "bs={bs}: rows derive from bs above the floor");
            // positions [0,4R) then seq_slot [4R,8R): no overlap.
            assert_eq!(l.seq_slot_off(), l.positions_off() + 4 * r);
            // slots i64 begins exactly after seq_slot and is 8-byte aligned.
            assert_eq!(l.slots_off(), l.seq_slot_off() + 4 * r);
            assert_eq!(l.slots_off() % 8, 0);
            // seq_lens begins exactly after the 8R-byte slots region.
            assert_eq!(l.seq_lens_off(), l.slots_off() + 8 * r);
            // block table begins after seq_lens (4R) + the scaled legacy pad (4R).
            assert_eq!(l.block_table_off(), l.seq_lens_off() + 8 * r);
            assert_eq!(l.meta_bytes(257), 24 * r + r * 257 * 4);
        }
    }

    #[test]
    fn policy_ceiling_and_floor_are_exact() {
        assert_eq!(DECODE_META_MIN_ROWS, 32);
        assert_eq!(DECODE_META_MAX_ROWS, 128);
    }
}
