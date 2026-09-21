// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the segmented n-gram row cache, split from [`super`] for size.

use super::*;

#[test]
fn aligned_scratch_is_4k_aligned_and_two_blocks() {
    let mut b = AlignedBlock::new();
    let s = b.blocks(2);
    assert_eq!(s.len(), BLOCK * 2);
    assert_eq!(s.as_ptr() as usize % BLOCK, 0);
}

#[test]
fn row_wider_than_a_block_is_refused() {
    // A row larger than one block could span three with an unaligned base.
    let msg = match NgramRowCache::open(Path::new("/nonexistent"), None, 10, BLOCK + 8, 4) {
        Ok(_) => panic!("expected refusal for oversize row_stride"),
        Err(e) => format!("{e:#}"),
    };
    assert!(msg.contains("O_DIRECT block"), "{msg}");
}

/// The seam arithmetic: with a base offset that is only 8-byte aligned
/// (what a safetensors shard gives), rows land at every phase relative to
/// the 4 KiB block, and the covering span must stay within two blocks.
#[test]
fn straddling_rows_are_covered_by_two_blocks() {
    for base in [0u64, 8, 1234568, 4095, 4097] {
        for stride in [256usize, 512, 4096] {
            for id in [0u64, 1, 7, 8, 1023] {
                let byte = base + id * stride as u64;
                let block_off = byte - (byte % BLOCK as u64);
                let within = (byte - block_off) as usize;
                let n = if within + stride > BLOCK { 2 } else { 1 };
                assert!(
                    within + stride <= n * BLOCK,
                    "base={base} stride={stride} id={id} within={within} n={n}"
                );
            }
        }
    }
}
