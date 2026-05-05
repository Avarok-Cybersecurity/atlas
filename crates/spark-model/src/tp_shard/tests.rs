// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Validate the slice-offset math without exercising the GPU.
#[test]
fn column_parallel_offsets() {
    // Out=4096, in=3072, BF16, tp=4, rank=2.
    // local_out = 1024; row_bytes = 6144; local_bytes = 6_291_456;
    // src_offset = 2 * 1024 * 6144 = 12_582_912.
    let out_dim = 4096usize;
    let in_dim = 3072usize;
    let tp_size = 4usize;
    let tp_rank = 2usize;
    let local_out = out_dim / tp_size;
    let row_bytes = in_dim * BF16_BYTES;
    let local_bytes = local_out * row_bytes;
    let src_offset = tp_rank * local_out * row_bytes;
    assert_eq!(local_out, 1024);
    assert_eq!(row_bytes, 6144);
    assert_eq!(local_bytes, 6_291_456);
    assert_eq!(src_offset, 12_582_912);
}

#[test]
fn row_parallel_offsets() {
    // Out=3072, in=4096, BF16, tp=2, rank=1.
    // local_in = 2048; local_row_bytes = 4096;
    // col_offset_bytes = 1 * 4096 = 4096.
    // For row 0: src_off = 0*8192 + 4096 = 4096; dst_off = 0.
    let _out_dim = 3072usize;
    let in_dim = 4096usize;
    let tp_size = 2usize;
    let tp_rank = 1usize;
    let local_in = in_dim / tp_size;
    let local_row_bytes = local_in * BF16_BYTES;
    let src_row_bytes = in_dim * BF16_BYTES;
    let col_offset_bytes = tp_rank * local_row_bytes;
    assert_eq!(local_in, 2048);
    assert_eq!(local_row_bytes, 4096);
    assert_eq!(src_row_bytes, 8192);
    assert_eq!(col_offset_bytes, 4096);

    // Row 5: src_off = 5*8192 + 4096 = 45_056; dst_off = 5*4096 = 20_480.
    let r = 5usize;
    assert_eq!(r * src_row_bytes + col_offset_bytes, 45_056);
    assert_eq!(r * local_row_bytes, 20_480);
}

#[test]
fn divisibility_check() {
    // The non-divisible cases are caught by `ensure!` at runtime, not at
    // compile time — verify the math fails the precondition.
    let out_dim = 4097usize;
    let tp_size = 4usize;
    assert_ne!(out_dim % tp_size, 0);
}
