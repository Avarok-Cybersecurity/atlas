// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
// Internal slice-plan helpers live in the `gdn` submodule (pub(crate));
// import them directly rather than re-exporting from the lib surface.
use super::gdn::{CopyOp, segment_copy_plan};

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

// ════════════════════════════════════════════════════════════════════
// GDN HeadParallel — segmented-slice math (no GPU; validates the copy plan
// the device slicers execute, plus a CPU reference re-concat).
// ════════════════════════════════════════════════════════════════════

/// Synthetic GDN dims: full_nk=4, kd=8, full_nv=8, vd=16, tp=2 → local
/// nk=2, nv=4. Q/K width = 32, V/Z width = 128. conv_dim = 2*32+128 = 192.
fn synth_dims(tp_rank: usize) -> TpGdnDims {
    TpGdnDims {
        tp_rank,
        tp_size: 2,
        h: 64,
        kd: 8,
        vd: 16,
        local_nk: 2,
        full_nk: 4,
        local_nv: 4,
        full_nv: 8,
    }
}

#[test]
fn gdn_dims_derived() {
    let d = synth_dims(0);
    assert_eq!(d.full_key_dim(), 32);
    assert_eq!(d.local_key_dim(), 16);
    assert_eq!(d.full_value_dim(), 128);
    assert_eq!(d.local_value_dim(), 64);
    assert_eq!(d.full_conv_dim(), 2 * 32 + 128); // 192
    assert_eq!(d.local_conv_dim(), 2 * 16 + 64); // 96
    assert_eq!(d.full_qkvz_out(), 192 + 128); // 320
    assert_eq!(d.local_qkvz_out(), 96 + 64); // 160
    assert_eq!(d.qkv_segments(), [32, 32, 128]);
    assert_eq!(d.qkvz_segments(), [32, 32, 128, 128]);
}

/// The QKVZ segmented plan must slice Q, K, V, Z each by the local head range
/// and pack them back-to-back — NOT take the first `out/tp` contiguous rows.
#[test]
fn qkvz_segment_plan_is_segmented_not_contiguous() {
    let d = synth_dims(1); // rank 1 of 2
    let row_bytes = d.h * BF16_BYTES; // 64 * 2 = 128
    let (ops, local_rows) =
        segment_copy_plan(&d.qkvz_segments(), row_bytes, d.tp_rank, d.tp_size).unwrap();
    assert_eq!(local_rows, d.local_qkvz_out()); // 160

    // Full segment starts (rows): Q@0, K@32, V@64, Z@192.
    // Rank-1 local halves: Q[16..32], K[48..64], V[128..192], Z[256..320].
    // Packed dst (rows): 0, 16, 32, 96.
    let want = [
        CopyOp {
            src_off: 16 * row_bytes,
            dst_off: 0,
            len: 16 * row_bytes,
        },
        CopyOp {
            src_off: 48 * row_bytes,
            dst_off: 16 * row_bytes,
            len: 16 * row_bytes,
        },
        CopyOp {
            src_off: 128 * row_bytes,
            dst_off: 32 * row_bytes,
            len: 64 * row_bytes,
        },
        CopyOp {
            src_off: 256 * row_bytes,
            dst_off: 96 * row_bytes,
            len: 64 * row_bytes,
        },
    ];
    assert_eq!(ops, want);

    // A naive contiguous "first half for rank 0 / second half for rank 1"
    // would put rank 1's Q source at row 160, NOT 16 — prove we differ.
    assert_ne!(ops[0].src_off, 160 * row_bytes);
}

/// Rank 0 + rank 1 slices must exactly tile the full buffer with no overlap
/// and no gap, per segment. Reference re-concat on a synthetic u16 buffer.
#[test]
fn qkvz_two_rank_reconcat_tiles_full() {
    let d0 = synth_dims(0);
    let d1 = synth_dims(1);
    let segs = d0.qkvz_segments();
    let full_rows: usize = segs.iter().sum(); // 320
    let h = d0.h;

    // Synthetic full weight: row r filled with value r (u16), h cols each.
    let full: Vec<u16> = (0..full_rows)
        .flat_map(|r| std::iter::repeat_n(r as u16, h))
        .collect();

    // CPU reference slice using the plan (byte offsets → row indices).
    let cpu_slice = |d: &TpGdnDims| -> Vec<u16> {
        let row_bytes = h * BF16_BYTES;
        let (ops, local_rows) = segment_copy_plan(&segs, row_bytes, d.tp_rank, d.tp_size).unwrap();
        let mut out = vec![0u16; local_rows * h];
        for op in &ops {
            let src_row = op.src_off / row_bytes;
            let dst_row = op.dst_off / row_bytes;
            let nrows = op.len / row_bytes;
            for i in 0..nrows {
                let s = (src_row + i) * h;
                let dd = (dst_row + i) * h;
                out[dd..dd + h].copy_from_slice(&full[s..s + h]);
            }
        }
        out
    };

    let r0 = cpu_slice(&d0);
    let r1 = cpu_slice(&d1);

    // Expected per-segment source rows for each rank.
    // Q: r0=[0..16]  r1=[16..32]
    // K: r0=[32..48] r1=[48..64]
    // V: r0=[64..128] r1=[128..192]
    // Z: r0=[192..256] r1=[256..320]
    let expect_rows_r0: Vec<u16> = (0..16)
        .chain(32..48)
        .chain(64..128)
        .chain(192..256)
        .map(|r| r as u16)
        .collect();
    let expect_rows_r1: Vec<u16> = (16..32)
        .chain(48..64)
        .chain(128..192)
        .chain(256..320)
        .map(|r| r as u16)
        .collect();

    let rows_of = |v: &[u16]| -> Vec<u16> { v.iter().step_by(h).copied().collect() };
    assert_eq!(rows_of(&r0), expect_rows_r0);
    assert_eq!(rows_of(&r1), expect_rows_r1);

    // Union of both ranks (per segment) == the full set of rows: no
    // overlap, no gap.
    let mut union: Vec<u16> = rows_of(&r0);
    union.extend(rows_of(&r1));
    union.sort_unstable();
    let all: Vec<u16> = (0..full_rows as u16).collect();
    assert_eq!(union, all);
}

/// conv1d uses the SAME [Q|K|V] 3-segment split but with row width = d_conv.
#[test]
fn conv_segment_plan_uses_qkv_segments() {
    let d = synth_dims(0);
    let d_conv = 4usize;
    let row_bytes = d_conv * BF16_BYTES; // 8
    let (ops, local_rows) =
        segment_copy_plan(&d.qkv_segments(), row_bytes, d.tp_rank, d.tp_size).unwrap();
    assert_eq!(local_rows, d.local_conv_dim()); // 96
    // Rank 0: Q[0..16], K[32..48], V[64..128]; packed at 0,16,32.
    let want = [
        CopyOp {
            src_off: 0,
            dst_off: 0,
            len: 16 * row_bytes,
        },
        CopyOp {
            src_off: 32 * row_bytes,
            dst_off: 16 * row_bytes,
            len: 16 * row_bytes,
        },
        CopyOp {
            src_off: 64 * row_bytes,
            dst_off: 32 * row_bytes,
            len: 64 * row_bytes,
        },
    ];
    assert_eq!(ops, want);
}

/// BA is per-group interleaved but the rank boundary lands on a group
/// boundary → a single contiguous slice. rank r → rows [r*2*local_nv, ...).
#[test]
fn ba_single_segment_group_aligned() {
    let d = synth_dims(1);
    let row_bytes = d.h * BF16_BYTES;
    let (ops, local_rows) =
        segment_copy_plan(&[2 * d.full_nv], row_bytes, d.tp_rank, d.tp_size).unwrap();
    assert_eq!(local_rows, 2 * d.local_nv); // 8
    // 2*full_nv = 16 rows; rank 1 → rows [8..16).
    assert_eq!(ops.len(), 1);
    assert_eq!(ops[0].src_off, 8 * row_bytes);
    assert_eq!(ops[0].dst_off, 0);
    assert_eq!(ops[0].len, 8 * row_bytes);
    // Group size = 2*vpg = 2*(nv/nk) = 2*2 = 4 rows; 8 is a multiple → the
    // slice starts on a group boundary.
    let vpg = d.full_nv / d.full_nk;
    assert_eq!((8) % (2 * vpg), 0);
}

/// Value-vector 1D shard: norm [full_nv*vd] BF16 and a_log/dt_bias [full_nv]
/// FP32, sliced on the value-head axis.
#[test]
fn value_vector_offsets() {
    let d = synth_dims(1);
    // norm: unit = vd = 16, bf16. full = 8*16 = 128, local = 4*16 = 64.
    let full_norm = d.full_nv * d.vd;
    let local_norm = d.local_nv * d.vd;
    assert_eq!(full_norm, 128);
    assert_eq!(local_norm, 64);
    let norm_src_off = d.tp_rank * local_norm * BF16_BYTES; // rank1 = 64*2
    assert_eq!(norm_src_off, 128);
    // a_log: unit = 1, fp32. full = 8, local = 4.
    let f32_bytes = 4usize;
    let a_log_src_off = d.tp_rank * d.local_nv * f32_bytes; // rank1 = 4*4
    assert_eq!(a_log_src_off, 16);
}

/// A non-divisible segment must be rejected loudly, not silently corrupt.
#[test]
fn segment_plan_rejects_indivisible() {
    // 33 rows can't split evenly across tp=2.
    let r = segment_copy_plan(&[32, 33], 128, 0, 2);
    assert!(r.is_err());
}
