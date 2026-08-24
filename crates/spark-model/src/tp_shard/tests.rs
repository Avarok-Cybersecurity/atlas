// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
// Internal slice-plan helpers live in the `gdn` submodule (pub(crate));
// import them directly rather than re-exporting from the lib surface.
use super::gdn::{CopyOp, segment_copy_plan};

#[test]
fn column_parallel_offsets() {
    let gpu = MockGpuBackend::new();
    let values: Vec<u16> = (0..12).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let src = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(&bytes, src).unwrap();

    let (dst, local_out, local_in) =
        shard_dense_bf16(src, 4, 3, TpShardKind::ColumnParallel, 1, 2, &gpu).unwrap();
    let mut got = vec![0u8; 6 * BF16_BYTES];
    gpu.copy_d2h(dst, &mut got).unwrap();

    assert_eq!((local_out, local_in), (2, 3));
    assert_eq!(got, bytes[6 * BF16_BYTES..].to_vec());
    assert_eq!(gpu.d2d_count(), 1);
}

#[test]
fn row_parallel_offsets() {
    let gpu = MockGpuBackend::new();
    let values: Vec<u16> = (0..12).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let src = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(&bytes, src).unwrap();

    let (dst, local_out, local_in) =
        shard_dense_bf16(src, 3, 4, TpShardKind::RowParallel, 1, 2, &gpu).unwrap();
    let mut got = vec![0u8; 6 * BF16_BYTES];
    gpu.copy_d2h(dst, &mut got).unwrap();
    let got: Vec<u16> = got
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
        .collect();

    assert_eq!((local_out, local_in), (3, 2));
    assert_eq!(got, [2, 3, 6, 7, 10, 11]);
    assert_eq!(gpu.d2d_count(), 3);
}

#[test]
fn divisibility_check() {
    let gpu = MockGpuBackend::new();
    let src = gpu.alloc(64).unwrap();
    let column = shard_dense_bf16(src, 5, 4, TpShardKind::ColumnParallel, 0, 2, &gpu)
        .unwrap_err()
        .to_string();
    let row = shard_dense_bf16(src, 4, 5, TpShardKind::RowParallel, 0, 2, &gpu)
        .unwrap_err()
        .to_string();
    let rank = shard_dense_bf16(src, 4, 4, TpShardKind::ColumnParallel, 2, 2, &gpu)
        .unwrap_err()
        .to_string();

    assert!(column.contains("out_dim 5 not divisible by tp_size 2"));
    assert!(row.contains("in_dim 5 not divisible by tp_size 2"));
    assert!(rank.contains("tp_rank 2 >= tp_size 2"));
    assert_eq!(
        gpu.alloc_count(),
        1,
        "invalid plans must not allocate output"
    );
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
    let bytes: Vec<u8> = full.iter().flat_map(|v| v.to_le_bytes()).collect();
    let gpu = MockGpuBackend::new();
    let src = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(&bytes, src).unwrap();
    let gpu_slice = |d: &TpGdnDims| -> Vec<u16> {
        let (dst, rows, cols) = shard_gdn_qkvz_rows(src, d, &gpu).unwrap();
        assert_eq!((rows, cols), (d.local_qkvz_out(), h));
        let mut out = vec![0u8; rows * cols * BF16_BYTES];
        gpu.copy_d2h(dst, &mut out).unwrap();
        out.chunks_exact(2)
            .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
            .collect()
    };

    let r0 = gpu_slice(&d0);
    let r1 = gpu_slice(&d1);

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
    assert_eq!(gpu.d2d_count(), 8, "four segments per rank");
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
    let gpu = MockGpuBackend::new();

    let norm: Vec<u16> = (0..128).collect();
    let norm_bytes: Vec<u8> = norm.iter().flat_map(|v| v.to_le_bytes()).collect();
    let norm_src = gpu.alloc(norm_bytes.len()).unwrap();
    gpu.copy_h2d(&norm_bytes, norm_src).unwrap();
    let (norm_dst, norm_len) =
        shard_gdn_value_vector(norm_src, &d, d.vd, BF16_BYTES, &gpu).unwrap();
    let mut norm_got = vec![0u8; norm_len * BF16_BYTES];
    gpu.copy_d2h(norm_dst, &mut norm_got).unwrap();
    assert_eq!(norm_len, 64);
    assert_eq!(norm_got, norm_bytes[64 * BF16_BYTES..]);

    let a_log: Vec<u32> = (100..108).collect();
    let a_log_bytes: Vec<u8> = a_log.iter().flat_map(|v| v.to_le_bytes()).collect();
    let a_log_src = gpu.alloc(a_log_bytes.len()).unwrap();
    gpu.copy_h2d(&a_log_bytes, a_log_src).unwrap();
    let (a_log_dst, a_log_len) = shard_gdn_value_vector(a_log_src, &d, 1, 4, &gpu).unwrap();
    let mut a_log_got = vec![0u8; a_log_len * 4];
    gpu.copy_d2h(a_log_dst, &mut a_log_got).unwrap();
    assert_eq!(a_log_len, 4);
    assert_eq!(a_log_got, a_log_bytes[4 * 4..]);
    assert_eq!(gpu.d2d_count(), 2);
}

/// Dense-Holo dims (Holo-3.1-0.8B, kernels/gb10/holo-3.1-0.8b/MODEL.toml):
/// 16 key heads x kd=128, 16 value heads x vd=128, h=1024, tp=2. Exercises the
/// qwen35_dense.rs LinearAttention TP wiring's copy plan at REAL checkpoint
/// shapes (equal Q/K/V/Z segment widths — unlike the synthetic dims above,
/// where V/Z are 4x wider than Q/K).
#[test]
fn qkvz_plan_holo_dense_dims() {
    let mk = |tp_rank: usize| TpGdnDims {
        tp_rank,
        tp_size: 2,
        h: 1024,
        kd: 128,
        vd: 128,
        local_nk: 8,
        full_nk: 16,
        local_nv: 8,
        full_nv: 16,
    };
    let d = mk(1);
    assert_eq!(d.full_key_dim(), 2048);
    assert_eq!(d.full_value_dim(), 2048);
    assert_eq!(d.full_conv_dim(), 6144);
    assert_eq!(d.full_qkvz_out(), 8192);
    assert_eq!(d.local_qkvz_out(), 4096);
    assert_eq!(d.qkvz_segments(), [2048, 2048, 2048, 2048]);

    let row_bytes = d.h * BF16_BYTES; // 2048
    let (ops, local_rows) =
        segment_copy_plan(&d.qkvz_segments(), row_bytes, d.tp_rank, d.tp_size).unwrap();
    assert_eq!(local_rows, 4096);
    // Full segment starts (rows): Q@0, K@2048, V@4096, Z@6144.
    // Rank-1 halves: Q[1024..2048], K[3072..4096], V[5120..6144], Z[7168..8192];
    // packed dst rows: 0, 1024, 2048, 3072.
    let want = [
        CopyOp {
            src_off: 1024 * row_bytes,
            dst_off: 0,
            len: 1024 * row_bytes,
        },
        CopyOp {
            src_off: 3072 * row_bytes,
            dst_off: 1024 * row_bytes,
            len: 1024 * row_bytes,
        },
        CopyOp {
            src_off: 5120 * row_bytes,
            dst_off: 2048 * row_bytes,
            len: 1024 * row_bytes,
        },
        CopyOp {
            src_off: 7168 * row_bytes,
            dst_off: 3072 * row_bytes,
            len: 1024 * row_bytes,
        },
    ];
    assert_eq!(ops, want);

    // Rank 0 takes each segment's FIRST half; both ranks tile every segment.
    let d0 = mk(0);
    let (ops0, local_rows0) =
        segment_copy_plan(&d0.qkvz_segments(), row_bytes, d0.tp_rank, d0.tp_size).unwrap();
    assert_eq!(local_rows0, 4096);
    for (i, op) in ops0.iter().enumerate() {
        assert_eq!(op.src_off, i * 2048 * row_bytes);
        assert_eq!(op.dst_off, i * 1024 * row_bytes);
        assert_eq!(op.len, 1024 * row_bytes);
    }
}

/// A non-divisible segment must be rejected loudly, not silently corrupt.
#[test]
fn segment_plan_rejects_indivisible() {
    // 33 rows can't split evenly across tp=2.
    let r = segment_copy_plan(&[32, 33], 128, 0, 2);
    assert!(r.is_err());
}
