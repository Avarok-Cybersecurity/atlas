// SPDX-License-Identifier: AGPL-3.0-only

//! CPU tests for the 5..16-row decode W8A8 routing rule and the strided-output
//! write-extent arithmetic (#927).
//!
//! Everything here is a PURE function: the selector takes the lever, the kill
//! switch, the shape and the capacities as arguments rather than reading the
//! environment, which is what lets one test process drive both arms without
//! racing the `OnceLock`s production uses.

use super::*;
use crate::weight_map::WeightQuantFormat;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

/// Qwen3.8-27B, the shapes of the round-7 nsys table.
const H: u32 = 5120;
/// SSM fused `in_proj_qkvz` output width.
const QKVZ_N: u32 = 16384;
/// Attention `q_proj` output width (gated `[Q|gate]`, 24 heads × 256 × 2).
const Q_N: u32 = 12288;
/// One sequence's `[Q|K|V]` slot, in BF16 elements: 12288 + 1024 + 1024.
const PER_SEQ_QKV: u32 = 14336;

fn scratch() -> DecodeW8a8Scratch {
    DecodeW8a8Scratch {
        act_fp8: DevicePtr(0x1000),
        act_fp8_bytes: 1 << 20,
        act_scale: DevicePtr(0x2000),
        act_scale_bytes: 1 << 20,
        act_scale_kmajor: DevicePtr(0x3000),
        act_scale_kmajor_bytes: 1 << 20,
        quant_k: KernelHandle(0xA1),
        scale_kmajor_k: KernelHandle(0xA2),
    }
}

/// A contiguous SSM `in_proj_qkvz` at `rows`, with a generous output arena.
fn ssm_qkvz(rows: usize) -> DecodeW8a8Plan {
    DecodeW8a8Plan::contiguous(rows, QKVZ_N, H, 16 * QKVZ_N as usize * 2)
}

/// The attention `q_proj` writing into the `[16, per_seq_qkv]` QKV buffer.
fn attn_q(rows: usize) -> DecodeW8a8Plan {
    DecodeW8a8Plan::strided(rows, Q_N, H, PER_SEQ_QKV, 16 * PER_SEQ_QKV as usize * 2)
}

fn selected(armed: bool, disabled: bool, plan: &DecodeW8a8Plan) -> bool {
    decode_w8a8_selected(
        armed,
        disabled,
        plan,
        WeightQuantFormat::Fp8BlockScaled,
        &scratch(),
    )
}

// ───────────────────────── the row band ─────────────────────────

/// The band is 5..=16 on the PADDED n — the rungs of `padded_batch_n` that the
/// round-7 receipt measured at 357 GB/s on `w8a16_gemv_batch16`.
#[test]
fn the_w8a8_decode_arm_owns_five_to_sixteen_rows() {
    for rows in [5, 6, 8, 12, 16] {
        assert!(selected(true, false, &ssm_qkvz(rows)), "rows={rows}");
        assert!(selected(true, false, &attn_q(rows)), "rows={rows} strided");
    }
}

/// M<=4 stays on `w8a16_gemv_batch4` and M=1 on the bit-exact scalar GEMV: at
/// those widths the GEMV streams each weight once and no MMA tile beats it
/// (round 7 measured the N-column tier 16%/11% SLOWER at M=2/M=4). Keeping the
/// band closed below 5 is also what keeps M=1 decode bit-exact.
#[test]
fn the_w8a8_decode_arm_leaves_four_rows_and_below_alone() {
    for rows in [1, 2, 3, 4] {
        assert!(!selected(true, false, &ssm_qkvz(rows)), "rows={rows}");
        assert!(!selected(true, false, &attn_q(rows)), "rows={rows} strided");
    }
}

#[test]
fn the_w8a8_decode_arm_declines_above_the_band() {
    for rows in [17, 24, 32] {
        assert!(!selected(true, false, &ssm_qkvz(rows)), "rows={rows}");
    }
}

// ───────────────────────── lever + kill switch ─────────────────────────

/// The scoped lever decides which FAMILIES take the arm. `ATLAS_CUBLAS_GEMM=ffn`
/// arming the SSM or attention projections is the exact #917 failure (10.3 GiB
/// of off-ledger weight copies), so this clause is not a formality.
#[test]
fn the_w8a8_decode_arm_needs_its_family_armed() {
    assert!(!selected(false, false, &ssm_qkvz(16)));
    assert!(!selected(false, false, &attn_q(16)));
}

/// `ATLAS_NO_W8A8_DECODE_PROJ` (presence) drops every row count back to the
/// `w8a16_gemv_batch16` tiers, both families at once.
#[test]
fn the_kill_switch_deselects_every_row_count_and_family() {
    for rows in [5, 8, 16] {
        assert!(!selected(true, true, &ssm_qkvz(rows)), "rows={rows}");
        assert!(!selected(true, true, &attn_q(rows)), "rows={rows}");
    }
}

// ───────────────────────── shape + format clauses ─────────────────────────

/// A per-ROW `row_scale` is a different tensor shape; cuBLASLt is told the
/// weight scales are a `[N/128, K/128]` BLK128x128 grid and would read it as
/// garbage.
#[test]
fn the_w8a8_decode_arm_refuses_non_block_scaled_weights() {
    for format in [
        WeightQuantFormat::Fp8PerRow,
        WeightQuantFormat::Fp8SingleScale,
    ] {
        assert!(
            !decode_w8a8_selected(true, false, &ssm_qkvz(16), format, &scratch()),
            "{format:?}"
        );
    }
}

/// `K % 512 == 0` is cuBLAS's own requirement on the weight-scale column
/// stride (`K/128` a multiple of 4); `N % 128` and `K % 128` are the scale
/// grids themselves.
#[test]
fn the_w8a8_decode_arm_refuses_dims_the_scale_grids_do_not_cover() {
    let cases = [
        // N not a multiple of 128.
        DecodeW8a8Plan::contiguous(16, 12280, H, 1 << 24),
        // K not a multiple of 128.
        DecodeW8a8Plan::contiguous(16, QKVZ_N, 5000, 1 << 24),
        // K a multiple of 128 but NOT of 512 — K/128 = 5, stride not a
        // multiple of 4.
        DecodeW8a8Plan::contiguous(16, QKVZ_N, 640, 1 << 24),
    ];
    for plan in cases {
        assert!(!selected(true, false, &plan), "{plan:?}");
    }
}

// ───────────────────────── the strided write extent ─────────────────────────

/// The rule itself, stated once: the library writes `n` elements of EACH of
/// the `m_pad` columns, so the last element touched is at
/// `(m_pad - 1) * ldc + n`. Not `m_pad * ldc` (counts a trailing gap the GEMM
/// never writes) and not `m_pad * n` (ignores the pitch).
#[test]
fn the_strided_write_extent_is_the_last_row_plus_its_width() {
    // Contiguous: the pitch IS the width, so the extent is the whole block.
    assert_eq!(strided_out_extent_elems(16, 5120, 5120), 16 * 5120);
    // Strided q_proj: 15 full slots plus q_proj's 12288 columns.
    assert_eq!(
        strided_out_extent_elems(16, PER_SEQ_QKV, Q_N),
        15 * PER_SEQ_QKV as usize + Q_N as usize
    );
    // A single row touches exactly its own width, gaps excluded.
    assert_eq!(
        strided_out_extent_elems(1, PER_SEQ_QKV, Q_N),
        Q_N as usize,
        "one row must not be charged for a pitch it never crosses"
    );
}

/// THE PHANTOM-ROW GUARD. `cublas_fp8_proj_prequant` hands cuBLASLt `ceil16(M)`
/// and those phantom rows ARE written. At a strided output they land in decode
/// slots `rows..16` — slots belonging to sequences that are not in this step,
/// which is in-bounds and harmless ONLY while the buffer really has 16 slots.
/// A buffer sized for the LIVE rows is therefore refused, not silently
/// overrun: at rows=5 the padded write reaches element
/// `15 * per_seq_qkv + q_proj_dim`, three times past a 5-slot buffer.
#[test]
fn a_strided_output_sized_for_the_live_rows_only_is_refused() {
    for rows in [5, 8, 12] {
        let live_only = rows * PER_SEQ_QKV as usize * 2;
        let plan = DecodeW8a8Plan::strided(rows, Q_N, H, PER_SEQ_QKV, live_only);
        assert!(
            !selected(true, false, &plan),
            "rows={rows}: padded write extent {} B must not fit in {live_only} B",
            plan.write_extent_bytes()
        );
        // The same shape with all 16 slots present is accepted.
        let full =
            DecodeW8a8Plan::strided(rows, Q_N, H, PER_SEQ_QKV, 16 * PER_SEQ_QKV as usize * 2);
        assert!(selected(true, false, &full), "rows={rows} with 16 slots");
    }
}

/// K and V sit at byte offsets inside each slot, so their buffer capacity is
/// measured from THEIR base — the tail of the arena after `q_proj_bytes` /
/// `+ kv_bytes`. The extent rule must be applied to each projection's own base,
/// and the last one (V) is the tightest.
#[test]
fn each_projection_is_bounded_from_its_own_base() {
    const KV_N: u32 = 1024;
    let arena = 16 * PER_SEQ_QKV as usize * 2;
    let q_bytes = Q_N as usize * 2;
    let kv_bytes = KV_N as usize * 2;
    let v = DecodeW8a8Plan::strided(
        16,
        KV_N,
        H,
        PER_SEQ_QKV,
        arena - q_bytes - kv_bytes, // V's base is that far into the arena
    );
    assert!(selected(true, false, &v), "V must fit: {v:?}");
    assert_eq!(
        v.write_extent_bytes(),
        (15 * PER_SEQ_QKV as usize + KV_N as usize) * 2
    );
    // One BF16 element less of arena and it must refuse rather than write past.
    let tight = DecodeW8a8Plan::strided(16, KV_N, H, PER_SEQ_QKV, v.write_extent_bytes() - 2);
    assert!(!selected(true, false, &tight));
}

/// A leading dimension shorter than the column is rejected by the library;
/// catching it here names the bug instead of surfacing a cuBLAS status code.
#[test]
fn a_row_pitch_narrower_than_the_output_is_refused() {
    let plan = DecodeW8a8Plan::strided(16, Q_N, H, Q_N - 128, 1 << 30);
    assert!(!selected(true, false, &plan));
}

// ───────────────────────── scratch capacity ─────────────────────────

/// The scratch is READ at `ceil16(M)` rows, not `M` — the phantom activation
/// rows are zeroed but they are read. A scratch sized for the live rows only
/// must decline.
#[test]
fn the_activation_scratch_is_checked_at_the_padded_row_count() {
    let mut s = scratch();
    let plan = ssm_qkvz(5);
    s.act_fp8_bytes = 5 * H as usize; // live rows only
    assert!(!decode_w8a8_selected(
        true,
        false,
        &plan,
        WeightQuantFormat::Fp8BlockScaled,
        &s
    ));
    s.act_fp8_bytes = 16 * H as usize;
    assert!(decode_w8a8_selected(
        true,
        false,
        &plan,
        WeightQuantFormat::Fp8BlockScaled,
        &s
    ));
}

/// Missing quantizer, missing k-major adapter kernel, missing k-major scratch,
/// or a k-major scratch too small — each alone must drop the arm. Handing
/// cuBLASLt the quantizer's `[M, K/128]` order instead of the token-contiguous
/// one it documents is fast and WRONG (H100 2026-09-11: rel_rms 7.7e-2 /
/// ~33 000 BF16 ULP on identical FP8 bytes), so falling back is the only safe
/// answer when any piece is absent.
#[test]
fn a_missing_scale_layout_adapter_drops_the_arm() {
    let plan = ssm_qkvz(16);
    let mutate: [(&str, fn(&mut DecodeW8a8Scratch)); 5] = [
        ("no quantizer", |s| s.quant_k = KernelHandle(0)),
        ("no fp8 scratch", |s| s.act_fp8 = DevicePtr(0)),
        ("no scale scratch", |s| s.act_scale = DevicePtr(0)),
        ("no kmajor kernel", |s| s.scale_kmajor_k = KernelHandle(0)),
        ("kmajor too small", |s| s.act_scale_kmajor_bytes = 4),
    ];
    for (what, apply) in mutate {
        let mut s = scratch();
        apply(&mut s);
        // The two k-major clauses only bind when the k-major layout is the
        // selected one, which is the default and what production runs.
        if !cublas_scale_layout_kmajor() && what.contains("kmajor") {
            continue;
        }
        assert!(
            !decode_w8a8_selected(true, false, &plan, WeightQuantFormat::Fp8BlockScaled, &s),
            "{what}"
        );
    }
}

/// The M pad is 16 across the whole band, which is why one buffer sized for 16
/// slots serves every rung — and why the phantom-row question is the same
/// question at rows=5 and rows=15.
#[test]
fn every_row_in_the_band_pads_to_sixteen() {
    for rows in 5..=16usize {
        assert_eq!(ssm_qkvz(rows).m_pad(), 16, "rows={rows}");
    }
}
