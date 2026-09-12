// SPDX-License-Identifier: AGPL-3.0-only

//! The fused attention Q/K/V decode rule (#927), as a pure function: WHICH arm
//! each width takes, which shapes are concatenable at all, and the `ldc ==
//! fused_n` identity the whole layout argument rests on.
//!
//! The numerics are the GPU microtest's job
//! (`examples/native_fp8_attn_qkv_fused_microtest.rs`), which asserts BYTE
//! equality of the three slices and of both consumers rather than a tolerance.
//! The launch-count and allocation contracts are in
//! `trait_impl/multi_seq/w8a8_decode_tests.rs`, where the layer lives.

use super::{attn_qkv_fused_selected, fused_n, qkv_fused_shape_ok};
use spark_runtime::buffers::{ATTN_QKV_FUSED_MAX_M, BufferSizes};

/// Qwen3.8-27B attention: hidden 5120, 24 q-heads / 4 kv-heads, head_dim 256,
/// output gate on. `q_proj` is the interleaved `[Q|gate]` at 12288, k/v are
/// 1024 each, and one sequence's `[Q|K|V]` slot is 14336 BF16 elements.
const H: u32 = 5120;
const Q_N: u32 = 12288;
const KV_N: u32 = 1024;
const LDC: u32 = Q_N + 2 * KV_N;

/// Every clause defaulted to its SELECTING value, so each case perturbs
/// exactly one thing and a failure names the clause.
fn selected(rows: usize) -> bool {
    attn_qkv_fused_selected(rows, Q_N, KV_N, H, LDC, true, true, true)
}

/// The receipt's own width: `q_proj` 12288 plus two 1024s is the 14336 the
/// round-13 trace names as `per_seq_qkv`.
#[test]
fn the_fused_width_is_q_plus_two_kv() {
    assert_eq!(fused_n(Q_N, KV_N), 14336);
    assert_eq!(
        fused_n(Q_N, KV_N),
        LDC,
        "the slot layout IS the fused width"
    );
    // Ungated Q (no `[Q|gate]` doubling) still concatenates.
    assert_eq!(fused_n(6144, KV_N), 8192);
}

/// The band: 5..=16 padded decode rows, the W8A8 cuBLASLt arm's own. Below 5
/// the `w8a16_gemv_batch4_strided` tier owns the width and already makes one
/// weight pass per projection; above 16 the arm is inert — the n=1 GEMV path
/// already runs at 76.6% of HBM (§C.5) and prefill is compute-bound (§A.4).
#[test]
fn the_fused_arm_claims_exactly_the_five_to_sixteen_row_decode_band() {
    for rows in [5, 6, 8, 12, 15, 16] {
        assert!(selected(rows), "rows={rows} is inside the decode band");
    }
    for rows in [1, 2, 4] {
        assert!(!selected(rows), "rows={rows} belongs to the batch4 tier");
    }
    for rows in [17, 25, 32, 1168, 4576] {
        assert!(!selected(rows), "rows={rows} is a prefill width");
    }
    assert_eq!(
        ATTN_QKV_FUSED_MAX_M, 16,
        "the band's upper edge and the row extent `qkv_output` must already \
         hold are ONE constant"
    );
}

/// `ATLAS_ATTN_QKV_FUSED=0`, and every target but hopper.
#[test]
fn the_lever_off_declines_at_every_width() {
    for rows in 1..=32 {
        assert!(!attn_qkv_fused_selected(
            rows, Q_N, KV_N, H, LDC, false, true, true
        ));
    }
}

/// The two runtime absences. `fused_installed` is false on any checkpoint or
/// route the loader did not build the concat for — including a target that
/// declares the lever `false`, where `ATLAS_ATTN_QKV_FUSED=1` arms the arm but
/// no weight exists. `w8a8_selected` false means the three separate GEMMs
/// would NOT have taken the cuBLASLt arm, and fusing there would change the
/// ARITHMETIC of the projections and not only their launch count.
#[test]
fn a_missing_weight_or_an_unselected_w8a8_arm_declines() {
    assert!(!attn_qkv_fused_selected(
        8, Q_N, KV_N, H, LDC, true, false, true
    ));
    assert!(!attn_qkv_fused_selected(
        8, Q_N, KV_N, H, LDC, true, true, false
    ));
}

/// THE LAYOUT IDENTITY. The fused GEMM writes `ceil16(m)` CONTIGUOUS rows of
/// `fused_n` columns at pitch `ldc`; that is the `[n, per_seq_qkv]` slot layout
/// only while `ldc == fused_n`. A wider pitch would leave K and V at the wrong
/// column offsets for the KV-cache write, and a narrower one would overlap the
/// next row — so this is an equality, not a `>=`, and declining is sound.
#[test]
fn a_row_pitch_that_is_not_the_fused_width_declines() {
    assert!(selected(16));
    for ldc in [LDC - 128, LDC + 128, Q_N, LDC * 2] {
        assert!(
            !attn_qkv_fused_selected(16, Q_N, KV_N, H, ldc, true, true, true),
            "ldc={ldc} is not the fused width"
        );
    }
}

/// Both N seams must fall on a 128-block boundary: the concat APPENDS the
/// `[N/128, K/128]` scale grids, and `ceil` of a sum is not the sum of the
/// `ceil`s. K is on the same grid, on the contract side.
#[test]
fn only_whole_block_extents_are_concatenable() {
    assert!(qkv_fused_shape_ok(Q_N, KV_N, H));
    for (q, kv, k) in [
        (Q_N + 64, KV_N, H),
        (Q_N, KV_N + 64, H),
        (Q_N, KV_N, H + 64),
        (0, KV_N, H),
        (Q_N, 0, H),
        (Q_N, KV_N, 0),
    ] {
        assert!(
            !qkv_fused_shape_ok(q, kv, k),
            "q={q} kv={kv} k={k} must decline"
        );
        assert!(!attn_qkv_fused_selected(
            16,
            q,
            kv,
            k,
            fused_n(q, kv),
            true,
            true,
            true
        ));
    }
}

/// THE ARENA PIN. The fused arm adds NO buffer: its `[ceil16(m), 14336]`
/// output IS `qkv_output`, which `BufferSizes` already sizes at
/// `ceil16(max_batch_tokens) * qkv_dim * 2`. This is the assertion that keeps
/// that true — a decode arena must hold the band's full PADDED extent, because
/// cuBLASLt writes `ceil16(m)` rows whatever `m` is.
#[test]
fn the_qkv_arena_already_holds_the_bands_padded_fused_extent() {
    let mut cfg = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    cfg.hidden_size = H as usize;
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 4;
    cfg.head_dim = 256;
    cfg.attn_gated = true;
    // A decode-shaped arena: one row per sequence, 16 sequences.
    let sizes = BufferSizes::from_config(&cfg, ATTN_QKV_FUSED_MAX_M, 4096, 16, 16);
    let need = ATTN_QKV_FUSED_MAX_M * LDC as usize * 2;
    assert_eq!(
        need,
        16 * 14336 * 2,
        "ceil16(16) rows of the fused N, in BF16"
    );
    assert_eq!(
        sizes.qkv_output, need,
        "`qkv_output` IS the fused output — its width is the same \
         `q_proj_dim + 2*kv_dim` and its row extent is the same cuBLASLt M-pad"
    );
    // And the narrow rungs of the band pay for the same padded extent, which
    // is why the buffer is sized once and not per width.
    for rows in [5usize, 8, 12] {
        let s = BufferSizes::from_config(&cfg, rows, 4096, 16, 16);
        assert_eq!(s.qkv_output, need, "rows={rows}: ceil16 is 16 all the way");
    }
}
