// SPDX-License-Identifier: AGPL-3.0-only

//! The 17+ row rung of the batched native-FP8 Q/K/V tier (G18 lever A): the
//! 32-row M-tile twin `w8a16_gemm_pipelined_m32_strided`. Child of
//! `qkv_fp8_batch_tests.rs`, sharing its harness; the numerics contract is the
//! GPU oracle (`examples/native_fp8_qkv_batch_microtest`, M32 leg).

use super::{
    BATCH4_K, BATCH16_K, Case, Expect, M16TC_STRIDED_K, M32_STRIDED_K, check_dispatch,
    qkv_phase_launches,
};
use crate::layers::qwen3_attention::attn_ncol_gemv::NcolWidth;

/// THE lever: every width past the GEMVs' MAX_M is ONE strided launch per
/// projection on the twin — the MoE's R=32 (C=16 x k=2) and the R=64 the
/// K=3 ladder or C=32 produce, where the kernel tiles M itself. The shared
/// assertions pin the argument layout (A pitch = hidden, C pitch =
/// per_seq_qkv) and that no projection is launched more than once.
#[test]
fn native_fp8_qkv_takes_the_m32_tile_above_sixteen_rows() {
    for rows in [17, 24, 32, 48, 64] {
        check_dispatch(&Case::new(rows), Expect::Batched(M32_STRIDED_K));
    }
}

/// The twin's lower edge is the GEMVs' MAX_M: 16 stays on batch16, 4 on
/// batch4 — the tiers below are untouched by the new rung.
#[test]
fn native_fp8_qkv_m32_tile_leaves_the_gemv_bands_alone() {
    check_dispatch(&Case::new(16), Expect::Batched(BATCH16_K));
    check_dispatch(&Case::new(4), Expect::Batched(BATCH4_K));
}

/// The `m16` lever and the N-column lever are 5..=16 kernels (MAX_M 16); at
/// 17+ they yield to the twin rather than clamping rows away.
#[test]
fn native_fp8_qkv_m32_tile_outranks_the_sixteen_row_levers_above_their_band() {
    for rows in [17, 32] {
        check_dispatch(&Case::m16_tc(rows), Expect::Batched(M32_STRIDED_K));
        check_dispatch(
            &Case::ncol(rows, NcolWidth::Four),
            Expect::Batched(M32_STRIDED_K),
        );
    }
    // ...and inside their band they still win, so the twin took nothing away.
    check_dispatch(&Case::m16_tc(16), Expect::Batched(M16TC_STRIDED_K));
}

/// The tier's other guards apply to the new rung unchanged: the kill
/// switch, per-row scales and unaligned dims all decline at 32 rows exactly
/// as they do at 4.
#[test]
fn native_fp8_qkv_m32_tile_honours_the_shared_guards() {
    let mut off = Case::new(32);
    off.enabled = false;
    check_dispatch(&off, Expect::Scalar);
    let mut per_row = Case::new(32);
    per_row.format = crate::weight_map::WeightQuantFormat::Fp8PerRow;
    check_dispatch(&per_row, Expect::Scalar);
    let mut unaligned = Case::new(32);
    unaligned.width = 64;
    check_dispatch(&unaligned, Expect::Scalar);
}

/// THE per-row-loop pin, extended past 16: the whole Q/K/V phase costs the
/// same number of launches at 64 rows as at 2. This is the number G18
/// measured moving — 128 graph nodes per attention layer at R=32.
#[test]
fn native_fp8_qkv_phase_launch_count_is_row_independent_past_sixteen() {
    let baseline = qkv_phase_launches(&Case::new(2));
    for rows in [17, 24, 32, 64] {
        assert_eq!(
            qkv_phase_launches(&Case::new(rows)),
            baseline,
            "M32 tile route, rows={rows}"
        );
    }
}
