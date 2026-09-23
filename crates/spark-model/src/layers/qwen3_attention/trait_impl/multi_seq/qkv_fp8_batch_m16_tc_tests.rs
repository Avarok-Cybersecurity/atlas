// SPDX-License-Identifier: AGPL-3.0-only

//! The `AVAROK_ATTN_M16_TC` rung of the batched native-FP8 Q/K/V tier (#927,
//! round 6). Child of `qkv_fp8_batch_tests.rs`, sharing its harness — moved
//! here by exact copy when the 17+ row tier pushed the parent past the
//! 500-line cap.

use super::{
    BATCH4_K, BATCH16_K, Case, Expect, M16TC_STRIDED_K, check_dispatch, qkv_phase_launches,
};
use crate::layers::qwen3_attention::attn_ncol_gemv::NcolWidth;

/// ROUND 6's SPLIT. `AVAROK_ATTN_M16_TC` turns THIS tier on — the one that
/// measured −21.7% on the H100 — and it takes exactly the band
/// `w8a16_gemv_batch16_strided` owns: one strided launch per projection, same
/// argument layout, a different kernel.
#[test]
fn native_fp8_qkv_attn_m16_tc_takes_the_batch16_band() {
    for rows in [5, 8, 12, 16] {
        check_dispatch(&Case::m16_tc(rows), Expect::Batched(M16TC_STRIDED_K));
    }
}

/// The tensor-core tier sits AHEAD of the bit-exact N-column tier: an operator
/// who sets `AVAROK_ATTN_M16_TC` is asking for the MMA route explicitly.
#[test]
fn native_fp8_qkv_attn_m16_tc_outranks_the_ncol_tier() {
    let mut case = Case::m16_tc(16);
    case.ncol = Some(NcolWidth::Four);
    check_dispatch(&case, Expect::Batched(M16TC_STRIDED_K));
}

/// Below the band `w8a16_gemv_batch4_strided` still owns the rows — the tier's
/// MAX_M is 16 and its lower edge is where the ALU wall starts, neither of
/// which the lever moves.
#[test]
fn native_fp8_qkv_attn_m16_tc_leaves_small_batches_on_batch4() {
    for rows in [2, 3, 4] {
        check_dispatch(&Case::m16_tc(rows), Expect::Batched(BATCH4_K));
    }
}

/// A shadow without `w8a16_gemm_m16_strided` keeps the batch16 GEMV rather than
/// launching a zero handle.
#[test]
fn native_fp8_qkv_attn_m16_tc_declines_without_its_entry_point() {
    let mut case = Case::m16_tc(16);
    case.m16_tc_handles = false;
    check_dispatch(&case, Expect::Batched(BATCH16_K));
}

/// ...and with the lever unset the tier is invisible, which is the default.
#[test]
fn native_fp8_qkv_without_the_attn_lever_stays_on_batch16() {
    for rows in [5, 16] {
        check_dispatch(&Case::new(rows), Expect::Batched(BATCH16_K));
    }
}

/// The tier is a pure kernel swap, so the phase still costs the same number of
/// launches at 16 rows as at 2 — the per-row-loop pin, on this route too.
#[test]
fn native_fp8_qkv_attn_m16_tc_phase_launch_count_is_row_independent() {
    let baseline = qkv_phase_launches(&Case::new(2));
    for rows in [4, 8, 12, 16] {
        assert_eq!(
            qkv_phase_launches(&Case::m16_tc(rows)),
            baseline,
            "tensor-core route, rows={rows}"
        );
    }
}
