// SPDX-License-Identifier: AGPL-3.0-only

//! CPU dispatch tests for the native-FP8 decode down-projection arm (#928).
//!
//! The rule is a pure function precisely so these can run without a GPU: they
//! pin which arm every handle/lever combination selects, and in particular
//! that the split-SiLU default cannot be reached on a target that lacks the
//! kernels it needs.

use super::fp8_down::{
    FP8_DOWN_FUSED_SILU_KEY, FP8_DOWN_FUSED_SILU_MSG, FP8_DOWN_SPLIT_SILU_KEY,
    FP8_DOWN_SPLIT_SILU_MSG, Fp8DownArm, fp8_down_arm, log_fp8_down_route,
};
use crate::layers::ops::ModelStats;

/// Handles a fully-equipped target resolves: dual, fused silu, `moe_silu_mul`
/// and the plain scalar GEMV all present.
fn full(lever: bool) -> Fp8DownArm {
    fp8_down_arm(true, true, true, true, true, lever)
}

#[test]
fn the_split_silu_arm_is_the_default_on_a_complete_target() {
    assert_eq!(full(true), Fp8DownArm::SplitSilu);
}

#[test]
fn the_kill_switch_restores_the_fused_kernel() {
    // `ATLAS_NO_DECODE_SPLIT_SILU` clears `decode_split_silu`; the fused
    // kernel is still resolved, so the arm must go back to it bit-for-bit
    // rather than fall all the way to the 4-launch path.
    assert_eq!(full(false), Fp8DownArm::FusedSilu);
}

#[test]
fn a_target_without_the_staging_kernels_cannot_take_the_split_arm() {
    // No `moe_silu_mul` -> nothing can stage silu(gate)*up.
    assert_eq!(
        fp8_down_arm(true, true, true, false, true, true),
        Fp8DownArm::FusedSilu
    );
    // No plain `w8a16_gemv` -> nothing can consume a staged activation.
    assert_eq!(
        fp8_down_arm(true, true, true, true, false, true),
        Fp8DownArm::FusedSilu
    );
}

#[test]
fn losing_both_fused_kernels_falls_to_the_per_projection_path() {
    assert_eq!(
        fp8_down_arm(true, true, false, false, true, true),
        Fp8DownArm::PerProjection
    );
    assert_eq!(
        fp8_down_arm(true, true, false, true, false, true),
        Fp8DownArm::PerProjection
    );
}

#[test]
fn the_split_arm_survives_a_missing_fused_kernel() {
    // The whole point of #928: a target that never built
    // `w8a16_gemv_silu_input` still gets the fast down projection, where
    // before it dropped to four launches.
    assert_eq!(
        fp8_down_arm(true, true, false, true, true, true),
        Fp8DownArm::SplitSilu
    );
}

#[test]
fn the_dual_gemv_gates_both_fused_arms() {
    // Without `w8a16_gemv_dual` there is no staged gate/up pair for either
    // fused arm to read, whatever else resolved.
    for lever in [true, false] {
        assert_eq!(
            fp8_down_arm(true, false, true, true, true, lever),
            Fp8DownArm::PerProjection
        );
    }
}

#[test]
fn a_non_silu_activation_never_reaches_the_fused_arms() {
    // GeLU has no fused down kernel; the SwiGLU-shaped arms must not claim it.
    assert_eq!(
        fp8_down_arm(false, true, true, true, true, true),
        Fp8DownArm::PerProjection
    );
}

// ── The `log_fp8_down_route` route lines (H100 round 9) ────────────────────
//
// What these DO NOT cover: reaching `log_fp8_down_route` from a real decode
// step needs a `DenseFfnLayer` with resolved FP8 kernel handles and a GPU
// backend (`dense_ffn_m16_tc_tests.rs::run_tiled` shows the shape of that
// harness for the sibling M16-TC tier). What IS covered is the reporting
// layer itself: given an `Fp8DownArm`, does the right key fire exactly once,
// does the OTHER arm's key stay untouched, and does the text name the lever,
// the kernel and the off switch — the actual round-9 complaint ("There is no
// route line for the split-SiLU default ... An operator cannot confirm from
// the log which `down` path a server is running").

#[test]
fn split_silu_arm_fires_its_route_exactly_once() {
    let stats = ModelStats::new();
    log_fp8_down_route(&stats, Fp8DownArm::SplitSilu);
    log_fp8_down_route(&stats, Fp8DownArm::SplitSilu);
    assert!(
        !stats.once(FP8_DOWN_SPLIT_SILU_KEY),
        "two calls on the same arm must consume the latch exactly once"
    );
}

#[test]
fn fused_silu_arm_fires_its_route_exactly_once() {
    let stats = ModelStats::new();
    log_fp8_down_route(&stats, Fp8DownArm::FusedSilu);
    log_fp8_down_route(&stats, Fp8DownArm::FusedSilu);
    assert!(!stats.once(FP8_DOWN_FUSED_SILU_KEY));
}

#[test]
fn per_projection_arm_emits_no_route_line() {
    // The 4-launch fallback is not governed by ATLAS_NO_DECODE_SPLIT_SILU —
    // it runs the same regardless of the lever — so there is nothing to
    // disambiguate and neither key should be touched.
    let stats = ModelStats::new();
    log_fp8_down_route(&stats, Fp8DownArm::PerProjection);
    assert!(stats.once(FP8_DOWN_SPLIT_SILU_KEY));
    assert!(stats.once(FP8_DOWN_FUSED_SILU_KEY));
}

#[test]
fn each_arm_fires_its_own_key_and_not_the_others() {
    let stats = ModelStats::new();
    log_fp8_down_route(&stats, Fp8DownArm::SplitSilu);
    assert!(
        stats.once(FP8_DOWN_FUSED_SILU_KEY),
        "SplitSilu must not also consume the FusedSilu key"
    );

    let stats = ModelStats::new();
    log_fp8_down_route(&stats, Fp8DownArm::FusedSilu);
    assert!(
        stats.once(FP8_DOWN_SPLIT_SILU_KEY),
        "FusedSilu must not also consume the SplitSilu key"
    );
}

#[test]
fn split_silu_message_names_the_default_lever_kernel_and_receipt() {
    let msg = FP8_DOWN_SPLIT_SILU_MSG;
    assert!(
        msg.contains("split-SiLU + w8a16_gemv (default"),
        "must name the default"
    );
    assert!(
        msg.contains("ATLAS_NO_DECODE_SPLIT_SILU restores the fused kernel"),
        "must name the off switch"
    );
    assert!(msg.contains("1.79x"), "must cite the speed receipt");
    assert!(
        msg.contains("843 -> 1513 GB/s"),
        "must cite the measured rates"
    );
    assert!(
        msg.contains("native_fp8_ffn_down_gemv_microtest"),
        "must cite the microtest that measured it"
    );
    assert!(
        msg.contains("NOT bit-identical"),
        "must state the numerics caveat, not just the speed win"
    );
    assert!(
        msg.contains("max_ulp=31195"),
        "must cite the ULP receipt for the numerics change"
    );
}

#[test]
fn fused_silu_message_names_what_it_restores_and_what_it_departs_from() {
    let msg = FP8_DOWN_FUSED_SILU_MSG;
    assert!(
        msg.contains("ATLAS_NO_DECODE_SPLIT_SILU"),
        "must name the lever"
    );
    assert!(
        msg.contains("w8a16_gemv_silu_input"),
        "must name the fused kernel actually running"
    );
    assert!(
        msg.contains("split-SiLU + w8a16_gemv default"),
        "must name the default this line means the server is NOT running, \
         so either line alone tells an operator both paths' names"
    );
}
