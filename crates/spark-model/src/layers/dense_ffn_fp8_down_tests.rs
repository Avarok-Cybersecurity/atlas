// SPDX-License-Identifier: AGPL-3.0-only

//! CPU dispatch tests for the native-FP8 decode down-projection arm (#928).
//!
//! The rule is a pure function precisely so these can run without a GPU: they
//! pin which arm every handle/lever combination selects, and in particular
//! that the split-SiLU default cannot be reached on a target that lacks the
//! kernels it needs.

use super::fp8_down::{Fp8DownArm, fp8_down_arm};

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
