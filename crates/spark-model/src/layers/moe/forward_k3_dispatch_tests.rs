// SPDX-License-Identifier: AGPL-3.0-only
//
// Focused dispatch tests for `forward_k3`'s E8M0 guard (`k3_e8m0_needs_per_token`),
// mirroring forward_k2_dispatch_tests.rs. Included via `#[path]` from forward_k3.rs.

use super::k3_e8m0_needs_per_token;
use crate::weight_map::WeightQuantFormat;

#[test]
fn e8m0_takes_per_token_path_not_gs16_batch3() {
    // Mxfp4E8m0 → per-token unified-T (GS32 _e8m0), never the GS16 batch3_t kernel.
    assert!(k3_e8m0_needs_per_token(WeightQuantFormat::Mxfp4E8m0));
}

#[test]
fn nvfp4_still_takes_batch3_kernel() {
    // NVFP4 (GS16) is compatible with the batch3_t kernel → stays on it.
    assert!(!k3_e8m0_needs_per_token(WeightQuantFormat::Nvfp4));
}

#[test]
fn bf16_and_fp8_k3_dispatch_unchanged() {
    // BF16 / FP8 K3 dispatch is decided by earlier gates (bf16_gate_weight_ptrs,
    // fp8_gate_weight_ptrs) — the E8M0 guard must never divert them.
    for f in [
        WeightQuantFormat::Bf16,
        WeightQuantFormat::Fp8SingleScale,
        WeightQuantFormat::Fp8PerRow,
    ] {
        assert!(!k3_e8m0_needs_per_token(f));
    }
}

#[test]
fn no_e8m0_tensor_reaches_gs16_batch3() {
    // Exhaustive over the format enum: the ONLY format routed away from the
    // batch3_t kernel by this guard is Mxfp4E8m0.
    let all = [
        WeightQuantFormat::Bf16,
        WeightQuantFormat::Fp8PerRow,
        WeightQuantFormat::Fp8BlockScaled,
        WeightQuantFormat::Fp8SingleScale,
        WeightQuantFormat::Nvfp4,
        WeightQuantFormat::Mxfp4E8m0,
    ];
    for f in all {
        assert_eq!(
            k3_e8m0_needs_per_token(f),
            f == WeightQuantFormat::Mxfp4E8m0,
            "only Mxfp4E8m0 diverts to the per-token path"
        );
    }
}

#[test]
fn e8m0_gs32_scale_alloc_is_half_the_gs16_batch3_kernel_read() {
    // The hazard the guard prevents — identical arithmetic to the K=2 case
    // (GROUP_SIZE is per-K-elements, independent of batch width): for any
    // (h, inter) an E8M0 (GS32) scale buffer is exactly half the GS16
    // batch3_t kernel's read span → 2× over-read.
    for (h, inter) in [(4096usize, 2048usize), (2048, 1408), (5120, 1536)] {
        let e8m0_alloc = inter * (h / 32); // transpose_for_gemm_gs, routed_gs=32
        let gs16_read = inter * (h / 16); // batch3_t kernel, GROUP_SIZE=16
        assert_eq!(gs16_read, 2 * e8m0_alloc);
    }
}
