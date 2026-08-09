// SPDX-License-Identifier: AGPL-3.0-only
//
// Focused dispatch tests for `forward_k2`'s E8M0 guard (`k2_e8m0_needs_per_token`).
// Included via `#[path]` from forward_k2.rs to keep that file ≤500 LoC.

use super::{batch2_block_width, k2_e8m0_needs_per_token};
use crate::weight_map::WeightQuantFormat;

#[test]
fn batch2_block_width_is_one_of_the_two_widths_the_kernel_implements() {
    // The kernel dispatches on blockDim.x and implements exactly 128 and 256.
    // Any other answer would silently fall back to the 128 decomposition and
    // waste the extra warps — the bug this helper exists to prevent.
    for hidden in [1024usize, 2048, 2560, 2688, 2816, 3072, 4096, 5120, 7168] {
        let w = batch2_block_width(hidden);
        assert!(w == 128 || w == 256, "hidden={hidden} → unsupported {w}");
    }
}

#[test]
fn batch2_block_width_widens_at_3072() {
    // 2048-hidden MoE models (qwen3.6-35b-a3b, holo-3.1-35b-a3b, qwen3-vl-30b)
    // stay narrow; the 3072/4096-hidden ones (122B, MiniMax-M2, step3.7,
    // nemotron-super/puzzle, 397B) go wide.
    assert_eq!(batch2_block_width(2048), 128);
    assert_eq!(batch2_block_width(2816), 128);
    assert_eq!(batch2_block_width(3071), 128);
    assert_eq!(batch2_block_width(3072), 256);
    assert_eq!(batch2_block_width(4096), 256);
    assert_eq!(batch2_block_width(5120), 256);
}

#[test]
fn e8m0_takes_per_token_path_not_gs16_batch2() {
    // Mxfp4E8m0 → per-token unified-T (GS32 _e8m0), never the GS16 batch2_t kernel.
    assert!(k2_e8m0_needs_per_token(WeightQuantFormat::Mxfp4E8m0));
}

#[test]
fn nvfp4_still_takes_batch2_kernel() {
    // NVFP4 (GS16) is compatible with the batch2_t kernel → stays on it.
    assert!(!k2_e8m0_needs_per_token(WeightQuantFormat::Nvfp4));
}

#[test]
fn bf16_and_fp8_dispatch_unchanged() {
    // BF16 / FP8 K2 dispatch is decided by earlier gates (bf16_gate_weight_ptrs,
    // fp8_gate_weight_ptrs) — the E8M0 guard must never divert them.
    for f in [
        WeightQuantFormat::Bf16,
        WeightQuantFormat::Fp8SingleScale,
        WeightQuantFormat::Fp8PerRow,
    ] {
        assert!(!k2_e8m0_needs_per_token(f));
    }
}

#[test]
fn no_e8m0_tensor_reaches_gs16_batch2() {
    // Exhaustive over the format enum: the ONLY format routed away from the
    // batch2_t kernel by this guard is Mxfp4E8m0.
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
            k2_e8m0_needs_per_token(f),
            f == WeightQuantFormat::Mxfp4E8m0,
            "only Mxfp4E8m0 diverts to the per-token path"
        );
    }
}

#[test]
fn e8m0_gs32_scale_alloc_is_half_the_gs16_kernel_read() {
    // The hazard the guard prevents: for any (h, inter) an E8M0 (GS32) scale
    // buffer is exactly half the GS16 kernel's read span → 2× over-read.
    for (h, inter) in [(4096usize, 2048usize), (2048, 1408), (5120, 1536)] {
        let e8m0_alloc = inter * (h / 32); // transpose_for_gemm_gs, routed_gs=32
        let gs16_read = inter * (h / 16); // batch2_t kernel, GROUP_SIZE=16
        assert_eq!(gs16_read, 2 * e8m0_alloc);
    }
}
