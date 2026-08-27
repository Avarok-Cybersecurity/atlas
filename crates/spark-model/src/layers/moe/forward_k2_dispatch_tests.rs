// SPDX-License-Identifier: AGPL-3.0-only
//
// Focused dispatch tests for `forward_k2`'s E8M0 guard (`k2_e8m0_needs_per_token`).
// Included via `#[path]` from forward_k2.rs to keep that file ≤500 LoC.

use super::{batch2_block_width, k2_e8m0_needs_per_token};
use crate::weight_map::WeightQuantFormat;

#[test]
fn batch2_block_width_switches_at_3072_for_supported_models() {
    // 2048-hidden MoE models (qwen3.6-35b-a3b, holo-3.1-35b-a3b, qwen3-vl-30b)
    // stay narrow; the 3072/4096-hidden ones (122B, MiniMax-M2, step3.7,
    // nemotron-super/puzzle, 397B) go wide.
    assert_eq!(batch2_block_width(1024), 128);
    assert_eq!(batch2_block_width(2048), 128);
    assert_eq!(batch2_block_width(2816), 128);
    assert_eq!(batch2_block_width(3071), 128);
    assert_eq!(batch2_block_width(3072), 256);
    assert_eq!(batch2_block_width(4096), 256);
    assert_eq!(batch2_block_width(5120), 256);
    assert_eq!(batch2_block_width(7168), 256);
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
