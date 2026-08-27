// SPDX-License-Identifier: AGPL-3.0-only
//
// Focused dispatch tests for `forward_k3`'s E8M0 guard (`k3_e8m0_needs_per_token`),
// mirroring forward_k2_dispatch_tests.rs. Included via `#[path]` from forward_k3.rs.

use super::k3_e8m0_needs_per_token;
use crate::weight_map::WeightQuantFormat;

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
