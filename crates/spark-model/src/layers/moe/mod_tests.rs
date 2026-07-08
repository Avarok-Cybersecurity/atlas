// SPDX-License-Identifier: AGPL-3.0-only

//! Tests split out of `moe/mod.rs` for the ≤500 LoC file-size cap.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn test_moe_kernel_loading() {
    let gpu = MockGpuBackend::new();
    assert!(gpu.kernel("gemv", "dense_gemv_bf16").is_ok());
    assert!(gpu.kernel("w4a16_gemv", "w4a16_gemv").is_ok());
    assert!(gpu.kernel("moe_topk", "moe_topk_softmax").is_ok());
    assert!(
        gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_gate_up")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_gate_up_2x")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_silu_down")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_silu_down_2x")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_shared_expert_fused", "moe_expert_gate_up_shared")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_shared_expert_fused", "moe_expert_silu_down_shared")
            .is_ok()
    );
    assert!(
        gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")
            .is_ok()
    );
    // K=2 batch dispatch
    assert!(gpu.kernel("moe_topk", "moe_topk_softmax_batched").is_ok());
}
