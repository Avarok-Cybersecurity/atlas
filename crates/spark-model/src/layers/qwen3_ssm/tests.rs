// SPDX-License-Identifier: AGPL-3.0-only

//! Extracted piecewise from `qwen3_ssm/mod.rs` (500-LoC cap).

use super::*;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn test_ssm_state_allocation_sizes() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let nv = config.linear_num_value_heads; // 32
    let vd = config.linear_value_head_dim; // 128
    let nk = config.linear_num_key_heads; // 16
    let kd = config.linear_key_head_dim; // 128
    let d_conv = config.linear_conv_kernel_dim; // 4

    let h_bytes = nv * vd * kd * 4;
    assert_eq!(h_bytes, 32 * 128 * 128 * 4); // 2 MB

    // conv_dim = 2*key_dim + value_dim = 2*2048 + 4096 = 8192
    let conv_dim = nk * kd * 2 + nv * vd;
    let conv_bytes = conv_dim * d_conv * 4;
    assert_eq!(conv_bytes, 8192 * 4 * 4); // 128 KB

    // Verify allocations
    let gpu = MockGpuBackend::new();
    let h_state = gpu.alloc(h_bytes).unwrap();
    let conv_state = gpu.alloc(conv_bytes).unwrap();
    assert!(!h_state.is_null());
    assert!(!conv_state.is_null());
}
