// SPDX-License-Identifier: AGPL-3.0-only

//! Fuse independent route casts and activation replication before expert GEMM.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Same buffers, indexing, and casts as stage_routing + replicate_a_bf16.
/// Launch geometry matches activation replication; no expert grid changes.
#[allow(clippy::too_many_arguments)]
pub fn exl3_moe_stage_ingress(
    gpu: &dyn GpuBackend,
    input_bf16: DevicePtr,
    indices_u32: DevicePtr,
    probs_f32: DevicePtr,
    b_indices: DevicePtr,
    b_weights: DevicePtr,
    out_f16: DevicePtr,
    local_start: usize,
    num_local: usize,
    num_tokens: usize,
    top_k: usize,
    hidden: usize,
    stream: u64,
) -> Result<()> {
    let h = gpu.kernel("exl3_matmul", "exl3_moe_stage_ingress")?;
    let slots = num_tokens * top_k;
    let total = slots * hidden;
    let grid = div_ceil(total as u32, 256).clamp(1, 4096);
    KernelLaunch::new(gpu, h)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(indices_u32)
        .arg_ptr(probs_f32)
        .arg_ptr(b_indices)
        .arg_ptr(b_weights)
        .arg_ptr(out_f16)
        .arg_i32(local_start as i32)
        .arg_i32(num_local as i32)
        .arg_i32(top_k as i32)
        .arg_u64(hidden as u64)
        .arg_u64(slots as u64)
        .arg_u64(total as u64)
        .launch(stream)
}

/// Keep the prior two launches available for one-variable serving A/B runs.
pub(super) fn fused_ingress_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ATLAS_NO_EXL3_FUSED_INGRESS").as_deref() != Ok("1"))
}
