// SPDX-License-Identifier: AGPL-3.0-only

//! Staging launches ahead of the expert GEMMs: the DECODE ingress (route
//! casts and activation replication fused into one launch) and the PREFILL
//! tier's sort-output staging into the fused kernel's LOCAL-expert-ordered
//! forms.

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

/// Stage Atlas's `moe_sort_by_expert` outputs into the fused kernel's
/// LOCAL-expert-ordered forms (plain launch; kernel contract at its
/// definition in `exl3_matmul.cu`).
#[allow(clippy::too_many_arguments)]
pub fn exl3_moe_stage_sorted(
    gpu: &dyn GpuBackend,
    token_to_perm: DevicePtr,
    probs_f32: DevicePtr,
    expert_offsets: DevicePtr,
    token_sorted: DevicePtr,
    weight_sorted: DevicePtr,
    expert_count: DevicePtr,
    local_start: usize,
    num_local: usize,
    top_k: usize,
    s: usize,
    stream: u64,
) -> Result<()> {
    let h = gpu.kernel("exl3_matmul", "exl3_moe_stage_sorted")?;
    let work = s.max(num_local + 1);
    let grid = div_ceil(work as u32, 256).clamp(1, 4096);
    KernelLaunch::new(gpu, h)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(token_to_perm)
        .arg_ptr(probs_f32)
        .arg_ptr(expert_offsets)
        .arg_ptr(token_sorted)
        .arg_ptr(weight_sorted)
        .arg_ptr(expert_count)
        .arg_i32(local_start as i32)
        .arg_i32(num_local as i32)
        .arg_i32(top_k as i32)
        .arg_u64(s as u64)
        .launch(stream)
}

/// Keep the prior two launches available for one-variable serving A/B runs.
pub(super) fn fused_ingress_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ATLAS_NO_EXL3_FUSED_INGRESS").as_deref() != Ok("1"))
}
