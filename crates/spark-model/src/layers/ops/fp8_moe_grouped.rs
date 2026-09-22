// SPDX-License-Identifier: AGPL-3.0-only

//! Launch wrappers for the cross-row GROUPED FP8 MoE decode kernels
//! (`moe_shared_expert_fused_fp8_grouped.cu`, `moe_fp8_grouped_blend.cu`).
//!
//! Rows are grouped by expert (`moe_sort_by_expert`): a CTA owns one expert
//! and streams each weight row once for every row routed to it, instead of the
//! batch2/3 kernels' one CTA per (token, slot). Intermediates are laid out by
//! SORTED position; the blend maps slots back through `token_to_perm`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::Fp8Weight;

/// Rows per register pass inside the grouped kernels — MUST equal `GROUP_ROWS`
/// in the `.cu`; the silu/down launch sizes its dynamic smem from it.
pub const FP8_GROUPED_ROWS_PER_PASS: u32 = 8;

/// Dynamic shared memory the silu/down kernel needs for intermediate width `k`.
pub fn fp8_grouped_silu_down_smem_bytes(k: u32) -> usize {
    FP8_GROUPED_ROWS_PER_PASS as usize * k as usize * 4
}

/// Grouped FP8 gate+up: grid `(ceil(n/8), num_experts+1, 2)`, block 128.
/// `gate_out`/`up_out` rows are SORTED positions (`[num_tokens*top_k, n]`).
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_fp8_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gp_w: DevicePtr,
    gp_s: DevicePtr,
    gate_out: DevicePtr,
    up_w: DevicePtr,
    up_s: DevicePtr,
    up_out: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    sh_gate: &Fp8Weight,
    sh_gate_out: DevicePtr,
    sh_up: &Fp8Weight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    num_experts: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), num_experts + 1, 2])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gp_w)
        .arg_ptr(gp_s)
        .arg_ptr(gate_out)
        .arg_ptr(up_w)
        .arg_ptr(up_s)
        .arg_ptr(up_out)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(sh_gate.weight)
        .arg_ptr(sh_gate.row_scale)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up.weight)
        .arg_ptr(sh_up.row_scale)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_experts)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// Grouped FP8 SiLU+down: grid `(ceil(n/8), num_experts+1, 1)`, block 128,
/// dynamic smem `fp8_grouped_silu_down_smem_bytes(k)`. `k` is the
/// intermediate width; rows of `gate_out`/`up_out`/`output` are SORTED positions.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_fp8_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    down_w: DevicePtr,
    down_s: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down: &Fp8Weight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    num_experts: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), num_experts + 1, 1])
        .block([128, 1, 1])
        .shared_mem(fp8_grouped_silu_down_smem_bytes(k) as u32)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(down_w)
        .arg_ptr(down_s)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.row_scale)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_experts)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// Grouped blend: grid `(ceil(hidden/256), num_tokens, 1)`, block 256.
/// `expert_out` rows are SORTED positions; `token_to_perm[token*top_k+k]` maps
/// a slot to its row. `gate_weight` may be NULL (ungated shared expert).
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_sum_blend_fp8_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    expert_out: DevicePtr,
    expert_weights: DevicePtr,
    token_to_perm: DevicePtr,
    shared_out: DevicePtr,
    input: DevicePtr,
    gate_weight: DevicePtr,
    hidden: u32,
    top_k: u32,
    k: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(hidden, 256), num_tokens, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(expert_out)
        .arg_ptr(expert_weights)
        .arg_ptr(token_to_perm)
        .arg_ptr(shared_out)
        .arg_ptr(input)
        .arg_ptr(gate_weight)
        .arg_u32(hidden)
        .arg_u32(top_k)
        .arg_u32(k)
        .launch(stream)
}
