// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Fused gate+up expert GEMV for N-token prefill with shared expert.
///
/// Grid: (ceil(inter/8), num_tokens*(top_k+1), 2)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr, // [num_tokens, H] BF16
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr, // [num_tokens*top_k, inter] BF16
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,         // [num_tokens*top_k, inter] BF16
    expert_indices: DevicePtr, // [num_tokens*top_k] u32
    sh_gate: &QuantizedWeight,
    sh_gate_out: DevicePtr, // [num_tokens, inter] BF16
    sh_up: &QuantizedWeight,
    sh_up_out: DevicePtr, // [num_tokens, inter] BF16
    n: u32,
    k: u32,
    top_k: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), num_tokens * (top_k + 1), 2])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(gate_out)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate.weight)
        .arg_ptr(sh_gate.weight_scale)
        .arg_f32(sh_gate.weight_scale_2)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up.weight)
        .arg_ptr(sh_up.weight_scale)
        .arg_f32(sh_up.weight_scale_2)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// Fused SiLU+down expert GEMV for N-token prefill with shared expert.
///
/// Grid: (ceil(hidden/64), num_tokens*(top_k+1), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr, // [num_tokens*top_k, inter] BF16
    up_out: DevicePtr,   // [num_tokens*top_k, inter] BF16
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,         // [num_tokens*top_k, H] BF16
    expert_indices: DevicePtr, // [num_tokens*top_k] u32
    sh_gate_in: DevicePtr,     // [num_tokens, inter] BF16
    sh_up_in: DevicePtr,       // [num_tokens, inter] BF16
    sh_down: &QuantizedWeight,
    sh_down_out: DevicePtr, // [num_tokens, H] BF16
    n: u32,
    k: u32,
    top_k: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), num_tokens * (top_k + 1), 1])
        .block([128, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.weight_scale)
        .arg_f32(sh_down.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// Fused weighted sum + sigmoid blend for N-token prefill.
///
/// Grid: (ceil(hidden/256), num_tokens, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_sum_blend_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,         // [num_tokens, hidden] BF16
    expert_out: DevicePtr,     // [num_tokens*top_k, hidden] BF16
    expert_weights: DevicePtr, // [num_tokens*top_k] f32
    shared_out: DevicePtr,     // [num_tokens, hidden] BF16
    input: DevicePtr,          // [num_tokens, K] BF16
    gate_weight: DevicePtr,    // [1, K] BF16 (shared)
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
        .arg_ptr(shared_out)
        .arg_ptr(input)
        .arg_ptr(gate_weight)
        .arg_u32(hidden)
        .arg_u32(top_k)
        .arg_u32(k)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// W4A16 dual GEMV: two projections sharing the same BF16 input, one launch.
///
/// blockIdx.z selects projection 0 vs 1. Both N dimensions must be equal.
///
/// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    weight2: &QuantizedWeight,
    output2: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_ptr(weight2.weight)
        .arg_ptr(weight2.weight_scale)
        .arg_f32(weight2.weight_scale_2)
        .arg_ptr(output2)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMV with fused SiLU input: silu(gate)*up as activation, GEMV with down weights.
///
/// Reads `gate_out[K]` and `up_out[K]` BF16, computes silu(gate)*up per element
/// inline, then multiplies by dequanted NVFP4 weights. Eliminates silu_mul kernel.
///
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_silu_input(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Sigmoid-gated blend reading gate scalar from device memory.
///
/// `output[i] += sigmoid(bf16_to_f32(*gate_ptr)) * src[i]`
///
/// Kernel: `bf16_sigmoid_blend_device(output, src, gate_ptr, n)`
/// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
pub fn sigmoid_blend_device(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    src: DevicePtr,
    gate_ptr: DevicePtr,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(src)
        .arg_ptr(gate_ptr)
        .arg_u32(num_elements)
        .launch(stream)
}

// ── MoE grouped GEMM (future) ──────────────────────────────────
