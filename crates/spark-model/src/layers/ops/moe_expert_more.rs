// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Fused SiLU+down expert GEMV, wide variant (16 outputs/block for small K).
///
/// Same semantics as `moe_expert_gemv_silu_down` but 4x more outputs per block
/// with sub-warp reduction. Optimal for K<=512 where the narrow kernel has
/// insufficient inner loop iterations for memory latency hiding.
///
/// Fused weighted sum + sigmoid blend + gate scalar GEMV.
///
/// Computes gate_scalar = dot(input, gate_weight) inline, then:
/// `output[j] = sum_e weights[e] * expert_out[e,j] + sigmoid(gate_scalar) * shared_out[j]`
///
/// Each block independently computes the gate scalar dot product (redundant but
/// only 8KB per block for K=2048 — negligible). Eliminates the separate dense_gemv
/// kernel for the shared expert gate scalar (saves 48 graph nodes).
///
/// Grid: (ceil(hidden/256), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_sum_blend(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    expert_out: DevicePtr,
    expert_weights: DevicePtr,
    shared_out: DevicePtr,
    input: DevicePtr,
    gate_weight: DevicePtr,
    hidden: u32,
    top_k: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(hidden, 256), 1, 1])
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
        .launch(stream)
}

// ═══════════════════════════════════════════════════════════════════
// K=2 batch MoE variants — process 2 tokens in single kernel launches
// ═══════════════════════════════════════════════════════════════════

/// Fused gate+up expert GEMV for K=2 tokens with shared expert.
///
/// Expands blockIdx.y from (top_k+1) to 2*(top_k+1) to process both tokens.
/// Token index = blockIdx.y / (top_k+1), expert slot = blockIdx.y % (top_k+1).
/// Shared expert blocks use direct weight pointers; routed use pointer table.
///
/// Grid: (ceil(N/8), 2*(top_k+1), 2)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr, // [2, H] BF16
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr, // [2*top_k, inter] BF16
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,         // [2*top_k, inter] BF16
    expert_indices: DevicePtr, // [2*top_k] u32
    sh_gate: &QuantizedWeight,
    sh_gate_out: DevicePtr, // [2, inter] BF16
    sh_up: &QuantizedWeight,
    sh_up_out: DevicePtr, // [2, inter] BF16
    n: u32,
    k: u32,
    top_k: u32,
    block_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 2 * (top_k + 1), 2])
        .block([block_size, 1, 1])
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
        .launch(stream)
}

/// Fused SiLU+down expert GEMV for K=2 tokens with shared expert.
///
/// Grid: (ceil(N/8), 2*(top_k+1), 1)  Block: (block_size, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr, // [2*top_k, inter] BF16
    up_out: DevicePtr,   // [2*top_k, inter] BF16
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,         // [2*top_k, H] BF16
    expert_indices: DevicePtr, // [2*top_k] u32
    sh_gate_in: DevicePtr,     // [2, inter] BF16
    sh_up_in: DevicePtr,       // [2, inter] BF16
    sh_down: &QuantizedWeight,
    sh_down_out: DevicePtr, // [2, H] BF16
    n: u32,
    k: u32,
    top_k: u32,
    block_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 2 * (top_k + 1), 1])
        .block([block_size, 1, 1])
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
        .launch(stream)
}

/// Fused weighted sum + sigmoid blend for K=2 tokens.
///
/// blockIdx.y = token index (0 or 1). Each block computes gate scalar
/// independently from per-token input and shared gate weight.
///
/// Grid: (ceil(hidden/256), 2, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_sum_blend_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,         // [2, hidden] BF16
    expert_out: DevicePtr,     // [2*top_k, hidden] BF16
    expert_weights: DevicePtr, // [2*top_k] f32
    shared_out: DevicePtr,     // [2, hidden] BF16
    input: DevicePtr,          // [2, K] BF16
    gate_weight: DevicePtr,    // [1, K] BF16 (shared)
    hidden: u32,
    top_k: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(hidden, 256), 2, 1])
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
        .launch(stream)
}

/// Fused gate+up expert GEMV for K=3 tokens with shared expert.
///
/// Grid: (ceil(N/8), 3*(top_k+1), 2)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr, // [3, H] BF16
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr, // [3*top_k, inter] BF16
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,         // [3*top_k, inter] BF16
    expert_indices: DevicePtr, // [3*top_k] u32
    sh_gate: &QuantizedWeight,
    sh_gate_out: DevicePtr, // [3, inter] BF16
    sh_up: &QuantizedWeight,
    sh_up_out: DevicePtr, // [3, inter] BF16
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 3 * (top_k + 1), 2])
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
        .launch(stream)
}

/// Fused SiLU+down expert GEMV for K=3 tokens with shared expert.
///
/// Grid: (ceil(N/8), 3*(top_k+1), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr, // [3*top_k, inter] BF16
    up_out: DevicePtr,   // [3*top_k, inter] BF16
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,         // [3*top_k, H] BF16
    expert_indices: DevicePtr, // [3*top_k] u32
    sh_gate_in: DevicePtr,     // [3, inter] BF16
    sh_up_in: DevicePtr,       // [3, inter] BF16
    sh_down: &QuantizedWeight,
    sh_down_out: DevicePtr, // [3, H] BF16
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 3 * (top_k + 1), 1])
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
        .launch(stream)
}

/// Fused weighted sum + sigmoid blend for K=3 tokens.
///
/// Grid: (ceil(hidden/256), 3, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_sum_blend_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,         // [3, hidden] BF16
    expert_out: DevicePtr,     // [3*top_k, hidden] BF16
    expert_weights: DevicePtr, // [3*top_k] f32
    shared_out: DevicePtr,     // [3, hidden] BF16
    input: DevicePtr,          // [3, K] BF16
    gate_weight: DevicePtr,    // [1, K] BF16 (shared)
    hidden: u32,
    top_k: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(hidden, 256), 3, 1])
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
        .launch(stream)
}

// ── MoE prefill (N-token batch) ──────────────────────────────────
