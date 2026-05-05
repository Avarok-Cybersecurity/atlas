// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Batched MoE expert W4A16 GEMV: runs top_k expert GEMVs in one launch.
///
/// Uses device-side pointer tables for weight indirection.
/// expert_indices come from GPU top-K (device memory).
///
/// input_stride: 0 = shared input (gate/up), K = per-expert input (down).
///
/// Kernel: `moe_expert_gemv(A, packed_ptrs, scale_ptrs, scale2_vals,
///          C, expert_indices, N, K, top_k, input_stride)`
/// Grid: (ceil(N/4), top_k, 1)  Block: (128, 1, 1)
pub fn moe_expert_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    input_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), top_k, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_u32(input_stride)
        .launch(stream)
}

/// Fused gate+up expert GEMV: both projections in one kernel launch.
///
/// blockIdx.z selects gate (0) vs up (1). Saves 48 launches per decode step.
///
/// Grid: (ceil(N/4), top_k, 2)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gemv_gate_up(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), top_k, 2])
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
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// Register-tiled fused gate+up expert GEMV: 2 output rows per thread.
///
/// Same as gate_up but each thread computes 2 adjacent output rows,
/// reusing the input vector from registers. Doubles weight reads per
/// iteration for better LPDDR5X bandwidth utilization.
///
/// Grid: (ceil(N/8), top_k, 2)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gemv_gate_up_2x(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), top_k, 2])
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
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// Fused SiLU+down expert GEMV: computes silu(gate)*up inline as activation.
///
/// Eliminates separate silu_mul kernel. Reads both gate_out and up_out,
/// computes silu(gate)*up per element, then GEMV with down weights.
///
/// Grid: (ceil(N/4), top_k, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gemv_silu_down(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), top_k, 1])
        .block([128, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// Register-tiled fused SiLU+down expert GEMV: 2 output rows per thread.
///
/// Same as silu_down but each thread computes 2 adjacent output rows,
/// reusing the SiLU(gate)*up activation from registers. Doubles weight
/// reads per iteration for better LPDDR5X bandwidth utilization.
///
/// Grid: (ceil(N/8), top_k, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gemv_silu_down_2x(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), top_k, 1])
        .block([128, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// Fused gate+up expert GEMV with shared expert as extra blockIdx.y slot.
///
/// blockIdx.y < top_k: routed expert (pointer table lookup).
/// blockIdx.y == top_k: shared expert (direct weight pointers).
/// Eliminates separate shared expert gate+up kernel launch.
///
/// Grid: (ceil(N/8), top_k+1, 2)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate: &QuantizedWeight,
    sh_gate_out: DevicePtr,
    sh_up: &QuantizedWeight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), top_k + 1, 2])
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

/// Fused SiLU+down expert GEMV with shared expert as extra blockIdx.y slot.
///
/// blockIdx.y < top_k: routed expert (pointer table + expert gate/up buffers).
/// blockIdx.y == top_k: shared expert (direct pointers + sh_gate_in/up_in).
/// Eliminates separate shared expert silu+down kernel launch.
///
/// Grid: (ceil(N/8), top_k+1, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down: &QuantizedWeight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), top_k + 1, 1])
        .block([128, 1, 1])
        .shared_mem(k * 4) // s_act[K] for precomputed SiLU(gate)*up activation
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

/// Fused gate+up expert GEMV with shared expert for FP8 weights.
///
/// FP8 variant of `moe_expert_gate_up_shared`: uses 2 pointer tables per
/// projection (weight_ptrs + scale_ptrs) instead of NVFP4's 3 (packed +
/// scale + scale2). Shared expert weights are passed as direct Fp8Weight
/// pointers.
///
/// Grid: (ceil(N/8), top_k+1, 2)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_weight_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_out: DevicePtr,
    up_weight_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate: &Fp8Weight,
    sh_gate_out: DevicePtr,
    sh_up: &Fp8Weight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), top_k + 1, 2])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_weight_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_out)
        .arg_ptr(up_weight_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate.weight)
        .arg_ptr(sh_gate.row_scale)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up.weight)
        .arg_ptr(sh_up.row_scale)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// Fused SiLU+down expert GEMV with shared expert for FP8 weights.
///
/// FP8 variant of `moe_expert_silu_down_shared`: uses 2 pointer tables
/// (weight_ptrs + scale_ptrs) instead of NVFP4's 3. Shared expert down
/// weight passed as direct Fp8Weight pointer.
///
/// Grid: (ceil(N/8), top_k+1, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    down_weight_ptrs: DevicePtr,
    down_scale_ptrs: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down: &Fp8Weight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), top_k + 1, 1])
        .block([128, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(down_weight_ptrs)
        .arg_ptr(down_scale_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.row_scale)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}
