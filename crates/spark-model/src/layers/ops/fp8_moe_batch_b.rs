// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// FP8 fused gate+up GEMV (transposed). K=2 batch.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_fp8_batch2_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_weight_t_ptrs: DevicePtr,
    gate_block_scale_t_ptrs: DevicePtr,
    gate_out: DevicePtr,
    up_weight_t_ptrs: DevicePtr,
    up_block_scale_t_ptrs: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_t: &Fp8Weight,
    sh_gate_out: DevicePtr,
    sh_up_t: &Fp8Weight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, T_BLOCK), 2 * (top_k + 1), 2])
        .block([T_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_weight_t_ptrs)
        .arg_ptr(gate_block_scale_t_ptrs)
        .arg_ptr(gate_out)
        .arg_ptr(up_weight_t_ptrs)
        .arg_ptr(up_block_scale_t_ptrs)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_t.weight)
        .arg_ptr(sh_gate_t.row_scale)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up_t.weight)
        .arg_ptr(sh_up_t.row_scale)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// FP8 fused SiLU+down GEMV (transposed). K=2 batch.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_fp8_batch2_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    weight_t_ptrs: DevicePtr,
    block_scale_t_ptrs: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down_t: &Fp8Weight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    let smem_bytes = (k as usize * std::mem::size_of::<f32>()) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, T_BLOCK), 2 * (top_k + 1), 1])
        .block([T_BLOCK, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(weight_t_ptrs)
        .arg_ptr(block_scale_t_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down_t.weight)
        .arg_ptr(sh_down_t.row_scale)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// FP8 fused gate+up GEMV (transposed). K=3 batch.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_fp8_batch3_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_weight_t_ptrs: DevicePtr,
    gate_block_scale_t_ptrs: DevicePtr,
    gate_out: DevicePtr,
    up_weight_t_ptrs: DevicePtr,
    up_block_scale_t_ptrs: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_t: &Fp8Weight,
    sh_gate_out: DevicePtr,
    sh_up_t: &Fp8Weight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, T_BLOCK), 3 * (top_k + 1), 2])
        .block([T_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_weight_t_ptrs)
        .arg_ptr(gate_block_scale_t_ptrs)
        .arg_ptr(gate_out)
        .arg_ptr(up_weight_t_ptrs)
        .arg_ptr(up_block_scale_t_ptrs)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_t.weight)
        .arg_ptr(sh_gate_t.row_scale)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up_t.weight)
        .arg_ptr(sh_up_t.row_scale)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// FP8 fused SiLU+down GEMV (transposed). K=3 batch.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_fp8_batch3_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    weight_t_ptrs: DevicePtr,
    block_scale_t_ptrs: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down_t: &Fp8Weight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    let smem_bytes = (k as usize * std::mem::size_of::<f32>()) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, T_BLOCK), 3 * (top_k + 1), 1])
        .block([T_BLOCK, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(weight_t_ptrs)
        .arg_ptr(block_scale_t_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down_t.weight)
        .arg_ptr(sh_down_t.row_scale)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}
