// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Sorted gate+up GEMV — routed experts only, L2-optimized via expert sorting.
///
/// Grid: (ceil(inter/8), total_expanded, 2)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn moe_sorted_gate_up(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_ptrs: &moe::ExpertPtrTable,
    up_ptrs: &moe::ExpertPtrTable,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    sorted_token_ids: DevicePtr,
    sorted_expert_ids: DevicePtr,
    inter: u32,
    hidden: u32,
    total_expanded: u32,
    stream: u64,
) -> Result<()> {
    let grid_x = div_ceil(inter, 8);
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, total_expanded, 2])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_ptrs.packed_ptrs)
        .arg_ptr(gate_ptrs.scale_ptrs)
        .arg_ptr(gate_ptrs.scale2_vals)
        .arg_ptr(gate_out)
        .arg_ptr(up_ptrs.packed_ptrs)
        .arg_ptr(up_ptrs.scale_ptrs)
        .arg_ptr(up_ptrs.scale2_vals)
        .arg_ptr(up_out)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(sorted_expert_ids)
        .arg_u32(inter)
        .arg_u32(hidden)
        .arg_u32(total_expanded)
        .launch(stream)
}

/// Sorted silu+down GEMV — routed experts only, L2-optimized via expert sorting.
///
/// Grid: (ceil(hidden/8), total_expanded, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn moe_sorted_silu_down(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    down_ptrs: &moe::ExpertPtrTable,
    output: DevicePtr,
    sorted_expert_ids: DevicePtr,
    hidden: u32,
    inter: u32,
    total_expanded: u32,
    stream: u64,
) -> Result<()> {
    let grid_x = div_ceil(hidden, 8);
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, total_expanded, 1])
        .block([128, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(down_ptrs.packed_ptrs)
        .arg_ptr(down_ptrs.scale_ptrs)
        .arg_ptr(down_ptrs.scale2_vals)
        .arg_ptr(output)
        .arg_ptr(sorted_expert_ids)
        .arg_u32(hidden)
        .arg_u32(inter)
        .arg_u32(total_expanded)
        .launch(stream)
}
