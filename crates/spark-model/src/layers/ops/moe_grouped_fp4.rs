// SPDX-License-Identifier: AGPL-3.0-only

//! Native-FP4 grouped MoE launcher. Split from `moe_grouped_a.rs` (500-LoC cap).

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::*;

/// Grouped W4A4 expert UP GEMM with fused relu^2 (native FP4 tensor cores).
/// A is the pre-quantized NVFP4 latent (packed E2M1 + per-16 E4M3 scales);
/// B comes from the per-expert NVFP4 pointer tables unchanged.
/// Grid: (ceil(n_out/128), max_m_tiles, num_experts)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a4_grouped_gemm_relu2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_sf: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 128), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_sf)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}
