// SPDX-License-Identifier: AGPL-3.0-only

//! Native FP4 (NVFP4 / mxf4nvf4) prefill launchers: activation quantization
//! and the W4A4 tensor-core GEMMs. Split from `gemm_dense.rs` (500-LoC cap).

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

use super::*;

/// Quantize a BF16 [M, K] matrix to NVFP4 (single-level, scale2=1.0): packed E2M1
/// `[M, K/2]` + per-group-16 E4M3 scales `[M, K/16]`. Prepares W4A4 prefill
/// activations. Grid = M rows (one block/row), block 128 (threads stride groups).
#[allow(clippy::too_many_arguments)]
pub fn quantize_bf16_to_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    packed_out: DevicePtr,
    scale_out: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([m, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(packed_out)
        .arg_ptr(scale_out)
        .arg_f32(1.0) // scale2 = 1.0 (single-level; activation range fits E4M3 group scales)
        .arg_u32(m) // kernel's N param = rows = tokens
        .arg_u32(k)
        .launch(stream)
}

/// W4A4 NVFP4 prefill GEMM (native FP4 tensor cores, sm_121a). Activation is
/// pre-quantized NVFP4 (`a_packed`/`a_scale`, scale2=1.0); weight is the native
/// NVFP4 `QuantizedWeight`. Output BF16 [M, N]. See kernels/.../w4a4_gemm.cu.
/// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (256, 1, 1).
#[allow(clippy::too_many_arguments)]
pub fn w4a4_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([256, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_ptr(output)
        .arg_f32(1.0) // scaleA2 (activation single-level)
        .arg_f32(weight.weight_scale_2) // scaleB2 (weight per-tensor)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMM with N_TILE=128: same kernel signature, wider N tile.
///
/// Grid: (ceil(N/128), ceil(M/64), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
/// `w4a4_gemm_mfast`: same W4A4 GEMM with M on the fast grid axis, so the
/// M-blocks sharing a B panel are co-resident and B streams from DRAM once.
pub fn w4a4_gemm_mfast(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(m, 128), div_ceil(n, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_ptr(output)
        .arg_f32(1.0) // scaleA2 (activation single-level)
        .arg_f32(weight.weight_scale_2) // scaleB2 (weight per-tensor)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
