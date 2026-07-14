// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 prefill launchers: NVFP4->FP8 weight pre-dequant, BF16->FP8 activation
//! cast, and the FP8-weight GEMMs. Split from `gemm_dense.rs` (500-LoC cap).

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

use super::*;

/// Pre-dequanted FP8 GEMM (prefill): C = A @ B_fp8.
///
/// A: [M, K] BF16, B_fp8: [N, K] FP8 E4M3 (pre-dequanted from NVFP4), C: [M, N] BF16.
/// Eliminates runtime NVFP4→FP8 dequant — only LOAD + FP8 MMA per K step.
///
/// Grid: (ceil(N/128), ceil(M/64), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Pre-dequant NVFP4 → FP8 E4M3.  One-time conversion at model load.
///
/// Reads B_packed[N, K/2] + B_scale[N, K/GROUP_SIZE] + scale2 → B_fp8[N, K].
///
/// Grid: (ceil(N*K/2 / 256), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
/// `fp8_gemm_t_mfast`: same GEMM as [`fp8_gemm_n128`] with the CTA grid axes
/// swapped so M is the fast axis. The M-blocks that share a B panel then run
/// co-resident and read it from L2 instead of DRAM; see the kernel comment.
pub fn fp8_gemm_n128_mfast(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(m, 64), div_ceil(n, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// `fp8_gemm_t_m128_mfast`: 128-row M tile (2 chunks/CTA), m on the fast axis.
/// Halves the B panel passes relative to [`fp8_gemm_n128_mfast`].
pub fn fp8_gemm_m128_mfast(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(m, 128), div_ceil(n, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// `fp8_fp8_gemm_t_m128_mfast`: FP8 A x FP8 B, 128-row M tile, m on the fast
/// axis. A must already be E4M3 (see `bf16_to_fp8`); the MMA consumed E4M3
/// either way, so pre-casting A is numerically identical to the BF16-A kernel.
pub fn fp8_fp8_gemm_m128_mfast(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(m, 128), div_ceil(n, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

pub fn predequant_nvfp4_to_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    b_packed: DevicePtr,
    b_scale: DevicePtr,
    scale2: f32,
    b_fp8: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let total = n * k / 2;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(b_packed)
        .arg_ptr(b_scale)
        .arg_f32(scale2)
        .arg_ptr(b_fp8)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Convert BF16 activations to FP8 E4M3 for FP8×FP8 GEMM.
///
/// Grid: (ceil(total_elements/2 / 256), 1, 1)  Block: (256, 1, 1)
pub fn bf16_to_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    total_elements: u32,
    stream: u64,
) -> Result<()> {
    let threads_needed = total_elements / 2;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(threads_needed, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(total_elements)
        .launch(stream)
}
