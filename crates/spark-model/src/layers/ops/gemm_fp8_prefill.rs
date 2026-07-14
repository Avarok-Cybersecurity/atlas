// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 prefill launchers: NVFP4->FP8 weight pre-dequant, BF16->FP8 activation
//! cast, and the FP8-weight GEMMs. Split from `gemm_dense.rs` (500-LoC cap).

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::{Fp8DenseWeight, QuantizedWeight};

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

/// Quantize a BF16 weight matrix `[N, K]` to FP8 E4M3 `[N, K]` with per-row
/// f32 scales `[N]`. One CTA per row, 256 threads — parallel absmax
/// reduction over K, then per-element saturating cast to E4M3.
///
/// Called **once at model load time**, never on the decode hot path.
///
/// Phase G (DFlash drafter FP8): converts each BF16 q/k/v/o/gate/up/down
/// weight at load time. Decode path then consumes the resulting
/// `Fp8DenseWeight` via `fp8_gemm_n128`.
///
/// Kernel: `quantize_bf16_to_fp8(input, output, row_scales, N, K)` —
/// `kernels/gb10/common/dense_gemv_fp8w.cu:36`.
/// Grid: (N, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn quantize_bf16_to_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    output: DevicePtr,
    row_scales: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(output)
        .arg_ptr(row_scales)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Small-M row-scaled FP8 GEMM (M ≤ 16) — single warp per CTA variant.
///
/// Same math as [`fp8_gemm_n128_row_scaled`] but M_TILE=16 instead of 64,
/// so all M rows are valid (no wasted MMA cycles on bounds-checked rows).
/// Uses 32 threads per CTA (1 warp) instead of 128, so 4× fewer threads
/// for the same useful work. Critical for the DFlash drafter lm_head
/// where M=γ=16 vs N=vocab_size=248320.
///
/// Kernel: `fp8_gemm_t_row_scaled_m16(A, B_fp8, row_scale, C, M, N, K)`.
/// Grid: (ceil(N/128), 1, 1)  Block: (32, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_n128_row_scaled_m16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), 1, 1])
        .block([32, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Row-scaled FP8 GEMM: `C[M, N] = A[M, K] @ (dequant(B_fp8[N, K]) * row_scale[N])`.
///
/// Same tiling and FP8 MMA as `fp8_gemm_n128` (BF16 × FP8 → BF16), with a
/// per-column scale multiply before the BF16 write-out. Consumes the
/// `Fp8DenseWeight` produced by [`crate::weight_map::DenseWeight::quantize_to_fp8`]
/// — the per-row scale on `Fp8DenseWeight` matches the kernel's
/// `row_scale` parameter.
///
/// Phase G (DFlash drafter FP8) hot-path GEMM. Replaces `dense_gemm` on
/// the seven dense-GEMM call sites in `forward_block_layer_pre_attn` /
/// `_post_attn` when `self.quant == DflashQuantization::Fp8Weights`.
///
/// Kernel: `fp8_gemm_t_row_scaled(A, B_fp8, row_scale, C, M, N, K)` —
/// `kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu`.
/// Grid: (ceil(N/128), ceil(M/64), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_n128_row_scaled(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
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
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
