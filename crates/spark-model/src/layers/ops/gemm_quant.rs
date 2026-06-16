// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// FP8×FP8 GEMM: A [M, K] FP8 × B [N, K] FP8 → C [M, N] BF16.
///
/// Both A (activations) and B (weights) are pre-converted FP8 E4M3.
/// No BF16→FP8 conversion in inner loop — pure MMA throughput.
/// Grid: (ceil(N/128), ceil(M/64))  Block: (128, 1, 1)
pub fn fp8_fp8_gemm_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,
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
        .arg_ptr(a_fp8)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// M128 variant of fp8_gemm_n128: halves B re-reads for large M (ISL > 128).
///
/// Each CTA covers 128 rows of A, loading B once for both 64-row halves.
/// ~2× speedup on out_proj (K=value_dim, N=h) at ISL≥128.
///
/// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_n128_m128(
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
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// M128 variant of fp8_fp8_gemm_n128: halves B re-reads for large M (ISL > 128).
///
/// Each CTA covers 128 rows of A, loading B once for both 64-row halves.
/// ~2× speedup on Q/K/V projections (FP8 activations × FP8 weights) at ISL≥128.
/// Compact FP8 A smem → 6 blocks/SM vs 3 for fp8_gemm_t_m128.
///
/// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn fp8_fp8_gemm_n128_m128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(a_fp8)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Dense BF16 GEMV (M=1): C = A @ B^T for single-row activations.
///
/// A: [1, K] BF16, B: [N, K] BF16, C: [1, N] BF16.
/// 8 outputs/block, 32 threads (1 warp) per output. Single-warp shuffle reduction.
///
/// Kernel: `dense_gemv_bf16(A, B, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn dense_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Dense FP8-weight GEMV (M=1): C = A @ (dequant(B_fp8) * row_scale).
///
/// A: `[1, K]` BF16, B: `[N, K]` FP8 E4M3, row_scale: `[N]` f32, C: `[1, N]` BF16.
/// Halves weight bandwidth vs dense_gemv (1 byte/weight instead of 2).
/// 4 outputs/block, 64 threads (2 warps) per output.
///
/// Kernel: `dense_gemv_fp8w(A, B, row_scale, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn dense_gemv_fp8w(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W8A16 GEMV (M=1): C = A @ dequant_lut(B_fp8) * row_scale for FP8 E4M3 weights.
///
/// A: `[1, K]` BF16, B: `[N, K]` FP8 E4M3 bytes, row_scale: `[N]` f32, C: `[1, N]` BF16.
/// Uses a 256-entry E4M3 LUT in shared memory for branchless dequant (no hardware
/// FP4/FP8 conversion PTX needed — works on SM121 without `cvt.rn.satfinite`).
/// 4 outputs/block, 64 threads (2 warps) per output. Cross-warp smem reduction.
///
/// Kernel: `w8a16_gemv(A, B, row_scale, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    row_scale: DevicePtr,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(row_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W8A16 GEMM (M>1): `C[M,N] = A[M,K] @ dequant(B[N,K])` for prefill.
///
/// Uses 256-entry E4M3 LUT + BF16 2D block scales.
/// Grid: (ceil(N/64), ceil(M/64), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Per-token-per-128-K-group FP8 activation quantization. Output: A_fp8
/// [M, K] FP8 E4M3 + a_scale [M, K/128] FP32. Matches vLLM's
/// `per_token_group_quant_fp8`.
///
/// Grid: (K/128, M, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn per_token_group_quant_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input_bf16: DevicePtr,
    output_fp8: DevicePtr,
    a_scale: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    // Grid: (M, K/128, 1). Putting M on grid X (max 2^31-1) avoids the
    // 65535 limit on grid Y for large MoE total_expanded counts.
    KernelLaunch::new(gpu, kernel)
        .grid([m, k / 128, 1])
        .block([128, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(output_fp8)
        .arg_ptr(a_scale)
        .arg_u32(m)
        .arg_u32(k)
        .launch(stream)
}

/// W8A8 + FP32 epilogue GEMM with per-token activation scales and
/// per-block weight scales — vLLM-equivalent FP8 numerics.
///
///   C[M, N] = bf16( Σ_g (FP8 MMA over K-group g) × a_scale[M, g] × b_scale[N/128, g] )
///
/// Inputs:
///   - `a_fp8`     [M, K] FP8 E4M3
///   - `a_scale`   [M, K/128] FP32 (from per_token_group_quant_fp8)
///   - `b_fp8`     [N, K] FP8 E4M3
///   - `b_scale`   [N/128, K/128] BF16 (existing checkpoint layout)
///   - `output`    [M, N] BF16
///
/// Grid: (ceil(N/128), ceil(M/64), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_t_blockscaled(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,
    a_scale: DevicePtr,
    b_fp8: DevicePtr,
    b_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(a_fp8)
        .arg_ptr(a_scale)
        .arg_ptr(b_fp8)
        .arg_ptr(b_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
