// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Dense BF16 GEMM: C = A @ B^T.
///
/// A: [M, K] row-major (activations)
/// B: [N, K] row-major (weights, HuggingFace layout)
/// C: [M, N] row-major (output)
///
/// Kernel: `dense_gemm_bf16(A, B, C, M, N, K)`
/// Grid: (ceil(N/16), ceil(M/16), 1)  Block: (16, 16, 1)
/// Tensor-core BF16 GEMM: m16n8k16 MMA for 3-5x speedup over scalar.
/// Grid: (ceil(N/64), ceil(M/16), 1), Block: (128, 1, 1)
pub fn dense_gemm_tc(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, 16), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Split-K GEMM: partial products over K_splits chunks, then reduce.
/// Uses FP32 workspace of size K_splits * M * N * 4 bytes.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_splitk(
    gpu: &dyn GpuBackend,
    partial_kernel: KernelHandle,
    reduce_kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    workspace: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    k_splits: u32,
    stream: u64,
) -> Result<()> {
    // Phase 1: partial products
    KernelLaunch::new(gpu, partial_kernel)
        .grid([div_ceil(n, 16), div_ceil(m, 16), k_splits])
        .block([16, 16, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(workspace)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(k_splits)
        .launch(stream)?;
    // Phase 2: reduce and write BF16
    KernelLaunch::new(gpu, reduce_kernel)
        .grid([div_ceil(n, 256), m, 1])
        .block([256, 1, 1])
        .arg_ptr(workspace)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k_splits)
        .launch(stream)
}

pub fn dense_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 16), div_ceil(m, 16), 1])
        .block([16, 16, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Pipelined tensor-core BF16 GEMM — drop-in faster `dense_gemm` (kernel
/// `dense_gemm_bf16_pipelined`): mma.sync.m16n8k16 + cp.async 2-stage, 128x128
/// tile. ~40x the scalar `dense_gemm` on large-M shapes (cosine=1.0, same math).
/// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_bf16_pipelined(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMM: C = A @ dequant(B).
///
/// A: [M, K] BF16 activations
/// B: NVFP4 packed weights (E2M1 + FP8 scales + FP32 per-tensor scale)
/// C: [M, N] BF16 output
///
/// Kernel: `w4a16_gemm(A, B_packed, B_scale, scale2, C, M, N, K)`
/// Grid: (ceil(N/64), ceil(M/64), 1)  Block: (128, 1, 1)
pub fn w4a16_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
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
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMM with N_TILE=128: same kernel signature, wider N tile.
///
/// Grid: (ceil(N/128), ceil(M/64), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
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
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMM v3: MiniMax-only shadow with K_STEP=64 (was 32 in v2).
/// Halves K-iteration count; doubles per-iter MMA count. 1 CTA/SM
/// (was 3 for v2) due to larger SMEM footprint.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_n128_m128_v3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
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
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMM v2: MiniMax-only shadow of `w4a16_gemm_n128_m128`.
///
/// Same CTA tile (M=128, N=128, K_STEP=32) but:
///   - blockDim 256 (8 warps) instead of 128 (4 warps)
///   - 3-stage cp.async pipeline instead of 2-stage
///   - Chunk 0 (rows 0-63) and chunk 1 (rows 64-127) MMAs run in parallel
///     across warps 0-3 and 4-7 instead of being serialized.
///
/// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (256, 1, 1)
/// SMEM: ~42.6 KB → 2 CTAs/SM (vs 3 for v1), but 2× warps/CTA.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_n128_m128_v2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
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
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMM: C = A @ B with 2-M-chunk CTA (M_TILE2=128).
///
/// Halves weight re-reads vs `w4a16_gemm_n128` for large M (ISL > 128):
/// each CTA covers 128 rows of A, loading B once for both 64-row halves.
/// ~2× speedup on qkvz (K=2048, N=12288) at ISL=1016.
///
/// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (128, 1, 1)
/// SMEM: ~29.8 KB → 3 blocks/SM (vs 5 for m64 at ~19.6 KB).
///
/// GRID CONTRACT — N is the FAST axis (blockIdx.x = N-block, blockIdx.y = M-block).
/// Every `w4a16_gemm_t_m128` kernel across all model dirs reads it this way. This
/// launcher is SHARED (qwen3_attention, dense_ffn, qwen3_ssm, nemotron_*), so the
/// axes must NOT be swapped here to suit one model: doing so silently mis-maps every
/// CTA for the other 18 kernels and produces garbage output with no error. If a model
/// wants the m-fast (L2-friendly) order, add a SEPARATELY NAMED kernel + launcher
/// (see `w4a4_gemm_mfast` / `fp8_gemm_t_m128_mfast`) rather than mutating this one.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_n128_m128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
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
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMM — LOSSLESS BF16 prefill variant of `w4a16_gemm_n128_m128`.
///
/// Identical launch config (grid/block/SMEM, M_TILE2=128) and weight layout
/// (transposed NVFP4) to `w4a16_gemm_n128_m128`, but launches the
/// `w4a16_gemm_t_m128_bf16` kernel: FP4→BF16 dequant + BF16 m16n8k16 MMA
/// (FP32 accum), i.e. the base `w4a16_gemm` math at the fast 128x128 tiling.
/// Unlike the default `t_m128` (which crushes weights+acts to FP8 E4M3 on
/// NVIDIA), this preserves prefill outputs bit-for-bit vs the base kernel.
///
/// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_n128_m128_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
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
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
