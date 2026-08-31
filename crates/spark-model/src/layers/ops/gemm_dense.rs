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

/// `output[m, n] += scale * bf16(input[m, k] @ weight[n, k]^T)` in ONE pass.
///
/// The fused epilogue of [`dense_gemm_tc`], for LoRA's expand+fold. The
/// unfused pair (GEMM into scratch, then `scaled_add`) writes an [m, n]
/// tensor, reads it back, and read-modify-writes the destination; this does
/// the last of those only. On a 27B prefill with n = intermediate = 17408
/// that scratch round-trip dominated — it measured as a 5.6x prefill
/// slowdown with a LoRA adapter resident.
///
/// BIT-IDENTICAL to the unfused pair: the kernel rounds the delta to BF16
/// before applying `scale`, exactly as storing to a BF16 scratch and running
/// `bf16_scaled_add` over it did.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_tc_scaled_acc(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    scale: f32,
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
        .arg_f32(scale)
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

/// Order-preserving register-blocked BF16 GEMM (kernel `dense_gemm_bf16_router`).
///
/// Same math AND the same per-output FP32 accumulation order (strict
/// k = 0..K-1) as the scalar `dense_gemm` — bit-identical output under the
/// kernel dir's `--fmad=false` build (verified 0 differing elements at the
/// router shapes M=4510/M=2255, `[M,2048]x[2048,256]`) — at ~2x the speed via
/// register blocking + vectorized smem staging. This is the ONLY fast GEMM
/// that satisfies the 2026-08-12 router-numerics pin (see
/// `router_gate_gemm_dense`); tensor-core kernels reassociate and stay
/// forbidden there.
///
/// Grid: (ceil(N/64), ceil(M/16), 1)  Block: (16, 16, 1)
pub fn dense_gemm_router(
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

/// Dense BF16 prefill GEMM. Prefer the pipelined tensor-core kernel when the
/// selected target ships it, and retain the scalar kernel as an explicit
/// compatibility fallback for older targets.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_prefill(
    gpu: &dyn GpuBackend,
    fallback_kernel: KernelHandle,
    pipelined_kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if pipelined_kernel.0 != 0 {
        dense_gemm_bf16_pipelined(
            gpu,
            pipelined_kernel,
            input,
            weight,
            output,
            m,
            n,
            k,
            stream,
        )
    } else {
        dense_gemm(gpu, fallback_kernel, input, weight, output, m, n, k, stream)
    }
}

/// W4A16 GEMM: C = A @ dequant(B).
///
/// A: [M, K] BF16 activations
/// B: NVFP4 packed weights (E2M1 + FP8 scales + FP32 per-tensor scale)
/// C: [M, N] BF16 output
///
/// Kernel: `w4a16_gemm(A, B_packed, B_scale, scale2, C, M, N, K)`
/// Grid: (ceil(N/64), ceil(M/64), 1)  Block: (128, 1, 1)
///
/// Also the launcher for `w4a16_gemm_t_k64_n64_p3` — the deep-K twin carries
/// the same 64-wide N tile and the identical argument list, so the two share
/// this grid rather than duplicating it.
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
/// `w4a16_gemm_n128` with an explicit transposed-B ROW STRIDE.
///
/// Needed when N is not a multiple of 16: the kernel's B loads are 16-byte
/// `cp.async`, which requires 16-byte-aligned sources, and row r sits at
/// `r * ldb`. lm_head is the motivating case — its N is the vocab size, 248077
/// on this checkpoint, which is ODD and made 15 of every 16 k-rows fault with
/// CUDA_ERROR_MISALIGNED_ADDRESS (the campaign's long-standing "716").
/// Pass `ldb = align_up(n, 128)` with the pad columns zero-filled.
pub fn w4a16_gemm_n128_ldb(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    ldb: u32,
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
        .arg_u32(ldb)
        .launch(stream)
}

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
    // Packed case: the transposed B rows are exactly N apart.
    w4a16_gemm_n128_ldb(gpu, kernel, input, weight, output, m, n, k, n, stream)
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

/// W4A16 GEMM v2: shadow of `w4a16_gemm_n128_m128` (minimax, step3p7, and —
/// since the 27B port — qwen3.6-27b).
///
/// Same CTA tile (M=128, N=128, K_STEP=32) but:
///   - blockDim 256 (8 warps) instead of 128 (4 warps)
///   - Chunk 0 (rows 0-63) and chunk 1 (rows 64-127) MMAs run in parallel
///     across warps 0-3 and 4-7 instead of being serialized.
///
/// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (256, 1, 1)
/// SMEM: 30,336 B/CTA (2-stage pipeline, padded B_fp8 rows) → 3 CTAs/SM, same
/// footprint as v1 — 768 resident threads/SM vs v1's 384. (An earlier version
/// of this doc claimed 3-stage/42.6 KB/2 CTAs — that described a prototype,
/// not the shipped kernel.)
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
/// `w4a16_gemm_n128_m128_bf16` with an explicit transposed-B row stride, for the
/// LOSSLESS BF16-MMA path. Needed for the same reason as `w4a16_gemm_n128_ldb`:
/// the B loads are 16-byte `cp.async` and lm_head's N is the vocab size (248077,
/// odd), so an unpadded stride misaligns 15 of every 16 k-rows.
pub fn w4a16_gemm_n128_m128_bf16_ldb(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    ldb: u32,
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
        .arg_u32(ldb)
        .launch(stream)
}

/// 8-arg launcher for `w4a16_gemm_t_m128_bf16` (v1) ONLY. The `_v2` sibling's
/// compiled signature has a 9th `ldb` param — launching it through this helper
/// makes cuLaunchKernel read one-past-the-end of the param array (observed as
/// CUDA_ERROR_INVALID_VALUE or a host SIGSEGV depending on the neighboring
/// heap word). Launch v2 via `w4a16_gemm_n128_m128_bf16_ldb` (ldb = N when the
/// transposed twin is unpadded).
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

#[cfg(test)]
#[path = "gemm_dense_tests.rs"]
mod gemm_dense_tests;
