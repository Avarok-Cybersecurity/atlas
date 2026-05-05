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

/// Fused gate GEMV + topK softmax for M=1 decode.
///
/// Single kernel that computes `gate[num_experts] = A[K] @ B_gate[num_experts, K]`
/// then extracts top-K indices + softmax weights. Saves 1 launch vs separate
/// gate GEMV + topK kernels.
///
/// Grid: (1, 1, 1)  Block: (256, 1, 1) — single CTA, uses shared memory reduction
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_topk_fused(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_weight: &QuantizedWeight,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    k: u32,
    top_k: u32,
    normalize: u32,
    stream: u64,
) -> Result<()> {
    // Dynamic shared memory: K BF16 values for input broadcast
    let smem_bytes = k as usize * 2;
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .shared_mem(smem_bytes as u32)
        .arg_ptr(input)
        .arg_ptr(gate_weight.weight)
        .arg_ptr(gate_weight.weight_scale)
        .arg_f32(gate_weight.weight_scale_2)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_u32(normalize)
        .launch(stream)
}

/// FP8 grouped GEMM for sorted MoE prefill.
///
/// BF16 activations × FP8 E4M3 block-scaled expert weights via pointer table.
/// Grid: (ceil(N/64), max_m_tiles, num_experts)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_fp8_grouped_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,            // [total_tokens, K] BF16
    weight_ptrs: DevicePtr,      // [num_experts] → [N, K] FP8
    scale_ptrs: DevicePtr,       // [num_experts] → [N/128, K/128] BF16
    output: DevicePtr,           // [total_expanded, N] BF16
    expert_offsets: DevicePtr,   // [num_experts + 1]
    sorted_token_ids: DevicePtr, // [total_expanded]
    num_experts: u32,
    n: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W8A16 Transposed GEMM: `C[M,N] = A[M,K] @ dequant(B_t[K,N])` with coalesced reads.
///
/// Uses transposed FP8 weights `B_t[K,N]` and `block_scale_t[K/128, N/128]` for
/// coalesced N-dimension reads. ~14x faster than non-transposed w8a16_gemm at long M.
/// Grid: (ceil(N/64), ceil(M/64), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight_t: DevicePtr,      // [K, N] FP8 transposed
    block_scale_t: DevicePtr, // [K/128, N/128] BF16 transposed
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
        .arg_ptr(weight_t)
        .arg_ptr(block_scale_t)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Transpose FP8 weight matrix on GPU: `B[N,K]` → `B_t[K,N]`.
/// Grid: (ceil(N*K/256), 1, 1)  Block: (256, 1, 1)
pub fn transpose_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr, // [N, K]
    dst: DevicePtr, // [K, N]
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let total = n as u64 * k as u64;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Transpose block scales: [N/128, K/128] → [K/128, N/128].
pub fn transpose_block_scale(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    n_blocks: u32,
    k_blocks: u32,
    stream: u64,
) -> Result<()> {
    let total = n_blocks * k_blocks;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(n_blocks)
        .arg_u32(k_blocks)
        .launch(stream)
}

// ── Unified quantization dispatch ────────────────────────────────────
//
// These wrappers select the correct kernel based on the QuantWeight
// variant. Adding a new quant format requires only a new match arm here.
