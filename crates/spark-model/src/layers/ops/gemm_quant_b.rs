// SPDX-License-Identifier: AGPL-3.0-only

//! MoE-grouped GEMM + transpose/scale utility kernel-launch wrappers.
//!
//! Split out of `gemm_quant.rs` to keep both files under the 500 LoC cap.
//! Pure thin kernel-launch wrappers (no ordered-launch dependency between
//! them); re-exported alongside `gemm_quant::*` via `ops.rs`.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

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

/// W8A8 + FP32 epilogue grouped MoE GEMM (vLLM-equivalent).
///
/// A_fp8 must be pre-quantized via `per_token_group_quant_fp8`. Both
/// `a_scale` (per-token, FP32) and `b_scale` (per-block, BF16) are applied
/// in the FP32 epilogue per K=128 block.
#[allow(clippy::too_many_arguments)]
pub fn moe_w8a8_grouped_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,            // [total_tokens, K] FP8 E4M3
    a_scale: DevicePtr,          // [total_tokens, K/128] FP32
    weight_ptrs: DevicePtr,      // [num_experts] → [N, K] FP8
    scale_ptrs: DevicePtr,       // [num_experts] → [N/128, K/128] BF16
    output: DevicePtr,           // [total_expanded, N] BF16
    expert_offsets: DevicePtr,   // [num_experts + 1]
    sorted_token_ids: DevicePtr, // [total_expanded] or NULL
    num_experts: u32,
    n: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a_fp8)
        .arg_ptr(a_scale)
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

/// BF16 grouped GEMM for sorted MoE prefill (FP8-dequant-on-load path).
///
/// BF16 activations × BF16 expert weights via pointer table. No scale.
/// Used when expert weights have been dequanted from FP8 to BF16 at load
/// time (ATLAS_FP8_DEQUANT_MOE_TO_BF16=1). Eliminates the per-layer 0.989
/// cosine ceiling that comes from FP8 quantization itself.
///
/// Grid: (ceil(N/64), max_m_tiles, num_experts)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_bf16_grouped_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,            // [total_tokens, K] BF16
    weight_ptrs: DevicePtr,      // [num_experts] → [N, K] BF16
    output: DevicePtr,           // [total_expanded, N] BF16
    expert_offsets: DevicePtr,   // [num_experts + 1]
    sorted_token_ids: DevicePtr, // [total_expanded] or NULL
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

/// Widen an FP8 block-scale tensor to FP32 on the GPU.
///
/// `src` is `[total]` BF16 (`in_is_fp32 == false`) or FP32 (`in_is_fp32 ==
/// true`); `dst` is `[total]` FP32. Lossless BF16→FP32 widen / straight copy.
/// Run once at load so downstream FP8 block-scale kernels read `const float*`.
/// Grid: (ceil(total/256), 1, 1)  Block: (256, 1, 1)
pub fn widen_block_scale_f32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    total: u32,
    in_is_fp32: bool,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(total)
        .arg_u32(in_is_fp32 as u32)
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
