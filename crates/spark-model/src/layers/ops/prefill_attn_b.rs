// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Write K/V to paged NVFP4 cache (E2M1 data + per-group FP8 scales).
///
/// Kernel: `reshape_and_cache_flash_nvfp4(key, value, k_cache, v_cache,
///          slot_mapping, num_kv_heads, head_dim, block_size,
///          key_stride, value_stride, block_stride_bytes, data_section_bytes)`
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
pub fn reshape_and_cache_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    key: DevicePtr,
    value: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    key_stride: u32,
    value_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(slot_mapping)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_u32(key_stride)
        .arg_u32(value_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

/// Compute max absolute value of a BF16 buffer into a device-side f32.
///
/// Used for FP8 KV cache online scale calibration: accumulates max |K| and
/// max |V| during warmup tokens. The output f32 is updated via atomicMax,
/// so the caller must initialize it to 0.0 before the first call.
///
/// Kernel: `bf16_absmax(data, out_max, n_elems)`
/// Grid: (ceil(n_elems / (256*2)), 1, 1)  Block: (256, 1, 1)
pub fn bf16_absmax(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    out_max: DevicePtr,
    n_elems: u32,
    stream: u64,
) -> Result<()> {
    // Each thread handles multiple pairs; use enough blocks to cover the buffer.
    // 256 threads per block, each reads ~8 pairs in the inner loop.
    let grid_x = (n_elems as u64).div_ceil(256 * 2).min(256) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(data)
        .arg_ptr(out_max)
        .arg_u32(n_elems)
        .launch(stream)
}

/// Paged decode attention (NVFP4 KV cache, single/multi sequence).
///
/// Kernel: `paged_decode_attn_nvfp4(Q, K_cache, V_cache, O, block_tables,
///          seq_lens, max_blocks_per_seq, num_q_heads, num_kv_heads,
///          head_dim, block_size, inv_sqrt_d, q_stride,
///          block_stride_bytes, data_section_bytes)`
/// Grid: (num_q_heads, num_seqs, 1)  Block: (256, 1, 1)
pub fn paged_decode_attn_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_seqs: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    q_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_seqs, 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(q_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

/// Split-K paged decode attention (NVFP4 KV cache).
///
/// Partitions the KV sequence across `num_splits` CTAs per (q_head, seq).
/// Each CTA computes partial softmax + weighted output, written to `workspace`.
///
/// Grid: (num_q_heads, num_splits, num_seqs)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_splitk_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    workspace: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    num_splits: u32,
    q_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    num_seqs: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_splits, num_seqs])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(workspace)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(num_splits)
        .arg_u32(q_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

/// Reduce split-K partials into final BF16 output.
///
/// Grid: (num_q_heads, num_seqs, 1)  Block: (32, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_reduce_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    workspace: DevicePtr,
    output: DevicePtr,
    seq_lens: DevicePtr,
    num_q_heads: u32,
    head_dim: u32,
    num_splits: u32,
    num_seqs: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_seqs, 1])
        .block([32, 1, 1])
        .arg_ptr(workspace)
        .arg_ptr(output)
        .arg_ptr(seq_lens)
        .arg_u32(num_q_heads)
        .arg_u32(head_dim)
        .arg_u32(num_splits)
        .launch(stream)
}

/// Split-K paged decode attention (FP8 KV cache).
///
/// Partitions the KV sequence across `num_splits` CTAs per (q_head, seq).
/// Each CTA computes partial softmax + weighted output, written to `workspace`.
///
/// Grid: (num_q_heads, num_splits, num_seqs)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_splitk_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    workspace: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    num_splits: u32,
    k_scale: f32,
    v_scale: f32,
    q_stride: u32,
    cache_stride: u64,
    num_seqs: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_splits, num_seqs])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(workspace)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(num_splits)
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u32(q_stride)
        .arg_u64(cache_stride)
        .launch(stream)
}

/// Reduce split-K partials into final BF16 output (FP8 variant).
///
/// Grid: (num_q_heads, num_seqs, 1)  Block: (32, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_reduce_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    workspace: DevicePtr,
    output: DevicePtr,
    seq_lens: DevicePtr,
    num_q_heads: u32,
    head_dim: u32,
    num_splits: u32,
    num_seqs: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_seqs, 1])
        .block([32, 1, 1])
        .arg_ptr(workspace)
        .arg_ptr(output)
        .arg_ptr(seq_lens)
        .arg_u32(num_q_heads)
        .arg_u32(head_dim)
        .arg_u32(num_splits)
        .launch(stream)
}

// ── SSM / Convolution ──────────────────────────────────────────────
