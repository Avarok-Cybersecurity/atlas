// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Flash Attention v2 prefill on contiguous Q/K/V.
///
/// Kernel: `inferspark_prefill(Q, K, V, O, seq_len, num_q_heads, num_kv_heads,
///          head_dim, inv_sqrt_d, causal)`
/// Grid: (num_q_heads, ceil(seq_len/32), batch)  Block: (128, 1, 1)
///
/// Layout: Q [batch, seq_len, num_q_heads, head_dim] BF16
///         K/V [batch, seq_len, num_kv_heads, head_dim] BF16
///         O [batch, seq_len, num_q_heads, head_dim] BF16
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    output: DevicePtr,
    seq_len: u32,
    batch: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    inv_sqrt_d: f32,
    causal: bool,
    sliding_window: u32, // 0 = no sliding limit; >0 = mask keys where q - k >= window
    stream: u64,
) -> Result<()> {
    // BR=16 for HDIM=512 (Gemma-4 full attention), BR=32 otherwise
    let br = if head_dim > 256 { 16u32 } else { 32u32 };
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(seq_len, br), batch])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(v)
        .arg_ptr(output)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_f32(inv_sqrt_d)
        .arg_u32(if causal { 1 } else { 0 })
        .arg_u32(sliding_window)
        .launch(stream)
}

/// Contiguous prefill Flash Attention — BF16, BR=64 (256 threads).
///
/// Larger tile size halves CTA count and causal KV iterations for long sequences.
/// Grid: (num_q_heads, ceil(seq_len/64), batch)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_64(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    output: DevicePtr,
    seq_len: u32,
    batch: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    inv_sqrt_d: f32,
    causal: bool,
    sliding_window: u32,
    stream: u64,
) -> Result<()> {
    let br = 64u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(seq_len, br), batch])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(v)
        .arg_ptr(output)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_f32(inv_sqrt_d)
        .arg_u32(if causal { 1 } else { 0 })
        .arg_u32(sliding_window)
        .launch(stream)
}

/// Contiguous prefill Flash Attention — FP8 E4M3 K/V variant (BR=64).
///
/// Q is BF16, K/V are FP8 E4M3 (dequantized to BF16 in shared memory).
/// Halves K/V memory reads compared to the BF16 kernel.
///
/// Grid: (num_q_heads, ceil(seq_len/64), batch)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_fp8kv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_fp8: DevicePtr,
    v_fp8: DevicePtr,
    output: DevicePtr,
    seq_len: u32,
    batch: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    inv_sqrt_d: f32,
    causal: bool,
    stream: u64,
) -> Result<()> {
    let br = 64u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(seq_len, br), batch])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_fp8)
        .arg_ptr(v_fp8)
        .arg_ptr(output)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_f32(inv_sqrt_d)
        .arg_u32(if causal { 1 } else { 0 })
        .launch(stream)
}

/// Paged prefill Flash Attention — reads K/V from paged KV cache via block_table.
///
/// For chunked prefill chunk 1+: Q comes from GEMM (contiguous), K/V reside
/// in the paged cache from prior chunks. Replaces per-token paged decode loop
/// with a single Flash Attention pass (O(N) per chunk instead of O(N^2) total).
///
/// Kernel: `inferspark_prefill_paged(Q, K_cache, V_cache, O, block_table,
///          q_len, kv_len, q_offset, nq, nkv, hd, block_size, inv_sqrt_d)`
/// Grid: (num_q_heads, ceil(q_len/32), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table: DevicePtr,
    q_len: u32,
    kv_len: u32,
    q_offset: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cache_block_size: u32,
    sliding_window: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    let br = 32u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table)
        .arg_u32(q_len)
        .arg_u32(kv_len)
        .arg_u32(q_offset)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(cache_block_size)
        .arg_u32(sliding_window)
        // causal_mask_enabled = 1 (default causal). DFlash γ-block kernels
        // pass 0 via dedicated dispatchers (`prefill_attention_paged_dflash_*`).
        .arg_u32(1u32)
        .arg_f32(inv_sqrt_d)
        .launch(stream)
}

/// Paged prefill Flash Attention — FP8 KV cache variant.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table: DevicePtr,
    q_len: u32,
    kv_len: u32,
    q_offset: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cache_block_size: u32,
    sliding_window: u32,
    inv_sqrt_d: f32,
    k_scale: f32,
    v_scale: f32,
    cache_stride: u64,
    stream: u64,
) -> Result<()> {
    let br = 32u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table)
        .arg_u32(q_len)
        .arg_u32(kv_len)
        .arg_u32(q_offset)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(cache_block_size)
        .arg_u32(sliding_window)
        // causal_mask_enabled = 1 (default causal). DFlash γ-block kernels
        // pass 0 via dedicated dispatchers (`prefill_attention_paged_dflash_*`).
        .arg_u32(1u32)
        .arg_f32(inv_sqrt_d)
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u64(cache_stride)
        .launch(stream)
}

/// DFlash γ-block paged Flash Attention — FP8 KV cache variant.
///
/// Same kernel binary as [`prefill_attention_paged_fp8`] but launched with
/// `causal_mask_enabled = 0`, producing bidirectional attention within the
/// γ-token query block. The prefix KV positions are still strictly less
/// than `q_offset` so they need no causal mask in this mode (every prefix
/// position is "older" than every query, which is the no-mask case anyway).
///
/// Used by `BlockDiffusionDraftHead::forward_block` once per drafter layer.
/// `q_len` is γ (typically 16). `q_offset` is the absolute starting index
/// of the γ-block in the drafter's logical sequence; the kernel uses it to
/// skip the now-disabled causal compare against `kv_start+col`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_fp8_dflash(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table: DevicePtr,
    q_len: u32,
    kv_len: u32,
    q_offset: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cache_block_size: u32,
    sliding_window: u32,
    inv_sqrt_d: f32,
    k_scale: f32,
    v_scale: f32,
    cache_stride: u64,
    stream: u64,
) -> Result<()> {
    let br = 32u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table)
        .arg_u32(q_len)
        .arg_u32(kv_len)
        .arg_u32(q_offset)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(cache_block_size)
        .arg_u32(sliding_window)
        .arg_u32(0u32) // causal_mask_enabled = 0 (DFlash bidirectional)
        .arg_f32(inv_sqrt_d)
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u64(cache_stride)
        .launch(stream)
}
