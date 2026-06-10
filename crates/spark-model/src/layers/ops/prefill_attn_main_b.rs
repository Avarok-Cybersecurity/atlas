// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Paged prefill Flash Attention — NVFP4 KV cache variant.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_nvfp4(
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
    block_stride_bytes: u64,
    data_section_bytes: u64,
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
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

/// Paged prefill Flash Attention for HDIM=512 (Gemma-4 full-attention) — BF16 KV.
///
/// Uses dynamic shared memory (101,120 B) opt-in. Single-buffered K, 8 warps.
/// Required for chunked long-context prefill on layers with `head_dim==512`
/// where the standard 4-warp template doesn't fit GB10's 99 KB smem cap.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_512(
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
        .block([256, 1, 1])
        .shared_mem(101_120)
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

/// Paged prefill Flash Attention — BF16 KV cache, BR=64 (256 threads).
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_64(
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
    let br = 64u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([256, 1, 1])
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

/// Paged prefill Flash Attention — FP8 KV cache, BR=64 (256 threads).
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_fp8_64(
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
    let br = 64u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([256, 1, 1])
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

/// Paged prefill Flash Attention — symmetric TurboQuant KV cache, BR=64.
/// Shared launch wrapper for the turbo8 / turbo4 / turbo3 `_64` kernel
/// entries: identical ABI, the caller selects the dtype via `kernel` and
/// passes that pool's block stride + data-section offset.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_turbo_64(
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
    block_stride_bytes: u64,
    data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    let br = 64u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([256, 1, 1])
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
        .arg_u32(1u32)
        .arg_f32(inv_sqrt_d)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

pub fn prefill_attention_paged_turbo2_64(
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
    block_stride_bytes: u64,
    data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    let br = 32u32; // try BR=32 entry first while debugging BR=64 OOB
    // BR=32 entry is sized for 128 threads (4 warps); 256 threads makes
    // warps 4-7 read past smem_V (OOB shared reads, results discarded).
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
        .arg_u32(1u32)
        .arg_f32(inv_sqrt_d)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

pub fn prefill_attention_paged_nvfp4_64(
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
    block_stride_bytes: u64,
    data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    let br = 64u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([256, 1, 1])
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
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

/// Paged prefill (BR=64) for Bf16K + Turbo3V asymmetric KV cache.
///
/// Reads K as BF16 (NHD contiguous) and V as 3-bit Lloyd-Max packed bytes
/// with FP8 per-group scale. Uses the asym prefill compute template
/// (prefill_paged_compute_asym.cuh) which takes separate LOAD_K_TILE +
/// LOAD_V_TILE macros — bf16 cp.async for K, sync dequant for V.
///
/// Kernel: `inferspark_prefill_paged_bf16k_turbo3v_64(Q, K_cache, V_cache,
///          O, block_table, q_len, kv_len, q_offset, num_q_heads,
///          num_kv_heads, head_dim, cache_block_size, sliding_window,
///          causal_mask_enabled, inv_sqrt_d, v_block_stride_bytes,
///          v_data_section_bytes)`
/// Grid: (num_q_heads, div_ceil(q_len, BR), 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_bf16k_turbo3v_64(
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
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    let br = 64u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([256, 1, 1])
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
        // causal_mask_enabled = 1 (default causal).
        .arg_u32(1u32)
        .arg_f32(inv_sqrt_d)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// Prefill paged attention — TurboQuant+ safer-asym Bf16K + Turbo4V (BR=64).
///
/// Same kernel ABI as `prefill_attention_paged_bf16k_turbo3v_64`; the
/// underlying kernel uses a 4-bit V dequant path in `LOAD_V_TILE`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_bf16k_turbo4v_64(
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
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    let br = 64u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([256, 1, 1])
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
        .arg_u32(1u32)
        .arg_f32(inv_sqrt_d)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// Prefill paged attention — TurboQuant+ safer-asym Bf16K + Turbo2V (BR=64).
///
/// 6.4x V compression. Same kernel ABI as `prefill_attention_paged_bf16k_turbo3v_64`;
/// kernel uses a 2-bit V dequant path in `LOAD_V_TILE`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_bf16k_turbo2v_64(
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
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    let br = 64u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([256, 1, 1])
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
        .arg_u32(1u32)
        .arg_f32(inv_sqrt_d)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}
