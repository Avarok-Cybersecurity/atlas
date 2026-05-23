// SPDX-License-Identifier: AGPL-3.0-only

//! TurboQuant+ both-sides-quantized Turbo*K + Turbo*V prefill (BR=64) ops wrappers.
//!
//! Sibling of `prefill_attn_main_b.rs` / `prefill_attn_fp8k.rs`. Each wrapper
//! takes BOTH K-side and V-side (block_stride_bytes, data_section_bytes) pairs
//! since the two pools have independent byte layouts.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Prefill paged attention — TurboQuant+ asym Turbo4K + Turbo3V (BR=64).
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_turbo4k_turbo3v_64(
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
    k_block_stride_bytes: u64,
    k_data_section_bytes: u64,
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
        .arg_u64(k_block_stride_bytes)
        .arg_u64(k_data_section_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// Prefill paged attention — TurboQuant+ asym Turbo4K + Turbo8V (BR=64).
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_turbo4k_turbo8v_64(
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
    k_block_stride_bytes: u64,
    k_data_section_bytes: u64,
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
        .arg_u64(k_block_stride_bytes)
        .arg_u64(k_data_section_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// Prefill paged attention — TurboQuant+ asym Turbo3K + Turbo8V (BR=64).
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_turbo3k_turbo8v_64(
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
    k_block_stride_bytes: u64,
    k_data_section_bytes: u64,
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
        .arg_u64(k_block_stride_bytes)
        .arg_u64(k_data_section_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}
