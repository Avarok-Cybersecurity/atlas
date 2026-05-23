// SPDX-License-Identifier: AGPL-3.0-only

//! TurboQuant+ asymmetric Fp8K + TurboNV prefill (BR=64) ops wrappers.
//!
//! Sibling of `prefill_attn_main_b.rs`: keeps the bf16k_* prefill wrappers
//! there from growing past the 500-LoC cap. Each wrapper mirrors the
//! corresponding bf16k_turbo*v_64 function plus a `k_scale` parameter
//! threaded into the kernel call (FP8 K dequant scale).

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Prefill paged attention — TurboQuant+ asym Fp8K + Turbo3V (BR=64).
///
/// K is read as FP8 (per-tensor `k_scale` dequant in LOAD_K_TILE),
/// V as turbo3 (3-bit packed + FP8 group scale).
///
/// Kernel: `inferspark_prefill_paged_fp8k_turbo3v_64(Q, K_cache, V_cache,
///          O, block_table, q_len, kv_len, q_offset, num_q_heads,
///          num_kv_heads, head_dim, cache_block_size, sliding_window,
///          causal_mask_enabled, inv_sqrt_d, k_scale,
///          v_block_stride_bytes, v_data_section_bytes)`
/// Grid: (num_q_heads, div_ceil(q_len, BR), 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_fp8k_turbo3v_64(
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
        .arg_f32(k_scale)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// Prefill paged attention — TurboQuant+ asym Fp8K + Turbo4V (BR=64).
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_fp8k_turbo4v_64(
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
        .arg_f32(k_scale)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// Prefill paged attention — TurboQuant+ asym Fp8K + Turbo2V (BR=64, 6.4x V comp).
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_fp8k_turbo2v_64(
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
        .arg_f32(k_scale)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}
