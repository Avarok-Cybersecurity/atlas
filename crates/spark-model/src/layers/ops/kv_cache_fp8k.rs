// SPDX-License-Identifier: AGPL-3.0-only

//! TurboQuant+ asymmetric Fp8K + TurboNV KV-cache ops wrappers (write + decode).
//!
//! Sibling of `kv_cache.rs`: keeps the bf16k_* paths there from growing past
//! the 500-LoC cap. Each wrapper mirrors the corresponding bf16k_turbo*v
//! function plus a `k_scale` parameter threaded into the kernel call.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Write K/V to paged Fp8K + Turbo3V (TurboQuant+ asym) cache.
///
/// K is written as FP8 E4M3 (per-tensor `k_scale`, NHD contiguous),
/// V as 3-bit Lloyd-Max + FP8 per-group scale with matched-norm correction.
/// K and V pools have separate strides (K 1 b/elem; V ~0.5 b/elem + scale).
///
/// Kernel: `reshape_and_cache_flash_fp8k_turbo3v(key, value, k_cache, v_cache,
///          slot_mapping, num_kv_heads, head_dim, block_size,
///          key_stride, value_stride, k_scale, k_block_stride_bytes,
///          v_block_stride_bytes, v_data_section_bytes)`
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_fp8k_turbo3v(
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
    k_scale: f32,
    k_block_stride_bytes: u64,
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
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
        .arg_f32(k_scale)
        .arg_u64(k_block_stride_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// Write K/V to paged Fp8K + Turbo4V (TurboQuant+ asym) cache.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_fp8k_turbo4v(
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
    k_scale: f32,
    k_block_stride_bytes: u64,
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
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
        .arg_f32(k_scale)
        .arg_u64(k_block_stride_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// Write K/V to paged Fp8K + Turbo2V (TurboQuant+ asym) cache (6.4x V comp).
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_fp8k_turbo2v(
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
    k_scale: f32,
    k_block_stride_bytes: u64,
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
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
        .arg_f32(k_scale)
        .arg_u64(k_block_stride_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// Paged decode attention for Fp8K + Turbo3V asymmetric KV cache.
///
/// K is read as FP8 NHD with per-tensor `k_scale` dequant, V as 3-bit
/// Lloyd-Max packed bytes + FP8 per-group scale (sparse-V threshold on
/// batched + remainder paths).
///
/// Kernel: `paged_decode_attn_fp8k_turbo3v(Q, K_cache, V_cache, O,
///          block_tables, seq_lens, max_blocks_per_seq, num_q_heads,
///          num_kv_heads, head_dim, block_size, inv_sqrt_d, k_scale,
///          q_stride, v_block_stride_bytes, v_data_section_bytes,
///          sliding_window)`
/// Grid: (num_q_heads, num_seqs, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_fp8k_turbo3v(
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
    k_scale: f32,
    q_stride: u32,
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
    sliding_window: u32,
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
        .arg_f32(k_scale)
        .arg_u32(q_stride)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .arg_u32(sliding_window)
        .launch(stream)
}

/// Paged decode attention for Fp8K + Turbo4V asymmetric KV cache.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_fp8k_turbo4v(
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
    k_scale: f32,
    q_stride: u32,
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
    sliding_window: u32,
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
        .arg_f32(k_scale)
        .arg_u32(q_stride)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .arg_u32(sliding_window)
        .launch(stream)
}

/// Paged decode attention for Fp8K + Turbo2V asymmetric KV cache (6.4x V comp).
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_fp8k_turbo2v(
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
    k_scale: f32,
    q_stride: u32,
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
    sliding_window: u32,
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
        .arg_f32(k_scale)
        .arg_u32(q_stride)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .arg_u32(sliding_window)
        .launch(stream)
}
