// SPDX-License-Identifier: AGPL-3.0-only

//! TurboQuant+ both-sides-quantized asymmetric Turbo*K + Turbo*V KV-cache
//! ops wrappers (write + decode). K and V both use turbo (byte-addressed,
//! data + scale section) layouts but with potentially different K-side and
//! V-side dtypes — three combos: turbo4k_turbo3v, turbo4k_turbo8v,
//! turbo3k_turbo8v.
//!
//! Sibling of `kv_cache.rs` / `kv_cache_fp8k.rs`. Each wrapper takes BOTH
//! K-side (k_block_stride_bytes, k_data_section_bytes) AND V-side
//! (v_block_stride_bytes, v_data_section_bytes) parameters since each pool
//! has its own byte layout.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Write K/V to paged Turbo4K + Turbo3V cache.
///
/// K written as turbo4 (4-bit packed + FP8 group scale, matched-norm L2)
/// and V as turbo3 (3-bit packed + FP8 group scale, matched-norm L2). The
/// pools have distinct byte strides — passed independently.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_turbo4k_turbo3v(
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
    k_block_stride_bytes: u64,
    k_data_section_bytes: u64,
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
        .arg_u64(k_block_stride_bytes)
        .arg_u64(k_data_section_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// Write K/V to paged Turbo4K + Turbo8V cache.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_turbo4k_turbo8v(
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
    k_block_stride_bytes: u64,
    k_data_section_bytes: u64,
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
        .arg_u64(k_block_stride_bytes)
        .arg_u64(k_data_section_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// Write K/V to paged Turbo3K + Turbo8V cache.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_turbo3k_turbo8v(
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
    k_block_stride_bytes: u64,
    k_data_section_bytes: u64,
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
        .arg_u64(k_block_stride_bytes)
        .arg_u64(k_data_section_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// Paged decode attention for Turbo4K + Turbo3V asymmetric KV cache.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_turbo4k_turbo3v(
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
    k_block_stride_bytes: u64,
    k_data_section_bytes: u64,
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
        .arg_u32(q_stride)
        .arg_u64(k_block_stride_bytes)
        .arg_u64(k_data_section_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .arg_u32(sliding_window)
        .launch(stream)
}

/// Paged decode attention for Turbo4K + Turbo8V asymmetric KV cache.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_turbo4k_turbo8v(
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
    k_block_stride_bytes: u64,
    k_data_section_bytes: u64,
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
        .arg_u32(q_stride)
        .arg_u64(k_block_stride_bytes)
        .arg_u64(k_data_section_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .arg_u32(sliding_window)
        .launch(stream)
}

/// Paged decode attention for Turbo3K + Turbo8V asymmetric KV cache.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_turbo3k_turbo8v(
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
    k_block_stride_bytes: u64,
    k_data_section_bytes: u64,
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
        .arg_u32(q_stride)
        .arg_u64(k_block_stride_bytes)
        .arg_u64(k_data_section_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .arg_u32(sliding_window)
        .launch(stream)
}
