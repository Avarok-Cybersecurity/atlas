// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Fill paged-KV slot mappings on-device from a persistent block table.
///
/// Kernel: `fill_slots_from_block_table(slots, block_table, start_pos, count, block_size)`
/// Grid: (ceil(count/256), 1, 1)  Block: (256, 1, 1)
pub fn fill_slots_from_block_table(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    slots: DevicePtr,
    block_table: DevicePtr,
    start_pos: u32,
    count: u32,
    block_size: u32,
    stream: u64,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(count, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(slots)
        .arg_ptr(block_table)
        .arg_u32(start_pos)
        .arg_u32(count)
        .arg_u32(block_size)
        .launch(stream)
}

// ── KV cache ───────────────────────────────────────────────────────

/// Write K/V to paged FP8 cache using slot_mapping.
///
/// Kernel: `reshape_and_cache_flash_fp8(key, value, k_cache, v_cache,
///          slot_mapping, num_kv_heads, head_dim, block_size,
///          k_scale, v_scale, key_stride, value_stride, cache_stride)`
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
///
/// `slot_mapping` is a device pointer to `i64[num_tokens]`.
/// BF16 reshape and cache — no quantization, direct BF16 copy.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache(
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
    _cache_stride: u64,
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
        .launch(stream)
}

/// `k_cache`/`v_cache` are the full pool base pointers.
/// `cache_stride` is in elements (block_size * num_kv_heads * head_dim).
pub fn reshape_and_cache_fp8(
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
    k_scale: f32,
    v_scale: f32,
    key_stride: u32,
    value_stride: u32,
    cache_stride: u64,
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
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u32(key_stride)
        .arg_u32(value_stride)
        .arg_u64(cache_stride)
        .launch(stream)
}

/// Paged decode attention (FP8 KV cache, single/multi sequence).
///
/// Kernel: `paged_decode_attn_fp8(Q, K_cache, V_cache, O, block_tables,
///          seq_lens, max_blocks_per_seq, num_q_heads, num_kv_heads,
///          head_dim, block_size, inv_sqrt_d, k_scale, v_scale,
///          q_stride, cache_stride)`
/// Grid: (num_q_heads, num_seqs, 1)  Block: (256, 1, 1)
///
/// `block_tables`: device ptr to `i32[num_seqs * max_blocks_per_seq]`
/// `seq_lens`: device ptr to `i32[num_seqs]`
/// `cache_stride` is in elements (u64).
/// BF16 paged decode attention — no FP8 quantization, direct BF16 KV cache.
/// MLA batched GEMV: output[head, n] = sum_k(weight[head, n, k] * input[head, k])
/// Replaces 32 sequential dense_gemv calls with a single kernel launch.
pub fn mla_batched_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    n_out: u32,
    k: u32,
    num_heads: u32,
    input_stride: u32,
    output_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 8), num_heads, 1]) // N_PER_BLOCK*2=8
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(output)
        .arg_u32(n_out)
        .arg_u32(k)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .launch(stream)
}

/// MLA Q_rope scatter: copy rope portion from q_full to strided q_absorbed_buf. 1 kernel replaces 32 D2D copies.
#[allow(clippy::too_many_arguments)]
pub fn mla_q_rope_scatter(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_full: DevicePtr,
    q_absorbed_buf: DevicePtr,
    q_rope_contiguous: DevicePtr,
    nq: u32,
    hd: u32,
    nope: u32,
    rope: u32,
    kv_lora: u32,
    mla_cache_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(q_full)
        .arg_ptr(q_absorbed_buf)
        .arg_ptr(q_rope_contiguous)
        .arg_u32(nq)
        .arg_u32(hd)
        .arg_u32(nope)
        .arg_u32(rope)
        .arg_u32(kv_lora)
        .arg_u32(mla_cache_dim)
        .launch(stream)
}

/// MLA Q_rope writeback: scatter RoPE'd rope portions to strided layout. 1 kernel replaces 32 D2D copies.
pub fn mla_q_rope_writeback(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_rope_direct: DevicePtr,
    q_absorbed_buf: DevicePtr,
    nq: u32,
    rope: u32,
    kv_lora: u32,
    mla_cache_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(q_rope_direct)
        .arg_ptr(q_absorbed_buf)
        .arg_u32(nq)
        .arg_u32(rope)
        .arg_u32(kv_lora)
        .arg_u32(mla_cache_dim)
        .launch(stream)
}

/// MLA cache assembly: fuse [kv_latent|k_rope]→K and [kv_latent|zeros]→V into 1 kernel.
pub fn mla_cache_assemble(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    kv_latent: DevicePtr,
    k_rope: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    kv_lora: u32,
    rope: u32,
    mla_cache_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([mla_cache_dim.max(256), 1, 1])
        .arg_ptr(kv_latent)
        .arg_ptr(k_rope)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_u32(kv_lora)
        .arg_u32(rope)
        .arg_u32(mla_cache_dim)
        .launch(stream)
}

// ── Batched prefill variants (N tokens) ──
