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

/// V-only paged cache write — companion to the fused K-path so the
/// K side of the cache stays exclusively owned by
/// `fused_k_norm_rope_cache_write_*`. Use this when the fused K kernel
/// is active to avoid the existing `reshape_and_cache` overwriting
/// the correct K values with a double-rounded copy.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_flash_v_only(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    value: DevicePtr,
    v_cache: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    value_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(value)
        .arg_ptr(v_cache)
        .arg_ptr(slot_mapping)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_u32(value_stride)
        .launch(stream)
}

/// Fused K-path: rms_norm → RoPE → BF16 paged cache write in one kernel.
///
/// Replaces the chained `ops::rms_norm + ops::rope + ops::reshape_and_cache`
/// sequence for the K projection. Keeps K in FP32 between the three
/// operations and BF16-rounds ONLY at cache write — vLLM-equivalent
/// precision regime. Eliminates the two intermediate BF16 rounding steps
/// that previously compounded at deep attention layers (L35-L39) where K
/// magnitudes peak ~18× vs L0, causing the documented BF16-KV cliff.
#[allow(clippy::too_many_arguments)]
pub fn fused_k_norm_rope_cache_write_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    k_in: DevicePtr,
    k_norm_weight: DevicePtr,
    positions: DevicePtr,
    k_cache: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    block_size: u32,
    rms_eps: f32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, num_kv_heads, 1])
        .block([head_dim, 1, 1])
        .arg_ptr(k_in)
        .arg_ptr(k_norm_weight)
        .arg_ptr(positions)
        .arg_ptr(k_cache)
        .arg_ptr(slot_mapping)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_u32(block_size)
        .arg_f32(rms_eps)
        .arg_f32(theta)
        .launch(stream)
}

/// MRoPE-interleaved variant — selects abs position from pos_t/pos_h/pos_w
/// based on `pair_idx % 3`. For text-only inputs (pos_h == pos_w == pos_t)
/// the result is bit-identical to the scalar-position variant.
#[allow(clippy::too_many_arguments)]
pub fn fused_k_norm_rope_cache_write_bf16_mrope(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    k_in: DevicePtr,
    k_norm_weight: DevicePtr,
    pos_t: DevicePtr,
    pos_h: DevicePtr,
    pos_w: DevicePtr,
    k_cache: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    block_size: u32,
    rms_eps: f32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, num_kv_heads, 1])
        .block([head_dim, 1, 1])
        .arg_ptr(k_in)
        .arg_ptr(k_norm_weight)
        .arg_ptr(pos_t)
        .arg_ptr(pos_h)
        .arg_ptr(pos_w)
        .arg_ptr(k_cache)
        .arg_ptr(slot_mapping)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_u32(block_size)
        .arg_f32(rms_eps)
        .arg_f32(theta)
        .launch(stream)
}

/// FP8-output sibling of [`fused_k_norm_rope_cache_write_bf16`]. Same
/// semantics; one fewer BF16 round before the saturating FP8 cast.
#[allow(clippy::too_many_arguments)]
pub fn fused_k_norm_rope_cache_write_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    k_in: DevicePtr,
    k_norm_weight: DevicePtr,
    positions: DevicePtr,
    k_cache_fp8: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    block_size: u32,
    rms_eps: f32,
    theta: f32,
    inv_scale: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, num_kv_heads, 1])
        .block([head_dim, 1, 1])
        .arg_ptr(k_in)
        .arg_ptr(k_norm_weight)
        .arg_ptr(positions)
        .arg_ptr(k_cache_fp8)
        .arg_ptr(slot_mapping)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_u32(block_size)
        .arg_f32(rms_eps)
        .arg_f32(theta)
        .arg_f32(inv_scale)
        .launch(stream)
}

/// Write K/V to paged Bf16K + Turbo3V (TurboQuant+ safer-asym) cache.
///
/// K is written as raw BF16 (NHD contiguous), V as 3-bit Lloyd-Max + FP8
/// per-group scale with matched-norm correction. K and V pools have separate
/// strides because K is 2 b/elem and V is ~0.5 b/elem + scale.
///
/// Kernel: `reshape_and_cache_flash_bf16k_turbo3v(key, value, k_cache, v_cache,
///          slot_mapping, num_kv_heads, head_dim, block_size,
///          key_stride, value_stride, k_block_stride_bytes,
///          v_block_stride_bytes, v_data_section_bytes)`
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_bf16k_turbo3v(
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
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// Paged decode attention for Bf16K + Turbo3V asymmetric KV cache.
///
/// K is read as BF16 NHD (vector loads), V as 3-bit Lloyd-Max packed bytes
/// with FP8 per-group scale (sparse V on batched AND remainder paths).
///
/// Kernel: `paged_decode_attn_bf16k_turbo3v(Q, K_cache, V_cache, O,
///          block_tables, seq_lens, max_blocks_per_seq, num_q_heads,
///          num_kv_heads, head_dim, block_size, inv_sqrt_d, q_stride,
///          v_block_stride_bytes, v_data_section_bytes, sliding_window)`
/// Grid: (num_q_heads, num_seqs, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_bf16k_turbo3v(
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
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .arg_u32(sliding_window)
        .launch(stream)
}

/// Write K/V to paged Bf16K + Turbo4V (TurboQuant+ safer-asym) cache.
///
/// K written as raw BF16 NHD, V as 4-bit Lloyd-Max + FP8 per-group scale with
/// matched-norm correction. K and V pools have separate strides.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_bf16k_turbo4v(
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
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// Write K/V to paged Bf16K + Turbo2V (TurboQuant+ safer-asym) cache (6.4x V comp).
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_bf16k_turbo2v(
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
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// Paged decode attention for Bf16K + Turbo4V asymmetric KV cache.
///
/// K read as BF16 NHD, V as 4-bit Lloyd-Max packed bytes + FP8 per-group scale
/// (sparse V on batched + remainder paths).
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_bf16k_turbo4v(
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
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .arg_u32(sliding_window)
        .launch(stream)
}

/// Paged decode attention for Bf16K + Turbo2V asymmetric KV cache (6.4x V comp).
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_bf16k_turbo2v(
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
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .arg_u32(sliding_window)
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

/// MLA Paged Decode — NVFP4 variant for DeepSeek-V4-Flash.
///
/// Kernel: `mla_paged_decode_nvfp4(Q, K_cache, V_cache, O, block_tables,
///          seq_lens, max_blocks_per_seq, num_q_heads, num_kv_heads,
///          q_head_dim, kv_cache_dim, block_size, inv_sqrt_d,
///          block_stride_bytes, data_section_bytes)`
/// Grid: (num_q_heads, num_seqs, 1)  Block: (256, 1, 1)
///
/// V4-Flash uses compressed KV cache (576 dims: 512 latent + 64 rope)
/// and flattened Q layout (32768 dims = 64 heads × 512).
#[allow(clippy::too_many_arguments)]
pub fn mla_paged_decode_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    o: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    q_head_dim: u32,
    kv_cache_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    num_seqs: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_seqs, 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(o)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(q_head_dim)
        .arg_u32(kv_cache_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

/// MLA Paged Decode — FP8 variant for DeepSeek-V4-Flash with FP8 KV cache.
///
/// Kernel: `mla_paged_decode_fp8(Q, K_cache, V_cache, O, block_tables,
///          seq_lens, max_blocks_per_seq, num_q_heads, num_kv_heads,
///          q_head_dim, kv_cache_dim, block_size, inv_sqrt_d,
///          k_scale, v_scale, cache_stride)`
/// Grid: (num_q_heads, num_seqs, 1)  Block: (256, 1, 1)
///
/// V4-Flash uses compressed KV cache (576 dims: 512 latent + 64 rope)
/// and FP8 quantization with per-layer scales.
#[allow(clippy::too_many_arguments)]
pub fn mla_paged_decode_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    o: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    q_head_dim: u32,
    kv_cache_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    k_scale: f32,
    v_scale: f32,
    cache_stride: u64,
    num_seqs: u32,
    sliding_window: u32,
    sinks: DevicePtr,
    comp_pool: DevicePtr, // 4b: flat FP8 compressed-KV pool (NULL = no compressed arm)
    comp_block_count: u32, // 4b: # compressed blocks to attend (0 = no-op)
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_seqs, 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(o)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(q_head_dim)
        .arg_u32(kv_cache_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u64(cache_stride)
        .arg_u32(sliding_window)
        .arg_ptr(sinks)
        .arg_ptr(comp_pool)
        .arg_u32(comp_block_count)
        .launch(stream)
}

// ── Batched prefill variants (N tokens) ──
