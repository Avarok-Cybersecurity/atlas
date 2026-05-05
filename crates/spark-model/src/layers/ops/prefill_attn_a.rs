// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Batched Q rope extract: [N, nq, hd] → [N, nq, rope] at offset nope per head.
/// 1 kernel replaces N*nq D2D copies per layer.
#[allow(clippy::too_many_arguments)]
pub fn mla_q_rope_extract_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_full: DevicePtr,
    q_rope_out: DevicePtr,
    num_tokens: u32,
    nq: u32,
    hd: u32,
    nope: u32,
    rope: u32,
    q_dim: u32,
    stream: u64,
) -> Result<()> {
    let total = num_tokens * nq * rope;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(q_full)
        .arg_ptr(q_rope_out)
        .arg_u32(num_tokens)
        .arg_u32(nq)
        .arg_u32(hd)
        .arg_u32(nope)
        .arg_u32(rope)
        .arg_u32(q_dim)
        .launch(stream)
}

/// Batched Q rope writeback: [N, nq, rope] → [N, nq, hd] at offset nope per head.
#[allow(clippy::too_many_arguments)]
pub fn mla_q_rope_writeback_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_rope_in: DevicePtr,
    q_full: DevicePtr,
    num_tokens: u32,
    nq: u32,
    hd: u32,
    nope: u32,
    rope: u32,
    q_dim: u32,
    stream: u64,
) -> Result<()> {
    let total = num_tokens * nq * rope;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(q_rope_in)
        .arg_ptr(q_full)
        .arg_u32(num_tokens)
        .arg_u32(nq)
        .arg_u32(hd)
        .arg_u32(nope)
        .arg_u32(rope)
        .arg_u32(q_dim)
        .launch(stream)
}

/// Batched K/V assembly from kv_expanded + k_rope for N tokens.
/// 1 kernel replaces N*nkv*3 D2D copies per layer.
#[allow(clippy::too_many_arguments)]
pub fn mla_kv_assemble_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    kv_expanded: DevicePtr,
    k_rope_buf: DevicePtr,
    k_out: DevicePtr,
    v_out: DevicePtr,
    num_tokens: u32,
    nkv: u32,
    nope: u32,
    v_dim: u32,
    rope: u32,
    hd: u32,
    kv_expanded_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 2, 1])
        .block([256, 1, 1])
        .arg_ptr(kv_expanded)
        .arg_ptr(k_rope_buf)
        .arg_ptr(k_out)
        .arg_ptr(v_out)
        .arg_u32(nkv)
        .arg_u32(nope)
        .arg_u32(v_dim)
        .arg_u32(rope)
        .arg_u32(hd)
        .arg_u32(kv_expanded_stride)
        .launch(stream)
}

/// Batched MLA cache assembly for N tokens: K=[latent|rope], V=[latent|zeros].
/// 1 kernel replaces N*4 D2D copies+memsets per layer.
#[allow(clippy::too_many_arguments)]
pub fn mla_cache_assemble_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    kv_latent: DevicePtr,
    k_rope: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    num_tokens: u32,
    kv_lora: u32,
    rope: u32,
    mla_cache_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
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

/// Fused MLA prefill: Q_absorption + attention + V_extraction in one kernel.
/// Grid: (num_heads, seq_len, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn mla_fused_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_full: DevicePtr,
    q_rope: DevicePtr,
    kv_latent: DevicePtr,
    k_rope: DevicePtr,
    w_uk: DevicePtr,
    w_uv: DevicePtr,
    v_out: DevicePtr,
    k_cache_out: DevicePtr,
    v_cache_out: DevicePtr,
    seq_len: u32,
    nq: u32,
    nope: u32,
    rope: u32,
    kv_lora: u32,
    v_dim: u32,
    hd: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([nq, seq_len, 1])
        .block([256, 1, 1])
        .arg_ptr(q_full)
        .arg_ptr(q_rope)
        .arg_ptr(kv_latent)
        .arg_ptr(k_rope)
        .arg_ptr(w_uk)
        .arg_ptr(w_uv)
        .arg_ptr(v_out)
        .arg_ptr(k_cache_out)
        .arg_ptr(v_cache_out)
        .arg_u32(seq_len)
        .arg_u32(nq)
        .arg_u32(nope)
        .arg_u32(rope)
        .arg_u32(kv_lora)
        .arg_u32(v_dim)
        .arg_u32(hd)
        .arg_f32(inv_sqrt_d)
        .launch(stream)
}

/// Assemble Q_final from Q_absorbed + Q_rope: [absorbed|rope] per head per token.
#[allow(clippy::too_many_arguments)]
pub fn mla_q_final_assemble_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_absorbed: DevicePtr,
    q_rope: DevicePtr,
    q_final: DevicePtr,
    num_tokens: u32,
    nq: u32,
    kv_lora: u32,
    rope: u32,
    mla_cache_dim: u32,
    stream: u64,
) -> Result<()> {
    let total = num_tokens * nq * mla_cache_dim;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(q_absorbed)
        .arg_ptr(q_rope)
        .arg_ptr(q_final)
        .arg_u32(num_tokens)
        .arg_u32(nq)
        .arg_u32(kv_lora)
        .arg_u32(rope)
        .arg_u32(mla_cache_dim)
        .launch(stream)
}

/// Grouped GEMM for MLA: G independent `[M,K_g]@[N_g,K_g]^T→[M,N_g]` in one launch.
/// Grid: (M*G, ceil(N_g/4), 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn grouped_gemm_mla(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: u32,
    g: u32,
    k_g: u32,
    n_g: u32,
    a_stride: u32,
    c_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([m * g, div_ceil(n_g, 4), 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(g)
        .arg_u32(k_g)
        .arg_u32(n_g)
        .arg_u32(a_stride)
        .arg_u32(c_stride)
        .launch(stream)
}

/// MLA absorbed prefill attention (HDIM=320, simple scalar kernel).
/// Grid: (num_q_heads, ceil(seq_len/16), batch)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn mla_prefill_attention_320(
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
    stream: u64,
) -> Result<()> {
    let br = 16u32; // MLA_BR in the kernel
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
        .launch(stream)
}

pub fn paged_decode_attn_bf16(
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
    sliding_window: u32, // 0 = full attention; >0 = window size (Gemma-4 sliding layers)
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
        .arg_u32(sliding_window)
        .launch(stream)
}

pub fn paged_decode_attn_fp8(
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
    v_scale: f32,
    q_stride: u32,
    cache_stride: u64,
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
        .arg_f32(v_scale)
        .arg_u32(q_stride)
        .arg_u64(cache_stride)
        .launch(stream)
}
