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

/// DeepSeek-V4 full-attention (non-CSA) prefill with a per-head attention sink.
///
/// Same as [`prefill_attention`] for HDIM=512 (BR=16), but passes the per-head
/// `sinks` logit so the softmax denominator matches the decode path (which
/// always applies the sink). Launches the V4-specific `inferspark_prefill_512`
/// kernel. `sinks` may be `DevicePtr::NULL` for no sink.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_512_sink(
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
    sinks: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(seq_len, 16), batch])
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
        .arg_ptr(sinks)
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
    // BR64 = query rows processed per CTA. The kernel clamps this to 32 on the
    // AMD targets (gfx1151 64 KB LDS cap: inferspark_prefill.cu's
    // `#if __SCALE__ || __HIP_PLATFORM_AMD__ #define BR64 32`). The grid stride
    // MUST match the kernel's BR64, else CTAs are spaced 64 rows apart while each
    // writes only 32 → query rows 32..63 of every 64-row band are silently left
    // unwritten (gross attention corruption for any prompt >32 tokens). cfg!
    // (atlas_scale) is set for both `strix` and `strix-hip`; NVIDIA keeps 64
    // (byte-identical). See the @human-review note in inferspark_prefill.cu.
    let br = if cfg!(atlas_scale) { 32u32 } else { 64u32 };
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

/// DFlash γ-block paged Flash Attention — BF16 KV cache variant.
///
/// Same kernel binary as [`prefill_attention_paged`] but launched with
/// `causal_mask_enabled = 0`, producing bidirectional attention within the
/// γ-token query block. The prefix KV positions are strictly less than
/// `q_offset` so they need no causal mask in this mode (every prefix
/// position is "older" than every query, which is the no-mask case anyway).
///
/// Used by `BlockDiffusionDraftHead::forward_block` once per drafter layer
/// when the drafter KV cache is BF16 (current default — FP8 acceptance
/// collapses on SM12.x per `dflash_head.rs:82–86`).
///
/// `q_len` is γ (typically 16). `q_offset` is the absolute starting index
/// of the γ-block in the drafter's logical sequence; the kernel uses it to
/// skip the now-disabled causal compare against `kv_start+col`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_dflash(
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
        .arg_u32(0u32) // causal_mask_enabled = 0 (DFlash bidirectional)
        .arg_f32(inv_sqrt_d)
        // Non-indirect kernel: q_rope_pos is a local var (= q_offset) in the
        // .cuh body — no extra kernel arg. Fix applies via indirect path only.
        .launch(stream)
}

/// DFlash γ-block paged Flash Attention — BF16 KV cache, INDIRECT scalar args.
///
/// Phase 5 (CUDA graph) variant of [`prefill_attention_paged_dflash`]. Reads
/// `kv_len`, `q_offset`, and `q_rope_pos` from device pointers at kernel entry
/// instead of taking them as kernel scalar arguments. This makes the launch
/// graph-friendly: the host writes the dynamic triple into `kv_len_q_offset_dev`
/// (12 bytes: `[u32 kv_len, u32 q_offset, u32 q_rope_pos]`) BEFORE entering the
/// captured region, and the captured graph node binds only the pointer — replays
/// pick up whatever values the host wrote pre-launch.
/// `q_offset` = ctx_count (cache-block addressing); `q_rope_pos` = absolute
/// decode position (query RoPE rotation, decoupled from cache addressing).
///
/// Resolves to kernel `inferspark_prefill_paged_indirect`. The kernel binary
/// is otherwise identical to `inferspark_prefill_paged` (`causal_mask_enabled
/// = 0` is still hardcoded here on the launch side).
///
/// Phase B note: defined but NOT YET WIRED IN to forward_block_layer_paged.
/// Phase C swaps the dispatcher; Phase D adds graph capture around it.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_dflash_bf16_indirect(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table: DevicePtr,
    q_len: u32,
    kv_len_q_offset_dev: DevicePtr,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cache_block_size: u32,
    sliding_window: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    let br = 32u32;
    // q_offset's only role in the captured-args path is grid sizing; we still
    // pass q_len (a model constant, γ) directly. The kernel will overwrite its
    // scalar `kv_len`/`q_offset` slots from the indirect buffer at entry, so
    // we feed placeholder zeros for those two args here — the values are
    // *ignored* once `KERNEL_PREAMBLE` runs in the .cu file.
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table)
        .arg_u32(q_len)
        .arg_u32(0u32) // kv_len placeholder — overwritten by KERNEL_PREAMBLE
        .arg_u32(0u32) // q_offset placeholder — overwritten by KERNEL_PREAMBLE
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(cache_block_size)
        .arg_u32(sliding_window)
        .arg_u32(0u32) // causal_mask_enabled = 0 (DFlash bidirectional)
        .arg_f32(inv_sqrt_d)
        .arg_ptr(kv_len_q_offset_dev) // KERNEL_EXTRA_PARAMS: kv_len_ptr
        .arg_ptr(kv_len_q_offset_dev.offset(4)) // KERNEL_EXTRA_PARAMS: q_offset_ptr
        .arg_ptr(kv_len_q_offset_dev.offset(8)) // KERNEL_EXTRA_PARAMS: q_rope_pos_ptr
        .launch(stream)
}
