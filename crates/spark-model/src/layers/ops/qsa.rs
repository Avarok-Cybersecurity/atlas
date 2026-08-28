// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the Qwen3.8-Flash-Next QSA indexer kernels
//! (`qsa_indexer.cu`): block-key pooling, decode-query prep, block scoring
//! and the selected-token K/V gather. See the .cu header for the semantics
//! and the scratch-as-paged-cache trick that lets the EXISTING paged decode
//! attention consume the selection.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Pool `n_new` freshly complete blocks starting at `first_block`:
/// mean over `ratio` raw keys -> RMSNorm*(1+w) -> rope at block-start pos.
#[allow(clippy::too_many_arguments)]
pub fn qsa_block_pool(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    raw_keys: DevicePtr,
    k_norm_w: DevicePtr,
    block_keys: DevicePtr,
    first_block: u32,
    n_new: u32,
    ratio: u32,
    hd: u32,
    rot: u32,
    theta: f32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    if n_new == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel)
        .grid([n_new, 1, 1])
        .block([hd, 1, 1])
        .shared_mem((hd + 32) * 4)
        .arg_ptr(raw_keys)
        .arg_ptr(k_norm_w)
        .arg_ptr(block_keys)
        .arg_u32(first_block)
        .arg_u32(ratio)
        .arg_u32(hd)
        .arg_u32(rot)
        .arg_f32(theta)
        .arg_f32(eps)
        .launch(stream)
}

/// One decode query: per head, RMSNorm*(1+w) + partial rope at `pos` -> FP32.
#[allow(clippy::too_many_arguments)]
pub fn qsa_qprep(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_in: DevicePtr,
    q_norm_w: DevicePtr,
    q_out: DevicePtr,
    n_heads: u32,
    hd: u32,
    rot: u32,
    pos: u32,
    theta: f32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_heads, 1, 1])
        .block([hd, 1, 1])
        .shared_mem((hd + 32) * 4)
        .arg_ptr(q_in)
        .arg_ptr(q_norm_w)
        .arg_ptr(q_out)
        .arg_u32(hd)
        .arg_u32(rot)
        .arg_u32(pos)
        .arg_f32(theta)
        .arg_f32(eps)
        .launch(stream)
}

/// scores[b] = sum_h relu(q_h . k_b) / sqrt(hd) over `n_blocks` blocks.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    block_keys: DevicePtr,
    scores: DevicePtr,
    n_blocks: u32,
    n_heads: u32,
    hd: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_blocks, 1, 1])
        .block([hd, 1, 1])
        .shared_mem(32 * 4)
        .arg_ptr(q)
        .arg_ptr(block_keys)
        .arg_ptr(scores)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .launch(stream)
}

/// Pack the selected tokens' K/V rows into contiguous NHD scratch.
#[allow(clippy::too_many_arguments)]
pub fn qsa_gather(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    block_table: DevicePtr,
    sel: DevicePtr,
    k_out: DevicePtr,
    v_out: DevicePtr,
    n_sel: u32,
    block_size: u32,
    nkv: u32,
    hd: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_sel, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(block_table)
        .arg_ptr(sel)
        .arg_ptr(k_out)
        .arg_ptr(v_out)
        .arg_u32(block_size)
        .arg_u32(nkv)
        .arg_u32(hd)
        .launch(stream)
}

/// Stage 2: per-row q prep for a contiguous selective row range.
#[allow(clippy::too_many_arguments)]
pub fn qsa_qprep_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    qk: DevicePtr,
    q_norm_w: DevicePtr,
    q_out: DevicePtr,
    rows: u32,
    first_pos: u32,
    qkw: u32,
    n_heads: u32,
    hd: u32,
    rot: u32,
    theta: f32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, n_heads, 1])
        .block([hd, 1, 1])
        .shared_mem((hd + 32) * 4)
        .arg_ptr(qk)
        .arg_ptr(q_norm_w)
        .arg_ptr(q_out)
        .arg_u32(first_pos)
        .arg_u32(qkw)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .arg_u32(rot)
        .arg_f32(theta)
        .arg_f32(eps)
        .launch(stream)
}

/// Tensor-core `qsa_score_rows` (split-q). Geometry differs from the
/// per-(row,block) kernel: one CTA covers 16 rows x 64 blocks with 8 warps,
/// so the launch count drops from rows*blocks to ~1/1000th of that.
///
/// Measured 39.010 → 1.831 ms average per call under nsys on a 28K prefill
/// (qwen4_exp, GB10): ~21x. Selection-equivalent, not bit-exact — split-q
/// carries ~17 mantissa bits, and the consumer is a top-k.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score_rows_tc(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    block_keys: DevicePtr,
    scores: DevicePtr,
    rows: u32,
    n_blocks_max: u32,
    first_pos: u32,
    score_stride: u32,
    ratio: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows.div_ceil(16), n_blocks_max.div_ceil(64), 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(block_keys)
        .arg_ptr(scores)
        .arg_u32(first_pos)
        .arg_u32(score_stride)
        .arg_u32(ratio)
        .arg_u32(n_blocks_max)
        .launch(stream)
}

/// Stage 2: per-row block scores, -inf beyond each row's complete count.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    block_keys: DevicePtr,
    scores: DevicePtr,
    rows: u32,
    n_blocks_max: u32,
    first_pos: u32,
    score_stride: u32,
    ratio: u32,
    n_heads: u32,
    hd: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, n_blocks_max, 1])
        .block([hd, 1, 1])
        .shared_mem(32 * 4)
        .arg_ptr(q)
        .arg_ptr(block_keys)
        .arg_ptr(scores)
        .arg_u32(first_pos)
        .arg_u32(score_stride)
        .arg_u32(ratio)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .launch(stream)
}

/// Stage 2: per-row selected-set attention, overwriting the context rows.
#[allow(clippy::too_many_arguments)]
pub fn qsa_prefill_attn(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    block_table: DevicePtr,
    lists: DevicePtr,
    attn_out: DevicePtr,
    rows: u32,
    first_pos: u32,
    topk: u32,
    ratio: u32,
    block_size: u32,
    nq: u32,
    nkv: u32,
    hd: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    // 8 warps x [hd] acc partials + m/l per warp.
    let smem = (8 * hd + 16) * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([rows, nq, 1])
        .block([256, 1, 1])
        .shared_mem(smem)
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(block_table)
        .arg_ptr(lists)
        .arg_ptr(attn_out)
        .arg_u32(first_pos)
        .arg_u32(topk)
        .arg_u32(ratio)
        .arg_u32(block_size)
        .arg_u32(nq)
        .arg_u32(nkv)
        .arg_u32(hd)
        .arg_f32(inv_sqrt_d)
        .launch(stream)
}
