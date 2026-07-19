// SPDX-License-Identifier: AGPL-3.0-only

//! Manifold-Constrained Hyper-Connections (mHC) kernel dispatch (DeepSeek-V4).
//!
//! Wraps the `hyper_connection` module kernels (`hc_pre`, `hc_post`,
//! `hc_head`). The hidden state is stored BF16 as `[T, hc_mult, H]`
//! (stream-major per token). HC parameters are float32 device buffers.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Broadcast a single hidden state into `hc_mult` identical streams:
/// `streams[t, i, d] = hidden[t, d]`. One block per token.
pub fn hc_expand(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    streams: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(streams)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// Collapse `hc_mult` streams to one (RMS-rescaled mix → sigmoid `pre`
/// weighted sum) and emit `post` / `comb` (Sinkhorn) for the matching
/// `hc_post`. One block per token.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    hc_fn: DevicePtr,
    hc_scale: DevicePtr,
    hc_base: DevicePtr,
    y_out: DevicePtr,
    post_out: DevicePtr,
    comb_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    norm_eps: f32,
    hc_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(hc_fn)
        .arg_ptr(hc_scale)
        .arg_ptr(hc_base)
        .arg_ptr(y_out)
        .arg_ptr(post_out)
        .arg_ptr(comb_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(sinkhorn_iters)
        .arg_f32(norm_eps)
        .arg_f32(hc_eps)
        .launch(stream)
}

/// Expand the sublayer output back into `hc_mult` streams, mixing the saved
/// residual streams through the doubly-stochastic `comb`. `out` may alias
/// `residual`. One block per token.
#[allow(clippy::too_many_arguments)]
pub fn hc_post(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    block_out: DevicePtr,
    residual: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(residual)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// Final collapse before the LM head: a single learned sigmoid-weighted sum
/// over the `hc_mult` streams. One block per token.
#[allow(clippy::too_many_arguments)]
pub fn hc_head(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    head_fn: DevicePtr,
    head_scale: DevicePtr,
    head_base: DevicePtr,
    y_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    hc_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(head_fn)
        .arg_ptr(head_scale)
        .arg_ptr(head_base)
        .arg_ptr(y_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_f32(norm_eps)
        .arg_f32(hc_eps)
        .launch(stream)
}
