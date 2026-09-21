// SPDX-License-Identifier: AGPL-3.0-only

//! PLE kernel dispatch — the gate, the dilated depthwise conv, and the
//! highway add. See `kernels/gb10/qwen3.8-flash-next/nvfp4/ple.cu`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Gate the n-gram value by the highway, and emit both the gated value and
/// its `norm_conv`'d twin.
///
/// `hidden` is the FP32 mHC highway `[T, hc*H]`; `key`/`value` are the BF16
/// projection outputs. **Both outputs are FP32** — the whole PLE chain is,
/// because its result lands on the FP32 highway; see the PRECISION NOTE in
/// `ple.cu`. One block per token.
#[allow(clippy::too_many_arguments)]
pub fn ple_gate(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    norm_query_w: DevicePtr,
    norm_key_w: DevicePtr,
    norm_conv_w: DevicePtr,
    gated_out: DevicePtr,
    gated_normed: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(norm_query_w)
        .arg_ptr(norm_key_w)
        .arg_ptr(norm_conv_w)
        .arg_ptr(gated_out)
        .arg_ptr(gated_normed)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_f32(norm_eps)
        .launch(stream)
}

/// Depthwise causal conv, kernel `k_size`, **dilation `dilation`**, plus the
/// SiLU and the residual add against the un-normalized gated value.
///
/// Everything but `weight` is FP32 — see the PRECISION NOTE in `ple.cu`.
///
/// `state` is `[(k_size-1)*dilation, channels]` and is rolled in place, so
/// prefill and decode share one launch — there is no decode twin to drift.
#[allow(clippy::too_many_arguments)]
pub fn ple_conv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    gated: DevicePtr,
    weight: DevicePtr,
    state: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    channels: u32,
    k_size: u32,
    dilation: u32,
    stream: u64,
) -> Result<()> {
    let threads = 256u32;
    KernelLaunch::new(gpu, kernel)
        .grid([channels.div_ceil(threads), 1, 1])
        .block([threads, 1, 1])
        .arg_ptr(x)
        .arg_ptr(gated)
        .arg_ptr(weight)
        .arg_ptr(state)
        .arg_ptr(out)
        .arg_u32(num_tokens)
        .arg_u32(channels)
        .arg_u32(k_size)
        .arg_u32(dilation)
        .launch(stream)
}

/// `highway += ple_out`, in FP32. The reference adds PLE's output to the
/// residual before that layer's attention hyper-connection.
pub fn ple_add_highway(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    ple_out: DevicePtr,
    hidden: DevicePtr,
    n: u32,
    stream: u64,
) -> Result<()> {
    let threads = 256u32;
    KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(threads), 1, 1])
        .block([threads, 1, 1])
        .arg_ptr(ple_out)
        .arg_ptr(hidden)
        .arg_u32(n)
        .launch(stream)
}
