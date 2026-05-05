// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Causal conv1d update (decode step, supports batched sequences).
///
/// Kernel: `causal_conv1d_update(conv_state, new_input, weight, bias,
///          output, batch, dim, d_conv)`
/// Grid: (ceil(dim/256), batch, 1)  Block: (256, 1, 1)
///
/// For batch > 1, conv_state and input must be contiguous [batch, ...].
pub fn conv1d_update(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    batch_size: u32,
    stream: u64,
) -> Result<()> {
    let bias_ptr = DevicePtr::NULL;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), batch_size, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias_ptr)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .launch(stream)
}

/// Fused conv1d update + SiLU + L2 normalization for Q/K channels.
///
/// Combines `causal_conv1d_update` and `l2_norm_bf16` into a single kernel.
/// Q+K channels (0..qk_channels) get L2-normalized per head after SiLU.
/// V channels (qk_channels..d_inner) get SiLU only.
///
/// Saves 1 kernel launch per SSM layer (36 launches/step for 35B/80B).
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_l2norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    batch_size: u32,
    qk_channels: u32,
    head_dim: u32,
    l2_eps: f32,
    stream: u64,
) -> Result<()> {
    let bias_ptr = DevicePtr::NULL;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), batch_size, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias_ptr)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_f32(l2_eps)
        .launch(stream)
}

/// Multi-token conv1d sliding window update + SiLU for prefill.
///
/// Processes `seq_len` tokens sequentially per channel in registers.
/// Input/output may be non-contiguous (different strides between tokens).
///
/// Kernel: `causal_conv1d_update_prefill(conv_state, input, weight, bias,
///          output, dim, d_conv, seq_len, input_stride, output_stride)`
/// Grid: (ceil(dim/256), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    bias: DevicePtr,
    output: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    seq_len: u32,
    input_stride: u32,
    output_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias)
        .arg_ptr(output)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(seq_len)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .launch(stream)
}

/// Mamba-2 SSM prefill: sequential recurrence across `seq_len` tokens in a single kernel.
///
/// Same algorithm as decode but loops over tokens, avoiding per-token launch overhead.
/// Supports non-contiguous layouts via per-tensor strides (BF16 elements between tokens).
///
/// Grid: (num_heads, batch_size, 1)  Block: (state_size, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssm_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    x: DevicePtr,
    b_proj: DevicePtr,
    c_proj: DevicePtr,
    dt_raw: DevicePtr,
    a_log: DevicePtr,
    d_param: DevicePtr,
    dt_bias: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    state_size: u32,
    n_groups: u32,
    dt_min: f32,
    dt_max: f32,
    x_stride: u32,
    bc_stride: u32,
    dt_stride: u32,
    y_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, batch_size, 1])
        .block([state_size, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(x)
        .arg_ptr(b_proj)
        .arg_ptr(c_proj)
        .arg_ptr(dt_raw)
        .arg_ptr(a_log)
        .arg_ptr(d_param)
        .arg_ptr(dt_bias)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(state_size)
        .arg_u32(n_groups)
        .arg_f32(dt_min)
        .arg_f32(dt_max)
        .arg_u32(x_stride)
        .arg_u32(bc_stride)
        .arg_u32(dt_stride)
        .arg_u32(y_stride)
        .launch(stream)
}

/// Persistent Mamba-2 SSM prefill: H in shared memory, reduces global traffic.
/// Same parameters and launch config as mamba2_ssm_prefill.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssm_prefill_persistent(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    x: DevicePtr,
    b_proj: DevicePtr,
    c_proj: DevicePtr,
    dt_raw: DevicePtr,
    a_log: DevicePtr,
    d_param: DevicePtr,
    dt_bias: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    state_size: u32,
    n_groups: u32,
    dt_min: f32,
    dt_max: f32,
    x_stride: u32,
    bc_stride: u32,
    dt_stride: u32,
    y_stride: u32,
    stream: u64,
) -> Result<()> {
    // H_smem + smem_x + smem_warp
    let smem = head_dim * state_size * 4 + head_dim * 4 + 4 * head_dim * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, batch_size, 1])
        .block([state_size, 1, 1])
        .shared_mem(smem)
        .arg_ptr(h_state)
        .arg_ptr(x)
        .arg_ptr(b_proj)
        .arg_ptr(c_proj)
        .arg_ptr(dt_raw)
        .arg_ptr(a_log)
        .arg_ptr(d_param)
        .arg_ptr(dt_bias)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(state_size)
        .arg_u32(n_groups)
        .arg_f32(dt_min)
        .arg_f32(dt_max)
        .arg_u32(x_stride)
        .arg_u32(bc_stride)
        .arg_u32(dt_stride)
        .arg_u32(y_stride)
        .launch(stream)
}
