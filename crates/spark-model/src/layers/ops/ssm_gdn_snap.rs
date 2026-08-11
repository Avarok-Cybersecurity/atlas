// SPDX-License-Identifier: AGPL-3.0-only

//! Launch wrappers for the exact-verify `_snap` kernel twins (issue #435
//! route (a)): the fused-norm GDN decode kernels with an inline per-token
//! h-state rollback snapshot, and the FP32-output fused verify conv.
//!
//! All three are OPTIONAL kernels (`try_kernel`, model-shadow staged): the
//! exact-verify arm falls back to the parent kernel + `copy_d2d_async`
//! snapshots when a handle is 0 — same bits, more launches.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::DenseWeight;

/// [`super::gdn_decode_f32_norm`] + inline h-state snapshot.
///
/// `h_inter` receives the post-update (post-state-norm-clamp) H — the same
/// bits left in `h_state` — or is skipped when NULL (the final verify
/// position, whose snapshot index has no reader). Same grid/block and
/// argument order as the parent, with `h_inter` appended.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_norm_snap(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    z_gate: DevicePtr,
    norm_weight: DevicePtr,
    output: DevicePtr,
    h_inter: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(z_gate)
        .arg_ptr(norm_weight)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_f32(eps)
        .arg_ptr(h_inter)
        .launch(stream)
}

/// [`super::gdn_decode_f32_strided_norm`] + inline h-state snapshot, for the
/// batched-verify arm at `batch_size = n` sequences.
///
/// `h_inter` is the snapshot base for THIS token position; sequences are
/// `h_inter_seq_stride` FP32 elements apart (the ssm-pool per-slot
/// intermediate stride — passed, not inferred, because pool slots are
/// `num_intermediates` snapshots wide while H itself is dense). NULL skips.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_strided_norm_snap(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    z_gate: DevicePtr,
    norm_weight: DevicePtr,
    output: DevicePtr,
    h_inter: DevicePtr,
    h_inter_seq_stride: u64,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    z_stride: u32,
    out_stride: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(z_gate)
        .arg_ptr(norm_weight)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(z_stride)
        .arg_u32(out_stride)
        .arg_f32(eps)
        .arg_ptr(h_inter)
        .arg_u64(h_inter_seq_stride)
        .launch(stream)
}

/// FP32-output twin of [`super::gdn_verify_fused_conv_kn`]: one launch for
/// all K verify positions of conv1d+SiLU+L2norm, FP32 conv rows (what the
/// sequential-decode-exact GDN chain reads), every per-token conv-state
/// rollback snapshot written inline. `output_stride` is in FP32 elements.
///
/// Kernel: `gdn_verify_fused_conv_kn_f32(conv_state, new_input, weight,
///          output, conv_state_inter, num_tokens, dim, d_conv, qk_channels,
///          head_dim, input_stride, output_stride, inter_stride, l2_eps)`
/// Grid: (ceil(dim/256), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_verify_fused_conv_kn_f32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    new_input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    conv_state_inter: DevicePtr,
    num_tokens: u32,
    dim: u32,
    d_conv: u32,
    qk_channels: u32,
    head_dim: u32,
    input_stride: u32,
    output_stride: u32,
    inter_stride: u32,
    l2_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(dim, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(new_input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(conv_state_inter)
        .arg_u32(num_tokens)
        .arg_u32(dim)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .arg_u32(inter_stride)
        .arg_f32(l2_eps)
        .launch(stream)
}
