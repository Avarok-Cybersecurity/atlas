// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Gated delta rule decode (recurrent SSM update, supports batched sequences).
///
/// Kernel: `gated_delta_rule_decode(h_state, query, key, value,
///          gate, beta, output, batch_size, num_k_heads, num_v_heads,
///          k_dim, v_dim)`
/// Grid: (num_v_heads, batch_size, 1)  Block: (128, 1, 1)
///
/// For batch_size > 1, h_state layout: [batch, num_v_heads, k_dim, v_dim].
pub fn gdn_decode(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
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
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .launch(stream)
}

/// FP32 GDN decode fused with gated RMS norm.
///
/// Produces the same BF16 post-gated-norm output that a separate
/// `gdn_decode_f32` + `gated_rms_norm_f32_input` pair would produce, while
/// avoiding the intermediate FP32 global write/read.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_norm(
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
        .launch(stream)
}

/// Register-resident token-sequential prefill recurrence (warm-replay path).
///
/// Kernel `gated_delta_rule_prefill_regresident`: one WARP owns one v-column,
/// holding the 128 k-rows of H in registers (4/lane) — no smem-H, no per-token
/// barriers, >=2 CTA/SM. Token-equal to WY4 (cosine 1.0) and ~2.9x faster.
/// Grid: (num_v_heads, batch, v_dim / 4)  Block: (128, 1, 1)  (4 warps/block).
/// Requires k_dim == 128 and v_dim % 4 == 0.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_regresident(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, v_dim / 4])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}

/// Split-v_dim prefill: 2 CTAs per v-head, 64 threads each.
/// FUSED conv1d_update_l2norm + recurrence + gated-RMS-norm decode.
///
/// Collapses the per-seq SSM decode chain `conv1d_l2norm -> gdn -> gated_norm`
/// into one launch (shorter critical path on the chain-depth-bound decode).
/// Race-free per-k-head grid: each block owns k-head `kh` and its `head_repeat`
/// v-heads, conv-updating its own q/k AND v `conv_state` exclusively.
/// Requires `head_repeat * v_dim == block`, `2*k_dim <= block`, `k_dim == v_dim`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_conv_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    conv_state: DevicePtr,
    new_input: DevicePtr,
    conv_weight: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    z_gate: DevicePtr,
    norm_weight: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    conv_dim: u32,
    d_conv: u32,
    l2_eps: f32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    let head_repeat = num_v_heads / num_k_heads;
    KernelLaunch::new(gpu, kernel)
        .grid([num_k_heads, batch_size, 1])
        .block([head_repeat * v_dim, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(conv_state)
        .arg_ptr(new_input)
        .arg_ptr(conv_weight)
        .arg_ptr(DevicePtr::NULL) // conv_bias
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
        .arg_u32(conv_dim)
        .arg_u32(d_conv)
        .arg_f32(l2_eps)
        .arg_f32(eps)
        .launch(stream)
}

/// Strided FP32 GDN decode for concurrent sequence decode.
///
/// Q/K/V are read from strided rows, typically the FP32 conv output laid out as
/// `[batch, Q | K | V]`. Gate/beta and output are also strided by batch row.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    out_stride: u32,
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
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(out_stride)
        .launch(stream)
}

/// Strided FP32 GDN decode fused with gated RMS norm.
///
/// Same recurrent update as `gdn_decode_f32_strided`, but writes BF16
/// post-gated-norm output directly and skips the intermediate FP32 output
/// buffer plus per-token gated-rms-norm launches.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_strided_norm(
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
        .launch(stream)
}

/// Fused 2-token GDN decode (speculative verification).
///
/// Processes exactly 2 tokens through GDN in a single kernel launch.
/// Saves intermediate H_1 state for rollback on draft rejection.
/// Reads H_0 once, computes both outputs and H_2 in 3 passes (vs 4 for
/// 2× sequential decode), with H_1 intermediate staying in L2 cache.
///
/// Q/K/V/gate/beta are accessed via stride params (in elements, not bytes)
/// to support layouts where tokens are interleaved with other data.
///
/// Kernel: `gated_delta_rule_chunk2(h_state, query, key, value, gate, beta,
///          output, h_state_intermediate, batch_size, num_k_heads,
///          num_v_heads, k_dim, v_dim, qk_stride, v_stride, gb_stride)`
/// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_chunk2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    h_state_intermediate: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
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
        .arg_ptr(output)
        .arg_ptr(h_state_intermediate)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}
