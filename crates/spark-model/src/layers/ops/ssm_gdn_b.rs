// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Fused 3-token GDN decode (K=3 speculative verification).
///
/// Processes exactly 3 tokens through GDN in a single kernel launch.
/// Saves 2 intermediate H states (H_1, H_2) for rollback on draft rejection.
/// 4 passes vs 6 for 3× sequential decode.
///
/// Kernel: `gated_delta_rule_chunk3(h_state, query, key, value, gate, beta,
///          output, h_inter0, h_inter1, batch_size, num_k_heads,
///          num_v_heads, k_dim, v_dim, qk_stride, v_stride, gb_stride)`
/// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_chunk3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    h_state_inter0: DevicePtr,
    h_state_inter1: DevicePtr,
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
        .arg_ptr(h_state_inter0)
        .arg_ptr(h_state_inter1)
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

/// WY-chunkwise 2-token GDN decode (2-pass algorithm).
///
/// Drop-in replacement for `gdn_decode_chunk2`. Computes both H^T @ k_t
/// dot products in a single pass over H, then applies WY algebraic correction.
/// 2 passes vs 3, reducing memory traffic by 33%.
///
/// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_wy2(
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

/// WY-chunkwise 3-token GDN decode (2-pass algorithm).
///
/// Drop-in replacement for `gdn_decode_chunk3`. All 3 H^T @ k_t dot products
/// computed in a single pass. 2 passes vs 4, reducing memory traffic by 50%.
///
/// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_wy3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    h_state_inter0: DevicePtr,
    h_state_inter1: DevicePtr,
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
        .arg_ptr(h_state_inter0)
        .arg_ptr(h_state_inter1)
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

/// WY-chunkwise 4-token GDN decode (2-pass algorithm).
///
/// All 4 H^T @ k_t dot products computed in a single pass, then WY correction
/// derives v_new values. Second pass applies all 4 state updates + outputs.
/// 2 passes vs 5, reducing memory traffic by 60%.
///
/// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_wy4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    h_state_inter0: DevicePtr,
    h_state_inter1: DevicePtr,
    h_state_inter2: DevicePtr,
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
        .arg_ptr(h_state_inter0)
        .arg_ptr(h_state_inter1)
        .arg_ptr(h_state_inter2)
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

/// WY-Chunkwise Gated Delta Rule for K=17 verification (DFlash γ+1).
/// Computes 17 H·k dot products in 1 pass over H, applies WY algebraic
/// correction over 17 tokens (136 inter-token k-dot products), then
/// applies 17 state updates in a second fused pass writing
/// Hi_0..Hi_15 + final H.
///
/// `h_state_inter_base` points to a contiguous pool of (K-1)=16
/// intermediate H states per (layer, slot). Each Hi_t is at
/// `h_state_inter_base + t * inter_stride_floats` (per (b, vh) sub-region).
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_wy17(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    h_state_inter_base: DevicePtr,
    inter_stride_floats: u32,
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
        .arg_ptr(h_state_inter_base)
        .arg_u32(inter_stride_floats)
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

/// Fused 2-token conv1d sliding window update + SiLU.
///
/// Each thread handles one channel independently. The 2-token dependency
/// (token 1's window includes token 0's input) is resolved in registers.
/// Saves intermediate conv_state (after token 0) for rollback.
///
/// Kernel: `causal_conv1d_update_chunk2(conv_state, input, weight, bias,
///          output, conv_state_intermediate, batch, dim, d_conv)`
/// Grid: (ceil(dim/256), batch, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_chunk2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    conv_state_intermediate: DevicePtr,
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
        .arg_ptr(conv_state_intermediate)
        .arg_u32(batch_size)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .launch(stream)
}

// ── Activations / Element-wise ─────────────────────────────────────
