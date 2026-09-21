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
    // false = contiguous state bases indexed by (b*num_v_heads+vh);
    // true  = device pointer TABLES, one entry per sequence. See
    // `gdn_decode_wy4` for the full rationale — contiguous is only correct at
    // batch_size==1 because the intermediate's pool stride is
    // num_intermediates x h_state's.
    state_is_table: bool,
    stream: u64,
) -> Result<()> {
    // HARD guard, not debug_assert: this compiles out in release, which is
    // exactly where the corruption would be silent.
    anyhow::ensure!(
        state_is_table || batch_size == 1,
        "gdn_decode_wy2: contiguous state addressing is only valid at \
         batch_size==1 (got {batch_size}) — the intermediate's pool stride is \
         num_intermediates x h_state's, so sequence 1's Hi0 would land on \
         sequence 0's Hi1. Stage pointer tables and pass state_is_table=true."
    );
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
        .arg_u32(u32::from(state_is_table))
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
    // false = contiguous state bases indexed by (b*num_v_heads+vh);
    // true  = device pointer TABLES, one entry per sequence. See
    // `gdn_decode_wy4` for the full rationale — contiguous is only correct at
    // batch_size==1 because the intermediates' pool stride is
    // num_intermediates x h_state's.
    state_is_table: bool,
    stream: u64,
) -> Result<()> {
    // HARD guard, not debug_assert: this compiles out in release, which is
    // exactly where the corruption would be silent.
    anyhow::ensure!(
        state_is_table || batch_size == 1,
        "gdn_decode_wy3: contiguous state addressing is only valid at \
         batch_size==1 (got {batch_size}) — the intermediates' pool stride is \
         num_intermediates x h_state's, so sequence 1's Hi0 would land on \
         sequence 0's Hi1. Stage pointer tables and pass state_is_table=true."
    );
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
        .arg_u32(u32::from(state_is_table))
        .launch(stream)
}
