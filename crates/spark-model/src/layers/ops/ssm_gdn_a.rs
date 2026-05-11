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

/// Gated delta rule prefill (multi-token, sequential SSM update within kernel).
///
/// Processes `seq_len` tokens sequentially per (batch, head) pair.
/// Supports strided access: Q/K/V/gate/beta may have different strides
/// between tokens (e.g., from conv1d output with interleaved Q|K|V layout).
///
/// Kernel: `gated_delta_rule_prefill(h_state, query, key, value,
///          gate, beta, output, batch_size, seq_len, num_k_heads,
///          num_v_heads, k_dim, v_dim, qk_stride, v_stride, gb_stride)`
/// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill(
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
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .shared_mem(4 * k_dim * 4) // double-buffered k[128]+q[128] × 2 buffers × 4 bytes
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
///
/// Kernel: `gated_delta_rule_prefill_split(h_state, query, key, value,
///          gate, beta, output, batch_size, seq_len, num_k_heads,
///          num_v_heads, k_dim, v_dim, qk_stride, v_stride, gb_stride)`
/// Grid: (num_v_heads * 2, batch, 1)  Block: (64, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_split(
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
        .grid([num_v_heads * 2, batch_size, 1])
        .block([64, 1, 1])
        .shared_mem(4 * k_dim * 4) // double-buffered k[K_DIM]+q[K_DIM] × 2 buffers × 4 bytes
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

/// 4-way split prefill: 4 CTAs per v-head, 32 threads each (128 total CTAs).
///
/// Kernel: `gated_delta_rule_prefill_split4(h_state, query, key, value,
///          gate, beta, output, batch_size, seq_len, num_k_heads,
///          num_v_heads, k_dim, v_dim, qk_stride, v_stride, gb_stride)`
/// Grid: (num_v_heads * 4, batch, 1)  Block: (32, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_split4(
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
        .grid([num_v_heads * 4, batch_size, 1])
        .block([32, 1, 1])
        .shared_mem(4 * k_dim * 4) // double-buffered k[K_DIM]+q[K_DIM] × 2 buffers × 4 bytes
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

/// Persistent GDN prefill — h_state stays in shared memory for entire sequence.
///
/// Same parameters as gdn_prefill_split4 but uses persistent CTAs with
/// 128 threads and 67 KB shared memory. Each CTA processes ALL tokens for
/// one v_head, keeping h_state in shared memory (never written to global
/// until the end). Targets L2 bandwidth (~3 TB/s) instead of LPDDR5X (273 GB/s).
///
/// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
/// Shared: k_dim*v_dim*4 + 4*k_dim*4 bytes
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_persistent(
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
    let smem = k_dim * v_dim * 4 + 4 * k_dim * 4; // h_state + double-buffered k/q
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .shared_mem(smem)
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

/// Persistent GDN prefill with explicit shared memory size.
/// Used for WY4-persistent variant which needs more shared memory.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_persistent_smem(
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
    smem: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .shared_mem(smem)
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
