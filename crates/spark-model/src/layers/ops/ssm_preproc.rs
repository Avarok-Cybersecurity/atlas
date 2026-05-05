// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Deinterleave QKVZ projection output from GQA-grouped to sequential layout.
///
/// Input: interleaved [num_groups × (kd + kd + vpg*vd + vpg*vd)]
/// Output: sequential [Q_total | K_total | V_total | Z_total]
///
/// Kernel: `deinterleave_qkvz(interleaved, output, num_groups, head_k_dim,
///          vheads_per_group, head_v_dim)`
/// Grid: (ceil(total/256), 1, 1)  Block: (256, 1, 1)
pub fn deinterleave_qkvz(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    interleaved: DevicePtr,
    output: DevicePtr,
    num_tokens: u32,
    num_groups: u32,
    head_k_dim: u32,
    vheads_per_group: u32,
    head_v_dim: u32,
    stream: u64,
) -> Result<()> {
    let group_dim = 2 * head_k_dim + 2 * vheads_per_group * head_v_dim;
    let total = num_groups * group_dim;
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, div_ceil(total, 256), 1])
        .block([256, 1, 1])
        .arg_ptr(interleaved)
        .arg_ptr(output)
        .arg_u32(num_groups)
        .arg_u32(head_k_dim)
        .arg_u32(vheads_per_group)
        .arg_u32(head_v_dim)
        .launch(stream)
}

/// Deinterleave Q/Gate from per-head interleaved to contiguous layout (in-place).
///
/// Input layout:  [Q_h0(hd), G_h0(hd), Q_h1(hd), G_h1(hd), ...]
/// Output layout: [Q_h0(hd), Q_h1(hd), ..., G_h0(hd), G_h1(hd), ...]
///
/// Kernel: `deinterleave_qg(data, num_heads, head_dim)`
/// Grid: (1, 1, 1)  Block: (256, 1, 1)
/// Dynamic shared memory: num_heads * head_dim * 2 * 2 bytes
pub fn deinterleave_qg(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    stride: u32,
    stream: u64,
) -> Result<()> {
    let shared_bytes = num_heads * head_dim * 2 * 2; // BF16 = 2 bytes each
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .shared_mem(shared_bytes)
        .arg_ptr(data)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(stride)
        .launch(stream)
}

/// Deinterleave Q/Gate with split output — Q to separate contiguous buffer.
///
/// Same as [`deinterleave_qg`] but writes Q to `q_out` (contiguous [N, q_dim])
/// instead of in-place. Gate is still written back to `data` in-place.
/// Eliminates the per-token D2D copy loop.
pub fn deinterleave_qg_split(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    q_out: DevicePtr,
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    stride: u32,
    stream: u64,
) -> Result<()> {
    let shared_bytes = num_heads * head_dim * 2 * 2;
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .shared_mem(shared_bytes)
        .arg_ptr(data)
        .arg_ptr(q_out)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(stride)
        .launch(stream)
}

/// Fused deinterleave Q/Gate + per-head Q RMS norm.
///
/// Combines [`deinterleave_qg_split`] + Q RMS norm into a single kernel,
/// eliminating one global memory round-trip for Q data.
/// Gate is deinterleaved to `data[q_total..]`, Q is deinterleaved → normalized → `q_out`.
pub fn deinterleave_qg_split_qnorm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    q_out: DevicePtr,
    q_norm_weight: DevicePtr,
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    stride: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    let shared_bytes = num_heads * head_dim * 2 * 2;
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .shared_mem(shared_bytes)
        .arg_ptr(data)
        .arg_ptr(q_out)
        .arg_ptr(q_norm_weight)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(stride)
        .arg_f32(eps)
        .launch(stream)
}

/// Batched sigmoid gate multiply across multiple tokens.
///
/// Replaces per-token [`sigmoid_gate_mul`] launches with a single kernel.
/// `gate` is strided (gate_stride elements between tokens in gate buffer).
pub fn sigmoid_gate_mul_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    output: DevicePtr,
    dim: u32,
    gate_stride: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    let total = num_tokens * dim;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(output)
        .arg_u32(dim)
        .arg_u32(gate_stride)
        .arg_u32(total)
        .launch(stream)
}

/// Compute GDN gates from interleaved BA projection + learned A_log/dt_bias.
///
/// Outputs FP32 gate (decay) and beta (write gate) for each value head.
///
/// Kernel: `compute_gdn_gates(ba_interleaved, A_log, dt_bias, gate_out,
///          beta_out, num_v_heads, num_groups, vheads_per_group)`
/// Grid: (1, 1, 1)  Block: (num_v_heads, 1, 1)
pub fn compute_gdn_gates(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    ba_interleaved: DevicePtr,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    gate_out: DevicePtr,
    beta_out: DevicePtr,
    num_tokens: u32,
    num_v_heads: u32,
    num_groups: u32,
    vheads_per_group: u32,
    ba_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([num_v_heads, 1, 1])
        .arg_ptr(ba_interleaved)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(gate_out)
        .arg_ptr(beta_out)
        .arg_u32(num_v_heads)
        .arg_u32(num_groups)
        .arg_u32(vheads_per_group)
        .arg_u32(ba_stride)
        .launch(stream)
}

/// Fused BA projection + GDN gates: dense GEMV + gate/beta transforms.
///
/// Combines `dense_gemv(input, ba_weight, ba_out, N, K)` and
/// `compute_gdn_gates(ba_out, a_log, dt_bias, gate, beta)` into a single
/// kernel, eliminating the intermediate ba_out buffer and one graph node.
///
/// Kernel: `dense_gemv_ba_gates(A, B, A_log, dt_bias, gate, beta, N, K, vpg)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_ba_gates(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    ba_weight: &DenseWeight,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    gate_out: DevicePtr,
    beta_out: DevicePtr,
    n: u32,
    k: u32,
    vheads_per_group: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(ba_weight.weight)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(gate_out)
        .arg_ptr(beta_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(vheads_per_group)
        .launch(stream)
}

/// Fused BA GEMM + GDN gates for prefill (token-parallel).
///
/// Replaces `dense_gemm(normed, ba_weight) + compute_gdn_gates` in the prefill path.
/// Uses vectorized uint4 loads and warp-shuffle reduction per token, adding a token
/// dimension via blockIdx.y. Skips the intermediate ba_out buffer entirely.
///
/// Output layout (shared gate_out buffer):
///   gate_out[token * gate_stride + vh]      = gate (alpha→exp transform)
///   gate_out[token * gate_stride + nv + vh] = beta (sigmoid)
///
/// Kernel: `dense_gemm_ba_gates_prefill(A, B, A_log, dt_bias, gate_out, M, N, K,
///          K_stride, gate_stride, nv, vpg)`
/// Grid: (ceil(N/4), M_tokens, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_ba_gates_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,        // [M, K_stride] activations (BF16)
    ba_weight: &DenseWeight, // [N, K] BA weight (BF16, row-major)
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    gate_out: DevicePtr, // [M, gate_stride] FP32 unified gate+beta buffer
    m: u32,              // num_tokens
    n: u32,              // ba_size (64)
    k: u32,              // hidden_size (2048)
    k_stride: u32,       // BF16 elements between tokens in input (= k)
    gate_stride: u32,    // FP32 elements between tokens in gate_out (= 2*nv)
    nv: u32,             // num_v_heads (32)
    vheads_per_group: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), m, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(ba_weight.weight)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(gate_out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(k_stride)
        .arg_u32(gate_stride)
        .arg_u32(nv)
        .arg_u32(vheads_per_group)
        .launch(stream)
}

// ── Sampling ─────────────────────────────────────────────────────
