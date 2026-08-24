// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Fused SiLU activation: output = SiLU(gate) * up.
///
/// Kernel: `silu_mul_separate(gate, up, output, n)`
/// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
pub fn silu_mul(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    output: DevicePtr,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(output)
        .arg_u32(num_elements)
        .launch(stream)
}

/// Fused SiLU·mul + per-token-group(128) FP8-E4M3 quantization — replaces the
/// `silu_mul` → `per_token_group_quant_fp8` pair on the W8A8 prefill down-path
/// without materializing the BF16 intermediate. Bit-identical to the pair
/// (product rounds through BF16 before the group max; same reduction order,
/// scale floor, and SATFINITE encode).
///
/// `out_bf16` is nullable (`DevicePtr::NULL`): pass the post-SiLU BF16 buffer
/// only when a downstream consumer needs it (expert down_proj LoRA fold).
///
/// Kernel: `silu_mul_quant_fp8(gate, up, out_fp8, a_scale, out_bf16, M, K)`
/// Grid: (M, 1, 1)  Block: (128, 1, 1). Caller must ensure `k % 128 == 0`
/// and `k / 128 <= 16` (SILU_QUANT_MAX_GROUPS) — fall back to the unfused
/// pair otherwise.
#[allow(clippy::too_many_arguments)]
pub fn silu_mul_quant_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    out_fp8: DevicePtr,
    a_scale: DevicePtr,
    out_bf16: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([m, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(out_fp8)
        .arg_ptr(a_scale)
        .arg_ptr(out_bf16)
        .arg_u32(m)
        .arg_u32(k)
        .launch(stream)
}

/// L2 normalization (in-place): `data[i] = data[i] / sqrt(sum(data^2) + eps)`.
///
/// Applied per head: data is [num_heads, head_dim], each head normalized independently.
/// Required for Gated Delta Net Q/K normalization (use_qk_l2norm_in_kernel=True).
///
/// Kernel: `l2_norm_bf16(data, head_dim, eps)`
/// Grid: (num_heads, 1, 1)  Block: (min(head_dim, 1024), 1, 1)
pub fn l2_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    num_heads: u32,
    head_dim: u32,
    eps: f32,
    num_tokens: u32,
    stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, num_tokens, 1])
        .block([head_dim.min(1024), 1, 1])
        .arg_ptr(data)
        .arg_u32(head_dim)
        .arg_f32(eps)
        .arg_u32(stride)
        .launch(stream)
}

/// Element-wise sigmoid gate: `output[i] = input[i] * sigmoid(gate[i])`.
///
/// Used for gated attention in Qwen3: attn_output = attn_output * sigmoid(q_gate).
///
/// Kernel: `sigmoid_gate_mul(input, gate, output, n)`
/// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
pub fn sigmoid_gate_mul(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    output: DevicePtr,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(output)
        .arg_u32(num_elements)
        .launch(stream)
}

/// Per-head sigmoid gate multiply with broadcast over head_dim.
///
/// Step 3.7 attention gate: `g_proj` produces one BF16 scalar per head.
/// This kernel applies `output[t,h,d] = input[t,h,d] * sigmoid(gate[t,h])`
/// where the sigmoid gate is broadcast across all `hd` dimensions of each head.
///
/// Kernel: `sigmoid_gate_mul_head_broadcast(input, gate, output, nq, hd, total)`
/// Grid: (ceil(total/256), 1, 1)  Block: (256, 1, 1)
pub fn sigmoid_gate_mul_head_broadcast(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    output: DevicePtr,
    nq: u32,
    hd: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    let total = num_tokens * nq * hd;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(output)
        .arg_u32(nq)
        .arg_u32(hd)
        .arg_u32(total)
        .launch(stream)
}

/// Per-head softplus gate multiply with broadcast over `head_dim`.
#[allow(clippy::too_many_arguments)]
pub fn softplus_gate_mul_head_broadcast(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    output: DevicePtr,
    nq: u32,
    hd: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    let total = num_tokens * nq * hd;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(output)
        .arg_u32(nq)
        .arg_u32(hd)
        .arg_u32(total)
        .launch(stream)
}

/// BF16 residual add: `residual[i] += src[i]` (in-place).
///
/// Kernel: `bf16_residual_add(residual, src, n)`
/// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
pub fn residual_add(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    residual: DevicePtr,
    src: DevicePtr,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(residual)
        .arg_ptr(src)
        .arg_u32(num_elements)
        .launch(stream)
}

/// BF16 scaled accumulate: `output[i] += scale * src[i]`.
///
/// Kernel: `bf16_scaled_add(output, src, scale, n)`
/// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
pub fn scaled_add(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    src: DevicePtr,
    scale: f32,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(src)
        .arg_f32(scale)
        .arg_u32(num_elements)
        .launch(stream)
}

/// Sigmoid-gated blend: output = output + sigmoid_gate * src.
///
/// Kernel: `bf16_sigmoid_blend(output, src, sigmoid_gate, n)`
/// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
pub fn sigmoid_blend(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    src: DevicePtr,
    sigmoid_gate: f32,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(src)
        .arg_f32(sigmoid_gate)
        .arg_u32(num_elements)
        .launch(stream)
}

// ── SSM Preprocessing ─────────────────────────────────────────────
