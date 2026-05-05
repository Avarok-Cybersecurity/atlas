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

/// BF16 → FP32 conversion: `dst[i] = (float)src[i]`.
///
/// Kernel: `bf16_to_f32(src, dst, n)`
/// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
pub fn bf16_to_f32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
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
