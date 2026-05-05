// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

// ── Normalization ──────────────────────────────────────────────────

/// RMS normalization: output = rms_norm(input) * weight.
///
/// Kernel: `rms_norm(input, weight, output, hidden_size, eps)`
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
pub fn rms_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

/// Fused RMS norm + residual save: normed = rms_norm(input), residual = input.
///
/// Eliminates a separate D2D copy by writing the raw input to the residual
/// buffer in the same pass as the normalized output write.
///
/// Kernel: `rms_norm_residual(input, weight, output, residual, hidden_size, eps)`
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
pub fn rms_norm_residual(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    residual: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(residual)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

/// Fused residual add + RMS norm + residual save.
///
/// `hidden[i] += src[i]; normed = rms_norm(hidden) * (1+weight); residual = hidden`.
/// Eliminates one kernel launch per fusion site (48 per decode step).
///
/// Kernel: `residual_add_rms_norm(hidden, src, weight, output, residual, hidden_size, eps)`
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn residual_add_rms_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    src: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    residual: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(src)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(residual)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

/// Gated RMS norm (norm_before_gate=False, per-group):
///   output = rms_norm_per_group(input * silu(gate), weight, group_size)
///
/// Kernel: `gated_rms_norm(input, gate, weight, output, hidden_size, eps, gate_stride, group_size)`
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
pub fn gated_rms_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    gate_stride: u32,
    eps: f32,
    group_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .arg_u32(gate_stride)
        .arg_u32(group_size)
        .launch(stream)
}

/// Batched gated RMS norm for prefill: all (head, actual_token) pairs in one launch.
///
/// Grid: (heads_per_token, num_actual_tokens, 1)
/// Block: (min(head_dim, 1024), 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gated_rms_norm_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    heads_per_token: u32,
    head_dim: u32,
    eps: f32,
    num_actual_tokens: u32,
    input_token_stride: u32,
    gate_token_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([heads_per_token, num_actual_tokens, 1])
        .block([head_dim.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(head_dim)
        .arg_f32(eps)
        .arg_u32(input_token_stride)
        .arg_u32(gate_token_stride)
        .launch(stream)
}

// ── GEMM ───────────────────────────────────────────────────────────
