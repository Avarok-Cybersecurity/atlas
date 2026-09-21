// SPDX-License-Identifier: AGPL-3.0-only

//! Which experts, not the multiply.
//!
//! Split from `moe_grouped_a.rs` on the 500-line cap. The seam is the one the
//! file already drew a banner for: everything here answers "which experts does
//! this token go to", and what stays behind is the grouped GEMM that runs once
//! that is known. The three variants differ only in the normaliser the router
//! was trained with — softmax, sigmoid, sqrt-softplus — and choosing the wrong
//! one is silent, so they are worth reading side by side.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::super::*;

// ── Grouped MoE prefill ops ─────────────────────────────────────

/// Batched top-K softmax: N tokens in parallel.
///
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_softmax_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .launch(stream)
}

/// Batched sigmoid + correction-bias top-K MoE routing.
///
/// Kernel: `moe_topk_sigmoid_batched(gate_logits, bias, expert_indices,
///         expert_weights, num_experts, top_k, normalize, scaling_factor)`
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_sigmoid_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    bias: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    scaling_factor: f32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(bias)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .arg_f32(scaling_factor)
        .launch(stream)
}

/// Batched sqrtsoftplus + correction-bias routing (DeepSeek-V4 prefill).
///
/// Same I/O as [`moe_topk_sigmoid_batched`] but scores experts with
/// `sqrt(log(1+exp(logits)))` (matching the single-token decode path), so
/// V4 prefill and decode route identically. Grid (N) / Block (256).
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_sqrtsoftplus_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    bias: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    scaling_factor: f32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(bias)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .arg_f32(scaling_factor)
        .launch(stream)
}
