// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// GPU-side MoE top-K softmax.
///
/// Finds top-K experts from BF16 gate logits, computes softmax weights.
///
/// Kernel: `moe_topk_softmax(gate_logits, expert_indices, expert_weights,
///          num_experts, top_k, normalize)`
/// Grid: (1, 1, 1)  Block: (256, 1, 1)
pub fn moe_topk_softmax(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .launch(stream)
}

/// GPU-side MoE top-K sigmoid routing (Nemotron-H).
///
/// Uses sigmoid scoring (not softmax). Bias affects expert selection only,
/// not their weights. Weights come from pre-bias sigmoid scores.
///
/// Kernel: `moe_topk_sigmoid(gate_logits, bias, expert_indices, expert_weights,
///          num_experts, top_k, normalize, scaling_factor)`
/// Grid: (1, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_sigmoid(
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
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
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

// ── Batched MoE Expert GEMV ──────────────────────────────────
