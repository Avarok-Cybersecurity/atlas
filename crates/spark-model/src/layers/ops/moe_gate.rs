// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Largest `top_k` the sigmoid routing kernels can hold.
///
/// `moe_topk_sigmoid` stages the running top-K in `__shared__ float
/// s_top_vals[MAX_TOP_K]` / `s_top_idxs[MAX_TOP_K]`, so a larger `top_k` walks
/// off the end of those arrays and into the shared block that follows them.
/// Nothing on the device can report that; the arrays are sized at compile time
/// and the selection loop was bounded by the expert count, not by the array.
///
/// The authoritative value is `#define MAX_TOP_K` in
/// `kernels/gb10/common/moe_topk_sigmoid.cu`. This mirror is pinned to it by
/// `tests/moe_topk_sigmoid_bounds.rs`, which also fails if a model directory
/// reintroduces a shadow copy of that kernel with a different cap — the drift
/// that had the two Nemotron shadows capped at 24 against a common file of 32.
pub const MOE_TOPK_SIGMOID_MAX_TOP_K: usize = 32;

/// Largest `num_experts` the sigmoid routing kernels can hold, from
/// `#define MAX_EXPERTS` in the same file. Beyond it the kernel silently
/// considers only the first `MAX_EXPERTS` experts (`actual_n` is a `min`), so
/// routing stays memory-safe but stops matching the checkpoint.
pub const MOE_TOPK_SIGMOID_MAX_EXPERTS: usize = 512;

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
/// LongCat softmax + e_score_correction_bias router with the zero-expert
/// fold (single token). `zero_accum` receives the token's summed
/// zero-expert weight; folded slots are rewritten (expert 0, weight 0).
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_softmax_bias(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    bias: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    zero_accum: DevicePtr,
    num_logits: u32, // routed + zero
    num_routed: u32,
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
        .arg_ptr(zero_accum)
        .arg_u32(num_logits)
        .arg_u32(num_routed)
        .arg_u32(top_k)
        .arg_u32(normalize as u32)
        .arg_f32(scaling_factor)
        .launch(stream)
}

/// Batched twin: one block per token.
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_softmax_bias_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    bias: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    zero_accum: DevicePtr,
    num_logits: u32,
    num_routed: u32,
    top_k: u32,
    normalize: bool,
    scaling_factor: f32,
    n: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(bias)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_ptr(zero_accum)
        .arg_u32(num_logits)
        .arg_u32(num_routed)
        .arg_u32(top_k)
        .arg_u32(normalize as u32)
        .arg_f32(scaling_factor)
        .launch(stream)
}

/// `out[t, :] += zero_accum[t] * x[t, :]` — the identity-expert blend.
pub fn moe_zero_expert_add(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    out: DevicePtr,
    x: DevicePtr,
    zero_accum: DevicePtr,
    n: u32,
    h: u32,
    stream: u64,
) -> Result<()> {
    use spark_runtime::kernel_args::div_ceil;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n * h, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(out)
        .arg_ptr(x)
        .arg_ptr(zero_accum)
        .arg_u32(n)
        .arg_u32(h)
        .launch(stream)
}

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

/// GPU-side MoE top-K sqrtsoftplus routing (DeepSeek-V4).
///
/// Uses sqrtsoftplus scoring (not sigmoid/softmax). Bias affects expert
/// selection only, not their weights. Weights come from pre-bias
/// sqrtsoftplus scores.
///
/// Kernel: `moe_topk_sqrtsoftplus(gate_logits, bias, expert_indices, expert_weights,
///          num_experts, top_k, normalize, scaling_factor)`
/// Grid: (1, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_sqrtsoftplus(
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

/// GPU-side MoE hash routing (DeepSeek-V4 hash_moe layers).
///
/// Expert selection is a static `tid2eid[token_id]` lookup (frozen table);
/// the learned gate still supplies the sqrtsoftplus scores that weight the
/// selected experts. Mirrors [`moe_topk_sqrtsoftplus`] but with static
/// selection instead of top-K.
///
/// Kernel: `moe_hash_route(gate_logits, tid2eid, token_id_ptr, expert_indices,
///          expert_weights, num_experts, top_k, normalize, scaling_factor)`
/// Grid: (1, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_hash_route(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    tid2eid: DevicePtr,
    token_id_ptr: DevicePtr,
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
        .arg_ptr(tid2eid)
        .arg_ptr(token_id_ptr)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .arg_f32(scaling_factor)
        .launch(stream)
}

/// Batched GPU-side MoE hash routing (DeepSeek-V4 hash_moe layers, prefill).
///
/// One block per token; reads `token_ids[N]` and the static `tid2eid` table.
/// Grid: (N, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_hash_route_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    tid2eid: DevicePtr,
    token_ids: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    scaling_factor: f32,
    n: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(tid2eid)
        .arg_ptr(token_ids)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .arg_f32(scaling_factor)
        .launch(stream)
}

// ── Batched MoE Expert GEMV ──────────────────────────────────
