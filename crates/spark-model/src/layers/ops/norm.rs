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
/// Strided RMS norm: `num_groups` groups of `rows_per_group` rows in ONE launch,
/// groups `row_stride` ELEMENTS apart, rows packed at `hidden_size` inside a group.
///
/// `rms_norm` above assumes one packed [num_tokens, hidden_size] block. The
/// multi-seq q/k head-norms are packed only WITHIN a sequence — each sequence's
/// heads sit inside its own interleaved [Q|K|V|gate] block — so that path was
/// launching the packed kernel once per sequence (516 launches/step, 0.76 ms).
/// Bit-identical: one block per row either way, same math, same reduction.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    rows_per_group: u32,
    num_groups: u32,
    hidden_size: u32,
    eps: f32,
    row_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows_per_group, num_groups, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .arg_u32(row_stride)
        .launch(stream)
}

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

/// Warp-per-row RMS norm for SHORT rows — one warp per row instead of one
/// block, so the grid shrinks 8x and the reduction needs no shared memory or
/// barrier. Profitable exactly for the Qwen3 per-head `q_norm`/`k_norm` during
/// prefill (`num_rows = heads * seq`, `hidden_size = head_dim`), where the
/// block-per-row kernel measured ~43x above its bandwidth floor.
pub fn rms_norm_warp_row(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    num_rows: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    const ROWS_PER_BLOCK: u32 = 8;
    KernelLaunch::new(gpu, kernel)
        .grid([num_rows.div_ceil(ROWS_PER_BLOCK), 1, 1])
        .block([32 * ROWS_PER_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(num_rows)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

/// Gate for [`rms_norm_warp_row`]: short even rows, many of them.
/// Disable with `ATLAS_RMS_NORM_WARP_ROW=0`.
pub fn rms_norm_short_row_eligible(num_rows: u32, hidden_size: u32) -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var("ATLAS_RMS_NORM_WARP_ROW").as_deref() != Ok("0"));
    on && hidden_size <= 256 && hidden_size.is_multiple_of(2) && num_rows >= 1024
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

/// Dual-output fused residual add + RMS norm (ATLAS_FP32_ROUTING).
///
/// Same as `residual_add_rms_norm` (bf16 hidden/residual/output unchanged) but
/// ALSO writes the normed output in FP32 to `output_f32` for the MoE router GEMM,
/// removing the norm's bf16-store rounding from the routing-critical path.
///
/// Kernel: `residual_add_rms_norm_gatef32(hidden, src, weight, output,
///          output_f32, residual, hidden_size, eps)`
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn residual_add_rms_norm_gatef32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    src: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    output_f32: DevicePtr,
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
        .arg_ptr(output_f32)
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

/// DFlash2 grouped dynamic causal conv (`kernels/gb10/common/dflash2_conv.cu`).
///
/// `out[t,c] = sum_o (base[o,c] + dyn[t, (stage*ks+o)*groups + c/G]) * x[t-o,c]`
/// with causal zero-pad inside the block. `base` must be pre-offset to the
/// stage (`base_ptr + stage * ks * hidden * 2` bytes); `dyn` is the FULL
/// `[n, 2*ks*groups]` projection output — the kernel indexes the stage half.
///
/// `block_len` is the causal-reset stride: rows are treated as
/// `n / block_len` INDEPENDENT blocks so a cross-sequence batch cannot
/// convolve one sequence's first row with the previous sequence's tail.
/// Pass `n` (or 0) for the single-block case — bit-identical to the
/// pre-batch kernel.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_conv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    dyn_kernels: DevicePtr,
    base_stage: DevicePtr,
    out: DevicePtr,
    n: u32,
    hidden: u32,
    conv_kernel_size: u32,
    group_size: u32,
    stage: u32,
    block_len: u32,
    stream: u64,
) -> Result<()> {
    let total = n * hidden;
    KernelLaunch::new(gpu, kernel)
        .grid([total.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(dyn_kernels)
        .arg_ptr(base_stage)
        .arg_ptr(out)
        .arg_u32(n)
        .arg_u32(hidden)
        .arg_u32(conv_kernel_size)
        .arg_u32(group_size)
        .arg_u32(stage)
        .arg_u32(block_len)
        .launch(stream)
}

/// DFlash2 GPU selector, stage 1: per-draft-row top-16 of the tail logits.
/// grid.x = gamma-1 (block b handles row b+1), grid.y = `n_seq` bands.
/// `n_seq = 1` is the single-sequence launch, unchanged.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_selector_topk16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    cand_ids: DevicePtr,
    cand_vals: DevicePtr,
    gamma: u32,
    vocab: u32,
    n_seq: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([gamma - 1, n_seq.max(1), 1])
        .block([256, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(cand_ids)
        .arg_ptr(cand_vals)
        .arg_u32(vocab)
        .arg_u32(gamma)
        .launch(stream)
}

/// DFlash2 GPU selector, stage 2: `g[t] = hidden_projection @ normed[t]`.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_selector_proj(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden_projection: DevicePtr,
    normed_hidden: DevicePtr,
    g_out: DevicePtr,
    gamma: u32,
    rank: u32,
    hidden: u32,
    n_seq: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rank, gamma - 1, n_seq.max(1)])
        .block([128, 1, 1])
        .arg_ptr(hidden_projection)
        .arg_ptr(normed_hidden)
        .arg_ptr(g_out)
        .arg_u32(rank)
        .arg_u32(hidden)
        .arg_u32(gamma)
        .launch(stream)
}

/// DFlash2 GPU selector, stage 3: greedy chain walk; rewrites `drafts[1..]`.
///
/// grid.y = `n_seq` independent chains (band b walks its own rows seeded
/// from `anchors[b]`). Pass `anchors = DevicePtr::NULL` with `n_seq = 1` for
/// the single-sequence launch, which then reads the scalar `anchor` — the
/// pre-batch behaviour exactly.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_selector_chain(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    cand_ids: DevicePtr,
    cand_vals: DevicePtr,
    g: DevicePtr,
    pred_cb: DevicePtr,
    succ_cb: DevicePtr,
    drafts: DevicePtr,
    gamma: u32,
    rank: u32,
    anchor: u32,
    anchors: DevicePtr,
    n_seq: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, n_seq.max(1), 1])
        .block([256, 1, 1])
        .arg_ptr(cand_ids)
        .arg_ptr(cand_vals)
        .arg_ptr(g)
        .arg_ptr(pred_cb)
        .arg_ptr(succ_cb)
        .arg_ptr(drafts)
        .arg_u32(gamma)
        .arg_u32(rank)
        .arg_u32(anchor)
        .arg_ptr(anchors)
        .launch(stream)
}
