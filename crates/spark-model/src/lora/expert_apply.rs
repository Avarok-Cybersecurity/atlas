// SPDX-License-Identifier: AGPL-3.0-only

//! Feature-1 phase-1 correctness-first MoE LoRA apply.
//!
//! The routed base MoE GEMM stays BYTE-IDENTICAL: this module folds an additive
//! BF16 delta onto its output buffers via the existing `apply_lora_delta`
//! (GEMV/GEMM shrink→expand + `scaled_add`), so there is **no new CUDA kernel**.
//! Two injection shapes:
//!
//! - **router** (`apply_router_lora`): one `apply_lora_delta` folding
//!   `scale·(router_in @ Aᵀ) @ Bᵀ` onto the `[n, num_experts]` routing logits,
//!   BEFORE top-k selection — reproduces PEFT `mlp.gate` (a routing-logit delta).
//! - **experts** (`apply_expert_lora_sorted`): after the base grouped GEMM
//!   writes the sorted `[total_expanded, n_out]` output, loop the ADAPTED experts
//!   and fold each one's delta onto ITS contiguous row range (`expert_offsets`),
//!   BEFORE `moe_unpermute_reduce_indexed` so the router weight multiplies
//!   `base+delta` exactly like PEFT. For gate/up inject before `silu_mul`; for
//!   down inject after the down GEMM.
//!
//! PHASE-1 caveats (deliberate, throwaway scaffold — do NOT benchmark):
//!   * single-active adapter (installed-pair path; no per-request `seq_slot`
//!     routing over experts — that is the phase-2 2-D `(slot, expert)` grouped
//!     BGMV kernel);
//!   * the caller D2H-copies `expert_offsets` once per MoE layer to drive the
//!     host loop — this BREAKS CUDA-graph capture (legal in eager prefill;
//!     phase-2 removes it). See `MoeLayer` injection sites.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{ExpertLoraLayer, ExpertProj};
use crate::layers::ops::lora_delta::{LoraKernels, LoraPair, apply_lora_delta};

const BF16_BYTES: u64 = 2;

/// One expert's contiguous sorted-row range (the grouped-GEMM row block for
/// that expert). `rows == 0` experts are dropped by the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertWork {
    pub expert: u16,
    pub row_off: u32,
    pub rows: u32,
}

/// PURE: from the base MoE `expert_offsets` prefix-sum (`[num_experts + 1]`,
/// `expert_offsets[e]..expert_offsets[e+1]` = expert `e`'s sorted rows) and the
/// adapter's adapted-expert set, produce the (expert, row_off, rows) work-items
/// for the delta side-path. Experts with zero routed rows or a malformed offset
/// pair are skipped (never a panic). This is the correctness-critical mapping
/// and is unit-tested without a GPU.
pub fn expert_delta_workitems(expert_offsets: &[u32], adapted: &[u16]) -> Vec<ExpertWork> {
    let n_experts = expert_offsets.len().saturating_sub(1);
    let mut work = Vec::with_capacity(adapted.len());
    for &e in adapted {
        let e_us = e as usize;
        if e_us >= n_experts {
            continue; // out of range for this layer's routing table
        }
        let start = expert_offsets[e_us];
        let end = expert_offsets[e_us + 1];
        if end <= start {
            continue; // no tokens routed to this expert this step
        }
        work.push(ExpertWork {
            expert: e,
            row_off: start,
            rows: end - start,
        });
    }
    work
}

/// Fold one projection's delta over `rows` contiguous rows, CHUNKED so the
/// caller's scratch (`lora_xa >= max_rows·max_rank`, `lora_delta >=
/// max_rows·n_out` BF16) is never overrun. `rows > max_rows` is split into
/// `ceil(rows/max_rows)` `apply_lora_delta` folds over disjoint row blocks —
/// byte-identical to one fold (each row is independent).
#[allow(clippy::too_many_arguments)]
fn fold_chunked(
    gpu: &dyn GpuBackend,
    kernels: &LoraKernels,
    pair: &LoraPair,
    x: DevicePtr,
    base_out: DevicePtr,
    rows: u32,
    max_rows: u32,
    lora_xa: DevicePtr,
    lora_delta: DevicePtr,
    stream: u64,
) -> Result<()> {
    let step = max_rows.max(1);
    let mut done = 0u32;
    while done < rows {
        let m = (rows - done).min(step);
        let x_row = x.offset((done as u64 * pair.k_in as u64 * BF16_BYTES) as usize);
        let out_row = base_out.offset((done as u64 * pair.n_out as u64 * BF16_BYTES) as usize);
        apply_lora_delta(
            gpu, kernels, pair, x_row, out_row, m, lora_xa, lora_delta, stream,
        )?;
        done += m;
    }
    Ok(())
}

/// Fold the router (`mlp.gate`) LoRA delta onto the routing logits in place,
/// BEFORE top-k. `router_in` = `[n, hidden]`, `gate_logits` = `[n, num_experts]`
/// (modified in place). Chunked by `max_rows` (the scratch capacity in rows).
#[allow(clippy::too_many_arguments)]
pub fn apply_router_lora(
    gpu: &dyn GpuBackend,
    kernels: &LoraKernels,
    pair: &LoraPair,
    router_in: DevicePtr,
    gate_logits: DevicePtr,
    n: u32,
    max_rows: u32,
    lora_xa: DevicePtr,
    lora_delta: DevicePtr,
    stream: u64,
) -> Result<()> {
    fold_chunked(
        gpu,
        kernels,
        pair,
        router_in,
        gate_logits,
        n,
        max_rows,
        lora_xa,
        lora_delta,
        stream,
    )
}

/// Fold `proj`'s per-expert LoRA deltas onto the SORTED grouped-GEMM output.
///
/// `x` = the projection's sorted input (`[total_expanded, pair.k_in]`),
/// `base_out` = the projection's sorted output (`[total_expanded, pair.n_out]`,
/// modified in place). `expert_offsets_host` is the D2H copy of the device
/// `expert_offsets` (`[num_experts + 1]`). One `apply_lora_delta(m = rows)` per
/// adapted expert, over that expert's contiguous row block — byte-identical to
/// `rows` sequential `m=1` folds. Only experts with a pair for `proj` AND
/// non-zero routed rows launch.
#[allow(clippy::too_many_arguments)]
pub fn apply_expert_lora_sorted(
    gpu: &dyn GpuBackend,
    kernels: &LoraKernels,
    layer: &ExpertLoraLayer,
    proj: ExpertProj,
    expert_offsets_host: &[u32],
    x: DevicePtr,
    base_out: DevicePtr,
    max_rows: u32,
    lora_xa: DevicePtr,
    lora_delta: DevicePtr,
    stream: u64,
) -> Result<()> {
    let work = expert_delta_workitems(expert_offsets_host, &layer.adapted_experts());
    for w in work {
        let Some(pair) = layer.pair(w.expert, proj) else {
            continue; // this expert adapts a different projection only
        };
        let x_row = x.offset((w.row_off as u64 * pair.k_in as u64 * BF16_BYTES) as usize);
        let out_row = base_out.offset((w.row_off as u64 * pair.n_out as u64 * BF16_BYTES) as usize);
        fold_chunked(
            gpu, kernels, pair, x_row, out_row, w.rows, max_rows, lora_xa, lora_delta, stream,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "expert_apply_tests.rs"]
mod tests;
