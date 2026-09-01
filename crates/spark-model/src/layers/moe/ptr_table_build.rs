// SPDX-License-Identifier: AGPL-3.0-only

//! Device-side expert pointer-table builders (NVFP4 / BF16 / FP8).
//!
//! Extracted from `mod.rs` (Wave: ARM-2 native-MXFP4) to keep it under the
//! 500-LoC cap. One device pointer array per projection across all experts,
//! consumed by the batched/grouped MoE GEMMs. Re-exported from `mod.rs`
//! (`pub(crate) use ptr_table_build::*`), so all call sites are unchanged.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::exl3::{Exl3Codebook, Exl3Weight};

use super::tables::{
    EXL3_MOE_OVERFLOW_CHUNK_ROWS, EXL3_MOE_PREFILL_BATCH_TOKENS_DEFAULT,
    EXL3_MOE_SLOT_BATCH_TOKENS, Exl3ExpertPtrTable, Exl3MoeState,
};
use super::{ExpertPtrTable, Fp8ExpertPtrTable, MoeLayer};
use crate::layers::ops::Exl3LaunchState;
use crate::weight_map::{DenseWeight, ExpertWeight, Fp8ExpertWeight, Fp8Weight, QuantizedWeight};

/// Build a device-side pointer table from pre-transposed QuantizedWeight vec.
pub(crate) fn build_ptr_table_from_qw(
    weights: &[QuantizedWeight],
    gpu: &dyn GpuBackend,
) -> Result<ExpertPtrTable> {
    let n = weights.len();
    let packed_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|w| w.weight.0.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|w| w.weight_scale.0.to_le_bytes())
        .collect();
    let scale2_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|w| w.weight_scale_2.to_le_bytes())
        .collect();

    let packed_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&packed_bytes, packed_ptrs)?;
    let scale_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&scale_bytes, scale_ptrs)?;
    let scale2_vals = gpu.alloc(n * 4)?;
    gpu.copy_h2d(&scale2_bytes, scale2_vals)?;

    Ok(ExpertPtrTable {
        packed_ptrs,
        scale_ptrs,
        scale2_vals,
    })
}

/// Build a device-side pointer table for one projection across all experts.
pub(crate) fn build_ptr_table(
    experts: &[ExpertWeight],
    proj: impl Fn(&ExpertWeight) -> &crate::weight_map::QuantizedWeight,
    gpu: &dyn GpuBackend,
) -> Result<ExpertPtrTable> {
    let n = experts.len();

    // Build host-side arrays
    let packed_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight.0.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight_scale.0.to_le_bytes())
        .collect();
    let scale2_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight_scale_2.to_le_bytes())
        .collect();

    // Upload to device
    let packed_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&packed_bytes, packed_ptrs)?;

    let scale_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&scale_bytes, scale_ptrs)?;

    let scale2_vals = gpu.alloc(n * 4)?;
    gpu.copy_h2d(&scale2_bytes, scale2_vals)?;

    Ok(ExpertPtrTable {
        packed_ptrs,
        scale_ptrs,
        scale2_vals,
    })
}

/// Build a device-side FP8 pointer table for one projection across all experts.
///
/// FP8 experts store 2 arrays (weight + block_scale) per projection,
/// vs NVFP4's 3 (packed + scale + scale2).
/// Build a device-side BF16 pointer table for one projection across all
/// experts. Used by the FP8-dequant-to-BF16 MoE path; one device pointer
/// per expert pointing at that expert's `[N, K]` BF16 weight buffer.
pub(crate) fn build_bf16_ptr_table(
    experts: &[DenseWeight],
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let n = experts.len();
    let weight_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| e.weight.0.to_le_bytes())
        .collect();
    let ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&weight_bytes, ptrs)?;
    Ok(ptrs)
}

/// Build the DENSE local EXL3 pointer table for one projection
/// (`ATLAS_EXL3_NATIVE_MOE=1`).
///
/// `experts` is GLOBAL-indexed (`len == num_experts`); `None` marks a remote
/// expert under EP. The `Some` entries MUST form one contiguous run — the
/// EP-local range — and the table is built densely over exactly that run
/// (entry `i` = global `local_start + i`, no null entries; see
/// `tables.rs::Exl3ExpertPtrTable` for why nulls under the mgemm weighted
/// reduction are silent corruption). Uniform `(k_bits, cb)` and geometry
/// across the run are re-validated here (one mgemm launch = one template).
pub(crate) fn build_exl3_ptr_table(
    experts: &[Option<Exl3Weight>],
    gpu: &dyn GpuBackend,
) -> Result<Exl3ExpertPtrTable> {
    let locals: Vec<(usize, &Exl3Weight)> = experts
        .iter()
        .enumerate()
        .filter_map(|(i, w)| w.as_ref().map(|w| (i, w)))
        .collect();
    ensure!(
        !locals.is_empty(),
        "EXL3 expert ptr table: no local experts (every entry None)"
    );
    let local_start = locals[0].0;
    for (pos, (i, _)) in locals.iter().enumerate() {
        ensure!(
            *i == local_start + pos,
            "EXL3 expert ptr table: local experts are not one contiguous run \
             (gap before global id {i}; run starts at {local_start}) — the \
             dense-local/-1-index encoding requires the contiguous EP range"
        );
    }
    let first = locals[0].1;
    let cb = match first.cb {
        Exl3Codebook::Mcg => 1u32,
        Exl3Codebook::Mul1 => 2u32,
        Exl3Codebook::Inst3 => anyhow::bail!(
            "EXL3 expert ptr table: cb0/\"3inst\" has no compiled kernel \
             instances — the materialize pass should not have kept this layer"
        ),
    };
    for (i, w) in &locals {
        ensure!(
            w.k_bits == first.k_bits
                && w.cb == first.cb
                && w.in_dim == first.in_dim
                && w.out_dim == first.out_dim,
            "EXL3 expert ptr table: expert {i} (K={} cb={:?} [{}x{}]) differs \
             from the projection template (K={} cb={:?} [{}x{}]) — one mgemm \
             launch decodes at ONE template",
            w.k_bits,
            w.cb,
            w.in_dim,
            w.out_dim,
            first.k_bits,
            first.cb,
            first.in_dim,
            first.out_dim,
        );
    }

    let trellis_bytes: Vec<u8> = locals
        .iter()
        .flat_map(|(_, w)| w.trellis.0.to_le_bytes())
        .collect();
    let suh_bytes: Vec<u8> = locals
        .iter()
        .flat_map(|(_, w)| w.suh.0.to_le_bytes())
        .collect();
    let svh_bytes: Vec<u8> = locals
        .iter()
        .flat_map(|(_, w)| w.svh.0.to_le_bytes())
        .collect();

    // All-or-nothing: roll back earlier arrays if a later alloc/upload fails.
    let mut owned: Vec<DevicePtr> = Vec::with_capacity(3);
    let mut upload = |bytes: &[u8]| -> Result<DevicePtr> {
        let r = gpu.alloc(bytes.len()).and_then(|p| {
            owned.push(p);
            gpu.copy_h2d(bytes, p).map(|_| p)
        });
        if r.is_err() {
            for p in owned.drain(..) {
                gpu.free(p).ok();
            }
        }
        r
    };
    let trellis_ptrs = upload(&trellis_bytes)?;
    let suh_ptrs = upload(&suh_bytes)?;
    let svh_ptrs = upload(&svh_bytes)?;
    let host_ptrs: Vec<[u64; 3]> = locals
        .iter()
        .map(|(_, w)| [w.trellis.0, w.suh.0, w.svh.0])
        .collect();

    Ok(Exl3ExpertPtrTable {
        trellis_ptrs,
        suh_ptrs,
        svh_ptrs,
        host_ptrs,
        num_local: locals.len(),
        local_start,
        k_bits: first.k_bits,
        cb,
        in_dim: first.in_dim,
        out_dim: first.out_dim,
    })
}

impl Exl3MoeState {
    /// Allocate the per-model mgemm slot-batched slabs over the shared
    /// `launch` state (locks + fence), all-or-nothing with rollback (the
    /// `Exl3LmHead::new` pattern). One named call site so the alloc ledger
    /// shows one legible row.
    pub(crate) fn new(
        gpu: &dyn GpuBackend,
        launch: std::sync::Arc<Exl3LaunchState>,
        hidden: usize,
        inter: usize,
        top_k: usize,
        num_experts: usize,
    ) -> Result<Self> {
        ensure!(
            hidden >= inter && top_k >= 1 && num_experts >= 1,
            "EXL3 MoE state: geometry hidden={hidden} inter={inter} top_k={top_k} \
             num_experts={num_experts} (the shared A_had slab assumes inter <= hidden)"
        );
        let s_cap = EXL3_MOE_SLOT_BATCH_TOKENS * top_k;
        let sm_count = launch.sm_count;
        ensure!(
            sm_count >= 8,
            "EXL3 MoE state: the fused prefill kernel needs >= 8 SMs \
             (MOE_SMS_PER_EXPERT), have {sm_count}"
        );
        // Prefill-tier sizing: token-batch cap (env-overridable), fused
        // temp-slab count C = sm/8 (the kernel's max concurrency), fixed
        // overflow chunk.
        let pf_t_cap = std::env::var("ATLAS_EXL3_MOE_PREFILL_BATCH_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 1)
            .unwrap_or(EXL3_MOE_PREFILL_BATCH_TOKENS_DEFAULT);
        let pf_concurrency = (sm_count as usize / 8).clamp(1, 64);
        let pf_e_cap = num_experts; // >= any EP-local width
        let locks = launch.locks;
        let mut owned: Vec<DevicePtr> = Vec::new();
        let mut alloc = |bytes: usize| -> Result<DevicePtr> {
            match gpu.alloc(bytes) {
                Ok(p) => {
                    owned.push(p);
                    Ok(p)
                }
                Err(e) => {
                    for p in owned.drain(..) {
                        gpu.free(p).ok();
                    }
                    Err(e)
                }
            }
        };
        let a_f16 = alloc(s_cap * hidden * 2)?;
        let a_had_f16 = alloc(s_cap * hidden * 2)?;
        let c_gate_f32 = alloc(s_cap * inter * 4)?;
        let c_up_f32 = alloc(s_cap * inter * 4)?;
        let c_down_f32 = alloc(s_cap * hidden * 4)?;
        let inter_f16 = alloc(s_cap * inter * 2)?;
        let b_indices = alloc(s_cap * 8)?;
        let b_weights = alloc(s_cap * 4)?;
        // Prefill tier: ingress/accumulator + fused temp slabs + sorted-slot
        // maps + overflow chunk slabs.
        let ov = EXL3_MOE_OVERFLOW_CHUNK_ROWS;
        let pf_hidden_f16 = alloc(pf_t_cap * hidden * 2)?;
        let pf_out_f32 = alloc(pf_t_cap * hidden * 4)?;
        let pf_temp_state_g = alloc(pf_concurrency * 128 * hidden * 2)?;
        let pf_temp_state_u = alloc(pf_concurrency * 128 * hidden * 2)?;
        let pf_temp_inter_g = alloc(pf_concurrency * 128 * inter * 2)?;
        let pf_temp_inter_u = alloc(pf_concurrency * 128 * inter * 2)?;
        let pf_token_sorted = alloc(pf_t_cap * top_k * 8)?;
        let pf_weight_sorted = alloc(pf_t_cap * top_k * 2)?;
        let pf_expert_count = alloc((pf_e_cap + 1) * 8)?;
        let pf_ov_a = alloc(ov * hidden * 2)?;
        let pf_ov_a_had = alloc(ov * hidden * 2)?;
        let pf_ov_gate = alloc(ov * inter * 2)?;
        let pf_ov_up = alloc(ov * inter * 2)?;
        let pf_ov_down = alloc(ov * hidden * 4)?;
        let total: usize = s_cap * (hidden * 2 * 2 + inter * 4 * 2 + hidden * 4 + inter * 2 + 12)
            + pf_t_cap * (hidden * 6 + top_k * 10)
            + pf_concurrency * 128 * (hidden + inter) * 4
            + (pf_e_cap + 1) * 8
            + ov * (hidden * 8 + inter * 4);
        tracing::info!(
            "EXL3 native MoE state allocated: {s_cap} decode slots \
             ({EXL3_MOE_SLOT_BATCH_TOKENS} tokens x top_k {top_k}) + prefill \
             batch {pf_t_cap} tokens (fused C={pf_concurrency}, overflow \
             chunk {ov}), {:.1} MB slabs over the shared launch state (locks + \
             fence; shared across all MoE layers and the dense arms)",
            total as f64 / 1e6,
        );
        Ok(Self {
            launch,
            locks,
            a_f16,
            a_had_f16,
            c_gate_f32,
            c_up_f32,
            c_down_f32,
            inter_f16,
            b_indices,
            b_weights,
            s_cap,
            hidden,
            inter,
            top_k,
            sm_count,
            pf_hidden_f16,
            pf_out_f32,
            pf_temp_state_g,
            pf_temp_state_u,
            pf_temp_inter_g,
            pf_temp_inter_u,
            pf_token_sorted,
            pf_weight_sorted,
            pf_expert_count,
            pf_ov_a,
            pf_ov_a_had,
            pf_ov_gate,
            pf_ov_up,
            pf_ov_down,
            pf_t_cap,
            pf_e_cap,
            pf_concurrency,
        })
    }

    /// Get the model-shared state, creating it on first use over the
    /// model-shared [`Exl3LaunchState`] cache (also created here on first
    /// use). The loader threads BOTH `Option` caches through its per-layer
    /// loop so the MoE arm and the native dense arms serialize on ONE
    /// section (`weight_loader/qwen4_exp/exl3_dense.rs`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_create_with_launch(
        cache: &mut Option<std::sync::Arc<Exl3MoeState>>,
        launch_cache: &mut Option<std::sync::Arc<Exl3LaunchState>>,
        gpu: &dyn GpuBackend,
        hidden: usize,
        inter: usize,
        top_k: usize,
        num_experts: usize,
    ) -> Result<std::sync::Arc<Exl3MoeState>> {
        if let Some(s) = cache {
            ensure!(
                launch_cache
                    .as_ref()
                    .is_none_or(|l| std::sync::Arc::ptr_eq(l, &s.launch)),
                "EXL3 MoE state: a second launch state was passed for one model"
            );
            ensure!(
                s.hidden == hidden
                    && s.inter == inter
                    && s.top_k == top_k
                    && s.pf_e_cap == num_experts,
                "EXL3 MoE state: geometry changed between layers \
                 ({}x{} top_k {} E {} vs {hidden}x{inter} top_k {top_k} E \
                 {num_experts})",
                s.hidden,
                s.inter,
                s.top_k,
                s.pf_e_cap,
            );
            return Ok(s.clone());
        }
        let launch = Exl3LaunchState::get_or_create(launch_cache, gpu)?;
        let s = std::sync::Arc::new(Self::new(gpu, launch, hidden, inter, top_k, num_experts)?);
        *cache = Some(s.clone());
        Ok(s)
    }
}

impl MoeLayer {
    /// Install the native EXL3 routed-expert tables + shared launch state
    /// (loader, post-construction — the `set_lm_head_exl3` precedent).
    /// Order: `[gate, up, down]`.
    pub(crate) fn set_exl3_experts(
        &mut self,
        tables: [Exl3ExpertPtrTable; 3],
        state: std::sync::Arc<Exl3MoeState>,
    ) {
        self.exl3_expert_tables = Some(tables);
        self.exl3_moe_state = Some(state);
    }
}

pub(crate) fn build_fp8_ptr_table(
    experts: &[Fp8ExpertWeight],
    proj: impl Fn(&Fp8ExpertWeight) -> &Fp8Weight,
    gpu: &dyn GpuBackend,
) -> Result<Fp8ExpertPtrTable> {
    let n = experts.len();

    let weight_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight.0.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).row_scale.0.to_le_bytes())
        .collect();

    let weight_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&weight_bytes, weight_ptrs)?;

    let scale_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&scale_bytes, scale_ptrs)?;

    Ok(Fp8ExpertPtrTable {
        weight_ptrs,
        scale_ptrs,
    })
}
