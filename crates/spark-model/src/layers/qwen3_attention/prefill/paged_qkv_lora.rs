// SPDX-License-Identifier: AGPL-3.0-only

//! LoRA runtime delta of the paged-prefill Q/K/V projections — the tail of
//! `paged_qkv.rs::prefill_one_proj`, moved verbatim to keep that file under
//! the 500-LoC cap. Runs after WHICHEVER projection arm wrote `out`
//! (native EXL3 included).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use super::paged_qkv::Proj;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// `out[n, out_dim] += scale * (normed[n, h] @ A^T) @ B^T` for the
    /// projection's adapter (routed slot / request bgmv / installed pair).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_qkv_lora(
        &self,
        proj: Proj,
        normed: DevicePtr,
        out: DevicePtr,
        n: u32,
        out_dim: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // ── LoRA runtime delta: out += scale·(normed@Aᵀ)@Bᵀ (Q/K/V).
        // For gated Q this folds into the RAW interleaved `[Q|gate]` `out`
        // (out_dim = q_proj_dim) BEFORE the caller's `deinterleave_qg_split`
        // (paged.rs / cache_skip.rs) — the PEFT `lora_B` was trained against
        // exactly that interleaved basis, so Q folds like K/V, just wider.
        // Runs before the caller's ATLAS_OP_DUMP, so dumps show ADAPTED
        // outputs (what an HF+PEFT forward hook shows).
        if let Some(ref lw) = self.lora {
            let (pair, route, module) = match proj {
                Proj::Q => (
                    lw.q.as_ref(),
                    lw.q_route.as_ref(),
                    Some(crate::lora::LoraModule::QProj),
                ),
                Proj::K => (
                    lw.k.as_ref(),
                    lw.k_route.as_ref(),
                    Some(crate::lora::LoraModule::KProj),
                ),
                Proj::V => (
                    lw.v.as_ref(),
                    lw.v_route.as_ref(),
                    Some(crate::lora::LoraModule::VProj),
                ),
            };
            if let Some(pair) = pair {
                debug_assert_eq!(pair.k_in, h);
                debug_assert_eq!(pair.n_out, out_dim);
                // #30 routed-prefill precision: when this prefill routes to a
                // NON-active slot (`ctx.routed_lora_layers` Some), select THAT
                // slot's (global_layer, module) pair and fold it through the SAME
                // dense `apply_lora_delta` (dense_gemm_tc for m>1) the ACTIVE
                // adapter's prefill uses — numerically identical to serving that
                // adapter active, unlike the per-row bgmv whose accumulation order
                // tips razor-margin tokens. `lw.layer_idx` is the GLOBAL layer
                // index (not `attn_layer_idx`), matching the pool's GLOBAL-indexed
                // slice. `None` when the routed slot doesn't adapt this module →
                // fall through to the bgmv (base for that module) / installed pair.
                let routed_pair = ctx.routed_lora_layers.and_then(|ls| {
                    module.and_then(|m| crate::lora::select_routed_pair(ls, lw.layer_idx, m))
                });
                // Request-scoped routing: fold THIS request's adapter delta over
                // all `n` prompt tokens via the bgmv when the prefill uploaded a
                // per-request slot buffer (`seq_slot != 0`) and the module has a
                // route. `normed` is contiguous [n, h] and `out` is contiguous
                // [n, out_dim], so the bgmv (all rows = same slot) is
                // byte-identical to `n` single-row `apply_lora_delta`. No pool /
                // no route → the installed-active-pair path (pre-M2 behaviour).
                let seq_slot = ctx
                    .attn_metadata
                    .map(|m| m.seq_slot)
                    .unwrap_or(DevicePtr(0));
                if let Some(routed_pair) = routed_pair {
                    // #30 dense routed path (MUST be checked before the bgmv branch:
                    // a routed prefill satisfies BOTH conditions and the dense path
                    // must win). Same k_in/n_out/max_rank as the installed pair
                    // (uniform pool) — only a/b/scale differ (the request slot's).
                    debug_assert_eq!(routed_pair.k_in, h);
                    debug_assert_eq!(routed_pair.n_out, out_dim);
                    ops::lora_delta::apply_lora_delta(
                        ctx.gpu,
                        &lw.kernels,
                        routed_pair,
                        normed,
                        out,
                        n,
                        ctx.buffers.lora_xa(),
                        ctx.buffers.lora_delta(),
                        stream,
                    )?;
                } else if seq_slot.0 != 0
                    && let Some(route) = route
                    && crate::lora::prefill_bgmv_forced()
                {
                    // OPT-IN ONLY: `n` row-wise GEMVs vs ONE GEMM in `else`;
                    // a prefill is uniform-slot. Kept for PER-ROW slots.
                    ops::lora_delta::apply_lora_bgmv(
                        ctx.gpu,
                        &lw.kernels,
                        route,
                        normed,
                        out,
                        seq_slot,
                        n,
                        pair.k_in,
                        pair.n_out,
                        ctx.buffers.lora_xa(),
                        stream,
                    )?;
                } else {
                    ops::lora_delta::apply_lora_delta(
                        ctx.gpu,
                        &lw.kernels,
                        pair,
                        normed,
                        out,
                        n,
                        ctx.buffers.lora_xa(),
                        ctx.buffers.lora_delta(),
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }
}
