// SPDX-License-Identifier: AGPL-3.0-only

//! Boot-time expert loading, layer side: keep the router inside the set of
//! experts whose weights are actually resident.
//!
//! `--expert-category` makes the loaders skip most of a layer's experts. The
//! expert pointer table stays full-length — the kernels index it by raw
//! expert id — with NULL entries where nothing was loaded. So a top-k that
//! selected an absent expert would hand a kernel a null pointer.
//!
//! Two mechanisms, and the second exists because the first cannot be
//! verified by the compiler:
//!
//!  * [`MoeLayer::apply_bel_mask`] adds `-inf` to the unloaded experts'
//!    logits before top-k. The top-k kernels' existing renormalize step then
//!    spreads the weight across the survivors, so the layer computes a
//!    re-weighted blend of loaded experts rather than a partial sum.
//!  * [`MoeLayer::bel_guard`] refuses, by name, on a routing path that has
//!    no mask applied. There are a dozen routing paths and the mask is
//!    applied on the ones v1 supports; the rest must fail loudly rather than
//!    dereference null, and `bel_path_audit` (in the tests) pins the
//!    partition so a new path cannot quietly join the unmasked set.

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use super::MoeLayer;
use crate::layer::ForwardContext;

impl MoeLayer {
    /// Upload this layer's router mask, if the serve restricts its experts.
    ///
    /// Once, at construction: the mask is a boot-time constant, and a
    /// per-pass upload would have to happen inside CUDA-graph capture.
    pub(super) fn build_bel_mask(
        site: super::MoeSite,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
    ) -> Result<Option<DevicePtr>> {
        let (Some(plan), Some(layer)) = (config.bel.as_ref(), site.layer_idx()) else {
            return Ok(None);
        };
        let Some(mask) = plan.router_mask(layer) else {
            // This layer is unrestricted — every expert is resident, so
            // there is nothing to make unselectable.
            return Ok(None);
        };
        // The mask indexes a logits row by `expert_id`, so the row must BE
        // the expert space. LongCat's router also scores zero-computation
        // experts, making the row wider than `num_experts` — the mask would
        // then land on the wrong columns and leave real experts selectable.
        if config.zero_expert_num != 0 {
            anyhow::bail!(
                "--expert-category is not supported on this model: its router scores \
                 {} zero-computation experts in addition to the {} routed ones, so a \
                 per-expert mask does not line up with a logits row",
                config.zero_expert_num,
                config.num_experts,
            );
        }
        let bytes: Vec<u8> = mask.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dev = gpu.alloc(bytes.len())?;
        gpu.copy_h2d(&bytes, dev)?;
        Ok(Some(dev))
    }

    /// Make unloaded experts unselectable for this pass.
    ///
    /// Called immediately before top-k, on the same buffer the router wrote
    /// and after any LoRA delta has been folded in — the mask has to be the
    /// LAST thing to touch the logits, or a delta could lift a masked expert
    /// back above the selection threshold.
    ///
    /// A no-op when the serve loads every expert, and additive, so a
    /// category covering the whole expert set leaves the logits bit-identical
    /// to a run without the flag.
    pub(super) fn apply_bel_mask(
        &self,
        ctx: &ForwardContext<'_>,
        gate_logits: DevicePtr,
        n: usize,
        fp32_logits: bool,
        stream: u64,
    ) -> Result<()> {
        let Some(mask) = self.bel_mask_dev else {
            return Ok(());
        };
        let kernel = if fp32_logits {
            self.moe_bel_mask_f32_k
        } else {
            self.moe_bel_mask_bf16_k
        };
        if kernel.0 == 0 {
            // The serve restricts experts but this build has no mask kernel,
            // so nothing would stop the router selecting an expert with no
            // weights behind it. Refuse rather than run.
            bail!(
                "--expert-category is active but the {} router-mask kernel is missing from \
                 this build (kernels/gb10/common/moe_bel_mask.cu); without it the router \
                 could select an expert whose weights were never loaded",
                if fp32_logits { "f32" } else { "bf16" }
            );
        }
        // BEFORE the mask: the ratio between what the router wanted and what
        // this serve holds only exists while the absent experts' logits are
        // still there.
        if !fp32_logits
            && let Some(rho) = self.bel_rho_dev
            && self.moe_bel_resident_mass_k.0 != 0
        {
            if n <= super::BEL_RHO_MAX_ROWS {
                crate::layers::ops::moe_bel_resident_mass(
                    ctx.gpu,
                    self.moe_bel_resident_mass_k,
                    gate_logits,
                    mask,
                    rho,
                    n as u32,
                    self.num_experts_for_mask() as u32,
                    stream,
                )?;
            } else if ctx.stats.once("bel_rho_overflow") {
                tracing::warn!(
                    "expert-category rescale skipped for a {n}-row pass (scratch holds {}); \
                     those rows keep the renormalized weights",
                    super::BEL_RHO_MAX_ROWS
                );
            }
        }

        let total = n * self.num_experts_for_mask();
        crate::layers::ops::moe_bel_mask(
            ctx.gpu,
            kernel,
            gate_logits,
            mask,
            n as u32,
            self.num_experts_for_mask() as u32,
            total as u32,
            stream,
        )
    }

    /// Turn renormalized-over-survivors weights back into true weights.
    ///
    /// Called immediately AFTER top-k, on the weights it produced. Masking
    /// before selection and renormalizing over what survives gives each
    /// selected expert `w / rho`, so the mass that belonged to absent experts
    /// is silently handed to whichever residents were picked — including ones
    /// the router ranked far below the true top-k. Multiplying by `rho` is
    /// exactly the inverse: every selected expert ends up carrying the weight
    /// it has in the FULL softmax, and the routed branch contributes `rho` of
    /// its usual total rather than all of it.
    ///
    /// Softmax routing only. Sigmoid routing has no shared denominator, so
    /// there is no ratio to take and this must not run.
    pub(super) fn apply_bel_rescale(
        &self,
        ctx: &crate::layer::ForwardContext<'_>,
        expert_weights: DevicePtr,
        row_base: usize,
        n: usize,
        top_k: usize,
        stream: u64,
    ) -> Result<()> {
        if !ctx.levers.bel_rescale {
            // Off by default: measured worse than leaving the renormalization
            // alone (see `ModelLevers::bel_rescale`).
            return Ok(());
        }
        let Some(rho) = self.bel_rho_dev else {
            // Not a restricted layer: nothing was masked, so there is no
            // renormalization to undo.
            return Ok(());
        };
        if !self.routing_is_softmax(ctx) {
            // Sigmoid and hash routing do not normalize across experts, so
            // there is no shared denominator and no ratio to take. Masking
            // still works for them; only this correction does not apply.
            // Said once rather than skipped in silence — a silent skip is how
            // this went unnoticed for a whole measurement round.
            if ctx.stats.once("bel_rescale_not_softmax") {
                tracing::warn!(
                    "--expert-category: routing on this model is not softmax, so the \
                     resident-mass rescale does not apply; masked routing keeps the \
                     renormalized weights and absent experts' mass goes to substitutes"
                );
            }
            return Ok(());
        }
        if self.moe_bel_scale_weights_k.0 == 0 {
            anyhow::bail!(
                "--expert-category is active but this build has no moe_bel_scale_weights \
                 kernel, so the resident-mass rescale cannot run and absent experts' mass \
                 would be handed to substitutes without any indication"
            );
        }
        if row_base + n > super::BEL_RHO_MAX_ROWS {
            if ctx.stats.once("bel_rho_overflow_scale") {
                tracing::warn!(
                    "--expert-category rescale skipped for rows beyond {} — those rows keep \
                     the renormalized weights",
                    super::BEL_RHO_MAX_ROWS
                );
            }
            return Ok(());
        }
        crate::layers::ops::moe_bel_scale_weights(
            ctx.gpu,
            self.moe_bel_scale_weights_k,
            expert_weights,
            rho,
            row_base as u32,
            n as u32,
            top_k as u32,
            stream,
        )
    }

    /// Refuse a routing path that v1 does not mask.
    ///
    /// The alternative is a null-pointer dereference inside a GEMM when the
    /// router picks an expert that was never loaded — a serve-killing
    /// illegal memory access with nothing in the log to say why.
    pub(super) fn bel_guard(&self, path: &'static str) -> Result<()> {
        if self.bel_mask_dev.is_none() {
            return Ok(());
        }
        bail!(
            "--expert-category cannot serve this request: the `{path}` MoE routing path does \
             not apply the router mask, so it could select an expert whose weights were not \
             loaded. Serve without --expert-category, or use a configuration that routes \
             through the masked prefill/decode paths."
        )
    }

    /// Whether this layer routes by a softmax over all experts.
    ///
    /// Determined the same way the dispatch does it, NOT by reading
    /// `scoring_func` — that field is absent from most checkpoints' config
    /// (Qwen3.6-35B-A3B included) and defaults to the empty string, so
    /// comparing it against "softmax" answers false for the plain softmax
    /// models that are the majority. That mistake silently disabled this
    /// correction for a full measurement round.
    fn routing_is_softmax(&self, ctx: &crate::layer::ForwardContext<'_>) -> bool {
        // Hash-routed layers select from a static table, not a distribution.
        if self.tid2eid_dev.is_some() {
            return false;
        }
        // A correction bias means sigmoid scoring, unless the model declares
        // softmax-with-bias (LongCat) — which BEL refuses at boot anyway.
        if self.correction_bias_dev.is_some() && ctx.config.scoring_func != "softmax" {
            return false;
        }
        ctx.config.scoring_func != "sqrtsoftplus"
    }

    /// Expert count the mask is indexed by. The mask is built over the
    /// model's expert count, which is the id space the top-k selects from.
    fn num_experts_for_mask(&self) -> usize {
        self.bel_num_experts
    }
}
