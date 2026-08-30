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

    /// Expert count the mask is indexed by. The mask is built over the
    /// model's expert count, which is the id space the top-k selects from.
    fn num_experts_for_mask(&self) -> usize {
        self.bel_num_experts
    }
}
