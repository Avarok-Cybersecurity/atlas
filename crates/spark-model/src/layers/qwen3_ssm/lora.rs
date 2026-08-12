// SPDX-License-Identifier: AGPL-3.0-only

//! Feature-1 MoE LoRA install on the GDN / linear-attention layer.
//!
//! Linear-attention layers carry NO attention LoRA (their projections are
//! rejected at classify), but the MoE FFN exists on every layer, so a real
//! all-layer MoE adapter installs its router + routed-expert deltas here too —
//! the same `MoeLayer::set_lora_weights` path the full-attention layer uses.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::Qwen3SsmLayer;
use crate::layers::FfnComponent;
use crate::layers::ops::lora_delta::{LoraKernels, LoraPair};
use crate::lora::ExpertLoraLayer;

impl Qwen3SsmLayer {
    /// Install this GDN layer's MoE router + routed-expert LoRA onto its
    /// `FfnComponent::Moe`. Hard-rejects (never silently drops) when the layer's
    /// FFN is not MoE — an expert/router delta on a dense-FFN GDN layer is a
    /// loader/adapter mismatch.
    pub fn set_moe_lora_weights(
        &mut self,
        router: Option<LoraPair>,
        experts: ExpertLoraLayer,
        kernels: LoraKernels,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        if let FfnComponent::Moe(m) = &mut self.ffn {
            return m.set_lora_weights(router, experts, kernels, gpu);
        }
        anyhow::bail!(
            "LoRA: router/expert deltas installed on a linear-attention layer whose \
             FFN is not MoE (loader/adapter mismatch)"
        )
    }
}
