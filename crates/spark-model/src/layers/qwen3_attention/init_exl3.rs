// SPDX-License-Identifier: AGPL-3.0-only

//! Native EXL3 attention family (`ATLAS_EXL3_NATIVE_DENSE=1`) install + the
//! per-site arm accessor for [`Qwen3AttentionLayer`]. Sibling of `init.rs`
//! (500-LoC cap).

use anyhow::{Result, ensure};

use super::types::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::exl3_dense::Exl3AttnWeights;

impl Qwen3AttentionLayer {
    /// Install the packed attention family. Every materialized projection
    /// slot — the BF16 denses in `attn`, the NVFP4 `o_proj`, the decode
    /// `QuantWeight`s, the transposed prefill twins and the FP8 predequants
    /// — must be EMPTY: a layer carrying both copies would double-hold memory
    /// and make "which arm ran" ambiguous. Refused rather than resolved by
    /// priority.
    pub fn set_exl3_attn_weights(&mut self, w: Exl3AttnWeights) -> Result<()> {
        ensure!(
            self.attn.q_proj.weight.is_null()
                && self.attn.k_proj.weight.is_null()
                && self.attn.v_proj.weight.is_null()
                && self.attn.o_proj.is_null()
                && self.q_weight.is_none()
                && self.k_weight.is_none()
                && self.v_weight.is_none()
                && self.o_weight.is_none()
                && self.o_dense_bf16.is_none()
                && self.qkv_nvfp4_t.is_none()
                && self.q_nvfp4_t.is_none()
                && self.k_nvfp4_t.is_none()
                && self.v_nvfp4_t.is_none()
                && self.o_nvfp4_t.is_none()
                && self.q_fp8.is_none()
                && self.k_fp8.is_none()
                && self.v_fp8.is_none()
                && self.o_fp8.is_none()
                && self.q_fp8w_t.is_none()
                && self.k_fp8w_t.is_none()
                && self.v_fp8w_t.is_none()
                && self.o_fp8w_t.is_none(),
            "EXL3 native attention: the layer already carries a materialized q/k/v/o \
             copy — the loader must leave every dense/quantized projection slot null \
             when it keeps the family packed"
        );
        ensure!(
            self.mla.is_none() && !self.k_eq_v,
            "EXL3 native attention: only the plain (non-MLA, separate-V) q/k/v/o \
             projection layout is served natively"
        );
        self.exl3_attn = Some(w);
        Ok(())
    }

    /// The installed packed attention family, if any.
    pub fn exl3_attn_weights(&self) -> Option<&Exl3AttnWeights> {
        self.exl3_attn.as_ref()
    }

    /// The native arm for a projection site: `Some(family)` when this layer
    /// serves q/k/v/o from packed trellis (the site then MUST dispatch
    /// through the family's funnels — every materialized slot is null), or
    /// `None` for the existing arms.
    ///
    /// Cooperative launches are never CUDA-graph-capturable, so a capturing
    /// pass reaching a native layer is refused here by name (the layer's
    /// `exl3_graph_veto` keeps the capturing decode paths away in the first
    /// place; this is the loud backstop, not the mechanism).
    pub(super) fn exl3_attn_arm(
        &self,
        ctx: &ForwardContext,
        site: &str,
    ) -> Result<Option<&Exl3AttnWeights>> {
        let Some(w) = self.exl3_attn.as_ref() else {
            return Ok(None);
        };
        ensure!(
            !ctx.graph_capture,
            "Qwen3AttentionLayer::{site}: native EXL3 attention launches cooperatively \
             and cannot be captured into a CUDA graph — the model-level \
             exl3_graph_veto must keep this layer off the capturing path"
        );
        Ok(Some(w))
    }
}
