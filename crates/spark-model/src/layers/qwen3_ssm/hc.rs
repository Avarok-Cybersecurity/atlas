// SPDX-License-Identifier: AGPL-3.0-only

//! mHC on the GDN layer.
//!
//! Qwen3.8-Flash-Next carries a `hc_mult`-wide residual highway on ALL 48
//! layers, 36 of which are GDN. DeepSeek-V4 — the model Atlas built mHC for —
//! is all-attention, so `Qwen3SsmLayer` never needed to know about it.
//!
//! The forward paths live in `trait_prefill_hc.rs` and `trait_decode_hc.rs`;
//! `trait_layer.rs` routes to them when `hc` is present. What is left here is
//! attachment and the one refusal that remains.

use super::Qwen3SsmLayer;
impl Qwen3SsmLayer {
    /// Attach mHC weights. Both concrete layer types carry them on this
    /// model: the 12 full-attention layers are `Qwen3AttentionLayer`, the 36
    /// GDN layers are this one.
    pub fn set_hc_weights(&mut self, hc: crate::layers::qwen3_attention::HcWeights) {
        self.hc = Some(hc);
    }

    /// Attach the PLE n-gram injection to this layer. Exactly one model layer
    /// carries it.
    pub fn set_ple(&mut self, ple: crate::layers::ple::PleLayer) {
        self.ple = Some(ple);
    }

    /// Refuse the batched and multi-sequence decode paths while the highway
    /// is live.
    ///
    /// Those paths keep their own residual bookkeeping, which the highway
    /// replaces — running them would add each block output to the residual a
    /// second time. v1 is C=1 only on this model (Avarok #753), and refusing
    /// is the point: a batched GDN step on an unmixed stream produces
    /// plausible, wrong activations with nothing in the log.
    ///
    /// This is what is LEFT of the blanket `ensure_no_unwired_hc` guard —
    /// prefill and single-token decode now run the highway rather than
    /// refusing it.
    pub(crate) fn refuse_batched_under_hc(&self, path: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.hc.is_none(),
            "qwen3_ssm::{path}: the mHC highway has no batched GDN path yet. \
             This model serves at concurrency 1; the batched paths maintain \
             their own residual, which the highway replaces, so running them \
             would count every block output twice. Avarok #753 item B."
        );
        Ok(())
    }
}
