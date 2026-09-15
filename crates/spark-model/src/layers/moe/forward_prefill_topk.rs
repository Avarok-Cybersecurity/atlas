// SPDX-License-Identifier: AGPL-3.0-only

//! Batched top-k dispatch of `MoeLayer::forward_prefill` — hoisted verbatim
//! from `forward_prefill.rs` on the 500-LoC cap (behavior identical).
//!
//! DeepSeek-V3 / MiniMax-M2 use sigmoid + correction bias (detected via
//! `correction_bias_dev`); DeepSeek-V4 uses hash routing (`tid2eid_dev`) or
//! sqrtsoftplus/softmax scoring with a bias; every other model takes the
//! softmax path.

use super::*;

impl MoeLayer {
    /// Stage the prefill routing state (`indices_dev` u32 `[n*top_k]`,
    /// `weights_dev` f32 `[n*top_k]`) from the gate logits.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_topk_dispatch(
        &self,
        gate_logits: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        num_experts: u32,
        top_k: u32,
        n: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(tid2eid) = self.tid2eid_dev {
            // DeepSeek-V4 hash routing (hash_moe layer): static
            // `tid2eid[token_id]` selection, sqrtsoftplus-weighted.
            let token_ids = ctx.token_ids.ok_or_else(|| {
                anyhow::anyhow!(
                    "DeepSeek-V4 hash-MoE layer requires ForwardContext.token_ids (prefill grouped)"
                )
            })?;
            ops::moe_hash_route_batched(
                ctx.gpu,
                self.moe_hash_route_batched_k,
                gate_logits,
                tid2eid,
                token_ids,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                ctx.config.routed_scaling_factor as f32,
                n,
                stream,
            )?;
        } else if let Some(bias) = self.correction_bias_dev {
            // DeepSeek-V4 scores experts with sqrtsoftplus (NOT sigmoid); the
            // bias selects experts, weights gather pre-bias scores. Other
            // sigmoid+bias models (DeepSeek-V3 / MiniMax-M2) keep sigmoid.
            if ctx.config.scoring_func == "sqrtsoftplus" {
                ops::moe_topk_sqrtsoftplus_batched(
                    ctx.gpu,
                    self.moe_topk_sqrtsoftplus_batched_k,
                    gate_logits,
                    bias,
                    indices_dev,
                    weights_dev,
                    num_experts,
                    top_k,
                    ctx.config.norm_topk_prob,
                    ctx.config.routed_scaling_factor as f32,
                    n,
                    stream,
                )?;
            } else if ctx.config.scoring_func == "softmax" {
                self.router_softmax_bias_batched(
                    gate_logits,
                    bias,
                    indices_dev,
                    weights_dev,
                    num_experts,
                    top_k,
                    n,
                    ctx,
                    stream,
                )?;
            } else {
                ops::moe_topk_sigmoid_batched(
                    ctx.gpu,
                    self.moe_topk_sigmoid_batched_k,
                    gate_logits,
                    bias,
                    indices_dev,
                    weights_dev,
                    num_experts,
                    top_k,
                    ctx.config.norm_topk_prob,
                    ctx.config.routed_scaling_factor as f32,
                    n,
                    stream,
                )?;
            }
        } else {
            ops::moe_topk_softmax_batched(
                ctx.gpu,
                self.moe_topk_batched,
                gate_logits,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                n,
                stream,
            )?;
        }
        Ok(())
    }
}
