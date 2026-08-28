// SPDX-License-Identifier: AGPL-3.0-only

//! LongCat-Flash router for batched prefill: softmax over
//! `num_experts + zero_expert_num` logits, `e_score_correction_bias` for
//! SELECTION only, and the zero-computation (identity) experts folded inside
//! the kernel. Decode twin lives in `forward.rs`.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::MoeLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn router_softmax_bias_batched(
        &self,
        gate_logits: DevicePtr,
        bias: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        num_experts: u32,
        top_k: u32,
        n: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ops::moe_topk_softmax_bias_batched(
            ctx.gpu,
            self.moe_topk_softmax_bias_batched_k,
            gate_logits,
            bias,
            indices_dev,
            weights_dev,
            self.zero_accum_dev,
            self.router_logits_n,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            ctx.config.routed_scaling_factor as f32,
            n,
            stream,
        )
    }

    /// The whole `correction_bias` scoring dispatch for ONE token, as used by
    /// `forward_batched`'s row-at-a-time loop: sqrt-softplus (DeepSeek-V4),
    /// softmax+bias (LongCat), or sigmoid+bias (DeepSeek-V3 / MiniMax-M2).
    ///
    /// Lives here rather than inline so `forward_batched` stays under the
    /// per-file cap, and so the three scoring functions are readable side by
    /// side — picking the wrong one is a silent mis-route, not an error.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn router_bias_one(
        &self,
        gate_t: DevicePtr,
        bias: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        num_experts: u32,
        top_k: u32,
        t: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match ctx.config.scoring_func.as_str() {
            "sqrtsoftplus" => ops::moe_topk_sqrtsoftplus(
                ctx.gpu,
                self.moe_topk_sqrtsoftplus_k,
                gate_t,
                bias,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                ctx.config.routed_scaling_factor as f32,
                stream,
            ),
            // LongCat: softmax + bias for SELECTION, unbiased softmax for the
            // blend weights, identity experts folded into `zero_accum[t]`.
            "softmax" => self.router_softmax_bias_one(
                gate_t,
                bias,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                t,
                ctx,
                stream,
            ),
            _ => ops::moe_topk_sigmoid(
                ctx.gpu,
                self.moe_topk_sigmoid_k,
                gate_t,
                bias,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                ctx.config.routed_scaling_factor as f32,
                stream,
            ),
        }
    }

    /// Per-TOKEN twin, for `forward_batched`'s row-at-a-time loop.
    ///
    /// `forward_batched` is where a short FP8/BF16-expert prefill lands
    /// (`forward_prefill` only takes the grouped GEMM above 64 tokens), so
    /// without this a LongCat request under 65 tokens refuses — which is most
    /// conversational turns.
    ///
    /// `zero_accum` is `[N]`, one scalar per token, and the single-token
    /// kernel writes index 0 of whatever it is handed — hence the `t*4`
    /// offset. Passing the base pointer would make every token overwrite
    /// token 0's identity-expert weight, and `apply_zero_expert` would then
    /// add one token's contribution to all of them.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn router_softmax_bias_one(
        &self,
        gate_t: DevicePtr,
        bias: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        num_experts: u32,
        top_k: u32,
        t: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ops::moe_topk_softmax_bias(
            ctx.gpu,
            self.moe_topk_softmax_bias_k,
            gate_t,
            bias,
            indices_dev,
            weights_dev,
            self.zero_accum_dev.offset(t * 4),
            self.router_logits_n,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            ctx.config.routed_scaling_factor as f32,
            stream,
        )
    }
}
