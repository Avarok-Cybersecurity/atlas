// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl MoeLayer {
    pub(super) fn route_k2(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        h: u32,
        num_experts: u32,
        top_k: u32,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let router_in = self.router_input(input, 2, h, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemv_batch2(
                ctx.gpu,
                self.w4a16_gemv_batch2,
                router_in,
                nvfp4,
                gate_logits,
                num_experts,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm,
                router_in,
                &self.weights.gate,
                gate_logits,
                2,
                num_experts,
                h,
                stream,
            )?;
        }

        let scratch = ctx.buffers.scratch();
        let indices = scratch;
        let weights = scratch.offset(2 * top_k as usize * 4);
        if let Some(bias) = self.correction_bias_dev {
            if ctx.config.scoring_func == "sqrtsoftplus" {
                for t in 0..2usize {
                    ops::moe_topk_sqrtsoftplus(
                        ctx.gpu,
                        self.moe_topk_sqrtsoftplus_k,
                        gate_logits.offset(t * num_experts as usize * 2),
                        bias,
                        indices.offset(t * top_k as usize * 4),
                        weights.offset(t * top_k as usize * 4),
                        num_experts,
                        top_k,
                        ctx.config.norm_topk_prob,
                        ctx.config.routed_scaling_factor as f32,
                        stream,
                    )?;
                }
            } else {
                ops::moe_topk_sigmoid_batched(
                    ctx.gpu,
                    self.moe_topk_sigmoid_batched_k,
                    gate_logits,
                    bias,
                    indices,
                    weights,
                    num_experts,
                    top_k,
                    ctx.config.norm_topk_prob,
                    ctx.config.routed_scaling_factor as f32,
                    2,
                    stream,
                )?;
            }
        } else {
            ops::moe_topk_softmax_batched(
                ctx.gpu,
                self.moe_topk_batched,
                gate_logits,
                indices,
                weights,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                2,
                stream,
            )?;
        }
        super::union_stats::maybe_sample_expert_union(ctx.gpu, indices, 2, top_k as usize, stream);
        Ok((indices, weights))
    }
}
