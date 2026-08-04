// SPDX-License-Identifier: AGPL-3.0-only

//! Purpose-built C=4 atomic-add MoE decode experiment.

use super::*;

impl MoeLayer {
    /// C=4 NVFP4 routed MoE decode with FP32 atomic accumulation.
    ///
    /// Gate/top-K remain batched. Gate+up reuses the token-major kernel, then
    /// routed down projections atomic-add weighted FP32 contributions into a
    /// tiny `[4,H]` scratch accumulator. Finalization casts routed output to
    /// BF16 and optionally blends shared expert output.
    pub fn forward_atomic_c4_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let has_shared = self.weights.shared_expert.gate_proj.weight.0 != 0
            && self.weights.shared_expert.up_proj.weight.0 != 0
            && self.weights.shared_expert.down_proj.weight.0 != 0;
        let nvfp4_supported = num_tokens == 4
            && self.moe_decode_atomic_c4_silu_down_accum_k.0 != 0
            && self.moe_decode_atomic_c4_finalize_k.0 != 0
            && self.bf16_gate_weight_ptrs.is_none()
            && self.fp8_gate_weight_ptrs.is_none()
            && !self.use_t_layout_for_decode()
            && self.pre_expert_norm.is_none()
            && has_shared;
        if !nvfp4_supported {
            return self.forward_batched(input, num_tokens, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = num_tokens as u32;

        let router_in = self.router_input(input, n, h, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                router_in,
                nvfp4,
                gate_logits,
                n,
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
                n,
                num_experts,
                h,
                stream,
            )?;
        }

        let scratch = ctx.buffers.scratch();
        let topk_bytes = num_tokens * top_k as usize * 4;
        let indices_dev = scratch;
        let weights_dev = scratch.offset(topk_bytes);
        let accum_off = (topk_bytes * 2 + 255) & !255;
        let accum_bytes = num_tokens * h as usize * 4;
        anyhow::ensure!(
            ctx.buffers.scratch_bytes() >= accum_off + accum_bytes,
            "scratch too small for ATLAS_MOE_ATOMIC_C4_DECODE: need {} bytes, have {}",
            accum_off + accum_bytes,
            ctx.buffers.scratch_bytes()
        );
        let routed_accum = scratch.offset(accum_off);

        if let Some(bias) = self.correction_bias_dev {
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

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let shared_down_out = ctx.buffers.attn_output();
        let output = ctx.buffers.moe_output();

        ops::moe_expert_gate_up_shared_prefill(
            ctx.gpu,
            self.moe_expert_gate_up_shared_token_major,
            input,
            self.gate_ptrs.packed_ptrs,
            self.gate_ptrs.scale_ptrs,
            self.gate_ptrs.scale2_vals,
            expert_gate_out,
            self.up_ptrs.packed_ptrs,
            self.up_ptrs.scale_ptrs,
            self.up_ptrs.scale2_vals,
            expert_up_out,
            indices_dev,
            &self.weights.shared_expert.gate_proj,
            shared_gate_scratch,
            &self.weights.shared_expert.up_proj,
            shared_up_scratch,
            inter,
            h,
            top_k,
            n,
            stream,
        )?;

        ctx.gpu.memset_async(routed_accum, 0, accum_bytes, stream)?;
        ops::moe_decode_atomic_c4_silu_down_accum(
            ctx.gpu,
            self.moe_decode_atomic_c4_silu_down_accum_k,
            expert_gate_out,
            expert_up_out,
            self.down_ptrs.packed_ptrs,
            self.down_ptrs.scale_ptrs,
            self.down_ptrs.scale2_vals,
            indices_dev,
            weights_dev,
            routed_accum,
            shared_gate_scratch,
            shared_up_scratch,
            &self.weights.shared_expert.down_proj,
            shared_down_out,
            h,
            inter,
            top_k,
            n,
            stream,
        )?;

        let is_ep = ctx.comm.is_some() && ctx.config.ep_world_size > 1;
        ops::moe_decode_atomic_c4_finalize(
            ctx.gpu,
            self.moe_decode_atomic_c4_finalize_k,
            output,
            routed_accum,
            shared_down_out,
            input,
            self.weights.shared_expert_gate.weight,
            h,
            n,
            !is_ep,
            stream,
        )?;

        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
        {
            if ctx.graph_capture {
                comm.all_reduce(output.0, num_tokens * h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, num_tokens * h as usize * 2, stream)?;
            }
            ops::moe_batched_blend(
                ctx.gpu,
                self.moe_batched_blend,
                output,
                shared_down_out,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                n,
                stream,
            )?;
        }

        Ok(())
    }
}
