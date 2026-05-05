// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward_k2 (verify K=2).

use super::*;

impl MoeLayer {
    /// Fused K=2 forward: process 2 tokens through MoE in 5 kernel launches.
    ///
    /// Gate GEMV batch2 → batched topK → fused expert gate+up → fused silu+down → fused wsum+blend.
    /// Expert buffers sized for 2*top_k slots. Shared expert buffers reuse logits/ssm_qkvz
    /// (sized for 2 tokens). Output at moe_output() [2, H].
    pub fn forward_k2(
        &self,
        input: DevicePtr, // [2, H] BF16 — normed MoE input for 2 tokens
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;

        // Gemma-4 router pre-norm (no-op for other models).
        let router_in = self.router_input(input, 2, h, ctx, stream)?;
        // 1. Gate GEMV batch2: reads gate weight once for 2 tokens
        let gate_logits = ctx.buffers.gate_logits(); // [2, 512] BF16
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

        // 2. Batched topK for 2 tokens: [2, 512] → [2*top_k] indices + [2*top_k] weights.
        //    Sigmoid+bias for MiniMax/DeepSeek-V3, softmax otherwise.
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch; // [2*top_k] u32
        let weights_dev = scratch.offset(2 * top_k as usize * 4); // [2*top_k] f32
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
                1.0,
                2,
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
                2,
                stream,
            )?;
        }

        // 3-5. Fused expert dispatch for 2 tokens
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_down_out = ctx.buffers.attn_output();
        let output = ctx.buffers.moe_output();

        let is_ep = ctx.comm.is_some_and(|c| c.world_size() > 1);

        if let (Some(gp), Some(up), Some(dp), Some(sh)) = (
            &self.fp8_gate_weight_ptrs,
            &self.fp8_up_weight_ptrs,
            &self.fp8_down_weight_ptrs,
            &self.fp8_shared_expert,
        ) {
            // FP8 batch2 path
            ops::moe_expert_gate_up_shared_fp8_batch2(
                ctx.gpu,
                self.moe_expert_gate_up_shared_fp8_batch2,
                input,
                gp.weight_ptrs,
                gp.scale_ptrs,
                expert_gate_out,
                up.weight_ptrs,
                up.scale_ptrs,
                expert_up_out,
                indices_dev,
                &sh.gate_proj,
                shared_gate_scratch,
                &sh.up_proj,
                shared_up_scratch,
                inter,
                h,
                top_k,
                stream,
            )?;
            ops::moe_expert_silu_down_shared_fp8_batch2(
                ctx.gpu,
                self.moe_expert_silu_down_shared_fp8_batch2,
                expert_gate_out,
                expert_up_out,
                dp.weight_ptrs,
                dp.scale_ptrs,
                expert_down_out,
                indices_dev,
                shared_gate_scratch,
                shared_up_scratch,
                &sh.down_proj,
                shared_down_out,
                h,
                inter,
                top_k,
                stream,
            )?;
            // EP fix: after silu_down, expert_gate_out is free — use as zero buffer
            // to exclude shared expert from blend (will add after all-reduce).
            let shared_for_blend = if is_ep && !shared_down_out.is_null() {
                ctx.gpu
                    .memset_async(expert_gate_out, 0, 2 * h as usize * 2, stream)?;
                expert_gate_out
            } else {
                shared_down_out
            };
            ops::moe_weighted_sum_blend_batch2(
                ctx.gpu,
                self.moe_weighted_sum_blend_fp8_batch2,
                output,
                expert_down_out,
                weights_dev,
                shared_for_blend,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                top_k,
                h,
                stream,
            )?;
        } else if self.use_t_layout_for_decode() {
            // Phase 8a unified-layout NVFP4 batch=2 verify (MTP K=2). Hybrid
            // mode skips this branch — small-N MTP verify wins on warp-
            // reduction originals.
            let gate_t = self
                .gate_ptrs_t
                .as_ref()
                .expect("gate_ptrs_t under unified_t");
            let up_t = self.up_ptrs_t.as_ref().expect("up_ptrs_t under unified_t");
            let down_t = self
                .down_ptrs_t
                .as_ref()
                .expect("down_ptrs_t under unified_t");
            let null_qw = QuantizedWeight::null();
            let sh_gate_t = self.shared_gate_t.as_ref().unwrap_or(&null_qw);
            let sh_up_t = self.shared_up_t.as_ref().unwrap_or(&null_qw);
            let sh_down_t = self.shared_down_t.as_ref().unwrap_or(&null_qw);
            ops::moe_expert_gate_up_shared_batch2_t(
                ctx.gpu,
                self.moe_expert_gate_up_shared_batch2_t_k,
                input,
                gate_t.packed_ptrs,
                gate_t.scale_ptrs,
                gate_t.scale2_vals,
                expert_gate_out,
                up_t.packed_ptrs,
                up_t.scale_ptrs,
                up_t.scale2_vals,
                expert_up_out,
                indices_dev,
                sh_gate_t,
                shared_gate_scratch,
                sh_up_t,
                shared_up_scratch,
                inter,
                h,
                top_k,
                stream,
            )?;
            ops::moe_expert_silu_down_shared_batch2_t(
                ctx.gpu,
                self.moe_expert_silu_down_shared_batch2_t_k,
                expert_gate_out,
                expert_up_out,
                down_t.packed_ptrs,
                down_t.scale_ptrs,
                down_t.scale2_vals,
                expert_down_out,
                indices_dev,
                shared_gate_scratch,
                shared_up_scratch,
                sh_down_t,
                shared_down_out,
                h,
                inter,
                top_k,
                stream,
            )?;
        } else {
            // NVFP4 batch2 path
            let batch2_block = if ctx.config.hidden_size >= 3072 {
                256u32
            } else {
                128u32
            };
            ops::moe_expert_gate_up_shared_batch2(
                ctx.gpu,
                self.moe_expert_gate_up_shared_batch2,
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
                batch2_block,
                stream,
            )?;
            ops::moe_expert_silu_down_shared_batch2(
                ctx.gpu,
                self.moe_expert_silu_down_shared_batch2,
                expert_gate_out,
                expert_up_out,
                self.down_ptrs.packed_ptrs,
                self.down_ptrs.scale_ptrs,
                self.down_ptrs.scale2_vals,
                expert_down_out,
                indices_dev,
                shared_gate_scratch,
                shared_up_scratch,
                &self.weights.shared_expert.down_proj,
                shared_down_out,
                h,
                inter,
                top_k,
                batch2_block,
                stream,
            )?;
            // EP fix: after silu_down, expert_gate_out is free — use as zero buffer
            let shared_for_blend = if is_ep && !shared_down_out.is_null() {
                ctx.gpu
                    .memset_async(expert_gate_out, 0, 2 * h as usize * 2, stream)?;
                expert_gate_out
            } else {
                shared_down_out
            };
            ops::moe_weighted_sum_blend_batch2(
                ctx.gpu,
                self.moe_weighted_sum_blend_batch2,
                output,
                expert_down_out,
                weights_dev,
                shared_for_blend,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                top_k,
                h,
                stream,
            )?;
        }

        // EP all-reduce: sum partial outputs for 2 tokens
        if let Some(comm) = ctx.comm
            && comm.world_size() > 1
        {
            if ctx.graph_capture {
                comm.all_reduce(output.0, 2 * h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, 2 * h as usize * 2, stream)?;
            }
            // Add shared expert with sigmoid gate (BUG #41 fix)
            if !shared_down_out.is_null() {
                if self.weights.shared_expert_gate.weight.0 == 0 {
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add,
                        output,
                        shared_down_out,
                        2 * h,
                        stream,
                    )?;
                } else {
                    ops::moe_batched_blend(
                        ctx.gpu,
                        self.moe_batched_blend,
                        output,
                        shared_down_out,
                        input,
                        self.weights.shared_expert_gate.weight,
                        h,
                        2,
                        stream,
                    )?;
                }
            }
        }

        Ok(())
    }
}
