// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward_prefill_fp8.

use super::*;

impl MoeLayer {
    /// EP token dispatch/combine forward pass (Workstream 3A scaffold).
    ///
    /// Instead of dense all-reduce, this:
    /// 1. Runs gate projection to get top-K routing
    /// 2. Builds a routing table partitioning tokens into local/remote
    /// 3. Dispatches remote tokens to partner rank
    ///
    /// FP8 sorted MoE prefill: grouped GEMM with FP8 expert weights.
    ///
    /// Same pipeline as NVFP4 forward_prefill but uses moe_fp8_grouped_gemm
    /// with FP8 pointer tables instead of NVFP4 pointer tables.
    pub(super) fn forward_prefill_fp8(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = num_tokens as u32;
        let total_expanded = n * top_k;
        let ne = num_experts as usize;

        let (gp, up, dp, sh) = match (
            &self.fp8_gate_weight_ptrs,
            &self.fp8_up_weight_ptrs,
            &self.fp8_down_weight_ptrs,
            &self.fp8_shared_expert,
        ) {
            (Some(g), Some(u), Some(d), Some(s)) => (g, u, d, s),
            _ => anyhow::bail!("FP8 expert pointer tables not set"),
        };

        // ── Shared expert (same as NVFP4 path) ──
        let has_shared = shared_inter > 0;
        if has_shared {
            let shared_gate_out = ctx.buffers.ssm_deinterleaved();
            let shared_up_out = ctx.buffers.ssm_qkvz();
            // FP8 GEMM for shared expert (M=num_tokens, single kernel each)
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                input,
                sh.gate_proj.weight,
                sh.gate_proj.row_scale,
                shared_gate_out,
                n,
                shared_inter,
                h,
                stream,
            )?;
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                input,
                sh.up_proj.weight,
                sh.up_proj.row_scale,
                shared_up_out,
                n,
                shared_inter,
                h,
                stream,
            )?;
            // Activation + down for shared expert (SiLU or GeGLU)
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                shared_gate_out,
                shared_up_out,
                shared_gate_out,
                n * shared_inter,
                stream,
            )?;
            let shared_down_out = ctx.buffers.attn_output();
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                shared_gate_out,
                sh.down_proj.weight,
                sh.down_proj.row_scale,
                shared_down_out,
                n,
                h,
                shared_inter,
                stream,
            )?;
        }

        // ── Routed expert path ──

        // Gemma-4 router pre-norm (no-op for other models).
        let router_in = self.router_input(input, n, h, ctx, stream)?;
        // 1. Gate GEMM
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

        // 2. Batched topK dispatch (sigmoid+bias for MiniMax/DeepSeek-V3,
        //    softmax for everyone else — selection by `correction_bias_dev`).
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(total_expanded as usize * 4);
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

        // 3. Sort tokens by expert
        let te = total_expanded as usize;
        let sorted_token_ids = gate_logits;
        let sorted_expert_ids = gate_logits.offset(te * 4);
        let expert_offsets = gate_logits.offset(te * 4 * 2);
        let token_to_perm = gate_logits.offset(te * 4 * 2 + (ne + 1) * 4);
        ops::moe_sort_by_expert(
            ctx.gpu,
            self.moe_sort_by_expert,
            indices_dev,
            sorted_token_ids,
            sorted_expert_ids,
            expert_offsets,
            token_to_perm,
            total_expanded,
            num_experts,
            top_k,
            stream,
        )?;

        // 4. Max M tiles (same heuristic as NVFP4)
        let avg_per_expert = (num_tokens * top_k as usize).div_ceil(ne);
        let max_m_tiles = (avg_per_expert * 2).div_ceil(64).max(1) as u32;

        // 5. FP8 grouped gate+up GEMM
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        // EP: zero expert buffers for remote experts
        if ctx.comm.is_some() {
            let gate_bytes = te * inter as usize * 2;
            ctx.gpu
                .memset_async(expert_gate_out, 0, gate_bytes, stream)?;
            ctx.gpu.memset_async(expert_up_out, 0, gate_bytes, stream)?;
            ctx.gpu.memset_async(
                ctx.buffers.expert_down_out(),
                0,
                te * h as usize * 2,
                stream,
            )?;
        }
        let fp8_grouped_k = self.fp8_grouped_kernel();
        if max_m_tiles > 0 {
            ops::moe_fp8_grouped_gemm(
                ctx.gpu,
                fp8_grouped_k,
                input,
                gp.weight_ptrs,
                gp.scale_ptrs,
                expert_gate_out,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                inter,
                h,
                max_m_tiles,
                stream,
            )?;

            ops::moe_fp8_grouped_gemm(
                ctx.gpu,
                fp8_grouped_k,
                input,
                up.weight_ptrs,
                up.scale_ptrs,
                expert_up_out,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                inter,
                h,
                max_m_tiles,
                stream,
            )?;
        }

        // 6. Activation+mul + down GEMM
        let expert_down_out = ctx.buffers.expert_down_out();
        if max_m_tiles > 0 {
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                expert_gate_out,
                expert_up_out,
                expert_gate_out,
                total_expanded * inter,
                stream,
            )?;

            ops::moe_fp8_grouped_gemm(
                ctx.gpu,
                fp8_grouped_k,
                expert_gate_out,
                dp.weight_ptrs,
                dp.scale_ptrs,
                expert_down_out,
                expert_offsets,
                spark_runtime::gpu::DevicePtr(0),
                num_experts,
                h,
                inter,
                max_m_tiles,
                stream,
            )?;
        }

        // 7. Unpermute + weighted reduce + shared blend
        let output = ctx.buffers.moe_output();
        ops::moe_unpermute_reduce_indexed(
            ctx.gpu,
            self.moe_unpermute_reduce,
            expert_down_out,
            output,
            token_to_perm,
            weights_dev,
            h,
            n,
            top_k,
            stream,
        )?;

        // EP all-reduce of routed-expert output FIRST.
        // Shared experts are NOT EP-sharded (every rank loads the full
        // shared_expert weights — see fast_weights/mod.rs:85-104), so
        // their down-projection output already contains the full
        // contribution and must be blended AFTER the routed-expert
        // allreduce — otherwise the shared term gets summed across ranks
        // (multiplied by world_size). Sibling of forward()/forward_k2()/
        // forward_k3() which already do this in the right order; mirrors
        // vllm PR #39181.
        if let Some(comm) = ctx.comm
            && comm.world_size() > 1
        {
            comm.all_reduce_async(output.0, num_tokens * h as usize * 2, stream)?;
        }

        // Shared expert blend (post-allreduce).
        if has_shared {
            let shared_down_out = ctx.buffers.attn_output();
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
