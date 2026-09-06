// SPDX-License-Identifier: AGPL-3.0-only

//! Token-major N-token MoE decode experiment.

use super::*;

impl MoeLayer {
    /// Token-major fused decode for small N>=4.
    ///
    /// This reuses the generic `moe_prefill` kernels without the sorted/grouped
    /// GEMM path. It batches gate/top-k and processes all `(token, expert-slot)`
    /// routes in three token-major kernels:
    ///
    /// gate GEMM -> batched topK -> gate+up -> silu+down -> wsum/blend.
    ///
    /// First pass is NVFP4 + shared-expert only, matching Holo's current decode
    /// path. FP8/BF16/unified-layout variants deliberately fall back to the
    /// existing implementation until they have equivalent generic kernels.
    pub fn forward_token_major_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // BEL is not wired on this routing path: it computes its own gate
        // logits without the mask, so the top-k could name an expert whose
        // weights were never loaded. Refuse by name — the alternative is a
        // null dereference inside a GEMM.
        self.bel_guard("forward_token_major_decode")?;
        // LongCat zero-experts are wired only on the single-token decode
        // + prefill paths (v1); this variant would silently mis-route the
        // 384-wide router. Named refusal, not silent wrongness.
        anyhow::ensure!(
            self.router_logits_n as usize == ctx.config.num_experts,
            "zero-expert MoE routing is not wired on this dispatch variant yet (forward_token_major)"
        );

        // SOLID Incr-4: the token-major fast path has no fold hooks. When a
        // MoE adapter is RESIDENT, delegate to the per-row batched fallback,
        // which folds router + gate/up/down route-agnostically (base rows
        // no-op device-side via the `moe_row_adapter` map; a homogeneous
        // batch without the map uses the request-granularity
        // `moe_route_gate`). PRESENCE-gated exactly like forward_k2/k3 —
        // install-time-fixed, so graph-safe (decode graphs drain on adapter
        // rotate/swap); gating on the per-step ROUTE here would bake a host
        // branch into a captured `padded_n` graph. A `Refuse` batch was
        // already bailed host-side before the ladder
        // (`ensure_decode_route_servable` in `decode_batch_compute_main`).
        if self.lora.is_some() {
            return self.forward_batched(input, num_tokens, ctx, stream);
        }
        let has_shared = self.weights.shared_expert.gate_proj.weight.0 != 0
            && self.weights.shared_expert.up_proj.weight.0 != 0
            && self.weights.shared_expert.down_proj.weight.0 != 0;
        let nvfp4_supported = self.bf16_gate_weight_ptrs.is_none()
            && !self.has_mixed_bf16_shared_expert()
            && self.fp8_gate_weight_ptrs.is_none()
            && !self.use_t_layout_for_decode()
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
        let indices_dev = scratch;
        let weights_dev = scratch.offset(num_tokens * top_k as usize * 4);
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
        let expert_down_out = ctx.buffers.expert_down_out();
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
        ops::moe_expert_silu_down_shared_prefill(
            ctx.gpu,
            self.moe_expert_silu_down_shared_token_major,
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
            n,
            stream,
        )?;

        let is_ep = ctx.comm.is_some() && ctx.config.ep_world_size > 1;
        let shared_for_blend = if is_ep {
            ctx.gpu
                .memset_async(expert_gate_out, 0, num_tokens * h as usize * 2, stream)?;
            expert_gate_out
        } else {
            shared_down_out
        };
        ops::moe_weighted_sum_blend_prefill(
            ctx.gpu,
            self.moe_weighted_sum_blend_token_major,
            output,
            expert_down_out,
            weights_dev,
            shared_for_blend,
            input,
            self.weights.shared_expert_gate.weight,
            h,
            top_k,
            h,
            n,
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
            if self.weights.shared_expert_gate.weight.0 == 0 {
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add,
                    output,
                    shared_down_out,
                    n * h,
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
                    n,
                    stream,
                )?;
            }
        }

        Ok(())
    }
}
