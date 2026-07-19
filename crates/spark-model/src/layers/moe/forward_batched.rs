// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward_batched.

use super::*;

impl MoeLayer {
    /// Batched forward: GEMM gate for N tokens, per-token expert dispatch.
    ///
    /// Gate projection reads weights once for N tokens (GEMM M=N).
    /// Expert dispatch remains per-token (data-dependent routing).
    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let bf16 = 2usize;
        let n = num_tokens as u32;

        // FP32 gate path (ATLAS_FP32_GATE): keep the router GEMM accumulator in
        // FP32 through top-K so two experts whose logits differ by less than a
        // BF16 ULP no longer flip routing (the cross-compiler routing-cascade
        // trigger on gfx1151). Only the softmax-routed dense-gate path is
        // covered — the NVFP4 gate and the sigmoid+bias path keep BF16. Falls
        // back to BF16 if the f32 kernels are absent on this target.
        // ATLAS_FP32_ROUTING: the SSM-side norm already wrote an FP32 router_in
        // (residual_add_rms_norm_gatef32 → moe_router_in_f32); the gate GEMM
        // reads it at full precision via dense_gemm_f32in. Supersedes the
        // gate-only ATLAS_FP32_GATE (which keeps the BF16 router_in but f32 gate
        // accumulation). Either way the gate logits + top-K run in FP32.
        let fp32_routing = self.fp32_routing_active();
        let fp32_gate = fp32_routing
            || (self.gate_nvfp4.is_none()
                && self.correction_bias_dev.is_none()
                && self.dense_gemm_f32out.0 != 0
                && self.moe_topk_f32.0 != 0
                && std::env::var("ATLAS_FP32_GATE").as_deref() == Ok("1"));
        let gate_elem = if fp32_gate { 4usize } else { bf16 };

        // Gemma-4 router pre-norm (no-op for other models).
        let router_in = self.router_input(input, n, h, ctx, stream)?;
        // Gate GEMM: [N, H] × [H, num_experts] → [N, num_experts]
        let gate_logits = if fp32_gate {
            ctx.buffers.gate_logits_f32() // [N, num_experts] FP32
        } else {
            ctx.buffers.gate_logits() // [N, num_experts] BF16
        };
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
        } else if fp32_routing {
            // FP32 router_in (from residual_add_rms_norm_gatef32) × bf16 gate.
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_f32in,
                ctx.buffers.moe_router_in_f32(),
                &self.weights.gate,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                if fp32_gate {
                    self.dense_gemm_f32out
                } else {
                    self.dense_gemm
                },
                router_in,
                &self.weights.gate,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        }
        // Routing-divergence diagnostic (no-op unless ATLAS_DUMP_EXPERT_IDS=1):
        // last-token gate logits, so the batched path can be compared to gb10
        // the same way the grouped paths are (HIP MoE routing-flip bisection).
        // The dump reads BF16; skip it on the FP32-gate path.
        if !fp32_gate {
            super::dump::dump_gate_logits(ctx.gpu, stream, gate_logits, n, num_experts)?;
        }

        // Per-token: topK routing + expert dispatch + weighted sum
        let h_usize = h as usize;
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        // ⚠ logits buffer aliased — see warning in moe/forward.rs:208-219
        // and project_batch_decode_corruption.md (bug 2). Concurrent
        // callers using `buffers.logits()` during the forward loop MUST
        // offset past `shared_expert_intermediate_size * 2` bytes.
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();

        for t in 0..num_tokens {
            let input_t = input.offset(t * h_usize * bf16);
            let gate_t = gate_logits.offset(t * num_experts as usize * gate_elem);
            let output_t = ctx.buffers.moe_output().offset(t * h_usize * bf16);

            let scratch = ctx.buffers.scratch();
            let indices_dev = scratch;
            let weights_dev = scratch.offset(top_k as usize * 4);

            if let Some(tid2eid) = self.tid2eid_dev {
                // DeepSeek-V4 hash routing: expert selection is static
                // `tid2eid[token_id]`; the learned gate weights the selection.
                // token IDs are uploaded [num_tokens] u32 in the SAME order as
                // this loop, so token t lives at offset t.
                let token_ids = ctx.token_ids.ok_or_else(|| {
                    anyhow::anyhow!(
                        "DeepSeek-V4 hash-MoE layer requires ForwardContext.token_ids (prefill)"
                    )
                })?;
                ops::moe_hash_route(
                    ctx.gpu,
                    self.moe_hash_route_k,
                    gate_t,
                    tid2eid,
                    token_ids.offset(t * 4),
                    indices_dev,
                    weights_dev,
                    num_experts,
                    top_k,
                    ctx.config.norm_topk_prob,
                    ctx.config.routed_scaling_factor as f32,
                    stream,
                )?;
            } else if let Some(bias) = self.correction_bias_dev {
                // DeepSeek-V4: sqrt-softplus expert scoring (replaces sigmoid).
                if ctx.config.scoring_func == "sqrtsoftplus" {
                    ops::moe_topk_sqrtsoftplus(
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
                    )?;
                } else {
                    ops::moe_topk_sigmoid(
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
                    )?;
                }
            } else {
                ops::moe_topk_softmax(
                    ctx.gpu,
                    if fp32_gate {
                        self.moe_topk_f32
                    } else {
                        self.moe_topk
                    },
                    gate_t,
                    indices_dev,
                    weights_dev,
                    num_experts,
                    top_k,
                    ctx.config.norm_topk_prob,
                    stream,
                )?;
            }
            // Last-token routing dump (no-op unless ATLAS_DUMP_EXPERT_IDS=1):
            // the token whose top-K determines the next prediction.
            if t == num_tokens - 1 {
                super::dump::dump_expert_ids(ctx.gpu, stream, indices_dev, weights_dev, 1, top_k)?;
            }

            let shared_out = ctx.buffers.attn_output();
            if let (Some(gp), Some(up), Some(dp), Some(sg), Some(su), Some(sd)) = (
                self.bf16_gate_weight_ptrs,
                self.bf16_up_weight_ptrs,
                self.bf16_down_weight_ptrs,
                self.bf16_shared_gate,
                self.bf16_shared_up,
                self.bf16_shared_down,
            ) {
                // BF16 path (FP8-dequant-on-load): same fused kernels as decode.
                ops::moe_expert_gate_up_shared_bf16(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared_bf16_k,
                    input_t,
                    gp,
                    expert_gate_out,
                    up,
                    expert_up_out,
                    indices_dev,
                    sg,
                    shared_gate_scratch,
                    su,
                    shared_up_scratch,
                    inter,
                    h,
                    top_k,
                    stream,
                )?;
                ops::moe_expert_silu_down_shared_bf16(
                    ctx.gpu,
                    self.moe_expert_silu_down_shared_bf16_k,
                    expert_gate_out,
                    expert_up_out,
                    dp,
                    expert_down_out,
                    indices_dev,
                    shared_gate_scratch,
                    shared_up_scratch,
                    sd,
                    shared_out,
                    h,
                    inter,
                    top_k,
                    stream,
                )?;
            } else if let (Some(gp), Some(up), Some(dp), Some(sh)) = (
                &self.fp8_gate_weight_ptrs,
                &self.fp8_up_weight_ptrs,
                &self.fp8_down_weight_ptrs,
                &self.fp8_shared_expert,
            ) {
                // FP8 path for batched decode
                ops::moe_expert_gate_up_shared_fp8(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared_fp8,
                    input_t,
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
                ops::moe_expert_silu_down_shared_fp8(
                    ctx.gpu,
                    self.moe_expert_silu_down_shared_fp8,
                    expert_gate_out,
                    expert_up_out,
                    dp.weight_ptrs,
                    dp.scale_ptrs,
                    expert_down_out,
                    indices_dev,
                    shared_gate_scratch,
                    shared_up_scratch,
                    &sh.down_proj,
                    shared_out,
                    h,
                    inter,
                    top_k,
                    stream,
                )?;
            } else if self.use_t_layout_for_prefill() {
                // Phase 8a unified-layout NVFP4 batched prefill — transposed
                // kernels coalesce well at large N. Hybrid mode lands here too.
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
                // ARM-2 Phase-K RIDER A1: the _e8m0 fused decode kernel is
                // <32,true,GROUP_SIZE,false> — routed E8M0, shared NVFP4. Assert
                // the shared expert really is NVFP4 before trusting that.
                if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
                    self.shared_experts_scale_kind.expect(
                        crate::weight_map::WeightQuantFormat::Nvfp4,
                        "decode fused _e8m0 kernel assumes an NVFP4 shared expert",
                    );
                }
                ops::moe_expert_gate_up_shared_t(
                    ctx.gpu,
                    self.e8m0_or(
                        self.moe_expert_gate_up_shared_t_k,
                        self.moe_expert_gate_up_shared_t_e8m0_k,
                        "decode gate_up_shared_t",
                    ),
                    input_t,
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
                ops::moe_expert_silu_down_shared_t(
                    ctx.gpu,
                    self.e8m0_or(
                        self.moe_expert_silu_down_shared_t_k,
                        self.moe_expert_silu_down_shared_t_e8m0_k,
                        "decode silu_down_shared_t",
                    ),
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
                    shared_out,
                    h,
                    inter,
                    top_k,
                    stream,
                )?;
            } else {
                // NVFP4 path
                ops::moe_expert_gate_up_shared(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared,
                    input_t,
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
                    stream,
                )?;
                ops::moe_expert_silu_down_shared(
                    ctx.gpu,
                    self.moe_expert_silu_down_shared,
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
                    shared_out,
                    h,
                    inter,
                    top_k,
                    stream,
                )?;
            }

            ops::moe_weighted_sum_blend(
                ctx.gpu,
                self.moe_weighted_sum_blend,
                output_t,
                expert_down_out,
                weights_dev,
                shared_out,
                input_t,
                self.weights.shared_expert_gate.weight,
                h,
                top_k,
                h,
                stream,
            )?;

            // EP all-reduce per-token partial output
            if let Some(comm) = ctx.comm
                && ctx.config.ep_world_size > 1
            {
                if ctx.graph_capture {
                    comm.all_reduce(output_t.0, h as usize * 2)?;
                } else {
                    comm.all_reduce_async(output_t.0, h as usize * 2, stream)?;
                }
            }
        }

        Ok(())
    }
}
