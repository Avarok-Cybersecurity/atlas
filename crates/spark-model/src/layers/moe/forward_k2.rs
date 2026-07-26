// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward_k2 (verify K=2).

use anyhow::Context as _;

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
        // BF16 (FP8-dequant-on-load) experts. The FP8/NVFP4 batch2 branches
        // below read expert weights that were FREED at dequant-load, so they
        // must NOT run for a dequanted model. When the fused BF16 batch2
        // kernels are present (and we're not EP), take the dedicated BF16
        // batch2 path (single-launch 2-token dispatch, same math as the
        // per-token BF16 decode kernels). Otherwise fall back to the per-token
        // BF16 batched path (SSOT: reuses the decode BF16 kernels via
        // forward_batched), which produces the same moe_output()[2,H].
        let is_ep = ctx.comm.is_some() && ctx.config.ep_world_size > 1;
        let use_bf16_batch2 = self.bf16_gate_weight_ptrs.is_some()
            && self.moe_expert_gate_up_shared_bf16_batch2_k.0 != 0
            && self.moe_expert_silu_down_shared_bf16_batch2_k.0 != 0
            && !is_ep;
        if self.bf16_gate_weight_ptrs.is_some() && !use_bf16_batch2 {
            return self.forward_batched(input, 2, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;

        // DIAG (ATLAS_K2_DIAG=1): synchronize checkpoints to localize the K2-verify
        // illegal access (the V4 NVFP4 batch2 verify path is exercised for the first
        // time by MTP). The label of the FIRST failing sync names the bad stage.
        let k2_diag = std::env::var("ATLAS_K2_DIAG").is_ok_and(|v| v == "1");
        if k2_diag {
            ctx.gpu
                .synchronize(stream)
                .context("K2 ENTRY: attention+norm BEFORE forward_k2")?;
        }

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
            // DeepSeek-V4 scores experts with sqrt(softplus(.)); sigmoid otherwise
            // (MiniMax/DeepSeek-V3). Must match the prefill/single-token paths or
            // decode routing diverges from prefill.
            if ctx.config.scoring_func == "sqrtsoftplus" {
                // Use the PROVEN non-batched sqrtsoftplus kernel per token (the
                // _batched variant is unexercised — the K2 verify is the only
                // user and it never ran for V4 before). gate_logits is BF16
                // [2, num_experts] (2-byte stride); indices/weights are
                // [2, top_k] (u32 / f32, 4-byte stride).
                for t in 0..2usize {
                    ops::moe_topk_sqrtsoftplus(
                        ctx.gpu,
                        self.moe_topk_sqrtsoftplus_k,
                        gate_logits.offset(t * num_experts as usize * 2),
                        bias,
                        indices_dev.offset(t * top_k as usize * 4),
                        weights_dev.offset(t * top_k as usize * 4),
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
                    indices_dev,
                    weights_dev,
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
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                2,
                stream,
            )?;
        }
        super::union_stats::maybe_sample_expert_union(
            ctx.gpu,
            indices_dev,
            2,
            top_k as usize,
            stream,
        );

        if k2_diag {
            ctx.gpu
                .synchronize(stream)
                .context("K2: gate-GEMV + topk")?;
        }

        // 3-5. Fused expert dispatch for 2 tokens
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_down_out = ctx.buffers.attn_output();
        let output = ctx.buffers.moe_output();

        if use_bf16_batch2
            && let (Some(gp), Some(up), Some(dp), Some(sg), Some(su), Some(sd)) = (
                self.bf16_gate_weight_ptrs,
                self.bf16_up_weight_ptrs,
                self.bf16_down_weight_ptrs,
                self.bf16_shared_gate,
                self.bf16_shared_up,
                self.bf16_shared_down,
            )
        {
            // BF16 batch2 path (FP8-dequant-on-load experts, MTP K=2 verify).
            // Single-launch 2-token dispatch mirroring the FP8 batch2 layout;
            // identical math to the per-token moe_expert_*_shared_bf16 kernels.
            // Non-EP only (guaranteed by use_bf16_batch2).
            ops::moe_expert_gate_up_shared_bf16_batch2(
                ctx.gpu,
                self.moe_expert_gate_up_shared_bf16_batch2_k,
                input,
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
            ops::moe_expert_silu_down_shared_bf16_batch2(
                ctx.gpu,
                self.moe_expert_silu_down_shared_bf16_batch2_k,
                expert_gate_out,
                expert_up_out,
                dp,
                expert_down_out,
                indices_dev,
                shared_gate_scratch,
                shared_up_scratch,
                sd,
                shared_down_out,
                h,
                inter,
                top_k,
                stream,
            )?;
            ops::moe_weighted_sum_blend_batch2(
                ctx.gpu,
                self.moe_weighted_sum_blend_batch2,
                output,
                expert_down_out,
                weights_dev,
                shared_down_out,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                top_k,
                h,
                stream,
            )?;
        } else if let (Some(gp), Some(up), Some(dp), Some(sh)) = (
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

        if k2_diag {
            ctx.gpu
                .synchronize(stream)
                .context("K2: expert dispatch (gate_up/silu_down/blend)")?;
        }

        // EP all-reduce: sum partial outputs for 2 tokens
        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
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
