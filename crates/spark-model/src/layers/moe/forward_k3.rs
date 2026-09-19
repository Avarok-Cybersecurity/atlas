// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward_k3 (verify K=3).

use super::*;

impl MoeLayer {
    /// Fused K=3 forward: process 3 tokens through MoE in 5 kernel launches.
    ///
    /// Gate GEMV batch3 → batched topK → fused expert gate+up → fused silu+down → fused wsum+blend.
    /// Expert buffers sized for 3*top_k slots. Output at moe_output() [3, H].
    pub fn forward_k3(
        &self,
        input: DevicePtr, // [3, H] BF16 — normed MoE input for 3 tokens
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // LongCat zero-experts are wired only on the single-token decode
        // + prefill paths (v1); this variant would silently mis-route the
        // 384-wide router. Named refusal, not silent wrongness.
        anyhow::ensure!(
            self.router_logits_n as usize == ctx.config.num_experts,
            "zero-expert MoE routing is not wired on this dispatch variant yet (forward_k3)"
        );

        // Feature-1: a resident MoE adapter forces the per-row batched fallback
        // (folds gate/up/down route-agnostically; base rows no-op; same
        // moe_output[3,H]), skipping any no-fold fast path. Install-time gate →
        // graph-safe (graphs drain on rotate/swap). Router adapter refused inside.
        if self.lora.is_some() {
            return self.forward_batched(input, 3, ctx, stream);
        }
        // BF16 (FP8-dequant-on-load) experts have no fused batch3 kernel.
        // The FP8 batch3 branch below would read expert weights that were
        // FREED at dequant-load → garbage MTP-verify logits → degenerate
        // repetition. Route the 3-token verify through the per-token BF16
        // batched path, which produces the same moe_output()[3,H]. (SSOT:
        // reuses the decode BF16 kernels via forward_batched.)
        if self.bf16_gate_weight_ptrs.is_some() {
            return self.forward_batched(input, 3, ctx, stream);
        }
        // Mixed NVFP4-routed / BF16-shared (Laguna): batch the routed half
        // through the _t kernels and run the shared expert as one batched BF16
        // pass afterwards. See forward_k2 for the rationale.
        let mixed_bf16_shared = self.has_mixed_bf16_shared_expert();
        if mixed_bf16_shared
            && !(self.use_t_layout_for_decode()
                && self.moe_expert_gate_up_shared_batch3_t_k.0 != 0
                && self.moe_expert_silu_down_shared_batch3_t_k.0 != 0
                && !(ctx.comm.is_some() && ctx.config.ep_world_size > 1))
        {
            return self.forward_batched(input, 3, ctx, stream);
        }
        // E8M0 (native MXFP4, per-32 E8M0 scale) routed experts MUST NOT reach
        // the unified-T batch3 kernel `moe_expert_gate_up_shared_batch3_t`: like
        // its K=2 twin it is an NVFP4 kernel that hardcodes GROUP_SIZE=16 and
        // would read `inter·h/16` scale bytes from the correctly-sized
        // `inter·h/32` E8M0 scale buffer — a 2× over-read →
        // CUDA_ERROR_ILLEGAL_ADDRESS (it also E4M3-decodes E8M0 scale bytes →
        // garbage even in-bounds). No E8M0 batch3 kernel exists, so route all 3
        // verify tokens through the per-token unified-T path (`forward_batched`),
        // whose `use_t_layout_for_prefill` branch selects the GS32 `_e8m0`
        // kernel via `e8m0_or` — the same correct path ordinary decode already
        // uses. Mirrors the K=2 guard at the top of `forward_k2`.
        if k3_e8m0_needs_per_token(self.experts_scale_kind) {
            return self.forward_batched(input, 3, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;

        // Gemma-4 router pre-norm (no-op for other models).
        let router_in = self.router_input(input, 3, h, ctx, stream)?;
        // 1. Gate GEMV batch3: reads gate weight once for 3 tokens
        let gate_logits = ctx.buffers.gate_logits();
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemv_batch3(
                ctx.gpu,
                self.w4a16_gemv_batch3,
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
                3,
                num_experts,
                h,
                stream,
            )?;
        }

        // 2. Batched topK for 3 tokens. Sigmoid+bias for MiniMax/DeepSeek-V3,
        //    softmax otherwise.
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(3 * top_k as usize * 4);
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
                3,
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
                3,
                stream,
            )?;
        }

        super::union_stats::maybe_sample_expert_union(ctx, indices_dev, 3, top_k as usize, stream);

        // 3-5. Fused expert dispatch for 3 tokens
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_down_out = ctx.buffers.attn_output();
        let output = ctx.buffers.moe_output();

        let is_ep = ctx.comm.is_some() && ctx.config.ep_world_size > 1;

        if let (Some(gp), Some(up), Some(dp), Some(sh)) = (
            &self.fp8_gate_weight_ptrs,
            &self.fp8_up_weight_ptrs,
            &self.fp8_down_weight_ptrs,
            &self.fp8_shared_expert,
        ) {
            // FP8 batch3 path
            ops::moe_expert_gate_up_shared_fp8_batch3(
                ctx.gpu,
                self.moe_expert_gate_up_shared_fp8_batch3,
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
            ops::moe_expert_silu_down_shared_fp8_batch3(
                ctx.gpu,
                self.moe_expert_silu_down_shared_fp8_batch3,
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
            let shared_for_blend = if is_ep && !shared_down_out.is_null() {
                ctx.gpu
                    .memset_async(expert_gate_out, 0, 3 * h as usize * 2, stream)?;
                expert_gate_out
            } else {
                shared_down_out
            };
            ops::moe_weighted_sum_blend_batch3(
                ctx.gpu,
                self.moe_weighted_sum_blend_fp8_batch3,
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
            // Phase 8a unified-layout NVFP4 batch=3 verify (MTP K=3). Hybrid
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
            // Mixed config: in-kernel shared expert off (NULL), computed in
            // BF16 below instead — the NVFP4 shared_*_t tables are load-time
            // placeholders and numerically wrong for this checkpoint.
            let (sh_gate_t, sh_up_t, sh_down_t) = if mixed_bf16_shared {
                (&null_qw, &null_qw, &null_qw)
            } else {
                (
                    self.shared_gate_t.as_ref().unwrap_or(&null_qw),
                    self.shared_up_t.as_ref().unwrap_or(&null_qw),
                    self.shared_down_t.as_ref().unwrap_or(&null_qw),
                )
            };
            ops::moe_expert_gate_up_shared_batch3_t(
                ctx.gpu,
                self.moe_expert_gate_up_shared_batch3_t_k,
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
            ops::moe_expert_silu_down_shared_batch3_t(
                ctx.gpu,
                self.moe_expert_silu_down_shared_batch3_t_k,
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
            if mixed_bf16_shared {
                let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
                self.run_bf16_shared_expert(
                    input,
                    3,
                    h,
                    shared_inter,
                    shared_gate_scratch,
                    shared_up_scratch,
                    shared_down_out,
                    ctx,
                    stream,
                )?;
            }
            // The _t branch previously returned without writing moe_output at
            // all — every sibling branch ends in this blend.
            let shared_for_blend = if is_ep && !shared_down_out.is_null() {
                ctx.gpu
                    .memset_async(expert_gate_out, 0, 3 * h as usize * 2, stream)?;
                expert_gate_out
            } else {
                shared_down_out
            };
            ops::moe_weighted_sum_blend_batch3(
                ctx.gpu,
                self.moe_weighted_sum_blend_batch3,
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
        } else {
            // NVFP4 batch3 path
            ops::moe_expert_gate_up_shared_batch3(
                ctx.gpu,
                self.moe_expert_gate_up_shared_batch3,
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
                stream,
            )?;
            ops::moe_expert_silu_down_shared_batch3(
                ctx.gpu,
                self.moe_expert_silu_down_shared_batch3,
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
                stream,
            )?;
            // EP fix: after silu_down, expert_gate_out is free — use as zero buffer
            let shared_for_blend = if is_ep && !shared_down_out.is_null() {
                ctx.gpu
                    .memset_async(expert_gate_out, 0, 3 * h as usize * 2, stream)?;
                expert_gate_out
            } else {
                shared_down_out
            };
            ops::moe_weighted_sum_blend_batch3(
                ctx.gpu,
                self.moe_weighted_sum_blend_batch3,
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

        // EP all-reduce: sum partial outputs for 3 tokens
        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
        {
            if ctx.graph_capture {
                comm.all_reduce(output.0, 3 * h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, 3 * h as usize * 2, stream)?;
            }
            // Add shared expert with sigmoid gate (BUG #41 fix)
            if !shared_down_out.is_null() {
                if self.weights.shared_expert_gate.weight.0 == 0 {
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add,
                        output,
                        shared_down_out,
                        3 * h,
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
                        3,
                        stream,
                    )?;
                }
            }
        }

        Ok(())
    }
}

/// K=3-verify MoE dispatch guard — the K=3 twin of `k2_e8m0_needs_per_token`
/// (`forward_k2.rs`). E8M0 (native MXFP4, per-32 E8M0 scale) routed experts
/// MUST take the per-token unified-T path (GS32 `_e8m0` kernel via `e8m0_or`),
/// NOT the GS16 NVFP4 `moe_expert_gate_up_shared_batch3_t` batch3 kernel: that
/// kernel reads `inter·h/16` scale bytes from the correctly-sized `inter·h/32`
/// E8M0 scale buffer — a 2× over-read → CUDA_ERROR_ILLEGAL_ADDRESS.
/// Pure decision, unit-tested and wired at the top of `forward_k3`.
pub(crate) fn k3_e8m0_needs_per_token(scale_kind: crate::weight_map::WeightQuantFormat) -> bool {
    matches!(scale_kind, crate::weight_map::WeightQuantFormat::Mxfp4E8m0)
}

// Focused dispatch tests live in a sibling file (same pattern as forward_k2).
#[cfg(test)]
#[path = "forward_k3_dispatch_tests.rs"]
mod k3_dispatch_tests;
