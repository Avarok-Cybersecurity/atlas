// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward_k2 (verify K=2).

use anyhow::Context as _;

use super::*;

mod originals;
mod unified_t;

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
        // LongCat zero-experts are wired only on the single-token decode
        // + prefill paths (v1); this variant would silently mis-route the
        // 384-wide router. Named refusal, not silent wrongness.
        anyhow::ensure!(
            self.router_logits_n as usize == ctx.config.num_experts,
            "zero-expert MoE routing is not wired on this dispatch variant yet (forward_k2)"
        );

        // Feature-1: the fused batch2 fast path has no fold hook. When a MoE
        // adapter is RESIDENT (install-time-fixed → graph-safe; graphs drain on
        // rotate/swap), route to the per-row batched fallback which folds
        // gate/up/down route-agnostically (base rows no-op) — same moe_output[2,H].
        // forward_batched itself refuses a router-adapted adapter.
        if self.lora.is_some() {
            return self.forward_batched(input, 2, ctx, stream);
        }
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
        // E8M0 (native MXFP4, per-32 E8M0 scale) routed experts MUST NOT reach the
        // unified-T batch2 kernel `moe_expert_gate_up_shared_batch2_t`: it is an
        // NVFP4 kernel that hardcodes GROUP_SIZE=16 and would read `inter·h/16`
        // scale bytes from the correctly-sized `inter·h/32` E8M0 scale buffer — a
        // 2× over-read → CUDA_ERROR_ILLEGAL_ADDRESS (it also E4M3-decodes E8M0
        // scale bytes → garbage even in-bounds). No E8M0 batch2 kernel exists, so
        // route both verify tokens through the per-token unified-T path
        // (`forward_batched`), whose `use_t_layout_for_prefill` branch selects the
        // GS32 `_e8m0` kernel via `e8m0_or` — the same correct path ordinary decode
        // already uses. Mirrors the BF16 fallback above.
        if k2_e8m0_needs_per_token(self.experts_scale_kind) {
            return self.forward_batched(input, 2, ctx, stream);
        }
        // Mixed NVFP4-routed / BF16-shared (Laguna): the fused batch2 kernels
        // cannot compute a BF16 shared expert alongside NVFP4 routed weights.
        // Under the transposed unified layout we still batch the routed half
        // through the _t kernels and run the shared expert as one batched BF16
        // GEMM pass afterwards (`mixed_bf16_shared` below). Every other layout
        // falls back to the per-token loop.
        let mixed_bf16_shared = self.has_mixed_bf16_shared_expert();
        // Either fused layout serves the mixed config. The originals-layout
        // kernels are usable only since 37e818ad NULL-guarded their shared
        // expert (their `_t` siblings always had that guard); before it, this
        // faulted with CUDA 700 on the first 2-sequence batch.
        let mixed_t_ok = self.use_t_layout_for_decode()
            && self.moe_expert_gate_up_shared_batch2_t_k.0 != 0
            && self.moe_expert_silu_down_shared_batch2_t_k.0 != 0;
        let mixed_orig_ok = !self.use_t_layout_for_decode()
            && self.moe_expert_gate_up_shared_batch2.0 != 0
            && self.moe_expert_silu_down_shared_batch2.0 != 0
            && !self.gate_ptrs.packed_ptrs.is_null();
        if mixed_bf16_shared && !((mixed_t_ok || mixed_orig_ok) && !is_ep) {
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
        super::union_stats::maybe_sample_expert_union(ctx, indices_dev, 2, top_k as usize, stream);

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
            && let (Some(gp), Some(up), Some(dp), Some(shared)) = (
                self.bf16_gate_weight_ptrs,
                self.bf16_up_weight_ptrs,
                self.bf16_down_weight_ptrs,
                self.bf16_shared_expert,
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
                shared.gate_proj.weight,
                shared_gate_scratch,
                shared.up_proj.weight,
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
                shared.down_proj.weight,
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
            self.forward_k2_unified_t(
                input,
                indices_dev,
                weights_dev,
                expert_gate_out,
                expert_up_out,
                expert_down_out,
                shared_gate_scratch,
                shared_up_scratch,
                shared_down_out,
                output,
                inter,
                h,
                top_k,
                is_ep,
                mixed_bf16_shared,
                ctx,
                stream,
            )?;
        } else {
            self.forward_k2_originals(
                input,
                indices_dev,
                weights_dev,
                expert_gate_out,
                expert_up_out,
                expert_down_out,
                shared_gate_scratch,
                shared_up_scratch,
                shared_down_out,
                output,
                inter,
                h,
                top_k,
                is_ep,
                mixed_bf16_shared,
                ctx,
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

// Pure dispatch helpers live in a sibling file (500-LoC cap).
mod forward_k2_helpers;
pub(crate) use forward_k2_helpers::{batch2_block_width, k2_e8m0_needs_per_token};

// Focused dispatch tests live in a sibling file to keep this file ≤500 LoC.
#[cfg(test)]
#[path = "forward_k2_dispatch_tests.rs"]
mod k2_dispatch_tests;
