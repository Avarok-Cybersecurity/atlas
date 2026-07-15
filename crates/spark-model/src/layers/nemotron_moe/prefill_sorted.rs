// SPDX-License-Identifier: AGPL-3.0-only

//! Nemotron MoE prefill — sorted-MoE expert dispatch path.
//!
//! Extracted from `NemotronMoeLayer::prefill` for file-size budget.
//! See `nemotron_moe.rs` for the surrounding setup; this helper handles the
//! batched grouped-GEMM dispatch when sorted-MoE kernels are available.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::NemotronMoeLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

/// Locals captured from `prefill` and passed to the sorted-MoE branch.
pub(super) struct SortedPrefillCtx {
    pub n: u32,
    pub num_tokens: usize,
    pub h: usize,
    pub inter: u32,
    pub shared_inter: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub scale: f32,
    pub latent: u32,
    pub gate_logits: DevicePtr,
    pub indices_dev: DevicePtr,
    pub weights_dev: DevicePtr,
    pub normed: DevicePtr,
    pub hidden: DevicePtr,
    pub latent_base: Option<DevicePtr>,
    pub shared_up_out_base: DevicePtr,
}

impl NemotronMoeLayer {
    pub(super) fn prefill_sorted_path(
        &self,
        p: &SortedPrefillCtx,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let total_expanded = p.n * p.top_k;
        let ne = p.num_experts as usize;
        let te = total_expanded as usize;

        // 5a. Batched routing: [N, E] → indices[N*p.top_k], weights[N*p.top_k]
        KernelLaunch::new(ctx.gpu, self.topk_sigmoid_batched_k)
            .grid([1, p.n, 1])
            .block([256, 1, 1])
            .arg_ptr(p.gate_logits)
            .arg_ptr(self.weights.e_score_correction_bias.weight)
            .arg_ptr(p.indices_dev)
            .arg_ptr(p.weights_dev)
            .arg_u32(p.num_experts)
            .arg_u32(p.top_k)
            .arg_u32(if ctx.config.norm_topk_prob { 1 } else { 0 })
            .arg_f32(p.scale)
            .arg_u32(p.n)
            .launch(stream)?;

        // 5b. Sort by expert → sorted_token_ids, expert_offsets
        // Reuse p.gate_logits buffer for sorted arrays (p.gate_logits no longer needed)
        let sorted_token_ids = p.gate_logits;
        let sorted_expert_ids = p.gate_logits.offset(te * 4);
        let expert_offsets = p.gate_logits.offset(te * 4 * 2);
        let token_to_perm = p.gate_logits.offset(te * 4 * 2 + (ne + 1) * 4);
        ops::moe_sort_by_expert(
            ctx.gpu,
            self.moe_sort_k,
            p.indices_dev,
            sorted_token_ids,
            sorted_expert_ids,
            expert_offsets,
            token_to_perm,
            total_expanded,
            p.num_experts,
            p.top_k,
            stream,
        )?;

        // Determine expert input source and dimensions based on LatentMoE vs direct.
        // LatentMoE (Super): experts operate in p.latent space [L], input from fc1_latent.
        // Direct (Nano): experts operate in p.hidden space [H], input from p.normed.
        let is_latent = self.moe_latent_size > 0;
        let expert_input = if is_latent {
            p.latent_base.unwrap()
        } else {
            p.normed
        };
        let expert_k = if is_latent { p.latent } else { p.h as u32 }; // input dim to expert UP
        let expert_out_dim = if is_latent { p.latent } else { p.h as u32 }; // output dim from expert DOWN

        // 5c. Grouped UP GEMM: [sorted, K_expert] → [sorted, p.inter]
        let expert_up_out = ctx.buffers.expert_up_out();
        let avg_per_expert = (p.num_tokens * p.top_k as usize).div_ceil(ne);
        let max_m_tiles = (avg_per_expert * 2).div_ceil(64).max(1) as u32;
        // Native FP4 up GEMM: latent activations quantized to NVFP4 once per layer,
        // expert weights consumed as raw E2M1 + scales (no LUT dequant), relu^2
        // fused. OPT-IN ONLY (ATLAS_MOE_W4A4=1): although the latent is a linear
        // (fc1) output, it is the input to EVERY routed expert, and quantizing it
        // to FP4 produced a systematic repetition tic in long-prompt A/B ("the
        // silence between notes, the silence between breaths, the silence...")
        // plus one outright story-collapse, while the BF16 path stayed clean on
        // the same prompts. Worth ~13ms at 1k if the numerics are ever fixed
        // (calibrated input_scale instead of dynamic scale2=1.0 is the candidate).
        let w4a4_up = is_latent
            && p.n >= 512
            && self.moe_w4a4_grouped_k.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && ctx.buffers.fp8_act_bytes() >= (p.n as usize) * (p.latent as usize)
            && std::env::var("ATLAS_MOE_W4A4").is_ok();
        if w4a4_up {
            let a4 = ctx.buffers.fp8_act();
            let a4_sf = a4.offset((p.n as usize) * (p.latent as usize) / 2);
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                expert_input,
                a4,
                a4_sf,
                p.n,
                p.latent,
                stream,
            )?;
            ops::moe_w4a4_grouped_gemm_relu2(
                ctx.gpu,
                self.moe_w4a4_grouped_k,
                a4,
                a4_sf,
                self.up_ptrs.packed_ptrs,
                self.up_ptrs.scale_ptrs,
                self.up_ptrs.scale2_vals,
                expert_up_out,
                expert_offsets,
                sorted_token_ids,
                p.num_experts,
                p.inter,
                expert_k,
                max_m_tiles,
                stream,
            )?;
        } else if let Some(ref upt) = self.up_ptrs_t {
            ops::moe_w4a16_grouped_gemm_ptrtable(
                ctx.gpu,
                self.moe_grouped_gemm_n128_k,
                expert_input,
                upt.packed_ptrs,
                upt.scale_ptrs,
                upt.scale2_vals,
                expert_up_out,
                expert_offsets,
                sorted_token_ids,
                p.num_experts,
                p.inter,
                expert_k,
                max_m_tiles,
                stream,
            )?;
            // This branch has no fused epilogue: apply relu^2 elementwise.
            let relu2_n = total_expanded * p.inter;
            KernelLaunch::new(ctx.gpu, self.moe_relu2_elementwise_k)
                .grid([div_ceil(relu2_n, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(expert_up_out)
                .arg_u32(relu2_n)
                .launch(stream)?;
        } else {
            // relu^2 fused into the UP GEMM's store when the epilogue variant is
            // compiled -- saves a full read+write of expert_up_out.
            let fused = self.moe_grouped_gemm_relu2_k.0 != 0;
            ops::moe_w4a16_grouped_gemm_ptrtable(
                ctx.gpu,
                if fused {
                    self.moe_grouped_gemm_relu2_k
                } else {
                    self.moe_grouped_gemm_k
                },
                expert_input,
                self.up_ptrs.packed_ptrs,
                self.up_ptrs.scale_ptrs,
                self.up_ptrs.scale2_vals,
                expert_up_out,
                expert_offsets,
                sorted_token_ids,
                p.num_experts,
                p.inter,
                expert_k,
                max_m_tiles,
                stream,
            )?;
            if !fused {
                let relu2_n = total_expanded * p.inter;
                KernelLaunch::new(ctx.gpu, self.moe_relu2_elementwise_k)
                    .grid([div_ceil(relu2_n, 256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(expert_up_out)
                    .arg_u32(relu2_n)
                    .launch(stream)?;
            }
        }

        // 5e. Grouped DOWN GEMM: [sorted, p.inter] → [sorted, expert_out_dim]
        let expert_down_out = ctx.buffers.expert_down_out();
        if let Some(ref dpt) = self.down_ptrs_t {
            ops::moe_w4a16_grouped_gemm_ptrtable(
                ctx.gpu,
                self.moe_grouped_gemm_n128_k,
                expert_up_out,
                dpt.packed_ptrs,
                dpt.scale_ptrs,
                dpt.scale2_vals,
                expert_down_out,
                expert_offsets,
                DevicePtr::NULL,
                p.num_experts,
                expert_out_dim,
                p.inter,
                max_m_tiles,
                stream,
            )?;
        } else {
            ops::moe_w4a16_grouped_gemm_ptrtable(
                ctx.gpu,
                self.moe_grouped_gemm_k,
                expert_up_out,
                self.down_ptrs.packed_ptrs,
                self.down_ptrs.scale_ptrs,
                self.down_ptrs.scale2_vals,
                expert_down_out,
                expert_offsets,
                DevicePtr::NULL,
                p.num_experts,
                expert_out_dim,
                p.inter,
                max_m_tiles,
                stream,
            )?;
        }

        // 5f. Unpermute + weighted reduce → [N, expert_out_dim]
        let routed_out = ctx.buffers.moe_output();
        ops::moe_unpermute_reduce_indexed(
            ctx.gpu,
            self.moe_unpermute_reduce_k,
            expert_down_out,
            routed_out,
            token_to_perm,
            p.weights_dev,
            expert_out_dim,
            p.n,
            p.top_k,
            stream,
        )?;

        // 5g. Shared expert: shared_up_out already computed in step 3.
        let shared_down_out = ctx.buffers.ssm_deinterleaved();
        let shared_relu2_n = p.n * p.shared_inter;
        KernelLaunch::new(ctx.gpu, self.moe_relu2_elementwise_k)
            .grid([div_ceil(shared_relu2_n, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(p.shared_up_out_base)
            .arg_u32(shared_relu2_n)
            .launch(stream)?;
        // OPT-IN only: shared_down's input is relu^2(x) -- squared, all-positive,
        // wide dynamic range -- and quantizing it to FP4 measurably degraded long-
        // prompt outputs (hallucinated MC options, repetition loops) while the
        // normed-input W4A4 GEMMs stayed clean. ATLAS_SHARED_W4A4_DOWN=1 to test.
        let w4a4_down = p.n >= 512
            && self.w4a4_gemm_k.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && ctx.buffers.fp8_act_bytes() >= (p.shared_inter as usize) * (p.n as usize)
            && std::env::var("ATLAS_SHARED_W4A4_DOWN").is_ok();
        if w4a4_down {
            let a4 = ctx.buffers.fp8_act();
            let a4_sf = a4.offset((p.n as usize) * (p.shared_inter as usize) / 2);
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                p.shared_up_out_base,
                a4,
                a4_sf,
                p.n,
                p.shared_inter,
                stream,
            )?;
            ops::w4a4_gemm_mfast(
                ctx.gpu,
                self.w4a4_gemm_k,
                a4,
                a4_sf,
                &self.weights.shared_down,
                shared_down_out,
                p.n,
                p.h as u32,
                p.shared_inter,
                stream,
            )?;
        } else if let Some(w_fp8) = self.shared_down_pd_fp8 {
            ops::fp8_gemm_m128_mfast(
                ctx.gpu,
                self.fp8_gemm_m128_k,
                p.shared_up_out_base,
                w_fp8,
                shared_down_out,
                p.n,
                p.h as u32,
                p.shared_inter,
                stream,
            )?;
        } else if let Some(ref sdt) = self.shared_down_t {
            if p.n > 128 && self.w4a16_gemm_t_m128_k.0 != 0 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    p.shared_up_out_base,
                    sdt,
                    shared_down_out,
                    p.n,
                    p.h as u32,
                    p.shared_inter,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    p.shared_up_out_base,
                    sdt,
                    shared_down_out,
                    p.n,
                    p.h as u32,
                    p.shared_inter,
                    stream,
                )?;
            }
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                p.shared_up_out_base,
                &self.weights.shared_down,
                shared_down_out,
                p.n,
                p.h as u32,
                p.shared_inter,
                stream,
            )?;
        }

        if is_latent {
            // 5h-p.latent. fc2_latent: routed_out [N, L] → [N, H], then blend with shared
            let fc2_out = ctx.buffers.attn_output();
            if let Some(w_fp8) = self.fc2_pd_fp8 {
                ops::fp8_gemm_m128_mfast(
                    ctx.gpu,
                    self.fp8_gemm_m128_k,
                    routed_out,
                    w_fp8,
                    fc2_out,
                    p.n,
                    p.h as u32,
                    p.latent,
                    stream,
                )?;
            } else {
                let fc2 = self.weights.fc2_latent_proj.as_ref().unwrap();
                self.dense_gemm_prefill(
                    ctx.gpu, routed_out, fc2, fc2_out, p.n, p.h as u32, p.latent, stream,
                )?;
            }
            // output = fc2_out + shared_down_out → p.hidden
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                fc2_out,
                shared_down_out,
                (p.num_tokens * p.h) as u32,
                stream,
            )?;
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                p.hidden,
                fc2_out,
                (p.num_tokens * p.h) as u32,
                stream,
            )?;
        } else {
            // 5h-direct. output = routed_out + shared_down_out → p.hidden
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                routed_out,
                shared_down_out,
                (p.num_tokens * p.h) as u32,
                stream,
            )?;
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                p.hidden,
                routed_out,
                (p.num_tokens * p.h) as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
