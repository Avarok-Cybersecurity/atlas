// SPDX-License-Identifier: AGPL-3.0-only

//! Nemotron MoE prefill — semi-batched per-token fallback path.
//!
//! Extracted from `NemotronMoeLayer::prefill` for file-size budget.
//! Used when sorted-MoE kernels are unavailable: routes all tokens in one
//! launch when possible, then dispatches expert GEMVs per-token.

use anyhow::Result;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::NemotronMoeLayer;
use super::prefill_sorted::SortedPrefillCtx;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl NemotronMoeLayer {
    pub(super) fn prefill_fallback_path(
        &self,
        p: &SortedPrefillCtx,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let scratch_buf = ctx.buffers.scratch();
        // ── Semi-batched: batch routing, per-token expert dispatch ──
        // Batch the routing for all N tokens (1 launch instead of N), then
        // run per-token expert GEMVs reading from the pre-computed arrays.
        let use_batched_routing = self.topk_sigmoid_batched_k.0 != 0 && p.num_tokens > 1;
        if use_batched_routing {
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
        }

        for t in 0..p.num_tokens {
            let token_offset_h = t * p.h * 2usize;
            let tok_k = p.top_k as usize;

            // Use pre-computed batched routing results, or compute per-token
            let tok_indices;
            let tok_weights;
            if use_batched_routing {
                tok_indices = p.indices_dev.offset(t * tok_k * 4);
                tok_weights = p.weights_dev.offset(t * tok_k * 4);
            } else {
                tok_indices = scratch_buf;
                tok_weights = scratch_buf.offset(tok_k * 4);
                let token_gate = p.gate_logits.offset(t * p.num_experts as usize * 2usize);
                ops::moe_topk_sigmoid(
                    ctx.gpu,
                    self.topk_sigmoid_k,
                    token_gate,
                    self.weights.e_score_correction_bias.weight,
                    tok_indices,
                    tok_weights,
                    p.num_experts,
                    p.top_k,
                    ctx.config.norm_topk_prob,
                    p.scale,
                    stream,
                )?;
            }

            if self.moe_latent_size > 0 {
                let token_latent = p
                    .latent_base
                    .unwrap()
                    .offset(t * p.latent as usize * 2usize);
                let expert_up_out = ctx.buffers.expert_up_out();
                ops::moe_expert_gemv(
                    ctx.gpu,
                    self.moe_expert_gemv_k,
                    token_latent,
                    self.up_ptrs.packed_ptrs,
                    self.up_ptrs.scale_ptrs,
                    self.up_ptrs.scale2_vals,
                    expert_up_out,
                    tok_indices,
                    p.inter,
                    p.latent,
                    p.top_k,
                    0,
                    stream,
                )?;
                let token_shared_up = p
                    .shared_up_out_base
                    .offset(t * p.shared_inter as usize * 2usize);
                let expert_down_out = ctx.buffers.expert_down_out();
                let shared_down_out = ctx.buffers.ssm_deinterleaved();
                let max_n = (p.h as u32).max(p.latent);
                let smem = (p.shared_inter.max(p.inter) as usize) * 4;
                KernelLaunch::new(ctx.gpu, self.relu2_down_shared_k)
                    .grid([div_ceil(max_n, 8), p.top_k + 1, 1])
                    .block([128, 1, 1])
                    .shared_mem(smem as u32)
                    .arg_ptr(expert_up_out)
                    .arg_ptr(self.down_ptrs.packed_ptrs)
                    .arg_ptr(self.down_ptrs.scale_ptrs)
                    .arg_ptr(self.down_ptrs.scale2_vals)
                    .arg_ptr(expert_down_out)
                    .arg_ptr(tok_indices)
                    .arg_ptr(token_shared_up)
                    .arg_ptr(self.weights.shared_down.weight)
                    .arg_ptr(self.weights.shared_down.weight_scale)
                    .arg_f32(self.weights.shared_down.weight_scale_2)
                    .arg_ptr(shared_down_out)
                    .arg_u32(p.latent)
                    .arg_u32(p.inter)
                    .arg_u32(p.shared_inter)
                    .arg_u32(p.h as u32)
                    .arg_u32(p.top_k)
                    .launch(stream)?;
                let combined_latent = ctx.buffers.attn_output();
                let dummy_shared = ctx.buffers.expert_gate_out();
                ctx.gpu
                    .memset_async(dummy_shared, 0, p.latent as usize * 2usize, stream)?;
                KernelLaunch::new(ctx.gpu, self.weighted_sum_scale_k)
                    .grid([div_ceil(p.latent, 256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(combined_latent)
                    .arg_ptr(expert_down_out)
                    .arg_ptr(tok_weights)
                    .arg_ptr(dummy_shared)
                    .arg_u32(p.latent)
                    .arg_u32(p.top_k)
                    .arg_f32(1.0f32)
                    .launch(stream)?;
                let fc2 = self.weights.fc2_latent_proj.as_ref().unwrap();
                let routed_out = ctx.buffers.moe_output();
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    combined_latent,
                    fc2,
                    routed_out,
                    p.h as u32,
                    p.latent,
                    stream,
                )?;
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    routed_out,
                    shared_down_out,
                    p.h as u32,
                    stream,
                )?;
                let token_hidden = p.hidden.offset(token_offset_h);
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    token_hidden,
                    routed_out,
                    p.h as u32,
                    stream,
                )?;
            } else {
                let token_normed = p.normed.offset(token_offset_h);
                let expert_up_out = ctx.buffers.expert_up_out();
                ops::moe_expert_gemv(
                    ctx.gpu,
                    self.moe_expert_gemv_k,
                    token_normed,
                    self.up_ptrs.packed_ptrs,
                    self.up_ptrs.scale_ptrs,
                    self.up_ptrs.scale2_vals,
                    expert_up_out,
                    tok_indices,
                    p.inter,
                    p.h as u32,
                    p.top_k,
                    0,
                    stream,
                )?;
                let token_shared_up = p
                    .shared_up_out_base
                    .offset(t * p.shared_inter as usize * 2usize);
                let expert_down_out = ctx.buffers.expert_down_out();
                let shared_down_out = ctx.buffers.ssm_deinterleaved();
                let smem = (p.shared_inter.max(p.inter) as usize) * 4;
                KernelLaunch::new(ctx.gpu, self.relu2_down_shared_k)
                    .grid([div_ceil(p.h as u32, 8), p.top_k + 1, 1])
                    .block([128, 1, 1])
                    .shared_mem(smem as u32)
                    .arg_ptr(expert_up_out)
                    .arg_ptr(self.down_ptrs.packed_ptrs)
                    .arg_ptr(self.down_ptrs.scale_ptrs)
                    .arg_ptr(self.down_ptrs.scale2_vals)
                    .arg_ptr(expert_down_out)
                    .arg_ptr(tok_indices)
                    .arg_ptr(token_shared_up)
                    .arg_ptr(self.weights.shared_down.weight)
                    .arg_ptr(self.weights.shared_down.weight_scale)
                    .arg_f32(self.weights.shared_down.weight_scale_2)
                    .arg_ptr(shared_down_out)
                    .arg_u32(p.h as u32)
                    .arg_u32(p.inter)
                    .arg_u32(p.shared_inter)
                    .arg_u32(p.h as u32)
                    .arg_u32(p.top_k)
                    .launch(stream)?;
                let output = ctx.buffers.moe_output();
                KernelLaunch::new(ctx.gpu, self.weighted_sum_scale_k)
                    .grid([div_ceil(p.h as u32, 256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(output)
                    .arg_ptr(expert_down_out)
                    .arg_ptr(tok_weights)
                    .arg_ptr(shared_down_out)
                    .arg_u32(p.h as u32)
                    .arg_u32(p.top_k)
                    .arg_f32(1.0f32)
                    .launch(stream)?;
                let token_hidden = p.hidden.offset(token_offset_h);
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    token_hidden,
                    output,
                    p.h as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
