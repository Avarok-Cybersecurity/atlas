// SPDX-License-Identifier: AGPL-3.0-only

//! Per-token decode helpers for [`super::NemotronMoeLayer`].
//!
//! Two MoE shapes:
//!   - `decode_direct_moe` (Nano 30B): experts operate in hidden space [H].
//!   - `decode_latent_moe`  (Super 120B): experts operate in latent space [L]
//!     with fc1/fc2 projections bridging hidden ↔ latent.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::NemotronMoeLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl NemotronMoeLayer {
    /// Single-token decode: norm + gate + routing, dispatch to direct or latent MoE.
    pub(super) fn decode_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let scale = ctx.config.routed_scaling_factor as f32;

        // 1. RMS norm (standard weight*x, saves residual)
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            1,
            h,
            eps,
            stream,
        )?;

        // 2. Gate GEMV: [1, H] x [H, num_experts]^T -> [num_experts] BF16
        let gate_logits = ctx.buffers.gate_logits();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &self.weights.gate,
            gate_logits,
            num_experts,
            h,
            stream,
        )?;

        // 3. Sigmoid routing
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(top_k as usize * 4);
        ops::moe_topk_sigmoid(
            ctx.gpu,
            self.topk_sigmoid_k,
            gate_logits,
            self.weights.e_score_correction_bias.weight,
            indices_dev,
            weights_dev,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            scale,
            stream,
        )?;

        if self.moe_latent_size > 0 {
            self.decode_latent_moe(
                hidden,
                normed,
                indices_dev,
                weights_dev,
                ctx,
                stream,
                h,
                inter,
                shared_inter,
                top_k,
            )
        } else {
            self.decode_direct_moe(
                hidden,
                normed,
                indices_dev,
                weights_dev,
                ctx,
                stream,
                h,
                inter,
                shared_inter,
                top_k,
            )
        }
    }

    /// Nano 30B: direct MoE — routed experts operate on hidden_size.
    pub(super) fn decode_direct_moe(
        &self,
        hidden: DevicePtr,
        normed: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        h: u32,
        inter: u32,
        shared_inter: u32,
        top_k: u32,
    ) -> Result<()> {
        // Batched routed UP GEMV
        let expert_up_out = ctx.buffers.expert_up_out();
        ops::moe_expert_gemv(
            ctx.gpu,
            self.moe_expert_gemv_k,
            normed,
            self.up_ptrs.packed_ptrs,
            self.up_ptrs.scale_ptrs,
            self.up_ptrs.scale2_vals,
            expert_up_out,
            indices_dev,
            inter,
            h,
            top_k,
            0,
            stream,
        )?;

        // Shared expert UP
        let shared_up_out = ctx.buffers.ssm_qkvz();
        ops::w4a16_gemv(
            ctx.gpu,
            self.w4a16_gemv_k,
            normed,
            &self.weights.shared_up,
            shared_up_out,
            shared_inter,
            h,
            stream,
        )?;

        // Fused relu²+down for all experts (routed + shared in one launch)
        let expert_down_out = ctx.buffers.expert_down_out();
        // Use ssm_deinterleaved (NOT attn_output) — attn_output is used by
        // Mamba-2 SSM y_ptr. MoE writing here during M-E prefill corrupts SSM output.
        let shared_down_out = ctx.buffers.ssm_deinterleaved();
        let smem = (shared_inter.max(inter) as usize) * 4;

        KernelLaunch::new(ctx.gpu, self.relu2_down_shared_k)
            .grid([div_ceil(h, 8), top_k + 1, 1])
            .block([128, 1, 1])
            .shared_mem(smem as u32)
            .arg_ptr(expert_up_out)
            .arg_ptr(self.down_ptrs.packed_ptrs)
            .arg_ptr(self.down_ptrs.scale_ptrs)
            .arg_ptr(self.down_ptrs.scale2_vals)
            .arg_ptr(expert_down_out)
            .arg_ptr(indices_dev)
            .arg_ptr(shared_up_out)
            .arg_ptr(self.weights.shared_down.weight)
            .arg_ptr(self.weights.shared_down.weight_scale)
            .arg_f32(self.weights.shared_down.weight_scale_2)
            .arg_ptr(shared_down_out)
            .arg_u32(h)
            .arg_u32(inter)
            .arg_u32(shared_inter)
            .arg_u32(h)
            .arg_u32(top_k)
            .launch(stream)?;

        // Weighted sum + residual add
        let output = ctx.buffers.moe_output();
        KernelLaunch::new(ctx.gpu, self.weighted_sum_scale_k)
            .grid([div_ceil(h, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(output)
            .arg_ptr(expert_down_out)
            .arg_ptr(weights_dev)
            .arg_ptr(shared_down_out)
            .arg_u32(h)
            .arg_u32(top_k)
            .arg_f32(1.0f32)
            .launch(stream)?;

        ops::residual_add(ctx.gpu, self.residual_add_k, hidden, output, h, stream)
    }

    /// Super 120B: LatentMoE — routed experts operate in latent space [moe_latent_size].
    ///
    /// fc1_latent(normed) → latent [L]
    /// routed up(latent) → [inter], relu²+down → [L]
    /// weighted_sum → combined [L]
    /// fc2_latent(combined) → routed_out [H]
    /// shared up(normed) → [shared_inter], relu²+down → shared_out [H]
    /// output = routed_out + shared_out
    pub(super) fn decode_latent_moe(
        &self,
        hidden: DevicePtr,
        normed: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        h: u32,
        inter: u32,
        shared_inter: u32,
        top_k: u32,
    ) -> Result<()> {
        let latent = self.moe_latent_size as u32;
        let fc1 = self.weights.fc1_latent_proj.as_ref().unwrap();
        let fc2 = self.weights.fc2_latent_proj.as_ref().unwrap();

        // 1. fc1_latent: normed [H] → latent [L]
        let latent_buf = ctx.buffers.ssm_ba();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            fc1,
            latent_buf,
            latent,
            h,
            stream,
        )?;

        // 2. Routed expert UP: latent [L] → [top_k, inter]
        let expert_up_out = ctx.buffers.expert_up_out();
        ops::moe_expert_gemv(
            ctx.gpu,
            self.moe_expert_gemv_k,
            latent_buf,
            self.up_ptrs.packed_ptrs,
            self.up_ptrs.scale_ptrs,
            self.up_ptrs.scale2_vals,
            expert_up_out,
            indices_dev,
            inter,
            latent,
            top_k,
            0,
            stream,
        )?;

        // 3. Shared expert UP: normed [H] → [shared_inter]
        let shared_up_out = ctx.buffers.ssm_qkvz();
        ops::w4a16_gemv(
            ctx.gpu,
            self.w4a16_gemv_k,
            normed,
            &self.weights.shared_up,
            shared_up_out,
            shared_inter,
            h,
            stream,
        )?;

        // 4. Fused relu²+down: routed → [top_k, L], shared → [H]
        //    The kernel supports N != N_shared natively.
        let expert_down_out = ctx.buffers.expert_down_out();
        // Use ssm_deinterleaved (NOT attn_output) — attn_output is used by
        // Mamba-2 SSM y_ptr. MoE writing here during M-E prefill corrupts SSM output.
        let shared_down_out = ctx.buffers.ssm_deinterleaved();
        let max_n = h.max(latent);
        let smem = (shared_inter.max(inter) as usize) * 4;

        KernelLaunch::new(ctx.gpu, self.relu2_down_shared_k)
            .grid([div_ceil(max_n, 8), top_k + 1, 1])
            .block([128, 1, 1])
            .shared_mem(smem as u32)
            .arg_ptr(expert_up_out)
            .arg_ptr(self.down_ptrs.packed_ptrs)
            .arg_ptr(self.down_ptrs.scale_ptrs)
            .arg_ptr(self.down_ptrs.scale2_vals)
            .arg_ptr(expert_down_out)
            .arg_ptr(indices_dev)
            .arg_ptr(shared_up_out)
            .arg_ptr(self.weights.shared_down.weight)
            .arg_ptr(self.weights.shared_down.weight_scale)
            .arg_f32(self.weights.shared_down.weight_scale_2)
            .arg_ptr(shared_down_out)
            .arg_u32(latent) // N: routed output dim = latent
            .arg_u32(inter) // K_routed
            .arg_u32(shared_inter) // K_shared
            .arg_u32(h) // N_shared: shared output dim = hidden_size
            .arg_u32(top_k)
            .launch(stream)?;

        // 5. Weighted sum in latent space: [top_k, L] → combined [L]
        //    Use latent_buf (ssm_ba, consumed by step 2) as output.
        //    Zero expert_gate_out[..L] as dummy shared input for the kernel
        //    (weighted_sum_scale unconditionally reads shared_down[idx]).
        let combined_latent = latent_buf; // reuse: step 2 output already consumed
        let dummy_shared = ctx.buffers.expert_gate_out();
        ctx.gpu
            .memset_async(dummy_shared, 0, latent as usize * 2, stream)?;
        KernelLaunch::new(ctx.gpu, self.weighted_sum_scale_k)
            .grid([div_ceil(latent, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(combined_latent)
            .arg_ptr(expert_down_out)
            .arg_ptr(weights_dev)
            .arg_ptr(dummy_shared) // zeroed: kernel reads but adds 0
            .arg_u32(latent)
            .arg_u32(top_k)
            .arg_f32(1.0f32) // weights already scaled by sigmoid kernel
            .launch(stream)?;

        // 6. fc2_latent: combined [L] → routed_out [H]
        let routed_out = ctx.buffers.moe_output();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            combined_latent,
            fc2,
            routed_out,
            h,
            latent,
            stream,
        )?;

        // 7. output = routed_out + shared_down_out, then residual add
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            routed_out,
            shared_down_out,
            h,
            stream,
        )?;
        ops::residual_add(ctx.gpu, self.residual_add_k, hidden, routed_out, h, stream)?;

        Ok(())
    }
}
