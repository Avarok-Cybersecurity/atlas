// SPDX-License-Identifier: AGPL-3.0-only

//! Generic per-expert MoE forward path for FP8/BF16 expert weights,
//! plus the deferred draft-token D2H readback used by the proposer trait impl,
//! plus a dense-FFN shortcut for non-MoE MTP heads (Qwen3.6-27B-FP8).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::MtpHead;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl MtpHead {
    /// Read the draft token ID that was stored on GPU by the last
    /// `forward_one` call with `draft_embed_target = Some(...)`.
    ///
    /// Performs a 4-byte D2H copy + stream sync. Call this lazily
    /// (after the verify step completes) to avoid blocking the GPU pipeline.
    pub(super) fn read_deferred_draft_token(&self, gpu: &dyn GpuBackend) -> Result<u32> {
        let mut buf = [0u8; 4];
        gpu.copy_d2h(self.draft_token_id_dev, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    /// Dense FFN forward for Qwen3.6-27B-FP8 (and other dense MTP heads):
    /// `out = down_proj( silu(gate_proj(x)) * up_proj(x) )`.
    ///
    /// One MLP, no router, no expert dispatch. Reuses the same FP8/BF16
    /// kernels (`dense_gemv_*`, `moe_silu_mul`) the per-expert path uses,
    /// so no new kernel wiring is required.
    pub(super) fn dense_ffn_forward_generic(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        // Dense MTP heads use the main model's dense `intermediate_size`
        // (Qwen3.6-27B: 17408). `moe_intermediate_size` is 0 for dense
        // checkpoints and would request a zero-element GEMV.
        let inter = if ctx.config.intermediate_size > 0 {
            ctx.config.intermediate_size as u32
        } else {
            ctx.config.moe_intermediate_size as u32
        };
        let (gate_w, up_w, down_w) = self
            .dense_ffn_generic
            .as_ref()
            .expect("dense_ffn_forward_generic called without dense_ffn_generic populated");

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        self.gemv(ctx.gpu, input, gate_w, gate_out, inter, h, stream)?;
        self.gemv(ctx.gpu, input, up_w, up_out, inter, h, stream)?;

        ops::moe_silu_mul(
            ctx.gpu,
            self.moe_silu_mul_k.unwrap(),
            gate_out,
            up_out,
            gate_out,
            inter,
            stream,
        )?;

        let output = ctx.buffers.moe_output();
        self.gemv(ctx.gpu, gate_out, down_w, output, h, inter, stream)?;
        Ok(output)
    }

    /// Generic MoE forward for FP8/BF16 expert weights.
    ///
    /// Loop-based per-expert dispatch using `dense_gemv` / `dense_gemv_fp8w`.
    /// Requires D2H copy of expert indices (~40 bytes, ~5µs).
    pub(super) fn moe_forward_generic(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;

        // 1. Gate GEMV: [1,h] → [1,num_experts] (always BF16 weights)
        let gate_logits = ctx.buffers.gate_logits();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k.unwrap(),
            input,
            &self.moe_gate,
            gate_logits,
            num_experts,
            h,
            stream,
        )?;

        // 2. TopK routing
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(top_k as usize * 4);
        ops::moe_topk_softmax(
            ctx.gpu,
            self.moe_topk_k.unwrap(),
            gate_logits,
            indices_dev,
            weights_dev,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            stream,
        )?;

        // 3. D2H expert indices
        ctx.gpu.synchronize(stream)?;
        let mut idx_buf = vec![0u8; top_k as usize * 4];
        ctx.gpu.copy_d2h(indices_dev, &mut idx_buf)?;
        let expert_ids: Vec<u32> = (0..top_k as usize)
            .map(|i| {
                u32::from_le_bytes([
                    idx_buf[i * 4],
                    idx_buf[i * 4 + 1],
                    idx_buf[i * 4 + 2],
                    idx_buf[i * 4 + 3],
                ])
            })
            .collect();

        let experts = self.moe_experts_generic.as_ref().unwrap();
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();

        // 4. Per-expert gate + up GEMVs
        for (slot, &eid) in expert_ids.iter().enumerate() {
            let (ref gate_w, ref up_w, _) = experts[eid as usize];
            let g_out = expert_gate_out.offset(slot * inter as usize * 2);
            let u_out = expert_up_out.offset(slot * inter as usize * 2);
            self.gemv(ctx.gpu, input, gate_w, g_out, inter, h, stream)?;
            self.gemv(ctx.gpu, input, up_w, u_out, inter, h, stream)?;
        }

        // 5. SiLU: gate_out = silu(gate_out) * up_out per expert slot
        for slot in 0..top_k as usize {
            let g = expert_gate_out.offset(slot * inter as usize * 2);
            let u = expert_up_out.offset(slot * inter as usize * 2);
            ops::moe_silu_mul(
                ctx.gpu,
                self.moe_silu_mul_k.unwrap(),
                g,
                u,
                g,
                inter,
                stream,
            )?;
        }

        // 6. Per-expert down GEMVs
        for (slot, &eid) in expert_ids.iter().enumerate() {
            let (_, _, ref down_w) = experts[eid as usize];
            let silu_out = expert_gate_out.offset(slot * inter as usize * 2);
            let d_out = expert_down_out.offset(slot * h as usize * 2);
            self.gemv(ctx.gpu, silu_out, down_w, d_out, h, inter, stream)?;
        }

        // 7. Shared expert
        let (sh_gate, sh_up, sh_down) = self.moe_shared_generic.as_ref().unwrap();
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        self.gemv(
            ctx.gpu,
            input,
            sh_gate,
            shared_gate_scratch,
            inter,
            h,
            stream,
        )?;
        self.gemv(ctx.gpu, input, sh_up, shared_up_scratch, inter, h, stream)?;
        ops::moe_silu_mul(
            ctx.gpu,
            self.moe_silu_mul_k.unwrap(),
            shared_gate_scratch,
            shared_up_scratch,
            shared_gate_scratch,
            inter,
            stream,
        )?;
        let shared_out = ctx.buffers.attn_output();
        self.gemv(
            ctx.gpu,
            shared_gate_scratch,
            sh_down,
            shared_out,
            h,
            inter,
            stream,
        )?;

        // 8. Weighted sum + blend
        let output = ctx.buffers.moe_output();
        ops::moe_weighted_sum_blend(
            ctx.gpu,
            self.moe_weighted_sum_blend_k.unwrap(),
            output,
            expert_down_out,
            weights_dev,
            shared_out,
            input,
            self.shared_expert_gate.weight,
            h,
            top_k,
            h,
            stream,
        )?;

        Ok(output)
    }
}
