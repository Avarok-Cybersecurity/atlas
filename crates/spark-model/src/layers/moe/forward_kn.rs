// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! MoeLayer::forward_km (verify K=m rows, m in 4..=8): the K-row MoE decode
//! arm of #1060. The batch3 shape with the row count as a launch argument:
//! router GEMV once for m rows, batched top-k, fused gate+up, fused silu+down,
//! fused weighted sum and shared-expert blend. Each row is bit-identical to
//! the batch3 row; weight traffic is linear in rows. Rows past 3 previously
//! fell to `forward_prefill`, which below 64 tokens is the per-token expert
//! loop (MTP-3 at 14.9 tok/s on Qwen3.8-Flash-Next).
//!
//! NVFP4 routed experts on the decode layout only. Every other configuration
//! (native EXL3, a resident adapter, BF16 or E8M0 experts, the unified `_t`
//! layout, a mixed BF16 shared expert) reports `can_forward_km == false` and
//! the caller keeps its existing path.

use super::*;

/// Widest row count the arm serves. The router GEMV is `w4a16_gemv_batch8`.
pub(crate) const MOE_KM_MAX_ROWS: u32 = 8;

impl MoeLayer {
    /// Whether `forward_km(m)` can run for this layer: the NVFP4 decode-layout
    /// routed path, the batchn kernels resolved, and `m` inside 4..=8.
    pub fn can_forward_km(&self, m: u32) -> bool {
        (4..=MOE_KM_MAX_ROWS).contains(&m)
            && self.moe_expert_gate_up_shared_batchn.0 != 0
            && self.moe_expert_silu_down_shared_batchn.0 != 0
            && self.moe_weighted_sum_blend_batchn.0 != 0
            && self.gate_nvfp4.is_some()
            && self.w4a16_gemv_batch8_k.0 != 0
            && !self.exl3_native_active()
            && self.lora.is_none()
            && self.bf16_gate_weight_ptrs.is_none()
            && !self.use_t_layout_for_decode()
            && !self.has_mixed_bf16_shared_expert()
            && !super::forward_k3::k3_e8m0_needs_per_token(self.experts_scale_kind)
            && self.fp8_gate_weight_ptrs.is_none()
    }

    /// Fused K=m forward for m rows of normed MoE input at `input` ([m, H]
    /// BF16). Output at `moe_output()` [m, H]. Caller checks `can_forward_km`.
    pub fn forward_km(
        &self,
        input: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.can_forward_km(m),
            "forward_km: arm unavailable for m={m} on this layer (caller must gate on can_forward_km)"
        );
        anyhow::ensure!(
            self.router_logits_n as usize == ctx.config.num_experts,
            "zero-expert MoE routing is not wired on this dispatch variant yet (forward_km)"
        );
        let n = m as usize;
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;

        let router_in = self.router_input(input, m, h, ctx, stream)?;
        // 1. Router GEMV: the gate weight read once for m rows.
        let gate_logits = ctx.buffers.gate_logits();
        let nvfp4 = self.gate_nvfp4.as_ref().expect("checked by can_forward_km");
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            self.w4a16_gemv_batch8_k,
            router_in,
            nvfp4,
            gate_logits,
            m,
            num_experts,
            h,
            stream,
        )?;

        // 2. Batched top-k for m rows.
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(n * top_k as usize * 4);
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
                m,
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
                m,
                stream,
            )?;
        }
        super::union_stats::maybe_sample_expert_union(ctx, indices_dev, n, top_k as usize, stream);

        // 3-5. Fused expert dispatch for m rows.
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_down_out = ctx.buffers.attn_output();
        let output = ctx.buffers.moe_output();
        let is_ep = ctx.comm.is_some() && ctx.config.ep_world_size > 1;

        ops::moe_expert_gate_up_shared_batchn(
            ctx.gpu,
            self.moe_expert_gate_up_shared_batchn,
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
            m,
            stream,
        )?;
        ops::moe_expert_silu_down_shared_batchn(
            ctx.gpu,
            self.moe_expert_silu_down_shared_batchn,
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
            m,
            stream,
        )?;
        // EP: after silu_down, expert_gate_out is free; use it as the zero buffer.
        let shared_for_blend = if is_ep && !shared_down_out.is_null() {
            ctx.gpu
                .memset_async(expert_gate_out, 0, n * h as usize * 2, stream)?;
            expert_gate_out
        } else {
            shared_down_out
        };
        ops::moe_weighted_sum_blend_batchn(
            ctx.gpu,
            self.moe_weighted_sum_blend_batchn,
            output,
            expert_down_out,
            weights_dev,
            shared_for_blend,
            input,
            self.weights.shared_expert_gate.weight,
            h,
            top_k,
            h,
            m,
            stream,
        )?;

        // EP all-reduce: sum partial outputs for m rows.
        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
        {
            if ctx.graph_capture {
                comm.all_reduce(output.0, n * h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, n * h as usize * 2, stream)?;
            }
            if !shared_down_out.is_null() {
                if self.weights.shared_expert_gate.weight.0 == 0 {
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add,
                        output,
                        shared_down_out,
                        m * h,
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
                        m,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }
}
