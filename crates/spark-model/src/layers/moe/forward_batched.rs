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

        // Gemma-4 router pre-norm (no-op for other models).
        let router_in = self.router_input(input, n, h, ctx, stream)?;
        // Gate GEMM: [N, H] × [H, num_experts] → [N, num_experts]
        let gate_logits = ctx.buffers.gate_logits(); // [N, 512] BF16
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
            let gate_t = gate_logits.offset(t * num_experts as usize * bf16);
            let output_t = ctx.buffers.moe_output().offset(t * h_usize * bf16);

            let scratch = ctx.buffers.scratch();
            let indices_dev = scratch;
            let weights_dev = scratch.offset(top_k as usize * 4);

            if let Some(bias) = self.correction_bias_dev {
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
            } else {
                ops::moe_topk_softmax(
                    ctx.gpu,
                    self.moe_topk,
                    gate_t,
                    indices_dev,
                    weights_dev,
                    num_experts,
                    top_k,
                    ctx.config.norm_topk_prob,
                    stream,
                )?;
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
                ops::moe_expert_gate_up_shared_t(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared_t_k,
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
                    self.moe_expert_silu_down_shared_t_k,
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
                && comm.world_size() > 1
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
