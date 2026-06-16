// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward (decode).

use super::*;

impl MoeLayer {
    /// Forward pass: gate → top-K routing → batched expert FFN → blend.
    ///
    /// All expert dispatch stays on device — zero D2H synchronization.
    /// 9 kernel launches per MoE layer (down from 58).
    ///
    /// When `gelu_activation` is true, falls back to the sorted prefill path
    /// (which uses separate activation kernel) to avoid fused SiLU decode kernels.
    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        // ── Phase 2.7 Tier C: Frankenstein decode-via-prefill dispatch ──
        // For DFlash capture layers only, when `ATLAS_FRANKENSTEIN_DECODE_VIA_PREFILL=1`
        // is set, route this layer's single-token MoE through `forward_prefill(M=1)`,
        // which uses the tensor-core grouped GEMM kernel (E2M1→E4M3 MMA) instead of
        // the scalar FP32 FMA decode path. Tests whether the numerical recipe of the
        // MoE kernel is the dominant cause of low DFlash drafter acceptance.
        //
        // Other (non-capture) layers fall through to the normal scalar decode path,
        // preserving Atlas's TPS on the bulk of the network. The 5 capture layers
        // pay ~250 µs each (microbench), totalling ≈1.25 ms per token (negligible
        // at Atlas's ~58 ms/token decode latency).
        if self.is_dflash_capture_layer
            && std::env::var("ATLAS_FRANKENSTEIN_DECODE_VIA_PREFILL")
                .ok()
                .as_deref()
                == Some("1")
        {
            // One-time per-process log so we can verify the env-gated route is hit.
            static LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(
                    "FRANKENSTEIN: routing DFlash capture-layer MoE decode through forward_prefill(M=1) (one-time log)"
                );
            }
            self.forward_prefill(input, 1, ctx, stream)?;
            return Ok(ctx.buffers.moe_output());
        }

        // GeGLU models: fused kernels now have GELU activation (model-specific override).
        // No longer need to redirect through sorted prefill path.
        // But we still need pre_expert_norm between routing and dispatch.
        // For the fused decode path, apply pre_expert_norm to the input before experts.
        // The gate GEMV already completed on the raw input in the fused path below.

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let _shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let profile = ctx.profile;

        macro_rules! prof {
            ($label:expr, $body:expr) => {{
                if profile {
                    let t = std::time::Instant::now();
                    let r = $body;
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!("    MoE {}: {:.0}μs", $label, t.elapsed().as_micros());
                    r
                } else {
                    $body
                }
            }};
        }

        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(top_k as usize * 4);

        // Note: moe_gate_topk_fused exists but uses single-CTA design,
        // too slow for 256 experts (serializes computation). Separate path is faster.
        {
            // Gemma-4 router pre-norm (no-op for other models).
            let router_in = self.router_input(input, 1, h, ctx, stream)?;
            let gate_logits = ctx.buffers.gate_logits();
            prof!("gate", {
                if let Some(ref nvfp4) = self.gate_nvfp4 {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv,
                        router_in,
                        nvfp4,
                        gate_logits,
                        num_experts,
                        h,
                        stream,
                    )
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv,
                        router_in,
                        &self.weights.gate,
                        gate_logits,
                        num_experts,
                        h,
                        stream,
                    )
                }
            })?;

            prof!("topk", {
                if let Some(bias) = self.correction_bias_dev {
                    // DeepSeek-V3 / MiniMax-M2 sigmoid + correction bias:
                    //   scores   = sigmoid(gate_logits)
                    //   indices  = topk(scores + bias)
                    //   weights  = scores[indices] / sum(scores[indices])
                    // Kernel does all three steps; norm_topk_prob toggles
                    // the final divide, scaling_factor = 1.0 (MiniMax has
                    // no routed_scaling_factor, unlike Nemotron-H's 2.5).
                    ops::moe_topk_sigmoid(
                        ctx.gpu,
                        self.moe_topk_sigmoid_k,
                        gate_logits,
                        bias,
                        indices_dev,
                        weights_dev,
                        num_experts,
                        top_k,
                        ctx.config.norm_topk_prob,
                        1.0,
                        stream,
                    )
                } else {
                    ops::moe_topk_softmax(
                        ctx.gpu,
                        self.moe_topk,
                        gate_logits,
                        indices_dev,
                        weights_dev,
                        num_experts,
                        top_k,
                        ctx.config.norm_topk_prob,
                        stream,
                    )
                }
            })?;
        }

        super::forward_dump::dump_routing(ctx.gpu, indices_dev, weights_dev, top_k, stream)?;

        // Apply pre-expert norm AFTER routing, BEFORE expert dispatch (Gemma-4 26B).
        // Write to scratch buffer to preserve original `input` (= residual in caller).
        let expert_input = if let Some(ref norm_w) = self.pre_expert_norm {
            let normed = ctx.buffers.ssm_deinterleaved();
            let eps = ctx.config.rms_norm_eps as f32;
            prof!("pre_expert_norm", {
                ops::rms_norm(
                    ctx.gpu,
                    self.pre_expert_norm_k,
                    input,
                    norm_w,
                    normed,
                    1,
                    h,
                    eps,
                    stream,
                )
            })?;
            normed
        } else {
            input
        };

        // ── Batched expert FFN: 3 GEMV + 1 activation + 1 weighted sum ──
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        // ⚠ `logits` aliased as shared-gate scratch — concurrent users
        // MUST offset past `shared_expert_intermediate_size * 2`
        // (decode_b.rs:197 uses .offset(65536)). See bug 2 in memory
        // `project_batch_decode_corruption.md` (2026-05-10).
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let shared_out = ctx.buffers.attn_output();

        if let (Some(gp), Some(up), Some(dp), Some(sg), Some(su), Some(sd)) = (
            self.bf16_gate_weight_ptrs,
            self.bf16_up_weight_ptrs,
            self.bf16_down_weight_ptrs,
            self.bf16_shared_gate,
            self.bf16_shared_up,
            self.bf16_shared_down,
        ) {
            // BF16 path: FP8-dequant-on-load. Eliminates the per-layer 0.989
            // FP8 cosine ceiling by serving experts as BF16 end-to-end.
            prof!("exp_gate_up_bf16", {
                ops::moe_expert_gate_up_shared_bf16(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared_bf16_k,
                    expert_input,
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
                )
            })?;
            prof!("exp_silu_down_bf16", {
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
                )
            })?;
        } else if let (Some(gp), Some(up), Some(dp), Some(sh)) = (
            &self.fp8_gate_weight_ptrs,
            &self.fp8_up_weight_ptrs,
            &self.fp8_down_weight_ptrs,
            &self.fp8_shared_expert,
        ) {
            // FP8 path: fused expert gate+up with FP8 weight/scale pointer tables
            prof!("exp_gate_up_fp8", {
                ops::moe_expert_gate_up_shared_fp8(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared_fp8,
                    expert_input,
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
                )
            })?;

            // FP8 path: fused silu+down
            prof!("exp_silu_down_fp8", {
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
                )
            })?;
        } else if self.use_t_layout_for_decode() {
            prof!("exp_unified_t", {
                self.dispatch_unified_t_decode(
                    ctx,
                    expert_input,
                    expert_gate_out,
                    expert_up_out,
                    expert_down_out,
                    shared_gate_scratch,
                    shared_up_scratch,
                    shared_out,
                    indices_dev,
                    h,
                    inter,
                    top_k,
                    stream,
                )
            })?;
        } else {
            // NVFP4 path: fused routed+shared gate+up
            prof!("exp_gate_up", {
                ops::moe_expert_gate_up_shared(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared,
                    expert_input,
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
                )
            })?;

            super::forward_dump::dump_gate_up(
                ctx.gpu,
                expert_gate_out,
                expert_up_out,
                shared_gate_scratch,
                shared_up_scratch,
                stream,
            )?;

            // NVFP4 path: fused routed+shared silu+down
            prof!("exp_silu_down", {
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
                )
            })?;
        }

        super::forward_dump::dump_down(ctx.gpu, expert_down_out, shared_out, stream)?;

        // Fused wsum+blend+gate: routed expert weighted sum + sigmoid(gate)*shared
        // Gate scalar GEMV is computed inline by each block (redundant but negligible).
        //
        // EP fix: for EP>1, the shared expert is computed identically on all ranks.
        // If we include it in the output before all-reduce, it gets summed world_size
        // times. Solution: pass NULL shared_out for EP, all-reduce the routed sum,
        // then add shared_out once after all-reduce.
        let output = ctx.buffers.moe_output();
        let is_ep = ctx.comm.is_some_and(|c| c.world_size() > 1);
        let shared_for_blend = if is_ep && !shared_out.is_null() {
            // EP: exclude shared expert from blend (will add after all-reduce).
            // Zero a temp buffer to pass as shared_out (kernel reads it even with NULL gate).
            let zero_buf = ctx.buffers.expert_gate_out(); // temp buffer, will be zeroed
            ctx.gpu.memset_async(zero_buf, 0, h as usize * 2, stream)?;
            zero_buf
        } else {
            shared_out
        };
        prof!("wsum_blend", {
            ops::moe_weighted_sum_blend(
                ctx.gpu,
                self.moe_weighted_sum_blend,
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
            )
        })?;

        // EP all-reduce: sum partial expert outputs across ranks.
        // Each rank only computed its local experts (remote → zero), so
        // SUM gives the correct global result.
        if let Some(comm) = ctx.comm
            && comm.world_size() > 1
        {
            if ctx.graph_capture {
                comm.all_reduce(output.0, h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, h as usize * 2, stream)?;
            }
            // Now add shared expert contribution ONCE (after all-reduce).
            // Must apply the sigmoid gate: output += sigmoid(dot(input, gate_w)) * shared_out.
            // Using moe_batched_blend with num_tokens=1 computes the gate and blends correctly.
            // BUG #41 fix: previous code used residual_add (ignoring the gate), producing
            // wrong output that compounded across 48 layers into gibberish.
            if !shared_out.is_null() {
                if self.weights.shared_expert_gate.weight.0 == 0 {
                    // No gate weight (e.g., Mistral): shared expert always at full strength.
                    ops::residual_add(ctx.gpu, self.residual_add, output, shared_out, h, stream)?;
                } else {
                    // Gated shared expert (e.g., Qwen3.5): apply sigmoid gate.
                    ops::moe_batched_blend(
                        ctx.gpu,
                        self.moe_batched_blend,
                        output,
                        shared_out,
                        input,
                        self.weights.shared_expert_gate.weight,
                        h,
                        1,
                        stream,
                    )?;
                }
            }
        }

        super::forward_dump::dump_output(ctx.gpu, output, stream)?;

        Ok(output)
    }
}
