// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward_prefill.

use super::*;

impl MoeLayer {
    /// N-token prefill via grouped GEMM: sort-by-expert → tensor-core GEMM per expert.
    ///
    /// Each expert's weight matrix is loaded once (not per-token), cutting LPDDR5X
    /// reads from ~6 GB (GEMV) to ~150 MB (grouped GEMM) at N=1024.
    ///
    /// Pipeline: gate → topK → sort → grouped gate/up GEMM → SiLU → grouped down GEMM
    ///           → unpermute + weighted reduce → shared expert blend.
    /// Shared expert uses standard w4a16_gemm (single-expert, M=N_tokens).
    #[allow(unused_assignments)]
    pub fn forward_prefill(
        &self,
        input: DevicePtr, // [num_tokens, H] BF16 — normed MoE input
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // BF16 experts (FP8-dequant-on-load path): same dispatch shape as
        // FP8 — grouped GEMM for long prefills, fused per-token for short.
        if self.bf16_gate_weight_ptrs.is_some() {
            if self.moe_bf16_grouped_gemm_k.0 != 0 && num_tokens > 64 {
                return self.forward_prefill_bf16(input, num_tokens, ctx, stream);
            }
            return self.forward_batched(input, num_tokens, ctx, stream);
        }

        // FP8 experts: use grouped GEMM for long prefills (>64 tokens),
        // fall back to per-token fused GEMV for short prefills where
        // the GEMM launch overhead exceeds the bandwidth savings.
        if self.fp8_gate_weight_ptrs.is_some() {
            if self.moe_fp8_grouped_gemm_k.0 != 0 && num_tokens > 64 {
                return self.forward_prefill_fp8(input, num_tokens, ctx, stream);
            }
            return self.forward_batched(input, num_tokens, ctx, stream);
        }

        // Lazy down_proj transpose: synchronous on the compute stream.
        // (See `kick_off_lazy_transpose` for an attempted overlap path
        // that regressed by 30 % on GB10 — SM contention dominated the
        // overlap savings, so the synchronous path is the shipped one.)
        let _t_xpose = if ctx.profile && self.down_t_scratch_packed.is_some() {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        self.transpose_down_into_scratch(ctx, stream)?;
        if let Some(t0) = _t_xpose {
            ctx.gpu.synchronize(stream)?;
            tracing::info!(
                "  MoE prefill [lazy_transpose_down] N={}: {}µs",
                num_tokens,
                t0.elapsed().as_micros(),
            );
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = num_tokens as u32;
        let total_expanded = n * top_k;

        // Profile helper macro
        #[allow(unused_macros)]
        macro_rules! prof {
            ($label:expr) => {
                if ctx.profile {
                    ctx.gpu.synchronize(stream)?;
                    let _t = std::time::Instant::now();
                    tracing::info!("  MoE prefill [{}] N={}", $label, num_tokens);
                }
            };
        }
        #[allow(unused_assignments)]
        let mut t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        macro_rules! prof_step {
            ($label:expr) => {
                if let Some(t) = t0.take() {
                    ctx.gpu.synchronize(stream)?;
                    let elapsed = t.elapsed().as_micros();
                    tracing::info!("  MoE prefill [{}] N={}: {}µs", $label, num_tokens, elapsed);
                    t0 = Some(std::time::Instant::now());
                }
            };
        }

        // ── Shared expert on secondary stream (overlaps with routed path) ──
        // Shared expert only reads `input` and writes to separate buffers
        // (ssm_deinterleaved, ssm_qkvz, attn_output) — no data conflict
        // with the routed expert path.  In profile mode, run sequentially
        // on the default stream for accurate per-step timing.
        //
        // Skip entirely when shared_inter == 0 (models without a shared expert,
        // e.g. Qwen3-VL-30B which has no shared_expert_intermediate_size).
        // Launching kernels with N=0 produces CUDA_ERROR_INVALID_VALUE (grid.x=0).
        let has_shared = shared_inter > 0;
        let use_overlap = false; // disabled: dual-stream contention worsens LPDDR5X bandwidth
        let aux = if use_overlap {
            self.prefill_stream
        } else {
            stream
        };

        if has_shared {
            self.run_shared_expert_prefill(
                input,
                n,
                h,
                shared_inter,
                aux,
                stream,
                use_overlap,
                ctx,
            )?;
        }
        prof_step!("shared_expert");

        // ── Routed expert path on default stream ──

        // Gemma-4 router pre-norm (no-op for other models).
        let router_in = self.router_input(input, n, h, ctx, stream)?;
        super::dump::dump_gate_input(ctx.gpu, stream, router_in, n, h)?;
        // 1. Gate GEMM: [N, H] × [H, num_experts] → [N, num_experts]
        let gate_logits = ctx.buffers.gate_logits();
        if let Some(fp8) = self.gate_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                router_in,
                fp8,
                gate_logits,
                n,
                num_experts,
                h,
                stream,
            )?;
        } else if let Some(ref nvfp4) = self.gate_nvfp4 {
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
        super::dump::dump_gate_logits(ctx.gpu, stream, gate_logits, n, num_experts)?;
        prof_step!("gate_gemm");

        // 2. Batched topK dispatch. DeepSeek-V3 / MiniMax-M2 use sigmoid
        //    + correction bias (detected via `correction_bias_dev`);
        //    every other model takes the softmax path (no behavior
        //    change — this is additive).
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(total_expanded as usize * 4);
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
                n,
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
                n,
                stream,
            )?;
        }
        super::dump::dump_expert_ids(ctx.gpu, stream, indices_dev, weights_dev, n, top_k)?;
        prof_step!("topk");

        // 3. Sort tokens by expert → L2-optimized ordering.
        let te = total_expanded as usize;
        let ne = num_experts as usize;
        let sorted_token_ids = gate_logits;
        let sorted_expert_ids = gate_logits.offset(te * 4);
        let expert_offsets = gate_logits.offset(te * 4 * 2);
        let token_to_perm = gate_logits.offset(te * 4 * 2 + (ne + 1) * 4);
        ops::moe_sort_by_expert(
            ctx.gpu,
            self.moe_sort_by_expert,
            indices_dev,
            sorted_token_ids,
            sorted_expert_ids,
            expert_offsets,
            token_to_perm,
            total_expanded,
            num_experts,
            top_k,
            stream,
        )?;
        prof_step!("sort");

        // 3.5. Pre-expert norm: norm the input for expert dispatch (Gemma-4 26B).
        // Router already used the raw input for routing; now norm for experts.
        // IMPORTANT: write to scratch (ssm_deinterleaved), NOT in-place — `input` is
        // the residual and must be preserved for the subsequent residual add.
        let expert_input = if let Some(ref norm_w) = self.pre_expert_norm {
            let normed_buf = ctx.buffers.ssm_deinterleaved();
            let n_tokens = num_tokens as u32;
            let eps = ctx.config.rms_norm_eps as f32;
            ops::rms_norm(
                ctx.gpu,
                self.pre_expert_norm_k,
                input,
                norm_w,
                normed_buf,
                n_tokens,
                h,
                eps,
                stream,
            )?;
            normed_buf
        } else {
            input
        };
        prof_step!("pre_expert_norm");

        // 4-6. Routed grouped-GEMM phase (grid sizing → grouped gate+up
        // GEMM → SiLU → grouped down GEMM). Hoisted to forward_prefill_routed.rs
        // to keep this file under the 500 LoC cap; behavior identical.
        self.run_routed_grouped_gemm(
            expert_input,
            expert_offsets,
            sorted_token_ids,
            n,
            h,
            inter,
            num_experts,
            top_k,
            num_tokens,
            ne,
            &mut t0,
            ctx,
            stream,
        )?;
        let expert_down_out = ctx.buffers.expert_down_out();

        // 7. Unpermute + weighted reduce: scatter sorted outputs to token order
        let output = ctx.buffers.moe_output();
        ops::moe_unpermute_reduce_indexed(
            ctx.gpu,
            self.moe_unpermute_reduce,
            expert_down_out,
            output,
            token_to_perm,
            weights_dev,
            h,
            n,
            top_k,
            stream,
        )?;

        // 8. Blend shared expert: output += sigmoid(dot(input, gate)) * shared
        // Skip when has_shared == false (no shared expert in this model config).
        // EP fix: defer shared expert blend until AFTER all-reduce to avoid doubling.
        let is_ep_prefill = ctx.comm.is_some_and(|c| c.world_size() > 1);
        if has_shared && !is_ep_prefill {
            let shared_down_out = ctx.buffers.attn_output();
            if use_overlap {
                ctx.gpu.stream_wait_event(stream, self.event_b)?;
            }
            super::dump::dump_routed_only(ctx.gpu, stream, output, n, h)?;
            super::dump::dump_shared_out(ctx.gpu, stream, shared_down_out, n, h)?;
            super::dump::dump_shared_gate(
                ctx.gpu,
                stream,
                input,
                self.weights.shared_expert_gate.weight,
                n,
                h,
            )?;
            ops::moe_batched_blend(
                ctx.gpu,
                self.moe_batched_blend,
                output,
                shared_down_out,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                n,
                stream,
            )?;
        }
        super::dump::dump_moe_out(ctx.gpu, stream, output, n, h)?;
        prof_step!("unpermute_blend");

        // EP all-reduce
        if let Some(comm) = ctx.comm
            && comm.world_size() > 1
        {
            let _t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            if ctx.graph_capture {
                comm.all_reduce(output.0, num_tokens * h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, num_tokens * h as usize * 2, stream)?;
            }
            if let Some(t0) = _t0 {
                ctx.gpu.synchronize(stream)?;
                tracing::info!(
                    "  EP allreduce (moe out) N={}: {}µs",
                    num_tokens,
                    t0.elapsed().as_micros(),
                );
            }
            // Add shared expert ONCE after all-reduce (prevents EP doubling)
            if has_shared {
                let shared_down_out = ctx.buffers.attn_output();
                if use_overlap {
                    ctx.gpu.stream_wait_event(stream, self.event_b)?;
                }
                ops::moe_batched_blend(
                    ctx.gpu,
                    self.moe_batched_blend,
                    output,
                    shared_down_out,
                    input,
                    self.weights.shared_expert_gate.weight,
                    h,
                    n,
                    stream,
                )?;
            }
        }

        Ok(())
    }
}
