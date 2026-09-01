// SPDX-License-Identifier: AGPL-3.0-only

//! Native EXL3 routed-expert PREFILL arm (`ATLAS_EXL3_NATIVE_MOE=1`) —
//! upstream ExLlamaV3's sort-by-expert tier mapped onto Atlas's prefill
//! machinery.
//!
//! Per token batch (`Exl3MoeState::pf_t_cap`, default 4096 — one batch for
//! the canonical prefill chunk): the SAME router + batched-topk numerics as
//! the grouped prefill path, Atlas's `moe_sort_by_expert` counting sort over
//! the batch's GLOBAL expert ids, then `ops::exl3_moe_prefill_routed`
//! (staging → fused `exl3_moe` persistent kernel for every local expert with
//! `0 < count <= 128` sorted rows → chunked `exl3_gemm` overflow for hotter
//! experts → fp32-accumulated, prob-weighted egress). The routing
//! probabilities are applied INSIDE the fp32 accumulator, so the tail must
//! not re-apply them; the shared expert stays NVFP4/FP8/BF16 and is blended
//! once after the (EP all-reduced) routed sums — the exact tail the decode
//! arm (`forward_exl3.rs`) ships.
//!
//! EP: the sort keeps GLOBAL ids; `exl3_moe_stage_sorted` rotates the local
//! span first and parks every remote slot in the sentinel tail bucket the
//! fused kernel never processes. A token whose experts are all remote
//! contributes an exact-zero row; the all-reduce completes it.
//!
//! Graph capture: the fused kernel spin-barriers on the shared locks buffer
//! (operationally cooperative) — refuse `ctx.graph_capture` loudly, exactly
//! like the decode arm.

use super::*;

impl MoeLayer {
    /// Full prefill through the native arm: router + batched topk + per-batch
    /// sort/fused/overflow routed phase + shared expert + blend + EP
    /// all-reduce. Output at `moe_output()` `[num_tokens, H]` BF16.
    pub(crate) fn forward_prefill_exl3(
        &self,
        input: DevicePtr, // [num_tokens, H] BF16 — normed MoE input
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            !ctx.graph_capture,
            "EXL3 native MoE prefill reached under CUDA-graph capture — the \
             fused exl3_moe kernel spin-barriers on the locks buffer and is \
             not capturable"
        );
        // Named refusals for machinery this arm does not wire (none of it
        // exists on qwen4_exp, the only EXL3-native target) — mirrors the
        // decode arm's set:
        anyhow::ensure!(
            self.router_logits_n as usize == ctx.config.num_experts && self.tid2eid_dev.is_none(),
            "EXL3 native MoE: zero-expert / hash routing is not wired on this arm"
        );
        anyhow::ensure!(
            self.lora.is_none(),
            "EXL3 native MoE has no LoRA fold hooks (the build refuses \
             --lora-adapter with ATLAS_EXL3_NATIVE)"
        );
        anyhow::ensure!(
            self.pre_expert_norm.is_none(),
            "EXL3 native MoE: pre-expert-norm models are not wired on this arm"
        );

        let st = self
            .exl3_moe_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("EXL3 MoE tables installed without launch state"))?;
        let _dispatch = st.dispatch_guard(ctx.gpu, stream)?;
        let tabs = self
            .exl3_expert_tables
            .as_ref()
            .expect("checked by exl3_native_active");
        let (local_start, num_local) = (tabs[0].local_start, tabs[0].num_local);
        debug_assert!(
            tabs.iter()
                .all(|t| t.local_start == local_start && t.num_local == num_local),
            "gate/up/down tables disagree on the EP-local range"
        );

        let h = ctx.config.hidden_size;
        let inter = ctx.config.moe_intermediate_size;
        let top_k = ctx.config.num_experts_per_tok;
        let num_experts = ctx.config.num_experts;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let n = num_tokens as u32;

        // ── Router: the PREFILL selection (fp8 / nvfp4 / pinned scalar
        // dense — see forward_prefill.rs for the 2026-08-12 numerics pin) ──
        let router_in = self.router_input(input, n, h as u32, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();
        if let Some(fp8) = self.gate_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                router_in,
                fp8,
                gate_logits,
                n,
                self.router_logits_n,
                h as u32,
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
                self.router_logits_n,
                h as u32,
                stream,
            )?;
        } else {
            self.router_gate_gemm_dense(
                router_in,
                gate_logits,
                n,
                self.router_logits_n,
                h as u32,
                ctx,
                stream,
            )?;
        }

        // ── Batched top-k (softmax, or sigmoid+bias — the decode arm's
        // envelope; other scoring functions refuse by falling to softmax
        // only when biasless) ──
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(num_tokens * top_k * 4);
        if let Some(bias) = self.correction_bias_dev {
            anyhow::ensure!(
                ctx.config.scoring_func != "sqrtsoftplus" && ctx.config.scoring_func != "softmax",
                "EXL3 native MoE: scoring_func {:?} with correction bias is \
                 not wired on this arm",
                ctx.config.scoring_func,
            );
            ops::moe_topk_sigmoid_batched(
                ctx.gpu,
                self.moe_topk_sigmoid_batched_k,
                gate_logits,
                bias,
                indices_dev,
                weights_dev,
                num_experts as u32,
                top_k as u32,
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
                num_experts as u32,
                top_k as u32,
                ctx.config.norm_topk_prob,
                n,
                stream,
            )?;
        }

        // ── Routed phase, per token batch: sort (GLOBAL ids) → staged fused
        // kernel + overflow → weighted BF16 rows at moe_output ──
        let output = ctx.buffers.moe_output();
        let proj = |t: &Exl3ExpertPtrTable| ops::Exl3MoeProj {
            trellis_ptrs: t.trellis_ptrs,
            suh_ptrs: t.suh_ptrs,
            svh_ptrs: t.svh_ptrs,
            k_bits: t.k_bits,
            cb: t.cb,
        };
        let proj_tables = [proj(&tabs[0]), proj(&tabs[1]), proj(&tabs[2])];
        let ov = ops::Exl3MoeOverflowCtx {
            gate_host: &tabs[0].host_ptrs,
            up_host: &tabs[1].host_ptrs,
            down_host: &tabs[2].host_ptrs,
        };
        let pf = st.prefill_scratch();
        let mut t0 = 0usize;
        while t0 < num_tokens {
            let tb = pf.t_cap.min(num_tokens - t0);
            let te_b = tb * top_k;
            // Per-batch counting sort over the batch's slice of the routing
            // state; outputs alias gate_logits exactly like the grouped path
            // (the logits were consumed by topk above).
            let sorted_token_ids = gate_logits;
            let sorted_expert_ids = gate_logits.offset(te_b * 4);
            let expert_offsets = gate_logits.offset(te_b * 8);
            let token_to_perm = gate_logits.offset(te_b * 8 + (num_experts + 1) * 4);
            ops::moe_sort_by_expert(
                ctx.gpu,
                self.moe_sort_by_expert,
                indices_dev.offset(t0 * top_k * 4),
                sorted_token_ids,
                sorted_expert_ids,
                expert_offsets,
                token_to_perm,
                te_b as u32,
                num_experts as u32,
                top_k as u32,
                stream,
            )?;
            let stats = ops::exl3_moe_prefill_routed(
                ctx.gpu,
                input.offset(t0 * h * 2),
                weights_dev.offset(t0 * top_k * 4),
                expert_offsets,
                token_to_perm,
                output.offset(t0 * h * 2),
                &proj_tables,
                &ov,
                &pf,
                st.locks,
                tb,
                top_k,
                h,
                inter,
                local_start,
                num_local,
                0.0, // qwen4_exp declares no activation clamp
                st.sm_count,
                stream,
            )?;
            tracing::trace!(
                "EXL3 MoE prefill batch [{t0}, {}): fused num_active={} \
                 overflow_experts={}",
                t0 + tb,
                stats.num_active,
                stats.overflow_experts,
            );
            t0 += tb;
        }

        // ── Shared expert (kept NVFP4/FP8/BF16) + blend, EP-aware — the
        // decode arm's tail: all-reduce the routed partials FIRST, then
        // blend the shared expert exactly once. ──
        let has_shared = shared_inter > 0;
        if has_shared {
            self.run_shared_expert_prefill(
                input,
                n,
                h as u32,
                shared_inter,
                stream,
                stream,
                false,
                ctx,
            )?;
        }
        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
        {
            // Stream-ordered variant only: graph capture is refused above.
            comm.all_reduce_async(output.0, num_tokens * h * 2, stream)?;
        }
        if has_shared {
            let shared_out = ctx.buffers.attn_output();
            ops::moe_batched_blend(
                ctx.gpu,
                self.moe_batched_blend,
                output,
                shared_out,
                input,
                self.weights.shared_expert_gate.weight,
                h as u32,
                n,
                stream,
            )?;
        }

        Ok(())
    }
}
