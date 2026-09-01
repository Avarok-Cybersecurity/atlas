// SPDX-License-Identifier: AGPL-3.0-only

//! Native EXL3 routed-expert DECODE arm (`ATLAS_EXL3_NATIVE_MOE=1`).
//!
//! Serves the routed experts straight from packed trellis via exactly three
//! `exl3_mgemm` calls per layer (`ops::exl3_moe_decode_routed` — upstream's
//! `run_bszN` decode tier), with the routing probabilities FOLDED into the
//! down call's fp32 grouped reduction. The shared expert stays NVFP4/FP8/
//! BF16 and runs as a separate dense pass (`run_shared_expert_prefill`, the
//! `run_bf16_shared_expert` unfusing precedent), then one `moe_batched_blend`
//! adds `sigmoid(input @ shared_gate) * shared` — so the routing probs are
//! applied exactly once, in fp32, inside the down mgemm.
//!
//! One arm covers EVERY decode entry point (`forward`, `forward_k2/k3`,
//! `forward_batched`, `forward_token_major_decode`, atomic-C4): with the
//! routed experts kept packed there are NO NVFP4 tables to fall back to, so
//! each entry delegates here when [`MoeLayer::exl3_native_active`]. Prefill
//! (> the decode tier) is a separate stage and refuses loudly for now.
//!
//! EP: remote experts are `-1` slots over the dense LOCAL tables (staged
//! device-side by `exl3_moe_stage_routing`, the canonical
//! `exl3_expert_slot_index` mapping); a token whose experts are all remote
//! contributes an exact-zero row, and the existing EP all-reduce sums the
//! per-rank partials BEFORE the shared expert is blended once — the same
//! tail every other arm uses.
//!
//! Graph capture: every launch in the mgemm pipeline is COOPERATIVE — the
//! decode-graph veto (`decode_graph_unsupported` on both layer kinds, plus
//! the verify-path use_graphs terms) keys on `exl3_native_active`, and this
//! file refuses `ctx.graph_capture` defensively rather than failing with
//! CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED mid-serve.

use super::*;

impl MoeLayer {
    /// True when this layer's routed experts are served natively from EXL3
    /// trellis (tables installed by the loader). Gates every decode dispatch
    /// AND the decode-graph vetoes (cooperative launches are not
    /// graph-capturable).
    pub fn exl3_native_active(&self) -> bool {
        self.exl3_expert_tables.is_some()
    }

    /// Full n-token decode through the native arm: batched routing (the
    /// `forward_token_major_decode` router mirror) + routed mgemm pipeline +
    /// shared expert + blend + EP all-reduce. Output at `moe_output()`
    /// `[num_tokens, H]` BF16.
    pub(crate) fn forward_exl3_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = num_tokens as u32;

        // Router (batched): same kernels/numerics as the shipping n>=4
        // token-major decode router.
        let router_in = self.router_input(input, n, h, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();
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

        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(num_tokens * top_k as usize * 4);
        if let Some(bias) = self.correction_bias_dev {
            // Same envelope as the prefill arm: a bias-carrying model with
            // softmax/sqrtsoftplus scoring must refuse here too, not silently
            // route through sigmoid numerics.
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

        self.forward_exl3_after_routing(input, num_tokens, indices_dev, weights_dev, ctx, stream)
            .map(|_| ())
    }

    /// The expert phase, entered with routing already staged at `scratch`
    /// (`indices_dev` u32 `[n*top_k]` GLOBAL ids, `weights_dev` f32
    /// `[n*top_k]` — both the C=1 `forward()` layout and the batched topk
    /// layout). Returns `moe_output()`.
    pub(crate) fn forward_exl3_after_routing(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        // Cooperative mgemm launches are not graph-capturable; reaching here
        // under capture means a veto site regressed — fail loud, not with an
        // opaque CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED.
        anyhow::ensure!(
            !ctx.graph_capture,
            "EXL3 native MoE decode reached under CUDA-graph capture — the \
             decode/verify graph vetoes must key on exl3_native_active()"
        );
        // Named refusals for machinery this arm does not wire (none of it
        // exists on qwen4_exp, the only EXL3-native target):
        anyhow::ensure!(
            self.router_logits_n as usize == ctx.config.num_experts && self.tid2eid_dev.is_none(),
            "EXL3 native MoE: zero-expert / hash routing is not wired on this arm"
        );
        anyhow::ensure!(
            self.lora.is_none(),
            "EXL3 native MoE has no LoRA fold hooks (the build refuses \
             --lora-adapter with ATLAS_EXL3_NATIVE)"
        );
        // `run_shared_expert_prefill` scratches ssm_deinterleaved(), which is
        // exactly where a pre-expert norm would put the expert input.
        anyhow::ensure!(
            self.pre_expert_norm.is_none(),
            "EXL3 native MoE: pre-expert-norm models are not wired on this arm"
        );

        let st = self
            .exl3_moe_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("EXL3 MoE tables installed without launch state"))?;
        // Held through the shared-expert blend: the routed output is read
        // from the shared slabs' downstream `output` only after the mgemm
        // chain, all on this one stream.
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
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let output = ctx.buffers.moe_output();

        // ── Routed experts: 3x exl3_mgemm, probs folded into DOWN ──
        let proj = |t: &Exl3ExpertPtrTable| ops::Exl3MoeProj {
            trellis_ptrs: t.trellis_ptrs,
            suh_ptrs: t.suh_ptrs,
            svh_ptrs: t.svh_ptrs,
            k_bits: t.k_bits,
            cb: t.cb,
        };
        let scratch = ops::Exl3MoeScratch {
            a_f16: st.a_f16,
            a_had_f16: st.a_had_f16,
            a_had_capacity_elems: st.s_cap * st.hidden,
            c_gate_f16: st.c_gate_f32,
            c_up_f16: st.c_up_f32,
            inter_f16: st.inter_f16,
            c_down_f32: st.c_down_f32,
            b_indices: st.b_indices,
            b_weights: st.b_weights,
            s_cap: st.s_cap,
        };
        ops::exl3_moe_decode_routed(
            ctx.gpu,
            input,
            indices_dev,
            weights_dev,
            output,
            &[proj(&tabs[0]), proj(&tabs[1]), proj(&tabs[2])],
            &scratch,
            st.locks,
            num_tokens,
            top_k,
            h,
            inter,
            local_start,
            num_local,
            0.0, // qwen4_exp declares no activation clamp
            st.sm_count,
            stream,
        )?;

        // ── Shared expert (kept NVFP4/FP8/BF16) + blend, EP-aware ──
        // The routed sums (probs already applied, fp32-reduced) are in
        // `output`; the shared expert is added exactly once:
        //   non-EP:  output += sigmoid(input @ w_sg) * shared
        //   EP:      all-reduce the routed partials FIRST, then the same
        //            blend once (the standard EP tail — remote experts were
        //            -1 slots contributing exact zeros).
        let has_shared = shared_inter > 0;
        if has_shared {
            // Writes shared_down_out = attn_output(); scratches
            // ssm_deinterleaved()/ssm_qkvz() (input is NOT in either — the
            // pre-expert-norm case is refused above).
            self.run_shared_expert_prefill(
                input,
                num_tokens as u32,
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
            // Always the stream-ordered variant: graph capture is refused
            // above, so the blocking capture-mode all_reduce can't apply.
            comm.all_reduce_async(output.0, num_tokens * h * 2, stream)?;
        }
        if has_shared {
            let shared_out = ctx.buffers.attn_output();
            // NULL gate weight -> sigmoid = 1.0 (ungated shared expert).
            ops::moe_batched_blend(
                ctx.gpu,
                self.moe_batched_blend,
                output,
                shared_out,
                input,
                self.weights.shared_expert_gate.weight,
                h as u32,
                num_tokens as u32,
                stream,
            )?;
        }

        Ok(output)
    }
}
