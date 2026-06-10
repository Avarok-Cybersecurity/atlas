// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward_prefill_fp8.

use super::*;

impl MoeLayer {
    /// EP token dispatch/combine forward pass (Workstream 3A scaffold).
    ///
    /// Instead of dense all-reduce, this:
    /// 1. Runs gate projection to get top-K routing
    /// 2. Builds a routing table partitioning tokens into local/remote
    /// 3. Dispatches remote tokens to partner rank
    ///
    /// FP8 sorted MoE prefill: grouped GEMM with FP8 expert weights.
    ///
    /// Same pipeline as NVFP4 forward_prefill but uses moe_fp8_grouped_gemm
    /// with FP8 pointer tables instead of NVFP4 pointer tables.
    pub(super) fn forward_prefill_fp8(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let n = num_tokens as u32;
        let total_expanded = n * top_k;
        let ne = num_experts as usize;

        let (gp, up, dp, sh) = match (
            &self.fp8_gate_weight_ptrs,
            &self.fp8_up_weight_ptrs,
            &self.fp8_down_weight_ptrs,
            &self.fp8_shared_expert,
        ) {
            (Some(g), Some(u), Some(d), Some(s)) => (g, u, d, s),
            _ => anyhow::bail!("FP8 expert pointer tables not set"),
        };

        // ── Shared expert ──
        // ATLAS_FP8_W8A8 path: per-token FP8 quant on activations +
        // fp8_gemm_t_blockscaled with both scales in the FP32 epilogue.
        // The shared expert is dense (every token), so we reuse the same
        // dense W8A8 GEMM that attention QKV/O proj already use.
        let force_w8a8_sh = matches!(std::env::var("ATLAS_FP8_W8A8").ok().as_deref(), Some("1"))
            && self.fp8_gemm_t_blockscaled_k.0 != 0
            && self.per_token_group_quant_fp8_k.0 != 0;
        let has_shared = shared_inter > 0;
        if has_shared && force_w8a8_sh {
            let shared_gate_out = ctx.buffers.ssm_deinterleaved();
            let shared_up_out = ctx.buffers.ssm_qkvz();
            let m_us: usize = n as usize;
            let a_fp8_bytes: usize = m_us * h as usize;
            let a_scale_bytes: usize = m_us * (h as usize / 128) * 4;
            let input_fp8 = ctx.gpu.alloc(a_fp8_bytes)?;
            let input_scale = ctx.gpu.alloc(a_scale_bytes)?;
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                input,
                input_fp8,
                input_scale,
                n,
                h,
                stream,
            )?;
            ops::fp8_gemm_t_blockscaled(
                ctx.gpu,
                self.fp8_gemm_t_blockscaled_k,
                input_fp8,
                input_scale,
                sh.gate_proj.weight,
                sh.gate_proj.row_scale,
                shared_gate_out,
                n,
                shared_inter,
                h,
                stream,
            )?;
            ops::fp8_gemm_t_blockscaled(
                ctx.gpu,
                self.fp8_gemm_t_blockscaled_k,
                input_fp8,
                input_scale,
                sh.up_proj.weight,
                sh.up_proj.row_scale,
                shared_up_out,
                n,
                shared_inter,
                h,
                stream,
            )?;
            ctx.gpu.synchronize(stream)?;
            ctx.gpu.free(input_fp8)?;
            ctx.gpu.free(input_scale)?;
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                shared_gate_out,
                shared_up_out,
                shared_gate_out,
                n * shared_inter,
                stream,
            )?;
            let shared_down_out = ctx.buffers.attn_output();
            // Quant the post-silu intermediate (K=shared_inter)
            let a2_bytes: usize = m_us * shared_inter as usize;
            let a2_scale_bytes: usize = m_us * (shared_inter as usize / 128) * 4;
            let down_in_fp8 = ctx.gpu.alloc(a2_bytes)?;
            let down_in_scale = ctx.gpu.alloc(a2_scale_bytes)?;
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                shared_gate_out,
                down_in_fp8,
                down_in_scale,
                n,
                shared_inter,
                stream,
            )?;
            ops::fp8_gemm_t_blockscaled(
                ctx.gpu,
                self.fp8_gemm_t_blockscaled_k,
                down_in_fp8,
                down_in_scale,
                sh.down_proj.weight,
                sh.down_proj.row_scale,
                shared_down_out,
                n,
                h,
                shared_inter,
                stream,
            )?;
            ctx.gpu.synchronize(stream)?;
            ctx.gpu.free(down_in_fp8)?;
            ctx.gpu.free(down_in_scale)?;
        } else if has_shared {
            let shared_gate_out = ctx.buffers.ssm_deinterleaved();
            let shared_up_out = ctx.buffers.ssm_qkvz();
            // Shared-expert dense GEMMs (gate/up/down, every token) always via
            // the bit-identical (cosine=1.0) ~4.6× faster pipelined kernel.
            let sh_gemm = |inp, w, sc, outp, mm, nn, kk| -> anyhow::Result<()> {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    inp,
                    w,
                    sc,
                    outp,
                    mm,
                    nn,
                    kk,
                    stream,
                )
            };
            // FP8 GEMM for shared expert (M=num_tokens, single kernel each)
            sh_gemm(
                input,
                sh.gate_proj.weight,
                sh.gate_proj.row_scale,
                shared_gate_out,
                n,
                shared_inter,
                h,
            )?;
            sh_gemm(
                input,
                sh.up_proj.weight,
                sh.up_proj.row_scale,
                shared_up_out,
                n,
                shared_inter,
                h,
            )?;
            // Activation + down for shared expert (SiLU or GeGLU)
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                shared_gate_out,
                shared_up_out,
                shared_gate_out,
                n * shared_inter,
                stream,
            )?;
            let shared_down_out = ctx.buffers.attn_output();
            sh_gemm(
                shared_gate_out,
                sh.down_proj.weight,
                sh.down_proj.row_scale,
                shared_down_out,
                n,
                h,
                shared_inter,
            )?;
        }

        // ── Routed expert path ──

        // Gemma-4 router pre-norm (no-op for other models).
        let router_in = self.router_input(input, n, h, ctx, stream)?;
        super::dump::dump_gate_input(ctx.gpu, stream, router_in, n, h)?;
        // 1. Gate GEMM
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

        super::dump::dump_gate_logits(ctx.gpu, stream, gate_logits, n, num_experts)?;

        // 2. Batched topK dispatch (sigmoid+bias for MiniMax/DeepSeek-V3,
        //    softmax for everyone else — selection by `correction_bias_dev`).
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

        // 3. Sort tokens by expert
        let te = total_expanded as usize;
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

        // 4. Max M tiles — sized for worst-case expert skew, not 2× avg.
        // The `(avg * 2)` heuristic silently truncated heavy experts:
        // observed avg=129, max=929 tokens for one expert (= 7× avg) in
        // a 4097-token chunk, dropping 609 rows for that expert and
        // under-counting routed-MoE output systematically (-14% at L0).
        // Now bumped to `(num_tokens * top_k).div_ceil(64)` which always
        // covers the absolute worst case (1 expert eats all tokens).
        // Cost: extra threadblocks for empty tiles (early-exit on
        // `m_idx >= M_expert`), low overhead vs the previous correctness
        // bug.
        let avg_per_expert = (num_tokens * top_k as usize).div_ceil(ne);
        let max_m_tiles = (num_tokens * top_k as usize).div_ceil(64).max(1) as u32;
        super::dump::dump_expert_load(
            ctx.gpu,
            stream,
            expert_offsets,
            ne,
            num_tokens,
            avg_per_expert,
            max_m_tiles,
        );

        // 5. FP8 grouped gate+up GEMM
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        // PM4_N_TILE / PM4_M_TILE — SSOT mirror of the kernel #defines. The
        // builder packs (m_tile<<6)|n_tile, so n_tiles uses PM4_N_TILE=64 and
        // the m granularity is PM4_M_TILE=128.
        const PM4_N_TILE: u32 = 64;
        const PM4_M_TILE: u32 = 128;
        // 2026-05-20: zero expert buffers unconditionally before the grouped
        // GEMMs. Even with worst-case `max_m_tiles` (which sizes the grid
        // for one-expert-eats-all), the kernel only writes rows where
        // `m_idx < M_expert` per expert — rows past the expert's actual
        // count keep stale data from the previous prefill (or uninit memory
        // on first prefill) and contaminate unpermute_reduce. Previously
        // guarded behind `ctx.comm.is_some()` (EP-only), making single-GPU
        // non-deterministic.
        {
            let gu_bytes = te * inter as usize * 2;
            ctx.gpu.memset_async(expert_gate_out, 0, gu_bytes, stream)?;
            ctx.gpu.memset_async(expert_up_out, 0, gu_bytes, stream)?;
            ctx.gpu.memset_async(
                ctx.buffers.expert_down_out(),
                0,
                te * h as usize * 2,
                stream,
            )?;
        }
        // ATLAS_FP8_W8A8: pre-quant input/intermediate to FP8 with per-token-
        // per-128 FP32 scale, use new W8A8 grouped GEMM (vLLM-equivalent).
        let force_w8a8 = matches!(std::env::var("ATLAS_FP8_W8A8").ok().as_deref(), Some("1"))
            && self.moe_w8a8_grouped_gemm_k.0 != 0
            && self.per_token_group_quant_fp8_k.0 != 0;

        if force_w8a8 && max_m_tiles > 0 {
            // Quant input [num_tokens, h] → input_fp8 + input_a_scale ONCE,
            // reuse for both gate and up.
            let m = num_tokens;
            let a_fp8_bytes = m * h as usize;
            let a_scale_bytes = m * (h as usize / 128) * 4;
            let input_fp8 = ctx.gpu.alloc(a_fp8_bytes)?;
            let input_a_scale = ctx.gpu.alloc(a_scale_bytes)?;
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                input,
                input_fp8,
                input_a_scale,
                m as u32,
                h,
                stream,
            )?;
            ops::moe_w8a8_grouped_gemm(
                ctx.gpu,
                self.moe_w8a8_grouped_gemm_k,
                input_fp8,
                input_a_scale,
                gp.weight_ptrs,
                gp.scale_ptrs,
                expert_gate_out,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                inter,
                h,
                max_m_tiles,
                stream,
            )?;
            ops::moe_w8a8_grouped_gemm(
                ctx.gpu,
                self.moe_w8a8_grouped_gemm_k,
                input_fp8,
                input_a_scale,
                up.weight_ptrs,
                up.scale_ptrs,
                expert_up_out,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                inter,
                h,
                max_m_tiles,
                stream,
            )?;
            ctx.gpu.synchronize(stream)?;
            ctx.gpu.free(input_fp8)?;
            ctx.gpu.free(input_a_scale)?;
        } else if max_m_tiles > 0 {
            // Routed-expert FP8 grouped gate+up GEMM via grid-compaction. Build
            // the work-list ONCE (gate and up share the same expert_offsets,
            // weight-pointer NULL-ness, N=inter, K=h tiling), reuse it for both
            // GEMMs, free after. Builder + both grouped-GEMM launches are on the
            // SAME `stream` (read-after-write of total_tiles/worklist — see the
            // moe_build_tile_worklist comment).
            let n_tiles_gu = inter.div_ceil(PM4_N_TILE);
            // Worst-case work-items: one expert can eat all te tokens
            // (te.div_ceil(128) m-tiles) plus a +ne+1 slack term covering
            // per-expert m-tile rounding when tokens are spread across all
            // experts. ×n_tiles n-tiles, ×2 words/item, ×4 bytes/word.
            let wl_cap_items = (te.div_ceil(PM4_M_TILE as usize) + ne + 1) * n_tiles_gu as usize;
            let wl_gu = ctx.gpu.alloc(wl_cap_items * 2 * 4)?;
            let tt_gu = ctx.gpu.alloc(4)?;
            ops::moe_build_tile_worklist(
                ctx.gpu,
                self.moe_build_tile_worklist_k,
                expert_offsets,
                gp.weight_ptrs,
                wl_gu,
                tt_gu,
                num_experts,
                n_tiles_gu,
                PM4_M_TILE,
                stream,
            )?;
            ops::moe_fp8_grouped_gemm(
                ctx.gpu,
                self.moe_fp8_grouped_gemm_k,
                input,
                gp.weight_ptrs,
                gp.scale_ptrs,
                expert_gate_out,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                inter,
                h,
                wl_gu,
                tt_gu,
                stream,
            )?;
            ops::moe_fp8_grouped_gemm(
                ctx.gpu,
                self.moe_fp8_grouped_gemm_k,
                input,
                up.weight_ptrs,
                up.scale_ptrs,
                expert_up_out,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                inter,
                h,
                wl_gu,
                tt_gu,
                stream,
            )?;
            ctx.gpu.synchronize(stream)?;
            ctx.gpu.free(wl_gu)?;
            ctx.gpu.free(tt_gu)?;
        }

        // 6. Activation+mul + down GEMM
        let expert_down_out = ctx.buffers.expert_down_out();
        if force_w8a8 && max_m_tiles > 0 {
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                expert_gate_out,
                expert_up_out,
                expert_gate_out,
                total_expanded * inter,
                stream,
            )?;
            // Quant the permuted post-silu intermediate. Length is
            // total_expanded, K is `inter` (down_proj input dim).
            let m: usize = total_expanded as usize;
            let a_fp8_bytes: usize = m * inter as usize;
            let a_scale_bytes: usize = m * (inter as usize / 128) * 4;
            let down_in_fp8 = ctx.gpu.alloc(a_fp8_bytes)?;
            let down_in_scale = ctx.gpu.alloc(a_scale_bytes)?;
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                expert_gate_out,
                down_in_fp8,
                down_in_scale,
                m as u32,
                inter,
                stream,
            )?;
            ops::moe_w8a8_grouped_gemm(
                ctx.gpu,
                self.moe_w8a8_grouped_gemm_k,
                down_in_fp8,
                down_in_scale,
                dp.weight_ptrs,
                dp.scale_ptrs,
                expert_down_out,
                expert_offsets,
                spark_runtime::gpu::DevicePtr(0),
                num_experts,
                h,
                inter,
                max_m_tiles,
                stream,
            )?;
            ctx.gpu.synchronize(stream)?;
            ctx.gpu.free(down_in_fp8)?;
            ctx.gpu.free(down_in_scale)?;
        } else if max_m_tiles > 0 {
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                expert_gate_out,
                expert_up_out,
                expert_gate_out,
                total_expanded * inter,
                stream,
            )?;
            // ── down-proj: separate work-list (N=h, K=inter → different
            // n_tiles than gate/up). sorted_token_ids=NULL keeps the
            // direct-index A-prefetch branch (R6). Builder + grouped GEMM on the
            // SAME `stream`. Down-proj uses dp.weight_ptrs for NULL-skip.
            let n_tiles_dn = h.div_ceil(PM4_N_TILE);
            let wl_cap_items = (te.div_ceil(PM4_M_TILE as usize) + ne + 1) * n_tiles_dn as usize;
            let wl_dn = ctx.gpu.alloc(wl_cap_items * 2 * 4)?;
            let tt_dn = ctx.gpu.alloc(4)?;
            ops::moe_build_tile_worklist(
                ctx.gpu,
                self.moe_build_tile_worklist_k,
                expert_offsets,
                dp.weight_ptrs,
                wl_dn,
                tt_dn,
                num_experts,
                n_tiles_dn,
                PM4_M_TILE,
                stream,
            )?;
            ops::moe_fp8_grouped_gemm(
                ctx.gpu,
                self.moe_fp8_grouped_gemm_k,
                expert_gate_out,
                dp.weight_ptrs,
                dp.scale_ptrs,
                expert_down_out,
                expert_offsets,
                spark_runtime::gpu::DevicePtr(0),
                num_experts,
                h,
                inter,
                wl_dn,
                tt_dn,
                stream,
            )?;
            ctx.gpu.synchronize(stream)?;
            ctx.gpu.free(wl_dn)?;
            ctx.gpu.free(tt_dn)?;
        }

        // 7. Unpermute + weighted reduce + shared blend
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

        // EP all-reduce of routed-expert output FIRST.
        // Shared experts are NOT EP-sharded (every rank loads the full
        // shared_expert weights — see fast_weights/mod.rs:85-104), so
        // their down-projection output already contains the full
        // contribution and must be blended AFTER the routed-expert
        // allreduce — otherwise the shared term gets summed across ranks
        // (multiplied by world_size). Sibling of forward()/forward_k2()/
        // forward_k3() which already do this in the right order; mirrors
        // vllm PR #39181.
        if let Some(comm) = ctx.comm
            && comm.world_size() > 1
        {
            comm.all_reduce_async(output.0, num_tokens * h as usize * 2, stream)?;
        }

        // Shared expert blend (post-allreduce).
        if has_shared {
            let shared_down_out = ctx.buffers.attn_output();
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

        Ok(())
    }
}
