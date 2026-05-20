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

        // ── Shared expert (same as NVFP4 path) ──
        let has_shared = shared_inter > 0;
        if has_shared {
            let shared_gate_out = ctx.buffers.ssm_deinterleaved();
            let shared_up_out = ctx.buffers.ssm_qkvz();
            // FP8 GEMM for shared expert (M=num_tokens, single kernel each)
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                input,
                sh.gate_proj.weight,
                sh.gate_proj.row_scale,
                shared_gate_out,
                n,
                shared_inter,
                h,
                stream,
            )?;
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                input,
                sh.up_proj.weight,
                sh.up_proj.row_scale,
                shared_up_out,
                n,
                shared_inter,
                h,
                stream,
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
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                shared_gate_out,
                sh.down_proj.weight,
                sh.down_proj.row_scale,
                shared_down_out,
                n,
                h,
                shared_inter,
                stream,
            )?;
        }

        // ── Routed expert path ──

        // Gemma-4 router pre-norm (no-op for other models).
        let router_in = self.router_input(input, n, h, ctx, stream)?;
        // ATLAS_DUMP_EXPERT_IDS: dump the gate INPUT (= post-attn-norm of
        // hidden + SSM out_proj). If this matches HF's input to .gate but
        // logits differ → gate matmul or weight loading is the bug.
        if std::env::var("ATLAS_DUMP_EXPERT_IDS").ok().as_deref() == Some("1") {
            ctx.gpu.synchronize(stream)?;
            let offset = (n - 1) as usize * h as usize * 2;
            let mut buf = vec![0u8; h as usize * 2];
            let _ = ctx.gpu.copy_d2h(router_in.offset(offset), &mut buf);
            let v: Vec<f32> = buf.chunks_exact(2).map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            }).collect();
            let norm = v.iter().map(|x| x*x).sum::<f32>().sqrt();
            tracing::info!(
                "ATLAS_GATE_INPUT last_tok: |x|={:.4}  first5={:?}",
                norm, &v[..5]
            );
        }
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

        // ATLAS_DUMP_EXPERT_IDS=1 also dumps gate_logits BEFORE softmax+topK
        // to attribute drift between gate matmul vs routing logic.
        if std::env::var("ATLAS_DUMP_EXPERT_IDS").ok().as_deref() == Some("1") {
            ctx.gpu.synchronize(stream)?;
            // gate_logits[n-1, :] = num_experts BF16 values for last token
            let logits_offset = (n - 1) as usize * num_experts as usize * 2;
            let mut buf = vec![0u8; num_experts as usize * 2];
            let _ = ctx.gpu.copy_d2h(gate_logits.offset(logits_offset), &mut buf);
            // Convert BF16 → float32 + print top 10 highest logits
            let logits: Vec<f32> = buf.chunks_exact(2).map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            }).collect();
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
            let top10: Vec<(usize, f32)> = idx.iter().take(10).map(|&i| (i, logits[i])).collect();
            tracing::info!(
                "ATLAS_GATE_LOGITS last_tok: top10_(idx,val)={:?} mean={:.4} std={:.4}",
                top10,
                logits.iter().sum::<f32>() / logits.len() as f32,
                {
                    let mean = logits.iter().sum::<f32>() / logits.len() as f32;
                    (logits.iter().map(|x| (x-mean).powi(2)).sum::<f32>() / logits.len() as f32).sqrt()
                }
            );
        }

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
                1.0,
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

        // ATLAS_DUMP_EXPERT_IDS=1 — log the last token's top-K expert
        // indices + weights for drift attribution vs HF.
        if std::env::var("ATLAS_DUMP_EXPERT_IDS").ok().as_deref() == Some("1") {
            ctx.gpu.synchronize(stream)?;
            let mut idx_buf = vec![0u8; top_k as usize * 4];
            let mut w_buf = vec![0u8; top_k as usize * 4];
            let idx_offset = (n - 1) as usize * top_k as usize * 4;
            let w_offset = idx_offset;
            let _ = ctx.gpu.copy_d2h(indices_dev.offset(idx_offset), &mut idx_buf);
            let _ = ctx.gpu.copy_d2h(weights_dev.offset(w_offset), &mut w_buf);
            let ids: Vec<u32> = idx_buf.chunks_exact(4).map(|b| u32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
            let ws: Vec<f32> = w_buf.chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
            tracing::info!(
                "ATLAS_EXPERT_IDS last_tok: indices={:?} weights={:?} sum={:.4}",
                ids, ws, ws.iter().sum::<f32>()
            );
        }

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

        // 4. Max M tiles (same heuristic as NVFP4)
        let avg_per_expert = (num_tokens * top_k as usize).div_ceil(ne);
        let max_m_tiles = (avg_per_expert * 2).div_ceil(64).max(1) as u32;

        // 5. FP8 grouped gate+up GEMM
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        // EP: zero expert buffers for remote experts
        if ctx.comm.is_some() {
            let gate_bytes = te * inter as usize * 2;
            ctx.gpu
                .memset_async(expert_gate_out, 0, gate_bytes, stream)?;
            ctx.gpu.memset_async(expert_up_out, 0, gate_bytes, stream)?;
            ctx.gpu.memset_async(
                ctx.buffers.expert_down_out(),
                0,
                te * h as usize * 2,
                stream,
            )?;
        }
        let fp8_grouped_k = self.fp8_grouped_kernel();
        // 2026-05-20: zero expert buffers unconditionally before the grouped
        // GEMMs. The `max_m_tiles = (avg*2).div_ceil(64)` heuristic assumes
        // peak-per-expert ≤ 2× average; skewed routing (especially long
        // chunks) violates this, leaving the un-processed rows uninitialized
        // and propagating stale data through unpermute_reduce. Without this
        // zeroing, the L0 MoE output magnitude is non-deterministic across
        // runs and chunk sizes (verified: ATLAS_ROUTED_ONLY at chunk-4 L0
        // varied 0.26-1.13 for the same prompt). Only EP-mode had this
        // memset.
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
        if max_m_tiles > 0 {
            ops::moe_fp8_grouped_gemm(
                ctx.gpu,
                fp8_grouped_k,
                input,
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

            ops::moe_fp8_grouped_gemm(
                ctx.gpu,
                fp8_grouped_k,
                input,
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
        }

        // 6. Activation+mul + down GEMM
        let expert_down_out = ctx.buffers.expert_down_out();
        if max_m_tiles > 0 {
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                expert_gate_out,
                expert_up_out,
                expert_gate_out,
                total_expanded * inter,
                stream,
            )?;
            ops::moe_fp8_grouped_gemm(
                ctx.gpu,
                fp8_grouped_k,
                expert_gate_out,
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
            // ATLAS_DUMP_EXPERT_IDS=1: dump routed-only output + shared output
            // BEFORE blend, so we can attribute the moe_out amplification.
            if std::env::var("ATLAS_DUMP_EXPERT_IDS").ok().as_deref() == Some("1") {
                ctx.gpu.synchronize(stream)?;
                let offset = (n - 1) as usize * h as usize * 2;
                // routed-only (in `output` before blend)
                let mut buf_r = vec![0u8; h as usize * 2];
                let _ = ctx.gpu.copy_d2h(output.offset(offset), &mut buf_r);
                let vr: Vec<f32> = buf_r.chunks_exact(2).map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                }).collect();
                let nr = vr.iter().map(|x| x*x).sum::<f32>().sqrt();
                tracing::info!(
                    "ATLAS_ROUTED_ONLY last_tok: |x|={:.4} first5={:?}",
                    nr, &vr[..5]
                );
                // shared_down_out
                let mut buf_s = vec![0u8; h as usize * 2];
                let _ = ctx.gpu.copy_d2h(shared_down_out.offset(offset), &mut buf_s);
                let vs: Vec<f32> = buf_s.chunks_exact(2).map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                }).collect();
                let ns = vs.iter().map(|x| x*x).sum::<f32>().sqrt();
                tracing::info!(
                    "ATLAS_SHARED_OUT last_tok: |x|={:.4} first5={:?}",
                    ns, &vs[..5]
                );
                // gate scalar: dot(normed[last], gate_weight) → sigmoid
                let mut buf_n = vec![0u8; h as usize * 2];
                let mut buf_g = vec![0u8; h as usize * 2];
                let _ = ctx.gpu.copy_d2h(input.offset(offset), &mut buf_n);
                let _ = ctx.gpu.copy_d2h(self.weights.shared_expert_gate.weight, &mut buf_g);
                let vn: Vec<f32> = buf_n.chunks_exact(2).map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                }).collect();
                let vg: Vec<f32> = buf_g.chunks_exact(2).map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                }).collect();
                let dot: f32 = vn.iter().zip(vg.iter()).map(|(a,b)| a*b).sum();
                let sig = 1.0 / (1.0 + (-dot).exp());
                tracing::info!(
                    "ATLAS_SHARED_GATE last_tok: dot={:.4} sigmoid={:.6}",
                    dot, sig
                );
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

        // ATLAS_DUMP_EXPERT_IDS=1 also dumps the FINAL moe output (routed +
        // shared blend) at the last token. Compared to HF L0_moe_out, this
        // localizes residual-stream amplification to (a) too-large moe out,
        // (b) too-large routed expert outputs, or (c) miscomputed shared gate.
        if std::env::var("ATLAS_DUMP_EXPERT_IDS").ok().as_deref() == Some("1") {
            ctx.gpu.synchronize(stream)?;
            let offset = (n - 1) as usize * h as usize * 2;
            let mut buf = vec![0u8; h as usize * 2];
            let _ = ctx.gpu.copy_d2h(output.offset(offset), &mut buf);
            let v: Vec<f32> = buf.chunks_exact(2).map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            }).collect();
            let norm = v.iter().map(|x| x*x).sum::<f32>().sqrt();
            tracing::info!(
                "ATLAS_MOE_OUT last_tok: |x|={:.4} first5={:?}",
                norm, &v[..5]
            );
        }

        Ok(())
    }
}
