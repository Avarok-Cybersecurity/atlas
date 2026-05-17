// SPDX-License-Identifier: AGPL-3.0-only

//! MLA branch of `prefill_attention_paged`. Mistral4-style 2-step
//! prefill: latent-rank Q/K/V projections, RoPE on the rope half,
//! assembled K/V for direct flash attention, compressed cache write.
//! Extracted from `paged.rs` to keep that file under 500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

#[allow(clippy::too_many_arguments)]
pub(super) struct MlaPrefillArgs {
    pub normed: DevicePtr,
    pub num_tokens: usize,
    pub n: u32,
    pub h: u32,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub kv_dim: usize,
    pub eps: f32,
    pub bf16: usize,
    pub bs: u32,
    pub stream: u64,
    /// Tokens already written to KV cache before this chunk (0 for the first chunk).
    pub seq_len_start: usize,
}

impl Qwen3AttentionLayer {
    /// Run the MLA prefill kernel chain (Q latent → expand → RoPE → cache
    /// write → flash attn → O proj). Returns the O-projection output
    /// pointer (`ctx.buffers.norm_output()`).
    pub(super) fn prefill_attention_paged_mla(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &MlaPrefillArgs,
    ) -> Result<DevicePtr> {
        let MlaPrefillArgs {
            normed,
            num_tokens,
            n,
            h,
            nq,
            nkv,
            hd,
            kv_dim,
            eps,
            bf16,
            bs,
            stream,
            seq_len_start,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("prefill_attention_paged_mla called without MLA config");

        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_nope = mla.nope as u32;
        let mla_v_dim = mla.v_dim as u32;
        let mla_rope = mla.rope as u32;
        let mla_cache_dim = kv_lora + mla_rope;
        let kv_len = seq_len_start + num_tokens; // full context length after this chunk

        // ── Step 1: Q projection (shared by both paths) ──────────────────────
        // Q: normed → wq_a → rms_norm → wq_b → [N, nq*hd] in [nope|rope] per head.
        // q_latent lives in ssm_ba. IMPORTANT: ssm_ba is later aliased by k_rope_buf
        // (line after wq_b). Any computation that needs q_latent must happen before
        // the k_rope_buf dense_gemm below.
        let q_latent = ctx.buffers.ssm_ba();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &mla.wq_a,
            q_latent,
            n,
            q_lora,
            h,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            q_latent,
            &mla.q_a_norm,
            q_latent,
            n,
            q_lora,
            eps,
            stream,
        )?;
        let qg_out = ctx.buffers.qkv_output();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            q_latent,
            &mla.wq_b,
            qg_out,
            n,
            nq * hd,
            q_lora,
            stream,
        )?;

        // ── Step 2: KV latent projection (shared by both paths) ──────────────
        let kv_latent = ctx.buffers.expert_gate_out();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &mla.wkv_a,
            kv_latent,
            n,
            kv_lora,
            h,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            kv_latent,
            &mla.kv_a_norm,
            kv_latent,
            n,
            kv_lora,
            eps,
            stream,
        )?;

        if seq_len_start == 0 {
            // ════════════════════════════════════════════════════════════════════
            // FIRST-CHUNK PATH (seq_len_start == 0): unabsorbed form.
            //
            // Expand KV via wkv_b, assemble contiguous K/V, and run flash
            // attention over the N new tokens only (no historical context).
            // ════════════════════════════════════════════════════════════════════

            let kv_expanded_dim = nkv * (mla_nope + mla_v_dim);
            let kv_expanded = ctx.buffers.ssm_deinterleaved();
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                kv_latent,
                &mla.wkv_b,
                kv_expanded,
                n,
                kv_expanded_dim,
                kv_lora,
                stream,
            )?;

            // K_rope: single shared head [N, rope] (MQA-style)
            let k_rope_buf = ctx.buffers.ssm_ba(); // aliases q_latent — OK, consumed above
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &mla.wkv_a_rope,
                k_rope_buf,
                n,
                mla_rope,
                h,
                stream,
            )?;

            let q_rope_tmp = ctx.buffers.ssm_conv_out_f32();
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                qg_out,
                q_rope_tmp,
                n,
                nq,
                hd,
                mla_nope,
                mla_rope,
                nq * hd,
                stream,
            )?;
            let rope_meta = ctx.attn_metadata.expect("MLA prefill requires metadata");
            ops::rope_yarn(
                ctx.gpu,
                self.rope_yarn_k,
                q_rope_tmp,
                k_rope_buf,
                rope_meta.positions,
                n,
                nq,
                1,
                mla_rope,
                mla_rope,
                mla.yarn_inv_freq,
                ctx.config.rope_theta as f32,
                stream,
            )?;
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                q_rope_tmp,
                qg_out,
                n,
                nq,
                hd,
                mla_nope,
                mla_rope,
                nq * hd,
                stream,
            )?;

            let k_contiguous = ctx.buffers.ssm_qkvz();
            let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);
            ops::mla_kv_assemble_batched(
                ctx.gpu,
                self.mla_kv_assemble_batched_k,
                kv_expanded,
                k_rope_buf,
                k_contiguous,
                v_contiguous,
                n,
                nkv,
                mla_nope,
                mla_v_dim,
                mla_rope,
                hd,
                nkv * (mla_nope + mla_v_dim),
                stream,
            )?;

            let mla_k_cache = ctx.buffers.expert_down_out();
            let mla_v_cache = mla_k_cache.offset(num_tokens * mla_cache_dim as usize * bf16);
            ops::mla_cache_assemble_batched(
                ctx.gpu,
                self.mla_cache_assemble_batched_k,
                kv_latent,
                k_rope_buf,
                mla_k_cache,
                mla_v_cache,
                n,
                kv_lora,
                mla_rope,
                mla_cache_dim,
                stream,
            )?;
            let meta = ctx.attn_metadata.expect("MLA prefill requires slot info");
            self.write_kv_cache(
                ctx.gpu,
                mla_k_cache,
                mla_v_cache,
                kv_cache,
                meta.slot,
                n,
                1,
                mla_cache_dim,
                bs,
                mla_cache_dim,
                mla_cache_dim,
                stream,
                ctx.graph_capture,
            )?;

            let attn_out = ctx.buffers.attn_output();
            let inv_sqrt_d = self.effective_attn_scale(hd);
            // For MLA unabsorbed path hd=128; HDIM=256 kernel reads K[k+1][0..127]
            // for d>=128, contaminating scores. Require the correct HDIM=128 kernel.
            anyhow::ensure!(
                hd > 128 || self.prefill_attn_128_k.0 != 0,
                "MLA paged prefill (first chunk): head_dim={hd} requires \
                 inferspark_prefill_hd128 (HDIM=256 over-reads adjacent K heads for hd<=128)",
            );
            let prefill_k = if hd > 256 && self.prefill_attn_512_k.0 != 0 {
                self.prefill_attn_512_k
            } else if hd <= 128 {
                self.prefill_attn_128_k
            } else {
                self.prefill_attn_k
            };
            ops::prefill_attention(
                ctx.gpu,
                prefill_k,
                qg_out,
                k_contiguous,
                v_contiguous,
                attn_out,
                n,
                1,
                nq,
                nkv,
                hd,
                inv_sqrt_d,
                true,
                self.sliding_window.unwrap_or(0),
                stream,
            )?;

            // O projection: [N, nq*hd] → [N, H]
            let o_out = ctx.buffers.norm_output();
            if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
                ops::w4a16_gemm(
                    ctx.gpu,
                    self.w4a16_gemm_k,
                    attn_out,
                    wo_nvfp4,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    attn_out,
                    &mla.wo,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
            return Ok(o_out);
        }

        // ════════════════════════════════════════════════════════════════════════
        // MULTI-CHUNK PATH (seq_len_start > 0): absorbed form with paged KV.
        //
        // Previous chunks have already written seq_len_start tokens into the paged
        // KV cache.  The current chunk (n tokens) must attend to the full context
        // (kv_len = seq_len_start + n tokens) using the compressed [kv_lora|rope]=320
        // dim cache format.
        //
        // Buffer plan (non-overlapping within this path):
        //   ssm_deinterleaved  → Q_absorbed [N, nq*kv_lora]  (before k_rope_buf aliases ssm_ba)
        //   ssm_conv_out_f32   → Q_rope     [N, nq*rope]     (post-RoPE)
        //   expert_down_out    → mla_k_cache / mla_v_cache   (compressed cache to write)
        //   attn_output        → Q_final    [N, nq, 320]     (assembled absorbed Q)
        //   ssm_deinterleaved  → attn_out   [N, nq, 320]     (paged attention output, reuse)
        //   attn_output        → v_extracted[N, nq, v_dim]   (V extraction output, reuse)
        //   norm_output        → o_out      [N, H]
        // ════════════════════════════════════════════════════════════════════════

        // ── Step A: Q_absorbed = q_latent @ w_qk_absorbed^T → ssm_deinterleaved ──
        // Must happen BEFORE k_rope_buf aliases ssm_ba and overwrites q_latent.
        // ssm_deinterleaved is free here: wkv_b is skipped for the absorbed path.
        let q_absorbed = ctx.buffers.ssm_deinterleaved();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            q_latent,
            &mla.w_qk_absorbed,
            q_absorbed,
            n,
            nq * kv_lora,
            q_lora,
            stream,
        )?;

        // ── Step B: K_rope (aliases ssm_ba, overwriting q_latent — OK, consumed above) ──
        let k_rope_buf = ctx.buffers.ssm_ba();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &mla.wkv_a_rope,
            k_rope_buf,
            n,
            mla_rope,
            h,
            stream,
        )?;

        // ── Step C: Extract Q_rope and apply RoPE ──
        let q_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_batched_k,
            qg_out,
            q_rope_tmp,
            n,
            nq,
            hd,
            mla_nope,
            mla_rope,
            nq * hd,
            stream,
        )?;
        let rope_meta = ctx.attn_metadata.expect("MLA prefill requires metadata");
        ops::rope_yarn(
            ctx.gpu,
            self.rope_yarn_k,
            q_rope_tmp,
            k_rope_buf,
            rope_meta.positions,
            n,
            nq,
            1,
            mla_rope,
            mla_rope,
            mla.yarn_inv_freq,
            ctx.config.rope_theta as f32,
            stream,
        )?;
        // Q_rope is now in q_rope_tmp [N, nq*rope]; no writeback to qg_out needed
        // (qg_out not used for attention in this path).

        // ── Step D: Write compressed KV cache ──
        let mla_k_cache = ctx.buffers.expert_down_out();
        let mla_v_cache = mla_k_cache.offset(num_tokens * mla_cache_dim as usize * bf16);
        ops::mla_cache_assemble_batched(
            ctx.gpu,
            self.mla_cache_assemble_batched_k,
            kv_latent,
            k_rope_buf,
            mla_k_cache,
            mla_v_cache,
            n,
            kv_lora,
            mla_rope,
            mla_cache_dim,
            stream,
        )?;
        let meta = ctx.attn_metadata.expect("MLA prefill requires slot info");
        self.write_kv_cache(
            ctx.gpu,
            mla_k_cache,
            mla_v_cache,
            kv_cache,
            meta.slot,
            n,
            1,
            mla_cache_dim,
            bs,
            mla_cache_dim,
            mla_cache_dim,
            stream,
            ctx.graph_capture,
        )?;

        // ── Step E: Assemble Q_final [N, nq, mla_cache_dim] in attn_output ──
        // q_absorbed = ssm_deinterleaved [N, nq*kv_lora]
        // q_rope     = ssm_conv_out_f32  [N, nq*rope]
        // q_final    = attn_output       [N, nq*mla_cache_dim]
        let q_final = ctx.buffers.attn_output();
        ops::mla_q_final_assemble_batched(
            ctx.gpu,
            self.mla_q_final_assemble_k,
            q_absorbed,
            q_rope_tmp,
            q_final,
            n,
            nq,
            kv_lora,
            mla_rope,
            mla_cache_dim,
            stream,
        )?;

        // ── Step F: Paged MLA prefill attention ──
        // Q = attn_output [N, nq, 320], KV from paged cache.
        // Output → ssm_deinterleaved [N, nq, 320] (q_absorbed buffer, now free).
        // Causal: Q[i] at global pos (seq_len_start + i) attends to KV 0..=seq_len_start+i.
        let attn_out = ctx.buffers.ssm_deinterleaved();
        let inv_sqrt_d = 1.0f32 / (mla_cache_dim as f32).sqrt();
        ops::mla_prefill_paged_320(
            ctx.gpu,
            self.mla_prefill_paged_k,
            q_final,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            attn_out,
            meta.block_table,
            n,
            kv_len as u32,
            seq_len_start as u32,
            nq,
            1, // num_kv_heads = 1 (MQA compressed cache)
            mla_cache_dim,
            bs,
            inv_sqrt_d,
            stream,
        )?;

        // ── Step G: V extraction — [N, nq, 320] → [N, nq, v_dim] ──
        // attn_out (ssm_deinterleaved) has absorbed attention output [N, nq, 320].
        // Only the first kv_lora=256 dims per head feed into V extraction.
        // Output → attn_output (q_final buffer, now free).
        let v_extracted = ctx.buffers.attn_output();
        ops::mla_v_extract_batched(
            ctx.gpu,
            self.mla_v_extract_batched_k,
            attn_out,
            mla.w_uv.weight,
            v_extracted,
            mla_v_dim,
            kv_lora,
            nq,
            mla_cache_dim, // input_head_stride: 320 (first kv_lora=256 dims used)
            mla_v_dim,     // output_head_stride: 128
            n,
            stream,
        )?;

        // ── Step H: O projection — [N, nq*v_dim] → [N, H] ──
        let o_out = ctx.buffers.norm_output();
        if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                v_extracted,
                wo_nvfp4,
                o_out,
                n,
                h,
                nq * mla_v_dim,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                v_extracted,
                &mla.wo,
                o_out,
                n,
                h,
                nq * mla_v_dim,
                stream,
            )?;
        }
        Ok(o_out)
    }
}
