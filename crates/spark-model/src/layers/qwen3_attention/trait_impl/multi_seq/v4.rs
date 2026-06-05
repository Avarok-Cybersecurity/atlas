// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash branch of multi-sequence batched decode.
//! Per-sequence loop reusing the same scratch buffers; mirrors
//! `mla.rs::ms_mla_decode` but with direct KV + low-rank O.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::ctx::MultiSeqCtx;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// Batched V4-Flash decode for `c.n` sequences.
    pub(super) fn ms_v4_decode(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<DevicePtr> {
        let mla = self
            .mla
            .as_ref()
            .expect("ms_v4_decode called without MLA config");

        let h = c.h as u32;
        let nq = c.nq;
        let nkv = c.nkv;
        let hd = c.hd;
        let eps = c.eps;
        let bf16 = c.bf16;
        let stream = c.stream;
        let bs = c.bs as usize;

        let q_lora = mla.q_lora_rank as u32;
        let mla_rope = mla.rope as u32;
        let o_lora = mla.o_lora_rank as u32;
        let q_dim = nq * hd;
        let inv_sqrt_d = self.effective_attn_scale(hd);

        let o_out = c.fwd.buffers.moe_output();

        for i in 0..c.n {
            let normed_i = c.normed.offset(i * c.h * bf16);
            let meta_i = AttnMetadataDev {
                positions: meta.positions.offset(i * 4),
                positions_h: meta.positions_h.offset(i * 4),
                positions_w: meta.positions_w.offset(i * 4),
                slot: meta.slot.offset(i * 8),
                seq_len: meta.seq_len.offset(i * 4),
                block_table: meta
                    .block_table
                    .offset(i * meta.max_blocks_per_seq as usize * 4),
                max_blocks_per_seq: meta.max_blocks_per_seq,
                num_seqs: 1,
            };
            let o_out_i = o_out.offset(i * c.h * bf16);
            self.ms_v4_decode_one(
                c, kv_cache, &meta_i, normed_i, o_out_i, mla, stream, h, nq, hd, q_lora, mla_rope,
                o_lora, q_dim, nkv, eps, bs, inv_sqrt_d,
            )?;
        }
        Ok(o_out)
    }

    #[allow(clippy::too_many_arguments)]
    fn ms_v4_decode_one(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: &AttnMetadataDev,
        normed: DevicePtr,
        o_out: DevicePtr,
        mla: &crate::layers::qwen3_attention::types::MlaWeights,
        stream: u64,
        h: u32,
        nq: u32,
        nkv: u32,
        hd: u32,
        q_lora: u32,
        mla_rope: u32,
        o_lora: u32,
        q_dim: u32,
        eps: f32,
        bs: usize,
        inv_sqrt_d: f32,
    ) -> Result<()> {
        let gpu = c.fwd.gpu;
        let buffers = c.fwd.buffers;

        // ── Step 1: Q latent → norm → expand ──
        let q_latent = buffers.ssm_ba();
        if let Some(ref wqa_nvfp4) = mla.wq_a_nvfp4 {
            ops::w4a16_gemv(
                gpu,
                self.w4a16_gemv_k,
                normed,
                wqa_nvfp4,
                q_latent,
                q_lora,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                normed,
                &mla.wq_a,
                q_latent,
                q_lora,
                h,
                stream,
            )?;
        }
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            q_latent,
            &mla.q_a_norm,
            q_latent,
            1,
            q_lora,
            eps,
            stream,
        )?;

        let q_out = buffers.qkv_output();
        if let Some(ref wqb_nvfp4) = mla.wq_b_nvfp4 {
            ops::w4a16_gemv(
                gpu,
                self.w4a16_gemv_k,
                q_latent,
                wqb_nvfp4,
                q_out,
                q_dim,
                q_lora,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                q_latent,
                &mla.wq_b,
                q_out,
                q_dim,
                q_lora,
                stream,
            )?;
        }

        // ── Step 2: Direct KV projection ──
        let k_out = q_out.offset((q_dim as usize) * 2);
        let v_out = k_out.offset(((nkv * hd) as usize) * 2);
        if let Some(ref wkva_nvfp4) = mla.wkv_a_nvfp4 {
            ops::w4a16_gemv(
                gpu,
                self.w4a16_gemv_k,
                normed,
                wkva_nvfp4,
                k_out,
                nkv * hd,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                normed,
                &mla.wkv_a,
                k_out,
                nkv * hd,
                h,
                stream,
            )?;
        }
        gpu.copy_d2d_async(k_out, v_out, ((nkv * hd) as usize) * 2, stream)?;

        // ── Step 3: RoPE ──
        ops::rope_yarn(
            gpu,
            self.rope_yarn_k,
            q_out,
            k_out,
            meta.positions,
            1,
            nq,
            nkv,
            hd,
            mla_rope,
            mla.yarn_inv_freq,
            c.fwd.config.rope_theta as f32,
            stream,
        )?;

        // ── Step 4: Write K/V to cache ──
        self.write_kv_cache(
            gpu,
            k_out,
            v_out,
            kv_cache,
            meta.slot,
            1,
            nkv,
            hd,
            bs as u32,
            nkv * hd,
            nkv * hd,
            stream,
            c.fwd.graph_capture,
        )?;

        // ── Step 5: Paged decode ──
        let attn_out = buffers.attn_output();
        self.run_paged_decode(
            gpu,
            q_out,
            kv_cache,
            attn_out,
            meta.block_table,
            meta.seq_len,
            meta.max_blocks_per_seq,
            1,
            nq,
            nkv,
            hd,
            bs as u32,
            inv_sqrt_d,
            nq * hd,
            buffers.splitk_workspace(),
            stream,
        )?;

        // ── Step 6: Low-rank O projection (wo_a → wo_b) ──
        let o_latent = buffers.ssm_qkvz();
        ops::dense_gemv(
            gpu,
            self.dense_gemv_k,
            attn_out,
            &mla.wo_a,
            o_latent,
            o_lora,
            nq * hd,
            stream,
        )?;
        ops::dense_gemv(
            gpu,
            self.dense_gemv_k,
            o_latent,
            &mla.wo_b,
            o_out,
            h,
            o_lora,
            stream,
        )?;

        Ok(())
    }
}
