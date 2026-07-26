// SPDX-License-Identifier: AGPL-3.0-only

//! Dual-token (K=2 MTP verify) DeepSeek-V4-Flash multi-seq decode.
//!
//! Amortizes NVFP4/FP8 GEMV weight traffic across both verify tokens:
//!   - w4a16_gemv_batch2 for NVFP4 wq_a / wq_b / wkv
//!   - w8a16_gemv_batch2 (batch4 M=2) for FP8 wo_b
//!   - batched rope extract/writeback + cache assemble (count=2)
//!
//! Q/O multi-seq: `mla_paged_decode_fp8` indexes
//! `Q/O[(seq_idx*nq+head)*hd]` (patched 2026-07-24). Pre-patch kernels
//! ignored seq_idx → token salad with num_seqs=2.
//!
//! Opt out of this whole path: unset ATLAS_V4_MS_N2 or set ATLAS_V4_MS_SEQ=1.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::ctx::MultiSeqCtx;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::layers::qwen3_attention::types::MlaWeights;

impl Qwen3AttentionLayer {
    /// Process both K=2 verify tokens with dual-token GEMV amortize.
    /// Writes `o_out[0..H]` and `o_out[H..2H]` (BF16).
    pub(super) fn ms_v4_flash_n2(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
        mla: &MlaWeights,
        o_out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let gpu = c.fwd.gpu;
        let h = c.h as u32;
        let nq = c.nq;
        let nkv = c.nkv;
        let hd = c.hd;
        let eps = c.eps;
        let bf16 = c.bf16;
        let bs = c.bs;
        let n = 2u32;
        let q_lora = mla.q_lora_rank as u32;
        let mla_rope = mla.rope as u32;
        let kv_dim = nkv * hd;
        let q_dim = nq * hd;
        let inv_sqrt_d = self.effective_attn_scale(hd);
        let substage_profile =
            std::env::var("ATLAS_PROFILE").is_ok_and(|v| v == "1" || v == "true");
        let mut substage_started = None;
        if substage_profile {
            gpu.synchronize(stream)?;
            substage_started = Some(std::time::Instant::now());
        }
        macro_rules! mla_stage {
            ($label:expr) => {
                if let Some(started) = substage_started.as_mut() {
                    gpu.synchronize(stream)?;
                    tracing::info!(
                        "K2_MLA_STAGE layer={} {} {:.3}ms",
                        self.attn_layer_idx,
                        $label,
                        started.elapsed().as_secs_f64() * 1000.0
                    );
                    *started = std::time::Instant::now();
                }
            };
        }

        // ── Buffers (packed [2, …] for batch2 GEMVs) ──
        // q_latent [2, q_lora] — ssm_ba sized for max_batch_tokens * q_lora
        let q_latent = c.fwd.buffers.ssm_ba();
        // q_out / k_out / v_out packed in ssm_qkvz: [Q0|Q1|K0|K1|V0|V1]
        let q_out = c.fwd.buffers.ssm_qkvz();
        let k_out = q_out.offset(2 * q_dim as usize * bf16);
        let v_out = k_out.offset(2 * kv_dim as usize * bf16);

        // ── 1. wq_a batch2 (NVFP4 first — matches attention_forward_v4) ──
        if let Some(ref wqa_nvfp4) = mla.wq_a_nvfp4 {
            ops::w4a16_gemv_batch2(
                gpu,
                self.w4a16_gemv_batch2_k,
                c.normed,
                wqa_nvfp4,
                q_latent,
                q_lora,
                h,
                stream,
            )?;
        } else if let Some(ref wqa) = mla.wq_a_fp8 {
            ops::w8a16_gemv_batch2(
                gpu,
                self.w8a16_gemv_batch4_k,
                c.normed,
                wqa.weight,
                wqa.row_scale,
                q_latent,
                q_lora,
                h,
                stream,
            )?;
        } else {
            for i in 0..2usize {
                ops::dense_gemv(
                    gpu,
                    self.dense_gemv_k,
                    c.normed.offset(i * c.h * bf16),
                    &mla.wq_a,
                    q_latent.offset(i * q_lora as usize * bf16),
                    q_lora,
                    h,
                    stream,
                )?;
            }
        }

        // ── 2. q_a_norm (n=2 rows) ──
        ops::rms_norm(
            gpu,
            self.rms_norm_w_k,
            q_latent,
            &mla.q_a_norm,
            q_latent,
            n,
            q_lora,
            eps,
            stream,
        )?;
        mla_stage!("wqa_norm");

        // ── 3. wq_b batch2 (NVFP4 first) ──
        if let Some(ref wqb_nvfp4) = mla.wq_b_nvfp4 {
            ops::w4a16_gemv_batch2(
                gpu,
                self.w4a16_gemv_batch2_k,
                q_latent,
                wqb_nvfp4,
                q_out,
                q_dim,
                q_lora,
                stream,
            )?;
        } else if let Some(ref wqb) = mla.wq_b_fp8 {
            ops::w8a16_gemv_batch2(
                gpu,
                self.w8a16_gemv_batch4_k,
                q_latent,
                wqb.weight,
                wqb.row_scale,
                q_out,
                q_dim,
                q_lora,
                stream,
            )?;
        } else {
            for i in 0..2usize {
                ops::dense_gemv(
                    gpu,
                    self.dense_gemv_k,
                    q_latent.offset(i * q_lora as usize * bf16),
                    &mla.wq_b,
                    q_out.offset(i * q_dim as usize * bf16),
                    q_dim,
                    q_lora,
                    stream,
                )?;
            }
        }

        // ── 4. q_b_norm: per-head unweighted RMS over (n*nq) heads ──
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            q_out,
            &crate::weight_map::DenseWeight {
                weight: c.fwd.buffers.norm_unit_w(),
            },
            q_out,
            n * nq,
            hd,
            eps,
            stream,
        )?;
        mla_stage!("wqb_norm");

        // ── 5. wkv batch2 (NVFP4 first) ──
        if let Some(ref wkva_nvfp4) = mla.wkv_a_nvfp4 {
            ops::w4a16_gemv_batch2(
                gpu,
                self.w4a16_gemv_batch2_k,
                c.normed,
                wkva_nvfp4,
                k_out,
                kv_dim,
                h,
                stream,
            )?;
        } else if let Some(ref wkv) = mla.wkv_a_fp8 {
            ops::w8a16_gemv_batch2(
                gpu,
                self.w8a16_gemv_batch4_k,
                c.normed,
                wkv.weight,
                wkv.row_scale,
                k_out,
                kv_dim,
                h,
                stream,
            )?;
        } else {
            for i in 0..2usize {
                ops::dense_gemv(
                    gpu,
                    self.dense_gemv_k,
                    c.normed.offset(i * c.h * bf16),
                    &mla.wkv_a,
                    k_out.offset(i * kv_dim as usize * bf16),
                    kv_dim,
                    h,
                    stream,
                )?;
            }
        }

        // kv_norm: nkv heads per token → n*nkv heads
        ops::rms_norm(
            gpu,
            self.rms_norm_w_k,
            k_out,
            &mla.kv_a_norm,
            k_out,
            n * nkv,
            kv_dim / nkv,
            eps,
            stream,
        )?;
        // K=V latent copy BEFORE rope writeback (SSOT: cache assemble uses v_out)
        gpu.copy_d2d_async(k_out, v_out, 2 * kv_dim as usize * bf16, stream)?;
        mla_stage!("wkv_norm_copy");

        // ── 6. RoPE (batched count=2) ──
        let q_rope_tmp = c.fwd.buffers.ssm_conv_out_f32();
        // k_rope reuses q_latent region after wq_b done
        let k_rope_tmp = q_latent;
        ops::mla_q_rope_extract_batched(
            gpu,
            self.mla_q_rope_extract_batched_k,
            q_out,
            q_rope_tmp,
            n,
            nq,
            hd,
            mla.nope as u32,
            mla_rope,
            nq * hd,
            stream,
        )?;
        ops::mla_q_rope_extract_batched(
            gpu,
            self.mla_q_rope_extract_batched_k,
            k_out,
            k_rope_tmp,
            n,
            1,
            hd,
            mla.nope as u32,
            mla_rope,
            hd,
            stream,
        )?;
        let inv_freq = if mla.compressor.is_none() {
            mla.main_inv_freq
        } else {
            mla.yarn_inv_freq
        };
        let mscale = if mla.compressor.is_none() {
            1.0f32
        } else {
            super::super::super::helpers::yarn_rope_mscale(c.fwd.config)
        };
        ops::rope_yarn(
            gpu,
            self.rope_yarn_interleaved_k,
            q_rope_tmp,
            k_rope_tmp,
            meta.positions,
            n,
            nq,
            1,
            mla_rope,
            mla_rope,
            inv_freq,
            mscale,
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            gpu,
            self.mla_q_rope_writeback_batched_k,
            q_rope_tmp,
            q_out,
            n,
            nq,
            hd,
            mla.nope as u32,
            mla_rope,
            nq * hd,
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            gpu,
            self.mla_q_rope_writeback_batched_k,
            k_rope_tmp,
            k_out,
            n,
            1,
            hd,
            mla.nope as u32,
            mla_rope,
            hd,
            stream,
        )?;

        // ── 7. Cache assemble + write (n=2) ──
        // v_cache must NOT alias ssm_qkvz (holds Q/K/V packing). Use MoE scratch.
        let k_cache = c.fwd.buffers.ssm_deinterleaved();
        let v_cache = c.fwd.buffers.expert_up_out();
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_cache_dim = kv_lora + mla_rope;
        ops::mla_cache_assemble_batched(
            gpu,
            self.mla_cache_assemble_batched_k,
            v_out,
            k_rope_tmp,
            k_cache,
            v_cache,
            n,
            kv_lora,
            mla_rope,
            mla_cache_dim,
            stream,
        )?;
        self.write_kv_cache(
            gpu,
            k_cache,
            v_cache,
            kv_cache,
            meta.slot,
            n,
            1,
            mla_cache_dim,
            bs,
            mla_cache_dim,
            mla_cache_dim,
            stream,
            c.fwd.graph_capture,
        )?;
        mla_stage!("rope_cache_write");

        // ── 8. Paged decode num_seqs=2 (Q/O multi-seq fixed in kernel) ──
        let attn_out = c.fwd.buffers.attn_output();
        self.run_paged_decode(
            gpu,
            q_out,
            kv_cache,
            attn_out,
            meta.block_table,
            meta.seq_len,
            meta.max_blocks_per_seq,
            n,
            nq,
            nkv,
            hd,
            bs,
            inv_sqrt_d,
            nq * hd,
            c.fwd.buffers.splitk_workspace(),
            stream,
        )?;
        mla_stage!("paged_attn");

        // ── 9. De-rotate attn output (count=2) ──
        {
            let o_rope_tmp = c.fwd.buffers.ssm_conv_out_f32();
            ops::mla_q_rope_extract_batched(
                gpu,
                self.mla_q_rope_extract_batched_k,
                attn_out,
                o_rope_tmp,
                n,
                nq,
                hd,
                mla.nope as u32,
                mla_rope,
                nq * hd,
                stream,
            )?;
            ops::rope_yarn(
                gpu,
                self.rope_yarn_interleaved_inv_k,
                o_rope_tmp,
                o_rope_tmp,
                meta.positions,
                n,
                nq,
                0,
                mla_rope,
                mla_rope,
                inv_freq,
                mscale,
                stream,
            )?;
            ops::mla_q_rope_writeback_batched(
                gpu,
                self.mla_q_rope_writeback_batched_k,
                o_rope_tmp,
                attn_out,
                n,
                nq,
                hd,
                mla.nope as u32,
                mla_rope,
                nq * hd,
                stream,
            )?;
        }
        mla_stage!("derope");

        self.ms_v4_flash_n2_output(c, mla, attn_out, o_out, nq, hd, bf16, stream)?;
        mla_stage!("output_proj");

        Ok(())
    }
}
