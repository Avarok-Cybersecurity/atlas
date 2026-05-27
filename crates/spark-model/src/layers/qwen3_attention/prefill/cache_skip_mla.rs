// SPDX-License-Identifier: AGPL-3.0-only

//! MLA branch of `prefill_attention_with_cache_skip`. Mistral4-style
//! absorbed MLA prefill: Q_absorption + causal attention + V_extraction
//! via `mla_fused_prefill` (HDIM=320 absorbed space). Extracted from
//! `cache_skip.rs` to keep that file under 500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

pub(super) struct CacheSkipMlaArgs {
    pub normed: DevicePtr,
    pub n: u32,
    pub h: u32,
    pub nq: u32,
    pub hd: u32,
    pub eps: f32,
    pub stream: u64,
    /// Number of token positions whose KV entries are already in the cache
    /// (prefix-cache hit). Only tokens `kv_write_start..n` need to be written.
    /// 0 = no cached prefix (all tokens are new).
    pub kv_write_start: usize,
}

impl Qwen3AttentionLayer {
    /// Run the cache-skip MLA prefill chain. Always returns the output
    /// pointer — caller short-circuits with `return Ok(out)`.
    pub(super) fn prefill_attention_cache_skip_mla(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &CacheSkipMlaArgs,
    ) -> Result<DevicePtr> {
        let CacheSkipMlaArgs { normed, n, h, nq, hd, eps, stream, kv_write_start } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("prefill_attention_cache_skip_mla called without MLA config");

        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_nope = mla.nope as u32;
        let mla_v_dim = mla.v_dim as u32;
        let mla_rope = mla.rope as u32;
        let use_tc = self.dense_gemm_tc_k.0 != 0;

        // Q: latent → norm → expand
        let q_latent = ctx.buffers.ssm_ba();
        if use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                &mla.wq_a,
                q_latent,
                n,
                q_lora,
                h,
                stream,
            )?;
        } else {
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
        }
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
        if use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                q_latent,
                &mla.wq_b,
                qg_out,
                n,
                nq * hd,
                q_lora,
                stream,
            )?;
        } else {
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
        }

        // KV latent + K_rope
        let kv_latent = ctx.buffers.expert_gate_out();
        if use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                &mla.wkv_a,
                kv_latent,
                n,
                kv_lora,
                h,
                stream,
            )?;
        } else {
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
        }
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
        let k_rope_buf = ctx.buffers.ssm_ba();
        if use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                &mla.wkv_a_rope,
                k_rope_buf,
                n,
                mla_rope,
                h,
                stream,
            )?;
        } else {
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
        }

        // Q rope extract → RoPE
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

        let mla_cache_dim = kv_lora + mla_rope;
        // Cache assembly (needed for decode regardless of path)
        let meta = ctx.attn_metadata.expect("MLA prefill requires metadata");
        let bs = kv_cache.block_size();
        let k_cache_assembled = ctx.buffers.expert_up_out();
        let v_cache_assembled = ctx.buffers.expert_down_out();
        ops::mla_cache_assemble_batched(
            ctx.gpu,
            self.mla_cache_assemble_batched_k,
            kv_latent,
            k_rope_buf,
            k_cache_assembled,
            v_cache_assembled,
            n,
            kv_lora,
            mla_rope,
            mla_cache_dim,
            stream,
        )?;
        // Only write the tokens that are NOT already in the cache.
        // kv_write_start tokens (prefix-cache hit) already have correct KV
        // entries at their physical slots; skip them to avoid redundant writes.
        // Mirror of the non-MLA `write_start` logic in cache_skip.rs.
        let write_count = (n as usize).saturating_sub(kv_write_start);
        if write_count > 0 {
            let bf16 = 2usize; // bytes per BF16 element
            let cache_elem_offset = kv_write_start * mla_cache_dim as usize;
            let slot_byte_offset = kv_write_start * 8; // 8 bytes per u64 slot entry
            self.write_kv_cache(
                ctx.gpu,
                k_cache_assembled.offset(cache_elem_offset * bf16),
                v_cache_assembled.offset(cache_elem_offset * bf16),
                kv_cache,
                meta.slot.offset(slot_byte_offset),
                write_count as u32,
                1,
                mla_cache_dim,
                bs as u32,
                mla_cache_dim,
                mla_cache_dim,
                stream,
                ctx.graph_capture,
            )?;
        }

        // MLA absorbed attention: fused Q_absorb + attention (320-dim) + V_extract.
        // inferspark_prefill_64 has compile-time HDIM=256; MLA kv_stride=nkv*hd=128 so
        // col>=128 aliases K[k+1][0..127] — corrupts attention scores over long contexts.
        // inv_sqrt_d: 1/sqrt(kv_lora + rope) = 1/sqrt(320) — absorbed dimension, NOT hd.
        // Using 1/sqrt(hd=128) would over-sharpen softmax by sqrt(128/320) ≈ 0.63.
        let attn_out_fb = ctx.buffers.attn_output();
        // inv_sqrt_d in the absorbed space: 1/sqrt(kv_lora + rope) = 1/sqrt(320).
        // Using 1/sqrt(hd=128) would over-sharpen softmax by sqrt(128/320) ≈ 0.63.
        let inv_sqrt_d_absorbed = 1.0f32 / ((kv_lora + mla_rope) as f32).sqrt();
        anyhow::ensure!(
            self.mla_fused_prefill_k.0 != 0,
            "MLA cache-skip prefill requires mla_fused_prefill kernel \
             (inferspark_prefill HDIM=256 is broken for MLA hd=128; \
              rebuild with kernels/gb10/mistral-small-4/nvfp4/mla_fused_prefill.cu)"
        );
        ops::mla_fused_prefill(
            ctx.gpu,
            self.mla_fused_prefill_k,
            qg_out,
            q_rope_tmp,
            kv_latent,
            k_rope_buf,
            mla.w_uk_t.weight,
            mla.w_uv.weight,
            attn_out_fb,
            DevicePtr::NULL,
            DevicePtr::NULL,
            n,
            nq,
            mla_nope,
            mla_rope,
            kv_lora,
            mla_v_dim,
            hd,
            inv_sqrt_d_absorbed,
            stream,
        )
        .map_err(|e| anyhow::anyhow!("MLA fused prefill: {e}"))?;
        // wo projection — output to qkv_output (norm_output aliases downstream)
        let o_out = ctx.buffers.qkv_output();
        if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                attn_out_fb,
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
                attn_out_fb,
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
