// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash prefill path using standard GQA FlashAttention.
//! Isolated to V4-Flash (o_lora_rank > 0); no other models reach this code.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    pub(super) fn prefill_attention_cache_skip_v4(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &super::cache_skip_mla::CacheSkipMlaArgs,
    ) -> Result<DevicePtr> {
        let super::cache_skip_mla::CacheSkipMlaArgs {
            normed,
            num_tokens: _,
            n,
            h,
            nq,
            nkv,
            hd: _,
            kv_dim: _,
            eps,
            bf16: _,
            stream,
        } = *args;
        let mla = self.mla.as_ref().expect("V4-Flash prefill requires MLA");
        let meta = ctx
            .attn_metadata
            .expect("V4-Flash prefill requires metadata");

        let nope = mla.nope as u32;
        let rope = mla.rope as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let _v_dim = mla.v_dim as u32;
        let q_lora = mla.q_lora_rank as u32;
        let o_lora = mla.o_lora_rank as u32;
        let _mla_cache_dim = kv_lora + rope;
        let hd_mla = nope + rope;
        let use_tc = self.dense_gemm_tc_k.0 != 0;

        // ── 1. Q latent → norm → expand ──
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
        ctx.gpu.synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: q_latent gemm sync failed: {e}"))?;
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
        ctx.gpu.synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: q_a_norm sync failed: {e}"))?;
        let q_full = ctx.buffers.qkv_output();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            q_latent,
            &mla.wq_b,
            q_full,
            n,
            nq * hd_mla,
            q_lora,
            stream,
        )?;
        ctx.gpu.synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: q_full gemm sync failed: {e}"))?;

        // ── 2. Direct KV projection (V4-Flash: K=V, no absorption) ──
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
        ctx.gpu.synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: kv_latent gemm sync failed: {e}"))?;
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
        ctx.gpu.synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: kv_a_norm sync failed: {e}"))?;

        // ── 3. RoPE on Q and K/V ──
        ops::rope_yarn(
            ctx.gpu,
            self.rope_yarn_k,
            q_full,
            kv_latent,
            meta.positions,
            n,
            nq,
            nkv,
            hd_mla,
            rope,
            mla.yarn_inv_freq,
            ctx.config.rope_theta as f32,
            stream,
        )?;
        ctx.gpu.synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: rope_yarn sync failed: {e}"))?;

        // ── 4. Standard GQA FlashAttention (intra-chunk) ──
        let attn_out = ctx.buffers.attn_output();
        ops::prefill_attention_64(
            ctx.gpu,
            self.prefill_attn_64_k,
            q_full,
            kv_latent,
            kv_latent,
            attn_out,
            n,
            1,
            nq,
            nkv,
            hd_mla,
            1.0f32 / (hd_mla as f32).sqrt(),
            true,
            0,
            stream,
        )
        .map_err(|e| anyhow::anyhow!("V4 attn: prefill_attention_64 failed: {e}"))?;
        ctx.gpu.synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: prefill_attention_64 sync failed: {e}"))?;

        // ── 5. Write KV cache (V4-Flash: direct K/V, no assembly) ──
        self.write_kv_cache(
            ctx.gpu,
            kv_latent,
            kv_latent,
            kv_cache,
            meta.slot,
            n,
            1,
            hd_mla,
            kv_cache.block_size() as u32,
            hd_mla,
            hd_mla,
            stream,
            ctx.graph_capture,
        )?;
        ctx.gpu.synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: write_kv_cache sync failed: {e}"))?;

        // ── 6. Grouped low-rank O projection ──
        let o_latent = ctx.buffers.norm_output();
        let o_out = ctx.buffers.qkv_output();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            attn_out,
            &mla.wo_a,
            o_latent,
            n,
            o_lora,
            nq * hd_mla,
            stream,
        )?;
        ctx.gpu.synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: wo_a gemm sync failed: {e}"))?;
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            o_latent,
            &mla.wo_b,
            o_out,
            n,
            h,
            o_lora,
            stream,
        )?;
        ctx.gpu.synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: wo_b gemm sync failed: {e}"))?;

        Ok(o_out)
    }
}
