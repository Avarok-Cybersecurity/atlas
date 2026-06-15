// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash chunk-1+ prefill using standard GQA FlashAttention.
//! Isolated to V4-Flash (o_lora_rank > 0); no other models reach this code.
//! NOTE: like paged_mla.rs, this only attends within the current chunk.
//! Full paged-cache attention for chunk-1+ requires a paged MLA kernel
//! that is not yet implemented.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use super::paged_mla::MlaPrefillArgs;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    pub(super) fn prefill_attention_paged_v4(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &MlaPrefillArgs,
        _seq_len_start: usize,
    ) -> Result<DevicePtr> {
        let MlaPrefillArgs {
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
            bs,
            stream,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("V4-Flash paged prefill requires MLA");
        let meta = ctx
            .attn_metadata
            .expect("V4-Flash paged prefill requires metadata");

        let nope = mla.nope as u32;
        let rope = mla.rope as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let _v_dim = mla.v_dim as u32;
        let q_lora = mla.q_lora_rank as u32;
        let o_lora = mla.o_lora_rank as u32;
        let _mla_cache_dim = kv_lora + rope;
        let hd_mla = nope + rope;

        // ── 1. Q latent → norm → expand ──
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

        // ── 2. Direct KV projection (V4-Flash: K=V, no absorption) ──
        // Layout in qkv_output: [Q | K | V]  (mirrors decode path)
        let q_dim = nq * hd_mla;
        let kv_dim = nkv * hd_mla;
        let k_out = q_full.offset((n * q_dim) as usize * 2);
        let v_out = k_out.offset((n * kv_dim) as usize * 2);
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &mla.wkv_a,
            k_out,
            n,
            kv_lora,
            h,
            stream,
        )?;
        // Copy K → V (V4-Flash: K and V share the same projection output)
        ctx.gpu
            .copy_d2d_async(k_out, v_out, (n * kv_dim) as usize * 2, stream)?;

        // ── 3. RoPE on Q and K (V is NOT RoPE'd) ──
        ops::rope_yarn(
            ctx.gpu,
            self.rope_yarn_k,
            q_full,
            k_out,
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

        // ── 4. Standard GQA FlashAttention (current chunk only) ──
        let attn_out = ctx.buffers.attn_output();
        let prefill_k = if hd_mla > 256 && self.prefill_attn_512_k.0 != 0 {
            tracing::info!(
                "V4-Flash paged prefill: using prefill_attn_512_k (hd_mla={})",
                hd_mla
            );
            self.prefill_attn_512_k
        } else {
            tracing::info!(
                "V4-Flash paged prefill: using prefill_attn_64_k fallback (hd_mla={}, 512_k loaded={})",
                hd_mla,
                self.prefill_attn_512_k.0 != 0
            );
            self.prefill_attn_64_k
        };
        ops::prefill_attention(
            ctx.gpu,
            prefill_k,
            q_full,
            k_out,
            v_out,
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
        .map_err(|e| anyhow::anyhow!("V4 paged: prefill_attention failed: {e}"))?;

        // ── 5. Write KV cache (V4-Flash: direct K/V, no assembly) ──
        self.write_kv_cache(
            ctx.gpu,
            k_out,
            v_out,
            kv_cache,
            meta.slot,
            n,
            1,
            hd_mla,
            bs,
            kv_dim,
            kv_dim,
            stream,
            ctx.graph_capture,
        )?;

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

        Ok(o_out)
    }
}
