// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash prefill path. Reuses low-rank Q projection
//! (wq_a→norm→wq_b) from the MLA path, but uses direct KV projection
//! (no absorption) and grouped low-rank O projection (wo_a→wo_b).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// Run the DeepSeek-V4-Flash prefill chain. Returns the output pointer.
    pub(super) fn prefill_attention_cache_skip_v4(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &super::cache_skip_mla::CacheSkipMlaArgs,
    ) -> Result<DevicePtr> {
        let super::cache_skip_mla::CacheSkipMlaArgs {
            normed,
            num_tokens,
            n,
            h,
            nq,
            nkv,
            hd,
            kv_dim: _,
            eps,
            bf16,
            stream,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("prefill_attention_cache_skip_v4 called without MLA config");
        let meta = ctx
            .attn_metadata
            .expect("V4-Flash prefill requires metadata");

        let q_lora = mla.q_lora_rank as u32;
        let mla_rope = mla.rope as u32;
        let o_lora = mla.o_lora_rank as u32;
        let use_tc = self.dense_gemm_tc_k.0 != 0;

        // ── 1. Q: latent → norm → expand ──
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

        // ── 2. Direct KV projection ──
        let k_contiguous = ctx.buffers.ssm_qkvz();
        let v_contiguous = k_contiguous.offset((num_tokens * (nkv * hd) as usize * bf16) as u64);
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &mla.wkv_a,
            k_contiguous,
            n,
            nkv * hd,
            h,
            stream,
        )?;
        // K=V: copy K to V
        ctx.gpu.copy_d2d_async(
            k_contiguous,
            v_contiguous,
            (num_tokens * (nkv * hd) as usize) * bf16,
            stream,
        )?;

        // ── 3. RoPE ──
        ops::rope_yarn(
            ctx.gpu,
            self.rope_yarn_k,
            qg_out,
            k_contiguous,
            meta.positions,
            n,
            nq,
            nkv,
            hd,
            mla_rope,
            mla.yarn_inv_freq,
            ctx.config.rope_theta as f32,
            stream,
        )?;

        // ── 4. Write K/V to paged cache ──
        let write_start = 0usize;
        let write_count = num_tokens;
        if write_count > 0 {
            let k_offset = write_start * (nkv * hd) as usize * bf16;
            let v_offset = write_start * (nkv * hd) as usize * bf16;
            let slot_offset = write_start * 8;
            self.write_kv_cache(
                ctx.gpu,
                k_contiguous.offset(k_offset as u64),
                v_contiguous.offset(v_offset as u64),
                kv_cache,
                meta.slot.offset(slot_offset as u64),
                write_count as u32,
                nkv,
                hd,
                kv_cache.block_size() as u32,
                nkv * hd,
                nkv * hd,
                stream,
                ctx.graph_capture,
            )?;
        }

        // ── 5. Flash Attention ──
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        let prefill_k = if hd > 256 && self.prefill_attn_512_k.0 != 0 {
            self.prefill_attn_512_k
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
        )
        .map_err(|e| anyhow::anyhow!("V4-Flash flash_attn: {e}"))?;

        // ── 6. Grouped low-rank O projection (wo_a → wo_b) ──
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
            nq * hd,
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
