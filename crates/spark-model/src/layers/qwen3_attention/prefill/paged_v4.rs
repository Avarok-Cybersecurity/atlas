// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash branch of `prefill_attention_paged`.
//! Direct KV projection + low-rank O, using the standard paged-attention
//! kernel for chunk-1+ prefill.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use super::paged_mla::MlaPrefillArgs;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// V4-Flash paged prefill for chunk-1+ (seq_len_start > 0).
    pub(super) fn prefill_attention_paged_v4(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &MlaPrefillArgs,
        seq_len_start: usize,
    ) -> Result<DevicePtr> {
        let MlaPrefillArgs {
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
            bs,
            stream,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("prefill_attention_paged_v4 called without MLA config");
        let meta = ctx
            .attn_metadata
            .expect("V4-Flash paged prefill requires metadata");

        let q_lora = mla.q_lora_rank as u32;
        let mla_rope = mla.rope as u32;
        let o_lora = mla.o_lora_rank as u32;

        // ── 1. Q: latent → norm → expand ──
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
        let q_contiguous = ctx.buffers.qkv_output();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            q_latent,
            &mla.wq_b,
            q_contiguous,
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
            q_contiguous,
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
        self.write_kv_cache(
            ctx.gpu,
            k_contiguous,
            v_contiguous,
            kv_cache,
            meta.slot,
            n,
            nkv,
            hd,
            bs,
            nkv * hd,
            nkv * hd,
            stream,
            ctx.graph_capture,
        )?;

        // ── 5. Paged Flash Attention ──
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        let kv_len = (seq_len_start + num_tokens) as u32;
        let empty_block_table = Vec::new();
        let mut empty_disk_block_ids = Vec::new();
        let mut empty_disk_last = Vec::new();
        let mut args = super::paged_attn::PagedAttnArgs {
            q_contiguous,
            k_contiguous,
            v_contiguous,
            attn_out,
            n,
            seq_len_start,
            num_tokens,
            nq,
            nkv,
            hd,
            bs,
            bf16,
            inv_sqrt_d,
            kv_len,
            meta: &meta,
            block_table: &empty_block_table,
            disk_block_ids: &mut empty_disk_block_ids,
            disk_last_offloaded_per_layer: &mut empty_disk_last,
            stream,
        };
        match self.prefill_attention_paged_attn(kv_cache, ctx, &mut args)? {
            super::paged_attn::PagedAttnOutcome::EarlyReturn(out) => return Ok(out),
            super::paged_attn::PagedAttnOutcome::Continue => {}
        }

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
