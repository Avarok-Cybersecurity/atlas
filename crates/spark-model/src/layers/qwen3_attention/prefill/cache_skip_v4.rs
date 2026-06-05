// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash prefill path using mla_fused_prefill kernel.
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
        let v_dim = mla.v_dim as u32;
        let q_lora = mla.q_lora_rank as u32;
        let o_lora = mla.o_lora_rank as u32;
        let mla_cache_dim = kv_lora + rope;
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

        // ── 2. Extract Q_rope ──
        let q_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_batched_k,
            q_full,
            q_rope_tmp,
            n,
            nq,
            hd_mla,
            nope,
            rope,
            nq * hd_mla,
            stream,
        )?;

        // ── 3. RoPE on Q_rope and K_rope ──
        ops::rope_yarn(
            ctx.gpu,
            self.rope_yarn_k,
            q_rope_tmp,
            q_rope_tmp,
            meta.positions,
            n,
            nq,
            1,
            rope,
            rope,
            mla.yarn_inv_freq,
            ctx.config.rope_theta as f32,
            stream,
        )?;

        let k_rope_buf = ctx.buffers.ssm_ba();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &mla.wkv_a_rope,
            k_rope_buf,
            n,
            rope,
            h,
            stream,
        )?;
        ops::rope_yarn(
            ctx.gpu,
            self.rope_yarn_k,
            q_rope_tmp,
            k_rope_buf,
            meta.positions,
            n,
            nq,
            1,
            rope,
            rope,
            mla.yarn_inv_freq,
            ctx.config.rope_theta as f32,
            stream,
        )?;

        // ── 4. KV latent ──
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

        // ── 5. Fused MLA prefill ──
        let v_out = ctx.buffers.attn_output();
        ops::mla_fused_prefill(
            ctx.gpu,
            self.mla_fused_prefill_k,
            q_full,
            q_rope_tmp,
            kv_latent,
            k_rope_buf,
            mla.w_uk_t.weight,
            mla.w_uv.weight,
            v_out,
            DevicePtr::NULL,
            DevicePtr::NULL,
            n,
            nq,
            nope,
            rope,
            kv_lora,
            v_dim,
            hd_mla,
            nkv,
            1.0f32 / (mla_cache_dim as f32).sqrt(),
            stream,
        )?;

        // ── 6. Write KV cache ──
        let k_cache = ctx.buffers.expert_up_out();
        let v_cache = ctx.buffers.expert_down_out();
        ops::mla_cache_assemble_batched(
            ctx.gpu,
            self.mla_cache_assemble_batched_k,
            kv_latent,
            k_rope_buf,
            k_cache,
            v_cache,
            n,
            kv_lora,
            rope,
            mla_cache_dim,
            stream,
        )?;
        self.write_kv_cache(
            ctx.gpu,
            k_cache,
            v_cache,
            kv_cache,
            meta.slot,
            n,
            1,
            mla_cache_dim,
            kv_cache.block_size() as u32,
            mla_cache_dim,
            mla_cache_dim,
            stream,
            ctx.graph_capture,
        )?;

        // ── 7. Grouped low-rank O projection ──
        let o_latent = ctx.buffers.norm_output();
        let o_out = ctx.buffers.qkv_output();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            v_out,
            &mla.wo_a,
            o_latent,
            n,
            o_lora,
            nq * v_dim,
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
