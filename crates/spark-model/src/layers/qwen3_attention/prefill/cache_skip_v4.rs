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
        let mla_cache_dim = kv_lora + rope;
        let hd_mla = nope + rope;
        let use_tc = self.dense_gemm_tc_k.0 != 0;
        let diag_all =
            std::env::var("ATLAS_DIAG_V4_ALL_LAYERS").is_ok_and(|v| v == "1" || v == "true");
        let diag_this = self.attn_layer_idx == 0 || diag_all;

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
        ctx.gpu
            .synchronize(stream)
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
        ctx.gpu
            .synchronize(stream)
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
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: q_full gemm sync failed: {e}"))?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                q_full,
                (nq * hd_mla) as usize,
                stream,
                &format!("V4-prefill L{} Q after proj", self.attn_layer_idx),
            );
        }

        // ── 2. Direct KV projection (V4-Flash: K=V, no absorption) ──
        // Layout in qkv_output: [Q | K | V]  (mirrors decode path)
        let q_dim = nq * hd_mla;
        let kv_dim = nkv * hd_mla;
        let k_out = q_full.offset((n * q_dim) as usize * 2);
        let v_out = k_out.offset((n * kv_dim) as usize * 2);
        let kv_latent = ctx.buffers.expert_gate_out(); // Capture latent for cache assembly
        if use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                &mla.wkv_a,
                kv_latent, // Write to kv_latent for cache assembly
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
                kv_latent, // Write to kv_latent for cache assembly
                n,
                kv_lora,
                h,
                stream,
            )?;
        }
        // Copy kv_latent → k_out (for attention computation)
        ctx.gpu
            .copy_d2d_async(kv_latent, k_out, n as usize * kv_lora as usize * 2, stream)?;
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: k_out gemm sync failed: {e}"))?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                kv_dim as usize,
                stream,
                &format!("V4-prefill L{} K after proj", self.attn_layer_idx),
            );
        }
        // Copy K → V (V4-Flash: K and V share the same projection output)
        ctx.gpu
            .copy_d2d_async(k_out, v_out, (n * kv_dim) as usize * 2, stream)?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                v_out,
                kv_dim as usize,
                stream,
                &format!("V4-prefill L{} V after copy", self.attn_layer_idx),
            );
        }

        // ── 3. RoPE on Q and K (V is NOT RoPE'd) ──
        // V4-Flash: rope dims are at offset `nope` per head (matching MLA layout),
        // not at the beginning. Extract → RoPE → writeback.
        let q_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        let k_rope_tmp = q_latent; // reuse after wq_b is done
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
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_batched_k,
            k_out,
            k_rope_tmp,
            n,
            nkv,
            hd_mla,
            nope,
            rope,
            nkv * hd_mla,
            stream,
        )?;
        ops::rope_yarn(
            ctx.gpu,
            self.rope_yarn_k,
            q_rope_tmp,
            k_rope_tmp,
            meta.positions,
            n,
            nq,
            nkv,
            rope,
            rope,
            mla.yarn_inv_freq,
            super::super::helpers::yarn_rope_mscale(ctx.config),
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_batched_k,
            q_rope_tmp,
            q_full,
            n,
            nq,
            hd_mla,
            nope,
            rope,
            nq * hd_mla,
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_batched_k,
            k_rope_tmp,
            k_out,
            n,
            nkv,
            hd_mla,
            nope,
            rope,
            nkv * hd_mla,
            stream,
        )?;
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: rope_yarn sync failed: {e}"))?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                kv_dim as usize,
                stream,
                &format!("V4-prefill L{} K after RoPE token0", self.attn_layer_idx),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out.offset((nope * 2) as usize),
                (kv_dim - nope) as usize,
                stream,
                &format!(
                    "V4-prefill L{} K rope after RoPE token0",
                    self.attn_layer_idx
                ),
            );
            let last_k_offset = ((n - 1) * kv_dim * 2) as usize;
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out.offset(last_k_offset),
                kv_dim as usize,
                stream,
                &format!("V4-prefill L{} K after RoPE last", self.attn_layer_idx),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out.offset(last_k_offset + (nope * 2) as usize),
                (kv_dim - nope) as usize,
                stream,
                &format!("V4-prefill L{} K rope after RoPE last", self.attn_layer_idx),
            );
        }

        // ── 4. Standard GQA FlashAttention (intra-chunk) ──
        let attn_out = ctx.buffers.attn_output();
        let prefill_k = if hd_mla > 256 {
            if self.prefill_attn_512_k.0 == 0 {
                anyhow::bail!(
                    "V4-Flash prefill: hd_mla={} > 256 but prefill_attn_512_k is not loaded (handle=0). \
                     The inferspark_prefill_512 kernel must be present in the PTX.",
                    hd_mla
                );
            }
            tracing::info!(
                "V4-Flash prefill: using prefill_attn_512_k (hd_mla={})",
                hd_mla
            );
            self.prefill_attn_512_k
        } else {
            tracing::info!(
                "V4-Flash prefill: using prefill_attn_64_k (hd_mla={})",
                hd_mla
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
        .map_err(|e| anyhow::anyhow!("V4 attn: prefill_attention failed: {e}"))?;
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: prefill_attention sync failed: {e}"))?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                attn_out,
                (nq * hd_mla) as usize,
                stream,
                &format!("V4-prefill L{} attn_out token0", self.attn_layer_idx),
            );
            let last_token_offset = ((n - 1) * nq * hd_mla * 2) as usize;
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                attn_out.offset(last_token_offset),
                (nq * hd_mla) as usize,
                stream,
                &format!("V4-prefill L{} attn_out last", self.attn_layer_idx),
            );
        }

        // ── 5. Assemble KV cache (V4-Flash: requires latent+rope assembly) ──
        // NOTE: k_out is 512-dim (complete K), but cache needs 576-dim (512 latent + 64 rope).
        // We need to extract the latent portion (first 512 dims), reassemble with rope, then write.
        let k_cache_assembled = ctx.buffers.expert_up_out();
        let v_cache_assembled = ctx.buffers.expert_down_out();
        ops::mla_cache_assemble_batched(
            ctx.gpu,
            self.mla_cache_assemble_batched_k,
            kv_latent,  // 512-dim latent (reused from step 2)
            k_rope_tmp, // 64-dim RoPE from K (reused from step 3)
            k_cache_assembled,
            v_cache_assembled,
            n,
            kv_lora,
            rope,
            mla_cache_dim,
            stream,
        )?;
        self.write_kv_cache(
            ctx.gpu,
            k_cache_assembled,
            v_cache_assembled,
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
        ctx.gpu
            .synchronize(stream)
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
        ctx.gpu
            .synchronize(stream)
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
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: wo_b gemm sync failed: {e}"))?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                o_out,
                h as usize,
                stream,
                &format!("V4-prefill L{} o_out token0", self.attn_layer_idx),
            );
            let last_token_offset = ((n - 1) * h * 2) as usize;
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                o_out.offset(last_token_offset),
                h as usize,
                stream,
                &format!("V4-prefill L{} o_out last", self.attn_layer_idx),
            );
        }

        Ok(o_out)
    }
}
