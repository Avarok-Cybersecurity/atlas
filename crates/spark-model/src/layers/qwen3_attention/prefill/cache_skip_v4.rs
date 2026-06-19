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

        // Per-token NaN scan of `normed` (post hc_pre + input_norm) — localizes
        // whether the K-FULL NaN originates upstream (hc_pre) or in the kv proj.
        if diag_this {
            let _ = ctx.gpu.synchronize(stream);
            let hh = h as usize;
            let mut buf = vec![0u8; (n as usize) * hh * 2];
            if ctx.gpu.copy_d2h(normed, &mut buf).is_ok() {
                let mut bad_tok = -1i64;
                for t in 0..n as usize {
                    let off = t * hh * 2;
                    if (0..hh).any(|i| {
                        let c = &buf[off + i * 2..off + i * 2 + 2];
                        let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                        !v.is_finite()
                    }) {
                        bad_tok = t as i64;
                        break;
                    }
                }
                tracing::info!(
                    "DIAG V4-prefill L{} NORMED first non-finite (nan/inf) token = {}",
                    self.attn_layer_idx,
                    bad_tok
                );
            }
        }

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
        // q_b_norm: per-head unweighted RMSNorm over head_dim (DeepSeek-V4),
        // each of the n*nq head vectors renormalized to unit RMS before rope.
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            q_full,
            &crate::weight_map::DenseWeight { weight: ctx.buffers.norm_unit_w() },
            q_full,
            n * nq,
            hd_mla,
            eps,
            stream,
        )?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                q_full,
                (nq * hd_mla) as usize,
                stream,
                &format!("V4-prefill L{} Q after q_b_norm token0", self.attn_layer_idx),
            );
            let q_last_off = ((n - 1) * nq * hd_mla * 2) as usize;
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                q_full.offset(q_last_off),
                (nq * hd_mla) as usize,
                stream,
                &format!("V4-prefill L{} Q after q_b_norm last", self.attn_layer_idx),
            );
        }

        // ── 2. Direct KV projection (V4-Flash: K=V, no absorption) ──
        // Layout in qkv_output: [Q | K | V]  (mirrors decode path)
        let q_dim = nq * hd_mla;
        let kv_dim = nkv * hd_mla;
        let k_out = q_full.offset((n * q_dim) as usize * 2);
        let v_out = k_out.offset((n * kv_dim) as usize * 2);
        let kv_latent = ctx.buffers.expert_gate_out(); // Capture latent for cache assembly
        // NOTE: dense_gemm_tc produces NON-DETERMINISTIC NaN for the wkv projection
        // here (varying token position across identical runs) — a latent TC-kernel
        // bug exposed once the upstream norms were corrected. Use the scalar
        // dense_gemm path for wkv until the TC kernel is fixed. wq_a above is
        // unaffected (TC output is correct there).
        #[allow(clippy::overly_complex_bool_expr)]
        if false && use_tc {
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
        if diag_this {
            let _ = ctx.gpu.synchronize(stream);
            let kl = kv_lora as usize;
            let mut wbuf = vec![0u8; kl * 2];
            let _ = ctx.gpu.copy_d2h(mla.kv_a_norm.weight, &mut wbuf);
            let wnan = (0..kl).any(|i| {
                let c = &wbuf[i * 2..i * 2 + 2];
                f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).is_nan()
            });
            let mut lbuf = vec![0u8; (n as usize) * kl * 2];
            let mut lnan = -1i64;
            if ctx.gpu.copy_d2h(kv_latent, &mut lbuf).is_ok() {
                for t in 0..n as usize {
                    if (0..kl).any(|i| {
                        let c = &lbuf[t * kl * 2 + i * 2..t * kl * 2 + i * 2 + 2];
                        f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).is_nan()
                    }) {
                        lnan = t as i64;
                        break;
                    }
                }
            }
            tracing::info!(
                "DIAG V4-prefill L{} PRE-kvnorm: kv_a_norm has NaN={}, kv_latent first NaN token={}",
                self.attn_layer_idx,
                wnan,
                lnan
            );
        }
        // kv_norm: weighted RMSNorm over each token's kv latent BEFORE rope and
        // before cache assembly (DeepSeek-V4: kv = kv_norm(kv_proj(h))). Applied to
        // kv_latent so the cached latent, k_out and v_out are all normalized.
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            kv_latent,
            &mla.kv_a_norm,
            kv_latent,
            n * nkv,
            kv_lora,
            eps,
            stream,
        )?;
        // Copy kv_latent → k_out (for attention computation)
        ctx.gpu
            .copy_d2d_async(kv_latent, k_out, n as usize * kv_lora as usize * 2, stream)?;
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: k_out gemm sync failed: {e}"))?;
        if diag_this {
            // Full-buffer K NaN/inf check across ALL n tokens (locates a bad token).
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                (n * kv_lora) as usize,
                stream,
                &format!("V4-prefill L{} K FULL ({} tokens)", self.attn_layer_idx, n),
            );
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
            // DeepSeek-V4 INTERLEAVED RoPE (rope_interleave=True): adjacent pairs
            // (2i, 2i+1), matching the HF reference. See attention_forward_v4.rs.
            self.rope_yarn_interleaved_k,
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

        // ── 6. Grouped low-rank O projection (block-diagonal wo_a → wo_b) ──
        // wo_a is block-diagonal over o_groups (DeepseekV4GroupedLinear); see
        // decode/attention_forward_v4.rs. Per-token×group GEMVs avoid the
        // strided-input limitation of dense_gemm; wo_b stays one GEMM.
        let o_groups = ctx.config.o_groups.max(1) as u32;
        let group_in = (nq * hd_mla) / o_groups;
        let latent_dim = o_groups * o_lora;
        let o_latent = ctx.buffers.o_latent();
        let o_out = ctx.buffers.qkv_output();
        for t in 0..n {
            for g in 0..o_groups {
                let in_g = attn_out.offset(((t * nq * hd_mla) + g * group_in) as usize * 2);
                let w_g = crate::weight_map::DenseWeight {
                    weight: mla.wo_a.weight.offset(
                        (g as usize) * (o_lora as usize) * (group_in as usize) * 2,
                    ),
                };
                let out_g = o_latent.offset(((t * latent_dim) + g * o_lora) as usize * 2);
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    in_g,
                    &w_g,
                    out_g,
                    o_lora,
                    group_in,
                    stream,
                )?;
            }
        }
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: wo_a grouped gemv sync failed: {e}"))?;
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            o_latent,
            &mla.wo_b,
            o_out,
            n,
            h,
            latent_dim,
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
