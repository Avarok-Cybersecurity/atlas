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
        // q_b_norm: per-head unweighted RMSNorm over head_dim (DeepSeek-V4),
        // each of the n*nq head vectors renormalized to unit RMS before rope.
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            q_full,
            &crate::weight_map::DenseWeight {
                weight: ctx.buffers.norm_unit_w(),
            },
            q_full,
            n * nq,
            hd_mla,
            eps,
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
        // kv_norm: weighted RMSNorm over each token's kv latent BEFORE rope
        // (DeepSeek-V4: kv = kv_norm(kv_proj(h))). n*nkv rows of kv_lora dims.
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            k_out,
            &mla.kv_a_norm,
            k_out,
            n * nkv,
            kv_lora,
            eps,
            stream,
        )?;
        // Copy K → V (V4-Flash: K and V share the same projection output)
        ctx.gpu
            .copy_d2d_async(k_out, v_out, (n * kv_dim) as usize * 2, stream)?;

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
            // Sliding layers (compressor==None) = main θ=10000, mscale=1; CSA/HCA
            // = θ=160000 yarn. Matches the eq.26 de-rotation below. (V4-exclusive
            // file, so compressor==None is a V4 sliding layer, never a non-V4 model.)
            if mla.compressor.is_none() {
                mla.main_inv_freq
            } else {
                mla.yarn_inv_freq
            },
            if mla.compressor.is_none() {
                1.0f32
            } else {
                super::super::helpers::yarn_rope_mscale(ctx.config)
            },
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

        // ── 4. Standard GQA FlashAttention (current chunk only) ──
        let attn_out = ctx.buffers.attn_output();
        let prefill_k = if hd_mla > 256 {
            if self.prefill_attn_512_k.0 == 0 {
                anyhow::bail!(
                    "V4-Flash paged prefill: hd_mla={} > 256 but prefill_attn_512_k is not loaded (handle=0). \
                     The inferspark_prefill_512 kernel must be present in the PTX.",
                    hd_mla
                );
            }
            tracing::info!(
                "V4-Flash paged prefill: using prefill_attn_512_k (hd_mla={})",
                hd_mla
            );
            self.prefill_attn_512_k
        } else {
            tracing::info!(
                "V4-Flash paged prefill: using prefill_attn_64_k (hd_mla={})",
                hd_mla
            );
            self.prefill_attn_64_k
        };
        // MLA passes kv as BOTH key and value (K==V). `k_out` carries the rotated
        // rope in its tail; `v_out` is the plain latent (kept for the cache
        // assembly below). Attention must use K==k_out for V too. The per-head
        // sink keeps the softmax denominator consistent with the decode path.
        ops::prefill_attention_512_sink(
            ctx.gpu,
            prefill_k,
            q_full,
            k_out,
            k_out,
            attn_out,
            n,
            1,
            nq,
            nkv,
            hd_mla,
            1.0f32 / (hd_mla as f32).sqrt(),
            true,
            0,
            mla.attn_sink,
            stream,
        )
        .map_err(|e| anyhow::anyhow!("V4 paged: prefill_attention failed: {e}"))?;

        // DeepSeek-V4 eq.26: de-rotate the attention output by each query
        // position (inverse interleaved YaRN RoPE on the trailing rope dims)
        // before the grouped o_proj, so each value's contribution is
        // relative-distance.
        {
            let o_rope_tmp = ctx.buffers.ssm_conv_out_f32();
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                attn_out,
                o_rope_tmp,
                n,
                nq,
                hd_mla,
                nope,
                rope,
                nq * hd_mla,
                stream,
            )?;
            ops::rope_yarn(
                ctx.gpu,
                self.rope_yarn_interleaved_inv_k,
                o_rope_tmp,
                o_rope_tmp,
                meta.positions,
                n,
                nq,
                0,
                rope,
                rope,
                // De-rotation must match the rope-in inv_freq above (else salad).
                if mla.compressor.is_none() {
                    mla.main_inv_freq
                } else {
                    mla.yarn_inv_freq
                },
                if mla.compressor.is_none() {
                    1.0f32
                } else {
                    super::super::helpers::yarn_rope_mscale(ctx.config)
                },
                stream,
            )?;
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                o_rope_tmp,
                attn_out,
                n,
                nq,
                hd_mla,
                nope,
                rope,
                nq * hd_mla,
                stream,
            )?;
        }

        // ── 5. Assemble KV cache (V4-Flash: latent+rope = 576-dim MLA) ──
        // v_out holds the original latent K (512-dim, before RoPE writeback).
        // k_rope_tmp (q_latent reuse) holds the rotated RoPE values.
        let k_cache_assembled = ctx.buffers.expert_up_out();
        let v_cache_assembled = ctx.buffers.expert_down_out();
        let mla_cache_dim = kv_lora + rope;
        ops::mla_cache_assemble_batched(
            ctx.gpu,
            self.mla_cache_assemble_batched_k,
            v_out,      // 512-dim latent K (V copy, unmodified)
            k_rope_tmp, // 64-dim RoPE from K extraction+rotation
            k_cache_assembled,
            v_cache_assembled,
            n,
            kv_lora,
            rope,
            mla_cache_dim,
            stream,
        )?;

        // ── 6. Write assembled K/V to paged cache ──
        self.write_kv_cache(
            ctx.gpu,
            k_cache_assembled,
            v_cache_assembled,
            kv_cache,
            meta.slot,
            n,
            1,
            mla_cache_dim,
            bs,
            mla_cache_dim,
            mla_cache_dim,
            stream,
            ctx.graph_capture,
        )?;

        // ── 7. Grouped low-rank O projection (block-diagonal wo_a → wo_b) ──
        // wo_a is block-diagonal over o_groups (DeepseekV4GroupedLinear): each
        // group_in slice of the attention output projects independently to o_lora,
        // giving an o_groups*o_lora latent that wo_b mixes back to hidden_size.
        // Per-token×group GEMVs avoid the strided-input limitation of dense_gemm;
        // wo_b stays a single GEMM since o_latent is contiguous [n, latent_dim].
        let o_groups = ctx.config.o_groups.max(1) as u32;
        let group_in = (nq * hd_mla) / o_groups;
        let latent_dim = o_groups * o_lora;
        let o_latent = ctx.buffers.o_latent();
        let o_out = ctx.buffers.qkv_output();
        for t in 0..n {
            for g in 0..o_groups {
                let in_g = attn_out.offset(((t * nq * hd_mla) + g * group_in) as usize * 2);
                let w_g = crate::weight_map::DenseWeight {
                    weight: mla
                        .wo_a
                        .weight
                        .offset((g as usize) * (o_lora as usize) * (group_in as usize) * 2),
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

        Ok(o_out)
    }
}
