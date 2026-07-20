// SPDX-License-Identifier: AGPL-3.0-only

//! Per-token MTP forward pass.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{MtpHead, MtpProposerState, MtpQuantization, ProjectionWeight};
use crate::layer::ForwardContext;
use crate::layers::ops;

/// MTP-debug (ATLAS_MTP_DEBUG_NORMS=1): L2 norm of a BF16 GPU buffer, for
/// localizing where the MTP forward produces NaN/0. NaN reads back as NaN.
fn mtp_dbg_l2(gpu: &dyn spark_runtime::gpu::GpuBackend, p: DevicePtr, n: usize) -> f64 {
    let mut b = vec![0u8; n * 2];
    if gpu.copy_d2h(p, &mut b).is_err() {
        return f64::NAN;
    }
    b.chunks_exact(2)
        .map(|c| {
            let f = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64;
            f * f
        })
        .sum::<f64>()
        .sqrt()
}

impl MtpHead {
    /// MTP forward pass for a single token.
    ///
    /// When `draft_embed_target` is `Some(ptr)`, the draft token's embedding
    /// is written directly to `ptr` on GPU via `embed_from_argmax`, and the
    /// token ID is stored in `self.draft_token_id_dev` for deferred readback.
    /// This eliminates the D2H sync that was previously required.
    ///
    /// When `draft_embed_target` is `None`, falls back to D2H readback.
    ///
    /// Visible to sibling modules (`mtp_multi`) so the multi-module
    /// proposer can dispatch per-draft to a different `MtpHead` while
    /// reusing the same per-token forward code path.
    pub(crate) fn forward_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let h = ctx.config.hidden_size as u32;
        let nq = ctx.config.num_attention_heads as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let hd = ctx.config.head_dim as u32;
        let eps = ctx.config.rms_norm_eps as f32;

        // 1. Embed token
        let embed_out = ctx.buffers.ssm_qkvz(); // reuse scratch
        let row_bytes = h as usize * 2;
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        ctx.gpu.copy_d2d_async(src, embed_out, row_bytes, stream)?;

        // 2. RMSNorm embedding and hidden separately
        let normed_embed = ctx.buffers.ssm_deinterleaved();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            embed_out,
            &self.pre_fc_norm_embedding,
            normed_embed,
            1,
            h,
            eps,
            stream,
        )?;

        // The saved hidden is always BF16 (the residual stream is BF16).
        let normed_hidden = ctx.buffers.ssm_gates();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            target_hidden,
            &self.pre_fc_norm_hidden,
            normed_hidden,
            1,
            h,
            eps,
            stream,
        )?;

        // 3. Concatenate: [normed_embed | normed_hidden] → [2*h]
        let concat_out = ctx.buffers.ssm_ba();
        ops::bf16_concat(
            ctx.gpu,
            self.bf16_concat_k,
            normed_embed,
            normed_hidden,
            concat_out,
            h,
            stream,
        )?;

        if std::env::var("ATLAS_MTP_DEBUG_NORMS").as_deref() == Ok("1") {
            ctx.gpu.synchronize(stream).ok();
            tracing::warn!(
                "MTP_DBG s1-embed ||={:.4} s2-n_embed ||={:.4} s2-n_hidden ||={:.4} s3-concat ||={:.4}",
                mtp_dbg_l2(ctx.gpu, embed_out, h as usize),
                mtp_dbg_l2(ctx.gpu, normed_embed, h as usize),
                mtp_dbg_l2(ctx.gpu, normed_hidden, h as usize),
                mtp_dbg_l2(ctx.gpu, concat_out, (h * 2) as usize),
            );
        }

        // 4. FC projection: [2*h] → [h]
        let hidden = ctx.buffers.hidden_states();
        self.gemv(ctx.gpu, concat_out, &self.fc, hidden, h, h * 2, stream)?;
        if std::env::var("ATLAS_MTP_DEBUG_NORMS").as_deref() == Ok("1") {
            ctx.gpu.synchronize(stream).ok();
            tracing::warn!(
                "MTP_DBG s4-fc_hidden ||={:.4}",
                mtp_dbg_l2(ctx.gpu, hidden, h as usize)
            );
        }

        // 5. Copy hidden to residual for residual stream
        let residual = ctx.buffers.residual();
        ctx.gpu
            .copy_d2d_async(hidden, residual, row_bytes, stream)?;

        // 6. Input layernorm
        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.input_layernorm,
            normed,
            1,
            h,
            eps,
            stream,
        )?;

        // 7. Attention: Q+Gate and K+V projections
        let q_out = ctx.buffers.qkv_output();
        let q_dim = nq * hd;
        let qg_dim = q_dim * 2;
        let qg_bytes = qg_dim as usize * 2;

        match self.quant {
            MtpQuantization::Nvfp4 => {
                // Fused GEMV + deinterleave in one kernel
                if let ProjectionWeight::Nvfp4(ref w) = self.q_proj {
                    ops::w4a16_gemv_qg(
                        ctx.gpu,
                        self.w4a16_gemv_qg_k,
                        normed,
                        w,
                        q_out,
                        qg_dim,
                        h,
                        nq,
                        hd,
                        stream,
                    )?;
                }
            }
            MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                // Separate GEMV + deinterleave kernel
                self.gemv(ctx.gpu, normed, &self.q_proj, q_out, qg_dim, h, stream)?;
                ops::deinterleave_qg(
                    ctx.gpu,
                    self.deinterleave_qg_k.unwrap(),
                    q_out,
                    1,
                    nq,
                    hd,
                    nq * hd * 2,
                    stream,
                )?;
            }
        }
        let gate_ptr = q_out.offset(q_dim as usize * 2);

        // K+V projections
        let k_out = q_out.offset(qg_bytes);
        let v_out = k_out.offset((nkv * hd) as usize * 2);

        match self.quant {
            MtpQuantization::Nvfp4 => {
                if let (ProjectionWeight::Nvfp4(kw), ProjectionWeight::Nvfp4(vw)) =
                    (&self.k_proj, &self.v_proj)
                {
                    ops::w4a16_gemv_dual(
                        ctx.gpu,
                        self.w4a16_gemv_dual_k,
                        normed,
                        kw,
                        k_out,
                        vw,
                        v_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                }
            }
            MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                self.gemv(ctx.gpu, normed, &self.k_proj, k_out, nkv * hd, h, stream)?;
                self.gemv(ctx.gpu, normed, &self.v_proj, v_out, nkv * hd, h, stream)?;
            }
        }

        // Q/K norms
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            q_out,
            &self.q_norm,
            q_out,
            nq,
            hd,
            eps,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            k_out,
            &self.k_norm,
            k_out,
            nkv,
            hd,
            eps,
            stream,
        )?;

        // 8. Upload attention metadata for MTP KV cache
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = (state.seq_len / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }

        let meta_base = ctx.buffers.scratch().offset(49152); // after target metadata
        let max_blocks = state.block_table.len() as u32;

        // Batch all metadata into a single H2D copy (saves 3 CUDA API calls).
        let block_idx = state.block_table[state.seq_len / bs];
        let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
        let actual_seq_len = (state.seq_len + 1) as i32;
        let bt_len = state.block_table.len() * 4;

        // Dynamic metadata buffer: 256 bytes header + block table.
        // Fixed 512-byte buffer overflows when seq_len > ~2000 (block table > 256 bytes).
        let meta_size = 256 + bt_len;
        let mut meta_buf = vec![0u8; meta_size];
        meta_buf[0..4].copy_from_slice(&(position as u32).to_le_bytes());
        meta_buf[8..16].copy_from_slice(&global_slot.to_le_bytes());
        meta_buf[16..20].copy_from_slice(&actual_seq_len.to_le_bytes());
        // Block table values are always < 2^31 (block indices), so u32 → i32 is lossless.
        let bt_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(state.block_table.as_ptr() as *const u8, bt_len) };
        meta_buf[256..256 + bt_len].copy_from_slice(bt_bytes);
        ctx.gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;

        // RoPE
        ops::rope(
            ctx.gpu,
            self.rope_k,
            q_out,
            k_out,
            meta_base, // positions
            1,
            nq,
            nkv,
            hd,
            ctx.config.rotary_dim() as u32,
            ctx.config.rope_theta as f32,
            stream,
        )?;

        // Reshape + cache + paged decode. BF16 KV (self.kv_bf16) matches the
        // main model; the FP8 path's hard-coded unit scales (1.0,1.0) collapse
        // the MTP attention to a constant on Qwen3.6-A3B → constant draft 0.
        let kv_stride = nkv * hd;
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = 1.0f32 / (hd as f32).sqrt();
        if self.kv_bf16 {
            ops::reshape_and_cache(
                ctx.gpu,
                self.reshape_cache_k,
                k_out,
                v_out,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                meta_base.offset(8), // slot
                1,                   // num_tokens
                nkv,
                hd,
                bs as u32, // block_size
                kv_stride,
                kv_stride,
                kv_cache.cache_stride() as u64,
                stream,
            )?;
            ops::paged_decode_attn_bf16(
                ctx.gpu,
                self.paged_decode_k,
                q_out,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta_base.offset(256), // block_table
                meta_base.offset(16),  // seq_len
                max_blocks,
                1, // num_seqs
                nq,
                nkv,
                hd,
                bs as u32, // block_size
                inv_sqrt_d,
                nq * hd, // q_stride
                0,       // sliding_window (full attention)
                stream,
            )?;
        } else {
            ops::reshape_and_cache_fp8(
                ctx.gpu,
                self.reshape_cache_k,
                k_out,
                v_out,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                meta_base.offset(8), // slot
                1,
                nkv,
                hd,
                bs as u32,
                1.0,
                1.0, // k_scale, v_scale (no pre-computed scales for MTP)
                kv_stride,
                kv_stride,
                kv_cache.cache_stride() as u64,
                stream,
            )?;
            ops::paged_decode_attn_fp8(
                ctx.gpu,
                self.paged_decode_k,
                q_out,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta_base.offset(256), // block_table
                meta_base.offset(16),  // seq_len
                max_blocks,
                1,
                nq,
                nkv,
                hd,
                bs as u32,
                inv_sqrt_d,
                1.0,
                1.0, // k_scale, v_scale
                nq * hd,
                kv_cache.cache_stride() as u64,
                stream,
            )?;
        }

        if std::env::var("ATLAS_MTP_DEBUG_NORMS").as_deref() == Ok("1") {
            ctx.gpu.synchronize(stream).ok();
            tracing::warn!(
                "MTP_DBG s7-attn_out(pre-gate) ||={:.4}  gate ||={:.4}",
                mtp_dbg_l2(ctx.gpu, attn_out, (nq * hd) as usize),
                mtp_dbg_l2(ctx.gpu, gate_ptr, (nq * hd) as usize)
            );
        }

        // Sigmoid gate: attn_out = attn_out * sigmoid(gate)
        ops::sigmoid_gate_mul(
            ctx.gpu,
            self.sigmoid_gate_mul_k,
            attn_out,
            gate_ptr,
            attn_out,
            nq * hd,
            stream,
        )?;

        // O projection: [nq*hd] → [h]
        let o_out = ctx.buffers.norm_output();
        self.gemv(ctx.gpu, attn_out, &self.o_proj, o_out, h, nq * hd, stream)?;

        // 9. Residual + post-attention norm
        let normed2 = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            o_out,
            &self.post_attn_layernorm,
            normed2,
            residual,
            1,
            h,
            eps,
            stream,
        )?;

        // 10. FFN: dense shortcut for non-MoE MTP heads (Qwen3.6-27B-FP8),
        //     otherwise routed MoE.
        let ffn_out = if self.dense_ffn_generic.is_some() {
            self.dense_ffn_forward_generic(normed2, ctx, stream)?
        } else {
            match self.quant {
                MtpQuantization::Nvfp4 => self
                    .moe_nvfp4
                    .as_ref()
                    .unwrap()
                    .forward(normed2, ctx, stream)?,
                MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                    self.moe_forward_generic(normed2, ctx, stream)?
                }
            }
        };
        ops::residual_add(ctx.gpu, self.residual_add_k, hidden, ffn_out, h, stream)?;

        // 11. Final norm
        let final_normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.norm,
            final_normed,
            1,
            h,
            eps,
            stream,
        )?;

        // 12. LM head (shared NVFP4, reduced vocab for faster propose)
        let v = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };
        let logits = ctx.buffers.logits();
        ops::w4a16_gemv(
            ctx.gpu,
            self.w4a16_gemv_k,
            final_normed,
            &self.lm_head_nvfp4,
            logits,
            v,
            h,
            stream,
        )?;

        // MTP-debug (ATLAS_MTP_DEBUG_NORMS=1): localize the constant-0 draft.
        // A true zero reads as 0.0 regardless of dtype, so these L2 norms
        // pinpoint the first stage to zero out: input_hidden (save bug) →
        // final_normed (forward bug) → logits (lm_head bug).
        if std::env::var("ATLAS_MTP_DEBUG_NORMS").as_deref() == Ok("1") {
            ctx.gpu.synchronize(stream).ok();
            let bf16_norm = |p: DevicePtr, n: usize| -> f64 {
                let mut b = vec![0u8; n * 2];
                if ctx.gpu.copy_d2h(p, &mut b).is_err() {
                    return -1.0;
                }
                b.chunks_exact(2)
                    .map(|c| {
                        let f =
                            f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64;
                        f * f
                    })
                    .sum::<f64>()
                    .sqrt()
            };
            // The residual stream is always BF16, so the saved hidden is BF16.
            let hin = bf16_norm(target_hidden, h as usize);
            tracing::warn!(
                "MTP_DEBUG_NORMS: ||input_hidden||={:.4} ||final_normed||={:.4} ||logits||={:.4}",
                hin,
                bf16_norm(final_normed, h as usize),
                bf16_norm(logits, v as usize)
            );
        }

        // 13a. Drafter chain confidence (ATLAS_MTP_DRAFT_CONF > 0):
        // observational only — token selection below is untouched. D2H the
        // BF16 logits (~200 us, the same cost the grammar-masked path pays)
        // and fold this draft's top-1 softmax prob into the propose-scoped
        // running MIN (`last_conf_bits`, reset by `propose`). The clamp that
        // acts on it lives in `run_mtp_propose_inner`.
        if crate::speculative::draft_conf_tau() > 0.0 {
            let vocab = v as usize;
            let mut bf16_buf = vec![0u8; vocab * 2];
            if ctx.gpu.copy_d2h(logits, &mut bf16_buf).is_ok() {
                let mut max = f32::NEG_INFINITY;
                for i in 0..vocab {
                    let hi = u16::from_le_bytes([bf16_buf[2 * i], bf16_buf[2 * i + 1]]);
                    let x = f32::from_bits((hi as u32) << 16);
                    if x > max {
                        max = x;
                    }
                }
                let mut denom = 0.0f64;
                for i in 0..vocab {
                    let hi = u16::from_le_bytes([bf16_buf[2 * i], bf16_buf[2 * i + 1]]);
                    let x = f32::from_bits((hi as u32) << 16);
                    denom += ((x - max) as f64).exp();
                }
                let top1 = (1.0 / denom.max(1.0)) as f32; // exp(max-max)=1 over Z
                let cur = f32::from_bits(
                    self.last_conf_bits
                        .load(std::sync::atomic::Ordering::Relaxed),
                );
                if top1 < cur {
                    self.last_conf_bits
                        .store(top1.to_bits(), std::sync::atomic::Ordering::Relaxed);
                }
            }
        }

        // 13. Argmax
        let out_ptr = ctx.buffers.scratch();

        let token_id = if let Some(bitmask) = grammar_bitmask {
            // Grammar-masked CPU argmax path.
            //
            // D2H the full logits vector (BF16), apply the XGrammar bitmask
            // (mask off ⇒ -inf), argmax on CPU. This adds ~200μs per draft
            // vs the GPU argmax, but the unmasked path sees ~0% draft
            // acceptance inside tool-call JSON — a 200μs overhead beats a
            // wasted 13.5ms verify step.
            //
            // We then H2D the chosen token id into `out_ptr` so the
            // downstream `embed_from_argmax` kernel can still gather the
            // embedding from the token table on GPU without a new kernel.
            let vocab = v as usize;
            let mut bf16_buf = vec![0u8; vocab * 2];
            ctx.gpu.copy_d2h(logits, &mut bf16_buf)?;

            // BF16 → f32 conversion. BF16 is the upper 16 bits of an f32.
            let mut f32_logits = vec![0.0f32; vocab];
            for i in 0..vocab {
                let lo = 0u16;
                let hi = u16::from_le_bytes([bf16_buf[2 * i], bf16_buf[2 * i + 1]]);
                f32_logits[i] = f32::from_bits(((hi as u32) << 16) | (lo as u32));
            }

            // Apply mask: bit `tok` set ⇒ allowed; unset ⇒ -inf.
            let mut any_allowed = false;
            for tok in 0..vocab {
                let word = tok / 32;
                let bit = tok % 32;
                let allowed = word < bitmask.len() && (bitmask[word] & (1i32 << bit)) != 0;
                if allowed {
                    any_allowed = true;
                } else {
                    f32_logits[tok] = f32::NEG_INFINITY;
                }
            }

            // Degenerate case: matcher gave us an empty allowed set. Don't
            // propose a real draft — return 0 (pad) as a sentinel. The
            // verifier almost certainly returns a non-zero target token, the
            // draft gets rejected, and the step falls through to target-only
            // decode. This is safer than re-emitting `last_token`, which
            // could be a special token (e.g. `<|im_end|>`) that the verifier
            // might happen to also pick — duplicating a role-boundary
            // token would poison the model's own context.
            if !any_allowed {
                tracing::warn!(
                    "MTP grammar mask allowed zero tokens at pos {position}; \
                     returning 0 as pad-draft (will be rejected at verify)."
                );
                0u32
            } else {
                // CPU argmax over masked logits.
                let mut best_tok = 0u32;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &v) in f32_logits.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best_tok = i as u32;
                    }
                }

                // If caller wants the embedding staged on GPU, stage the
                // chosen token id into `out_ptr` (4 bytes) and reuse the
                // existing embed_from_argmax kernel — it reads the argmax
                // result from `out_ptr` and gathers the embedding on GPU.
                if let Some(embed_target) = draft_embed_target {
                    let tok_bytes = best_tok.to_le_bytes();
                    ctx.gpu.copy_h2d(&tok_bytes, out_ptr)?;
                    ops::embed_from_argmax(
                        ctx.gpu,
                        self.embed_from_argmax_k,
                        out_ptr,
                        self.embed_tokens.weight,
                        embed_target,
                        self.draft_token_id_dev,
                        h,
                        stream,
                    )?;
                }
                best_tok
            }
        } else {
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out_ptr, v, stream)?;
            if let Some(embed_target) = draft_embed_target {
                // GPU-side embedding: write draft embedding to verify input buffer
                // and token ID to deferred readback buffer. No D2H sync needed.
                ops::embed_from_argmax(
                    ctx.gpu,
                    self.embed_from_argmax_k,
                    out_ptr,
                    self.embed_tokens.weight,
                    embed_target,
                    self.draft_token_id_dev,
                    h,
                    stream,
                )?;
                // Return 0 as placeholder — caller reads actual ID later via
                // read_deferred_draft_token().
                0u32
            } else {
                // Fallback: synchronous D2H readback.
                let mut buf = [0u8; 4];
                ctx.gpu.copy_d2h(out_ptr, &mut buf)?;
                u32::from_le_bytes(buf)
            }
        };

        state.seq_len += 1;
        // Pair-key bookkeeping (ATLAS_MTP_CATCHUP gap detection): this call
        // wrote the pair for sequence key `position - 1` at the row above.
        state.last_pair_key = Some(position.saturating_sub(1));
        Ok(token_id)
    }
}
