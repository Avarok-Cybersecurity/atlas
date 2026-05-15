// SPDX-License-Identifier: AGPL-3.0-only

//! Per-token MTP forward pass.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{MtpHead, MtpProposerState, MtpQuantization, ProjectionWeight};
use crate::layer::ForwardContext;
use crate::layers::ops;

/// True iff `ATLAS_MTP_DIVERGENCE_DUMP=1` is set in the environment.
/// Cached after the first call.
fn mtp_divergence_dump_enabled() -> bool {
    use std::sync::atomic::{AtomicI8, Ordering};
    static STATE: AtomicI8 = AtomicI8::new(-1);
    match STATE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = std::env::var("ATLAS_MTP_DIVERGENCE_DUMP")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            STATE.store(if on { 1 } else { 0 }, Ordering::Relaxed);
            on
        }
    }
}

/// True iff `ATLAS_MTP_AUDIT_PTRS=1` is set in the environment. Cached after the first call.
fn audit_ptrs_enabled() -> bool {
    use std::sync::atomic::{AtomicI8, Ordering};
    static STATE: AtomicI8 = AtomicI8::new(-1);
    match STATE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = std::env::var("ATLAS_MTP_AUDIT_PTRS")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            STATE.store(if on { 1 } else { 0 }, Ordering::Relaxed);
            on
        }
    }
}

/// True iff `ATLAS_MTP_SKIP_ATTN=1`. Cached after the first call.
fn skip_attn_enabled() -> bool {
    use std::sync::atomic::{AtomicI8, Ordering};
    static STATE: AtomicI8 = AtomicI8::new(-1);
    match STATE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = std::env::var("ATLAS_MTP_SKIP_ATTN")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            STATE.store(if on { 1 } else { 0 }, Ordering::Relaxed);
            on
        }
    }
}

/// True iff `ATLAS_MTP_SKIP_MLP=1`. Cached after the first call.
fn skip_mlp_enabled() -> bool {
    use std::sync::atomic::{AtomicI8, Ordering};
    static STATE: AtomicI8 = AtomicI8::new(-1);
    match STATE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = std::env::var("ATLAS_MTP_SKIP_MLP")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            STATE.store(if on { 1 } else { 0 }, Ordering::Relaxed);
            on
        }
    }
}

/// Returns true for the FIRST `n` calls (process-wide) — used to bound
/// the divergence dump's D2H-sync cost while still capturing the
/// early-decode samples where divergence is most diagnostic.
/// `position` parameter in `MtpHead::forward_one` is the absolute
/// next-token position (prompt_len + accepted_tokens + draft_idx),
/// which starts at ~50+ for typical prompts — so `position <= 10`
/// never fires. Using a call counter sidesteps that.
fn mtp_divergence_should_dump() -> bool {
    use std::sync::atomic::{AtomicU32, Ordering};
    const MAX_DUMPS: u32 = 12;
    static CALLS: AtomicU32 = AtomicU32::new(0);
    let n = CALLS.fetch_add(1, Ordering::Relaxed);
    n < MAX_DUMPS
}

/// Compute the L2 norm of a BF16 buffer on GPU by D2H-copying the
/// first `n_elements` BF16 values and reducing in F32. Only used
/// from `mtp_divergence_dump_enabled()`-gated paths because it
/// forces a stream sync and a D2H copy — cheap at small N (we
/// dump ≤ vocab_size BF16 per call) but never on the hot path.
fn bf16_l2_norm(gpu: &dyn GpuBackend, ptr: DevicePtr, n_elements: usize) -> f32 {
    let bytes = n_elements * 2;
    let mut buf = vec![0u8; bytes];
    if gpu.copy_d2h(ptr, &mut buf).is_err() {
        return f32::NAN;
    }
    let mut sum = 0.0f64;
    for c in buf.chunks_exact(2) {
        let bits = u16::from_le_bytes([c[0], c[1]]);
        let v = f32::from_bits((bits as u32) << 16) as f64;
        sum += v * v;
    }
    sum.sqrt() as f32
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

        // ATLAS_MTP_AUDIT_PTRS=1 — log embed/lm_head pointer values once at first call.
        // Compare against "MTP_PTRS build main_embed=..." in impl_a1_init.rs to detect
        // identity mismatches (e.g. MTP receiving lm_head weight instead of embed_tokens).
        if audit_ptrs_enabled() {
            use std::sync::atomic::{AtomicBool, Ordering};
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                let lm_head_ptr = match &self.lm_head {
                    ProjectionWeight::Nvfp4(w) => w.weight.0,
                    ProjectionWeight::Fp8(w) => w.weight.0,
                    ProjectionWeight::Fp8BlockScaled(w) => w.weight.0,
                    ProjectionWeight::Bf16(w) => w.weight.0,
                };
                tracing::info!(
                    "MTP_PTRS embed_tokens=0x{:016x} lm_head=0x{:016x} quant={:?} target_hidden=0x{:016x}",
                    self.embed_tokens.weight.0,
                    lm_head_ptr,
                    self.quant,
                    target_hidden.0,
                );
            }
        }

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

        // 3. Concatenate: order is checkpoint-dependent. Upstream Qwen
        // sometimes uses `[hidden | embed]` (DeepSeek-V3 style: prior
        // hidden first, new token's embed second) instead of the older
        // `[embed | hidden]`. `ATLAS_MTP_CONCAT_HIDDEN_FIRST=1` swaps
        // the args so the fc projection's first-half weight columns
        // multiply normed_hidden and second-half columns multiply
        // normed_embed.
        let concat_out = ctx.buffers.ssm_ba();
        let hidden_first = std::env::var("ATLAS_MTP_CONCAT_HIDDEN_FIRST")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let (concat_a, concat_b) = if hidden_first {
            (normed_hidden, normed_embed)
        } else {
            (normed_embed, normed_hidden)
        };
        ops::bf16_concat(
            ctx.gpu,
            self.bf16_concat_k,
            concat_a,
            concat_b,
            concat_out,
            h,
            stream,
        )?;

        // 4. FC projection: [2*h] → [h]
        // ATLAS_MTP_SKIP_FC=1 replaces the FC gemv with an identity copy of
        // normed_hidden → hidden (Step 2 bisection: isolates whether the FC
        // projection is the source of the hidden-state corruption seen when
        // cos(post_ffn_hidden, main_hidden) ≈ 0.22 despite target_hidden cos≈0.99).
        let hidden = ctx.buffers.hidden_states();
        let skip_fc = std::env::var("ATLAS_MTP_SKIP_FC")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if skip_fc {
            ctx.gpu.copy_d2d_async(normed_hidden, hidden, h as usize * 2, stream)?;
        } else {
            self.gemv(ctx.gpu, concat_out, &self.fc, hidden, h, h * 2, stream)?;
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
                // Fused GEMV + deinterleave for NVFP4 weights.
                // Fall through to the generic path when native FP8 block-scaled
                // weights are used (Fp8BlockScaled variant) — those are loaded
                // at native FP8 precision even when --mtp-quantization=nvfp4 is
                // set, because the FP8 checkpoint ships pre-quantized projections
                // and we skip the BF16→NVFP4 round-trip. The `quant` field only
                // governs the FC weight; the Q/K/V/O weights are dispatch-typed.
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
                } else {
                    // Native FP8 or BF16 Q weights — use the generic gemv
                    // dispatcher which handles Fp8BlockScaled correctly.
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
                } else {
                    // Native FP8 or BF16 K/V weights — fall back to generic dispatch.
                    self.gemv(ctx.gpu, normed, &self.k_proj, k_out, nkv * hd, h, stream)?;
                    self.gemv(ctx.gpu, normed, &self.v_proj, v_out, nkv * hd, h, stream)?;
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

        // Reshape + cache (FP8)
        let kv_stride = nkv * hd;
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

        // Paged decode attention
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = 1.0f32 / (hd as f32).sqrt();
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
        // ATLAS_MTP_SKIP_ATTN=1: zero o_out so attention contributes nothing to residual.
        // If post_ffn_hidden cos improves vs baseline → attention is the corruption source.
        if skip_attn_enabled() {
            ctx.gpu.memset_async(o_out, 0, h as usize * 2, stream)?;
        }

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

        // Post-attention hidden state capture for per-stage cos diagnostic.
        // D2H copy is synchronous and only runs when ATLAS_MTP_DIVERGENCE_COMPARE=1.
        // Stored and reported later in the MTP_COMPARE block.
        let post_attn_hidden_bytes: Vec<u8> = if crate::mtp_divergence::enabled() {
            let mut buf = vec![0u8; h as usize * 2];
            if ctx.gpu.copy_d2h(hidden, &mut buf).is_ok() {
                buf
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

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
        // ATLAS_MTP_SKIP_MLP=1: omit FFN contribution so hidden = post-attn residual only.
        // If post_ffn_hidden cos improves vs baseline → FFN is the corruption source.
        if !skip_mlp_enabled() {
            ops::residual_add(ctx.gpu, self.residual_add_k, hidden, ffn_out, h, stream)?;
        }

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

        // 12. LM head — dispatched through the shared `gemv()` helper
        // so the proposer uses whichever precision matches the
        // main-model verifier's LM head (Nvfp4 / Fp8BlockScaled /
        // Bf16). Matching precisions is what keeps proposer/verifier
        // logit distributions aligned; the prior hardcoded NVFP4
        // dispatch produced ~99% MTP K=2 rejection on dense Qwen 3.6
        // 27B once the main verifier switched to BF16 lm_head.
        let v = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };
        // Divergence telemetry (ATLAS_MTP_DIVERGENCE_DUMP=1, first 12 MTP calls):
        // dump L2 norms of every major MTP intermediate so the
        // hidden-state-comparison can localise where the dense path
        // diverges from what the main verifier expects. Forces a
        // stream sync per call — never on by default.
        let dump_this_call = mtp_divergence_dump_enabled() && mtp_divergence_should_dump();
        if dump_this_call {
            // Stream sync so the buffers we just wrote are visible
            // on the host. The D2H copies inside `bf16_l2_norm` go
            // through the same default stream, but explicit sync
            // here avoids any default-stream/explicit-stream race.
            let _ = ctx.gpu.synchronize(stream);
            let h_us = h as usize;
            let nq_us = nq as usize;
            let hd_us = hd as usize;
            let target_hidden_norm = bf16_l2_norm(ctx.gpu, target_hidden, h_us);
            let normed_embed_norm = bf16_l2_norm(ctx.gpu, normed_embed, h_us);
            let normed_hidden_norm = bf16_l2_norm(ctx.gpu, normed_hidden, h_us);
            // `hidden` at this point in the forward has been
            // overwritten twice (FC output, then post-attn residual);
            // skip dumping it to avoid confusion, and dump
            // `final_normed` (right before LM head) instead.
            let attn_out_norm = bf16_l2_norm(ctx.gpu, attn_out, nq_us * hd_us);
            let ffn_out_norm = bf16_l2_norm(ctx.gpu, ffn_out, h_us);
            let final_normed_norm = bf16_l2_norm(ctx.gpu, final_normed, h_us);
            tracing::info!(
                "MTP_DIV pos={position} target_hidden={target_hidden_norm:.4} \
                 normed_embed={normed_embed_norm:.4} normed_hidden={normed_hidden_norm:.4} \
                 attn_out={attn_out_norm:.4} ffn_out={ffn_out_norm:.4} \
                 final_normed={final_normed_norm:.4}",
            );
        }

        let logits = ctx.buffers.logits();
        self.gemv(ctx.gpu, final_normed, &self.lm_head, logits, v, h, stream)?;

        // After LM head: dump top-5 logits when telemetry is on.
        // The verifier will report its own top-5 in a sibling log
        // line; comparing the two columns shows whether MTP's
        // argmax matches what the verifier would have picked.
        //
        // For MTP_COMPARE we don't bump a counter — the main-side
        // tap is the one capped at 12 captures. The MTP side just
        // reads the latest snapshot whenever the compare env is on.
        let compare_this_call = crate::mtp_divergence::enabled();
        if dump_this_call || compare_this_call {
            let _ = ctx.gpu.synchronize(stream);
            let v_us = v as usize;
            let mut buf = vec![0u8; v_us * 2];
            let mtp_top5: Vec<(u32, f32)> = if ctx.gpu.copy_d2h(logits, &mut buf).is_ok() {
                let mut idx_val: Vec<(u32, f32)> = (0..v_us)
                    .map(|i| {
                        let bits = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
                        (i as u32, f32::from_bits((bits as u32) << 16))
                    })
                    .collect();
                idx_val.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                idx_val.into_iter().take(5).collect()
            } else {
                Vec::new()
            };

            if dump_this_call {
                tracing::info!("MTP_DIV pos={position} top5={mtp_top5:?}");
            }

            if compare_this_call
                && let Some(main_snap) = crate::mtp_divergence::peek_snapshot()
            {
                // Reject stale comparisons: in a healthy K=2 cycle the
                // MTP draft fires immediately after main's decode of the
                // previous token, so |mtp_pos - main_pos| ≤ 2. A large
                // delta means main paused (thinking phase finished and
                // emitted a chunk under graph capture without firing
                // MTP) — comparing across that gap is meaningless.
                let main_pos = main_snap.position as i64;
                let mtp_pos_i = position as i64;
                let pos_delta = (mtp_pos_i - main_pos).abs();
                if pos_delta > 3 {
                    tracing::info!(
                        "MTP_COMPARE skipped (stale snapshot): main_pos={} mtp_pos={position} delta={pos_delta}",
                        main_snap.position,
                    );
                } else {
                    let h_us = h as usize;

                    // Stage 0: target_hidden (= main's pre-norm hidden, MTP input).
                    // target_hidden is main's hidden_states() BEFORE final RMSNorm.
                    // main_snap.final_normed is AFTER RMSNorm — same direction, different
                    // scale. RMSNorm is direction-preserving so cos should be ~0.95-1.0
                    // when delta=0 (same position). If cos≈0 here → target_hidden is
                    // pointing at the wrong buffer (identity bug upstream of MTP).
                    let mut th_buf = vec![0u8; h_us * 2];
                    let cos_th = if ctx.gpu.copy_d2h(target_hidden, &mut th_buf).is_ok() {
                        let th_vec = crate::mtp_divergence::bf16_bytes_to_f32(&th_buf);
                        crate::mtp_divergence::cosine(&main_snap.final_normed, &th_vec)
                    } else {
                        f32::NAN
                    };
                    tracing::info!(
                        "MTP_STAGE_COS pos={position} delta={pos_delta} stage=target_hidden cos={cos_th:.4}"
                    );

                    // Stage 1: post_attn_hidden — hidden state after attention residual+norm
                    // but BEFORE FFN. Captured inline above to bisect attn vs FFN corruption.
                    if !post_attn_hidden_bytes.is_empty() {
                        let post_attn_vec = crate::mtp_divergence::bf16_bytes_to_f32(&post_attn_hidden_bytes);
                        let cos_post_attn = crate::mtp_divergence::cosine(&main_snap.final_normed, &post_attn_vec);
                        tracing::info!(
                            "MTP_STAGE_COS pos={position} delta={pos_delta} stage=post_attn_hidden cos={cos_post_attn:.4}"
                        );
                    }

                    // Stage 3: hidden (post-FFN residual, before final norm).
                    // At this point in the call, hidden = FC(concat) + O_proj(attn) + FFN_out
                    // — the full transformer block output before final RMSNorm.
                    // cos vs main's final_normed tells us if the MTP transformer block
                    // produces a plausible direction. If cos_th≈0.97 but cos_hid≈0 →
                    // the MTP block itself is scrambling the representation.
                    let mut hid_buf = vec![0u8; h_us * 2];
                    let cos_hid = if ctx.gpu.copy_d2h(hidden, &mut hid_buf).is_ok() {
                        let hid_vec = crate::mtp_divergence::bf16_bytes_to_f32(&hid_buf);
                        crate::mtp_divergence::cosine(&main_snap.final_normed, &hid_vec)
                    } else {
                        f32::NAN
                    };
                    tracing::info!(
                        "MTP_STAGE_COS pos={position} delta={pos_delta} stage=post_ffn_pre_norm cos={cos_hid:.4}"
                    );

                    // Stage 2: final_normed — full MTP_COMPARE metrics.
                    let mut hbuf = vec![0u8; h_us * 2];
                    if ctx.gpu.copy_d2h(final_normed, &mut hbuf).is_ok() {
                        let mtp_hidden = crate::mtp_divergence::bf16_bytes_to_f32(&hbuf);
                        let main_l2 = crate::mtp_divergence::l2_norm(&main_snap.final_normed);
                        let mtp_l2 = crate::mtp_divergence::l2_norm(&mtp_hidden);
                        let rel =
                            crate::mtp_divergence::rel_l2(&main_snap.final_normed, &mtp_hidden);
                        let cos =
                            crate::mtp_divergence::cosine(&main_snap.final_normed, &mtp_hidden);
                        let mtp_argmax = mtp_top5.first().map(|p| p.0);
                        let argmax_match = match (mtp_argmax, main_snap.argmax) {
                            (Some(a), Some(b)) => a == b,
                            _ => false,
                        };
                        let jaccard =
                            crate::mtp_divergence::topk_jaccard(&main_snap.top5, &mtp_top5, 5);
                        tracing::info!(
                            "MTP_COMPARE main_pos={} mtp_pos={position} delta={pos_delta} \
                             L2_main={main_l2:.4} L2_mtp={mtp_l2:.4} rel_diff={rel:.4} cos={cos:.4} \
                             argmax_main={:?} argmax_mtp={:?} argmax_match={argmax_match} \
                             top5_jaccard={jaccard:.3}",
                            main_snap.position, main_snap.argmax, mtp_argmax,
                        );
                    }
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
        Ok(token_id)
    }
}
