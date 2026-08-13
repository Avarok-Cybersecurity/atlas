// SPDX-License-Identifier: AGPL-3.0-only

//! Per-token MTP forward pass.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::propose_graph::{mtp_propose_graph_enabled, propose_graphable};
use super::{MtpHead, MtpProposerState};
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
        let row_bytes = h as usize * 2;

        // Token-dependent embed source — never baked into a CUDA graph.
        let embed_out = ctx.buffers.ssm_qkvz();
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        ctx.gpu.copy_d2d_async(src, embed_out, row_bytes, stream)?;
        // Stage hidden into a process-lifetime buffer so one graph covers
        // both `mtp_hidden_save` (draft 0) and `hidden_states()` (draft >0).
        ctx.gpu
            .copy_d2d_async(target_hidden, self.propose_in_hidden, row_bytes, stream)?;

        let kv = self.prepare_propose_kv(state, ctx, position, stream)?;

        let debug_norms = std::env::var("ATLAS_MTP_DEBUG_NORMS").as_deref() == Ok("1");
        let graphable = mtp_propose_graph_enabled()
            && propose_graphable(
                grammar_bitmask.is_some(),
                ctx.levers.shadow_topk,
                crate::speculative::draft_conf_tau(),
                ctx.profile,
                debug_norms,
            );

        if graphable {
            self.replay_or_capture_propose(ctx, stream, &kv)?;
        } else {
            self.propose_gpu_to_argmax(ctx, stream, &kv)?;
        }

        let v = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };
        let logits = ctx.buffers.logits();
        let final_normed = ctx.buffers.norm_output();

        if debug_norms {
            ctx.gpu.synchronize(stream).ok();
            tracing::warn!(
                "MTP_DEBUG_NORMS: ||input_hidden||={:.4} ||final_normed||={:.4} ||logits||={:.4}",
                mtp_dbg_l2(ctx.gpu, target_hidden, h as usize),
                mtp_dbg_l2(ctx.gpu, final_normed, h as usize),
                mtp_dbg_l2(ctx.gpu, logits, v as usize),
            );
        }

        // Drafter chain confidence (ATLAS_MTP_DRAFT_CONF > 0): observational
        // only. D2H the BF16 logits and fold this draft's top-1 softmax into
        // `last_conf_bits`. Ineligible for CUDA graphs (see propose_graphable).
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
                let top1 = (1.0 / denom.max(1.0)) as f32;
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

        let shadow_k = ctx.levers.shadow_topk;
        if shadow_k > 0 {
            let vocab = v as usize;
            let mut bf16_buf = vec![0u8; vocab * 2];
            if ctx.gpu.copy_d2h(logits, &mut bf16_buf).is_ok() {
                let at = |i: usize| -> f32 {
                    let hi = u16::from_le_bytes([bf16_buf[2 * i], bf16_buf[2 * i + 1]]);
                    f32::from_bits((hi as u32) << 16)
                };
                let mut top: Vec<(f32, usize)> = Vec::with_capacity(shadow_k + 1);
                for i in 0..vocab {
                    let x = at(i);
                    if top.len() < shadow_k || x > top.last().map(|t| t.0).unwrap_or(f32::MIN) {
                        let pos = top.partition_point(|t| t.0 >= x);
                        top.insert(pos, (x, i));
                        top.truncate(shadow_k);
                    }
                }
                let max = top.first().map(|t| t.0).unwrap_or(0.0);
                let mut denom = 0.0f64;
                for i in 0..vocab {
                    denom += ((at(i) - max) as f64).exp();
                }
                let ids: Vec<usize> = top.iter().map(|t| t.1).collect();
                let probs: Vec<f32> = top
                    .iter()
                    .map(|t| (((t.0 - max) as f64).exp() / denom.max(1e-30)) as f32)
                    .collect();
                tracing::info!("SHADOW_TOPK pos={position} ids={ids:?} probs={probs:?}");
            }
        }

        let out_ptr = ctx.buffers.scratch();
        let token_id = if let Some(bitmask) = grammar_bitmask {
            // Grammar-masked CPU argmax. D2H the full logits, mask, argmax.
            // GPU argmax already ran (writes scratch); this overwrites it.
            let vocab = v as usize;
            let mut bf16_buf = vec![0u8; vocab * 2];
            ctx.gpu.copy_d2h(logits, &mut bf16_buf)?;

            let mut f32_logits = vec![0.0f32; vocab];
            for i in 0..vocab {
                let hi = u16::from_le_bytes([bf16_buf[2 * i], bf16_buf[2 * i + 1]]);
                f32_logits[i] = f32::from_bits((hi as u32) << 16);
            }

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

            if !any_allowed {
                tracing::warn!(
                    "MTP grammar mask allowed zero tokens at pos {position}; \
                     returning 0 as pad-draft (will be rejected at verify)."
                );
                0u32
            } else {
                let mut best_tok = 0u32;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &val) in f32_logits.iter().enumerate() {
                    if val > best_val {
                        best_val = val;
                        best_tok = i as u32;
                    }
                }
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
        } else if let Some(embed_target) = draft_embed_target {
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
            0u32
        } else {
            let mut buf = [0u8; 4];
            ctx.gpu.copy_d2h(out_ptr, &mut buf)?;
            u32::from_le_bytes(buf)
        };

        state.seq_len += 1;
        state.last_pair_key = Some(position.saturating_sub(1));
        Ok(token_id)
    }
}
