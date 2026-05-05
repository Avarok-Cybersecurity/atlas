// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 5+6+7+8 — last-chunk finalization:
//!   • final RMS-norm on the last token's hidden state
//!   • LM head → logits buffer
//!   • diagnostic dumps (long-context / Gemma4 paths)
//!   • prefix-cache insert + Marconi snapshot save (with reclaim retry)
//!   • DFlash ctx-len bookkeeping

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::layers::ops;
use crate::traits::SequenceState;

impl TransformerModel {
    pub(super) fn prefill_b_finalize_last(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = self.config.hidden_size;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let hidden = self.buffers.hidden_states();
        let bs = kv_cache.block_size();

        // ── 6. Final norm on LAST token only ──
        let last_hidden = hidden.offset((proc_count - 1) * h * fp32);
        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            last_hidden,
            &self.final_norm,
            normed,
            1,
            h as u32,
            eps,
            stream,
        )?;

        // Diagnostic: post-norm hidden state
        if (chunk_start + chunk_len) > 16384
            || std::env::var("ATLAS_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true")
        {
            self.gpu.synchronize(stream)?;
            let (vals, norm) = self.readback_bf16(normed, h.min(16))?;
            tracing::warn!(
                "DIAG post-norm: norm={norm:.4} first2={:.4?}",
                &vals[..2.min(vals.len())]
            );
        }

        // ── 7. LM head on last token → logits ──
        self.lm_head(normed, stream)?;

        // Diagnostic: logits stats
        if (chunk_start + chunk_len) > 16384
            || std::env::var("ATLAS_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true")
        {
            self.gpu.synchronize(stream)?;
            let logits_ptr = self.buffers.logits();
            let n_logits = self.config.vocab_size;
            let mut buf = vec![0u8; n_logits * 2];
            self.gpu.copy_d2h(logits_ptr, &mut buf)?;
            let logit_vals: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let max = logit_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = logit_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let nan_count = logit_vals.iter().filter(|v| v.is_nan()).count();
            let mut idx: Vec<usize> = (0..logit_vals.len()).collect();
            idx.sort_by(|&a, &b| {
                logit_vals[b]
                    .partial_cmp(&logit_vals[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let top5: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, logit_vals[i])).collect();
            tracing::warn!(
                "DIAG logits[0..{}]: max={max:.4} min={min:.4} nan={nan_count} top5={top5:?}",
                n_logits,
            );
        }

        // ── 8. Insert into prefix cache + Marconi snapshot ──
        if self.ssm_snapshots.is_enabled() {
            let snap_result = match self.ssm_snapshots.save(
                seq.slot_idx,
                seq.session_hash,
                &self.ssm_pool,
                self.gpu.as_ref(),
                stream,
            ) {
                Ok(Some(id)) => Some(id),
                Ok(None) => {
                    tracing::debug!("Snapshot pool full, reclaiming...");
                    if self
                        .ssm_snapshots
                        .reclaim_from_cache(self.prefix_cache.as_ref(), kv_cache)
                    {
                        self.ssm_snapshots
                            .save(
                                seq.slot_idx,
                                seq.session_hash,
                                &self.ssm_pool,
                                self.gpu.as_ref(),
                                stream,
                            )
                            .ok()
                            .flatten()
                    } else {
                        tracing::debug!("Reclaim failed — no evictable snapshots");
                        None
                    }
                }
                Err(e) => {
                    tracing::warn!("SSM snapshot save error: {e}");
                    None
                }
            };
            if let Some(snap_id) = snap_result {
                if self.tokens_have_vision_pad(tokens) {
                    self.ssm_snapshots.free(snap_id);
                } else {
                    tracing::info!(
                        "Saved SSM snapshot {} for {} tokens ({} blocks) [chunk]",
                        snap_id,
                        tokens.len(),
                        seq.block_table.len(),
                    );
                    if let Some(old) = self.prefix_cache.insert_with_snapshot(
                        tokens,
                        &seq.block_table,
                        &seq.disk_block_ids,
                        bs,
                        snap_id,
                        seq.session_hash,
                        seq.cached_prefix_tokens,
                    ) {
                        self.ssm_snapshots.free(old);
                    }
                }
            } else if !self.tokens_have_vision_pad(tokens) {
                self.prefix_cache.insert(
                    tokens,
                    &seq.block_table,
                    &seq.disk_block_ids,
                    bs,
                    seq.cached_prefix_tokens,
                );
            }
        } else if !self.tokens_have_vision_pad(tokens) {
            self.prefix_cache.insert(
                tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                bs,
                seq.cached_prefix_tokens,
            );
        }

        // DFlash: advance ctx_len after the LAST chunk of chunked prefill.
        self.update_dflash_ctx_len_after_prefill(seq, chunk_start, chunk_len)?;

        Ok(self.decode_logits_ptr())
    }
}
