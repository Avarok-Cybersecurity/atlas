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
    pub(in crate::model) fn prefill_b_finalize_last(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_b_finalize_last_at(
            tokens,
            seq,
            kv_cache,
            chunk_start,
            chunk_len,
            proc_count,
            0,
            0,
            stream,
        )
    }

    /// Q12 Path B: stream-offset-aware finalize for the kernel-batched
    /// orchestrator. `hidden_stream_offset_tokens` is `b * chunk_len`
    /// where `b` is the stream's index in the batched dispatch.
    /// All other args identical to `prefill_b_finalize_last`.
    pub(in crate::model) fn prefill_b_finalize_last_at(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        proc_count: usize,
        hidden_stream_offset_tokens: usize,
        logits_row: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = self.config.hidden_size;
        let fp32 = 2usize;
        let hidden = self.buffers.hidden_states();
        let bs = kv_cache.block_size();

        if std::env::var("ATLAS_SSM_SAVE_DUMP").is_ok() {
            self.ssm_pool.debug_state_checksum(
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
                &format!("final_state@{}", tokens.len()),
            );
            // PER-LAYER KV fingerprint over the WHOLE block table, split at the
            // reused-prefix / recomputed-suffix boundary (`marconi_skip_to`
            // tokens → `/bs` blocks). On cache-OFF, `marconi_skip_to==0` so all
            // blocks are "suffix" (cold recompute). On cache-ON chained,
            // `[0, boundary)` are reused-prefix blocks carried from a prior turn
            // and `[boundary, end)` are recomputed. Compare cold-vs-chained
            // per (layer, region) to localize the first divergent K/V.
            let boundary_idx = seq.marconi_skip_to / bs;
            kv_cache.debug_kv_checksum_per_layer(
                &seq.block_table,
                boundary_idx,
                self.gpu.as_ref(),
                stream,
                &format!("final@{}/skip{}", tokens.len(), seq.marconi_skip_to),
            );
            // Per-LOGICAL-BLOCK fingerprint of a FIXED ABSOLUTE window
            // (logical blocks 250..266 — straddles the 4097/bs=256 restore
            // boundary) for L0 AND L9 so OFF and ON dump the SAME logical
            // positions and a per-position aliasing bug (identical region SUM
            // but wrong block→position mapping) is directly comparable.
            let _ = boundary_idx;
            let lo = 250usize.min(seq.block_table.len());
            let hi = 266usize.min(seq.block_table.len());
            if hi > lo {
                for layer_idx in [0usize, 9usize] {
                    kv_cache.debug_kv_per_block(
                        layer_idx,
                        &seq.block_table[lo..hi],
                        self.gpu.as_ref(),
                        stream,
                        &format!(
                            "abswin@{}/skip{}/lo{}",
                            tokens.len(),
                            seq.marconi_skip_to,
                            lo
                        ),
                    );
                }
            }
            let tail_lo = seq.block_table.len().saturating_sub(3);
            kv_cache.debug_kv_per_block(
                0,
                &seq.block_table[tail_lo..],
                self.gpu.as_ref(),
                stream,
                &format!("tail@{}/lo{}", tokens.len(), tail_lo),
            );
            tracing::warn!(
                "ATLAS_BTBL[final@{}/skip{}] nblk={} bt={:?}",
                tokens.len(),
                seq.marconi_skip_to,
                seq.block_table.len(),
                seq.block_table,
            );
        }

        // ── 6. Final norm on LAST token only ──
        let last_token_offset = hidden_stream_offset_tokens + proc_count - 1;
        let last_hidden = hidden.offset(last_token_offset * h * fp32);
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
        if std::env::var("ATLAS_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true") {
            self.gpu.synchronize(stream)?;
            let (vals, norm) = self.readback_bf16(normed, h.min(16))?;
            tracing::warn!(
                "DIAG post-norm: norm={norm:.4} first2={:.4?}",
                &vals[..2.min(vals.len())]
            );
        }

        // Per-layer divergence dump: final-norm output (input to lm_head).
        if let Ok(dir) = std::env::var("ATLAS_NEMO_DUMP")
            && !dir.is_empty()
        {
            self.gpu.synchronize(stream)?;
            let (vals, _) = self.readback_bf16(normed, h)?;
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::create_dir_all(&dir).ok();
            std::fs::write(
                std::path::Path::new(&dir).join("atlas_final_norm.bin"),
                &bytes,
            )
            .ok();
        }

        // ── 6b. Marconi exact-hit fixup ──
        // On an exact full-prompt leaf hit the last prompt token was re-run
        // for logits (proc_range Compute{N-1, 1}). For SSM layers that re-run
        // applies the last token's recurrent update a second time on top of
        // the restored state@N → double-advance, corrupting both the logits
        // computed above and the pool state decode will read. Undo it:
        //   (1) re-restore the pristine SSM state@N from the snapshot, and
        //   (2) overwrite `normed` with the snapshot's stashed last-token
        //       post-norm hidden so `lm_head` emits the correct first token.
        // The redundant 1-token forward is otherwise harmless (its KV write
        // duplicates already-cached values).
        if let Some(snap_id) = seq.marconi_exact_snap {
            self.ssm_snapshots.restore(
                snap_id,
                seq.slot_idx,
                &self.ssm_pool,
                self.gpu.as_ref(),
                stream,
            )?;
            if self.ssm_snapshots.has_hidden(snap_id) {
                self.ssm_snapshots
                    .restore_hidden(snap_id, normed, self.gpu.as_ref(), stream)?;
            } else {
                // Leaf snapshots always stash the hidden; absence means a
                // non-leaf snapshot was matched at full length. Fail loud
                // rather than silently emit double-advanced logits.
                tracing::warn!(
                    "Marconi exact hit on snapshot {snap_id} without stashed hidden — \
                     first-token logits may be degraded (SSM state restored)"
                );
            }
        }

        // ── 7. LM head on last token → logits ──
        // Co-dispatched streams must each write their OWN logits row, else all
        // streams' first token collapses to one shared buffer (cross-request
        // contamination). Row 0 (single-stream + batched stream 0) keeps the
        // byte-identical `lm_head` GEMV path; rows >0 write to their own offset.
        let logits_ptr = if logits_row == 0 {
            self.lm_head(normed, stream)?;
            self.decode_logits_ptr()
        } else {
            let v = self.config.vocab_size;
            let dst = self.buffers.logits().offset(logits_row * v * 2);
            self.lm_head_batched(normed, 1, dst, stream)?;
            dst
        };

        // Per-layer divergence dump: full logits vector + top-10 token IDs.
        if let Ok(dir) = std::env::var("ATLAS_NEMO_DUMP")
            && !dir.is_empty()
        {
            self.gpu.synchronize(stream)?;
            let n_logits = self.config.vocab_size;
            let mut buf = vec![0u8; n_logits * 2];
            self.gpu.copy_d2h(self.buffers.logits(), &mut buf)?;
            let logit_vals: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let lbytes: Vec<u8> = logit_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::create_dir_all(&dir).ok();
            std::fs::write(std::path::Path::new(&dir).join("atlas_logits.bin"), &lbytes).ok();
            let mut idx: Vec<usize> = (0..logit_vals.len()).collect();
            idx.sort_by(|&a, &b| logit_vals[b].partial_cmp(&logit_vals[a]).unwrap());
            let top: Vec<(usize, f32)> = idx.iter().take(10).map(|&i| (i, logit_vals[i])).collect();
            tracing::info!("ATLAS_NEMO_DUMP: top-10 logits = {top:?}");
        }

        // Diagnostic: logits stats
        if std::env::var("ATLAS_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true") {
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
        //
        // Stale-V cap: only COMPLETE blocks whose K/V was fully written this
        // (or a prior valid) prefill may be cached. `seq.kv_valid_tokens` is
        // the contiguous prefix length with guaranteed-written KV; any trailing
        // complete block past it (e.g. left stale by the `proc_count == 1`
        // last-chunk decode shortcut) is excluded so a future turn never reads
        // donor/zeroed V from it. When the whole prompt's complete blocks are
        // valid we keep the full token range (preserving the partial-suffix
        // sub-block TTFT optimization); otherwise we truncate to the
        // block-aligned valid prefix AND drop the SSM snapshot attach (the
        // snapshot, keyed at full prompt length, would be unreachable through
        // the shortened tree and would only orphan a pool slot).
        let full_blocks = tokens.len() / bs;
        let valid_blocks = seq.kv_valid_tokens / bs;
        let cache_blocks = full_blocks.min(valid_blocks);
        let cap_applied = cache_blocks < full_blocks;
        let cache_tokens_len = if cap_applied {
            cache_blocks * bs
        } else {
            tokens.len()
        };
        let cache_tokens = &tokens[..cache_tokens_len];
        let cache_block_table = &seq.block_table[..cache_blocks.min(seq.block_table.len())];
        let cache_disk_block_ids = if seq.disk_block_ids.is_empty() {
            &seq.disk_block_ids[..]
        } else {
            &seq.disk_block_ids[..cache_blocks.min(seq.disk_block_ids.len())]
        };
        if cap_applied {
            tracing::warn!(
                "Prefix-cache stale-V cap: caching {} of {} complete blocks \
                 (kv_valid_tokens={} < prompt_len={}); trailing blocks had \
                 unwritten K/V and are excluded",
                cache_blocks,
                full_blocks,
                seq.kv_valid_tokens,
                tokens.len(),
            );
        }

        if seq.marconi_exact_snap.is_some() {
            // nothing to save (handled in the exact-hit fixup above)
        } else if cache_blocks == 0 {
            // Nothing safe to cache (no complete block has fully-written KV).
        } else if cap_applied {
            // Cap forced — never attach the full-length snapshot to a
            // truncated tree (would be unreachable + leak a pool slot).
            if !self.tokens_have_vision_pad(cache_tokens) {
                let acquired = self.prefix_cache.insert(
                    cache_tokens,
                    cache_block_table,
                    cache_disk_block_ids,
                    bs,
                    seq.cached_prefix_tokens.min(cache_tokens_len),
                    seq.adapter_id,
                );
                super::super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
            }
        } else if self.ssm_snapshots.is_enabled() {
            if std::env::var("ATLAS_SSM_SAVE_DUMP").is_ok() {
                self.ssm_pool.debug_state_checksum(
                    seq.slot_idx,
                    self.gpu.as_ref(),
                    stream,
                    &format!("leaf_save@{}", tokens.len()),
                );
            }
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
                    if self.ssm_snapshots.reclaim_from_cache(
                        self.prefix_cache.as_ref(),
                        kv_cache,
                        self.ssm_tier_store.as_deref(),
                        self.gpu.as_ref(),
                    ) {
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
                    // Stash the last-token post-norm hidden so a future exact
                    // full-prompt hit can emit the first token's logits without
                    // re-running the last token through the SSM layers. `normed`
                    // still holds the post-final-norm last-token hidden here
                    // (lm_head reads it without mutating).
                    self.ssm_snapshots
                        .save_hidden(snap_id, normed, self.gpu.as_ref(), stream)?;
                    let (displaced, acquired) = self.prefix_cache.insert_with_snapshot(
                        tokens,
                        &seq.block_table,
                        &seq.disk_block_ids,
                        bs,
                        snap_id,
                        seq.session_hash,
                        seq.cached_prefix_tokens,
                        seq.adapter_id,
                    );
                    super::super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
                    if let Some(old) = displaced {
                        self.ssm_snapshots.free(old);
                    }
                }
            } else if !self.tokens_have_vision_pad(tokens) {
                let acquired = self.prefix_cache.insert(
                    tokens,
                    &seq.block_table,
                    &seq.disk_block_ids,
                    bs,
                    seq.cached_prefix_tokens,
                    seq.adapter_id,
                );
                super::super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
            }
        } else if !self.tokens_have_vision_pad(tokens) {
            let acquired = self.prefix_cache.insert(
                tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                bs,
                seq.cached_prefix_tokens,
                seq.adapter_id,
            );
            super::super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
        }

        // DFlash: advance ctx_len after the LAST chunk of chunked prefill.
        self.update_dflash_ctx_len_after_prefill(seq, chunk_start, chunk_len)?;

        Ok(logits_ptr)
    }
}
