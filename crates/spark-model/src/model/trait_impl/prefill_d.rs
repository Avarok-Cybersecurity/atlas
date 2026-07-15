// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! Post-prefill snapshot + prefix-cache insert helper.
//!
//! Hoisted from `prefill_c.rs` (`prefill_twophase_dispatch`) to keep that
//! file under the 500 LoC cap. The single helper here mirrors the original
//! "section 8" block 1:1 — Marconi snapshot save (with reclaim retry on
//! pool full) followed by a prefix-cache insert that prefers the snapshot
//! flavour when one was successfully saved.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::types::TransformerModel;
use crate::layers::ops;
use crate::traits::SequenceState;

impl TransformerModel {
    /// Full-prompt cache-hit fast path: re-embed just the last token,
    /// run final norm + LM head, insert into the prefix cache, and
    /// return the decode logits pointer. Hoisted from `prefill_c.rs`'s
    /// `if proc_count == 0` branch.
    pub(super) fn prefill_full_cache_hit(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        hidden: DevicePtr,
        h: u32,
        bs: usize,
        total_len: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        seq.tokens.extend_from_slice(tokens);
        seq.seq_len = total_len;
        // Re-embed just the last token at hidden[0] for final norm + LM head.
        let last_tok = tokens[total_len - 1];
        let last_tok_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(&last_tok as *const u32 as *const u8, 4) };
        let token_id_dev = self.buffers.scratch();
        self.gpu
            .copy_h2d_async(last_tok_bytes, token_id_dev, stream)?;
        ops::batched_embed(
            self.gpu.as_ref(),
            self.batched_embed_kernel,
            token_id_dev,
            self.embed_tokens.weight,
            hidden,
            1,
            h,
            stream,
        )?;
        self.scale_embeddings(hidden, 1usize, stream)?;
        // Run final norm + LM head on the single re-embedded token.
        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            hidden,
            &self.final_norm,
            normed,
            1,
            h,
            eps,
            stream,
        )?;
        self.lm_head(normed, stream)?;
        // Prefix cache insert (no new snapshot needed — SSM state unchanged).
        if !self.tokens_have_vision_pad(tokens) {
            let acquired = self.prefix_cache.insert(
                tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                bs,
                seq.cached_prefix_tokens,
                seq.adapter_id,
            );
            super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
        }
        Ok(self.decode_logits_ptr())
    }

    /// Same as [`Self::prefill_save_snapshot_and_insert`] but with the
    /// vision-pad gate that the standard `prefill_dispatch` path uses:
    /// when the prompt has vision-pad tokens, the SSM snapshot is image-
    /// tainted and gets freed (radix block also skipped) so a different
    /// image's token stream can't collide with this entry on lookup.
    pub(super) fn prefill_save_snapshot_with_vision_gate(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        bs: usize,
        stream: u64,
    ) {
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
                    // Pool exhausted — evict LRU entries to reclaim a slot
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
                        None
                    }
                }
                Err(_) => None,
            };
            if let Some(snap_id) = snap_result {
                // Order any later warm restore after this save's D2D (the
                // restore may run on a different stream under concurrency).
                if let Err(e) = self.record_snapshot_save_dispatch(stream) {
                    tracing::warn!("prefill snapshot save: record snapshot event: {e}");
                }
                if self.tokens_have_vision_pad(tokens) {
                    // Vision prefill: snapshot is image-tainted and the
                    // token stream collides across distinct images, so do
                    // not admit either the snapshot or the radix block.
                    self.ssm_snapshots.free(snap_id);
                } else {
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
                    super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
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
                super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
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
            super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
        }
    }

    /// Save a Marconi SSM snapshot for the prefilled sequence (reclaiming
    /// from the prefix cache on pool exhaustion) and insert into the
    /// prefix cache. Falls back to the snapshot-less insert when the
    /// snapshot pool is unavailable or vision-pad tokens are present.
    pub(super) fn prefill_save_snapshot_and_insert(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        bs: usize,
        stream: u64,
    ) {
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
                // Order any later warm restore after this save's D2D (the
                // restore may run on a different stream under concurrency).
                if let Err(e) = self.record_snapshot_save_dispatch(stream) {
                    tracing::warn!("prefill snapshot save [twophase]: record snapshot event: {e}");
                }
                if self.tokens_have_vision_pad(tokens) {
                    // Vision prefill: the SSM snapshot is image-tainted and
                    // the token stream collides across distinct images (the
                    // prefix-cache key hashes token IDs only, and image-pad
                    // placeholders are identical regardless of pixel content),
                    // so admitting this entry returns a stale image's result
                    // on the next same-prompt request (issue #58). Free the
                    // snapshot and skip the radix insert, matching the gated
                    // standard path in `prefill_save_snapshot_with_vision_gate`.
                    self.ssm_snapshots.free(snap_id);
                } else {
                    tracing::info!(
                        "Saved SSM snapshot {} for {} tokens ({} blocks) [twophase]",
                        snap_id,
                        tokens.len(),
                        seq.block_table.len(),
                    );
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
                    super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
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
                super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
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
            super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
        }
    }
}
