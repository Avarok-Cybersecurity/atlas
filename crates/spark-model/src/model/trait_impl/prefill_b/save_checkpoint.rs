// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 9 — non-last chunk: save SSM snapshot at chunked-prefill block
//! boundaries (Marconi intermediate checkpoint). On future partial
//! prefix hits, restoring from the deepest intermediate checkpoint
//! avoids recomputing SSM for the entire prefix.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    pub(super) fn prefill_b_save_checkpoint(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        stream: u64,
    ) -> Result<()> {
        if self.ssm_checkpoint_interval == 0 || !self.ssm_snapshots.is_enabled() {
            return Ok(());
        }
        let bs = kv_cache.block_size();
        let end_token = chunk_start + chunk_len;
        let end_block = end_token / bs;
        if end_block == 0 || !end_block.is_multiple_of(self.ssm_checkpoint_interval) {
            return Ok(());
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
                // Pool exhausted — try to reclaim from cache
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
                    tracing::warn!(
                        "SSM snapshot pool exhausted and no evictable cached entries — \
                         dropping checkpoint for this chunk. Long-context prefix-cache \
                         hits will recompute SSM state. Consider raising \
                         --ssm-cache-slots."
                    );
                    None
                }
            }
            Err(e) => {
                tracing::warn!("SSM snapshot save error: {e}");
                None
            }
        };
        let Some(snap_id) = snap_result else {
            return Ok(());
        };

        let boundary_tokens = &tokens[..end_token];
        // Phase 6.3 sliding-window: when HSS is engaged AND sliding has begun
        // (hss_window_start > 0), the front of the prefix is no longer
        // represented by physical HBM blocks — the rolling-window slice
        // would mis-represent the cached entry. Skip the prefix-cache insert
        // in that case; the SSM snapshot is freed to avoid leaks.
        let skip_boundary_insert = seq.hss_window_start() > 0 || end_block > seq.block_table.len();
        if skip_boundary_insert {
            self.ssm_snapshots.free(snap_id);
            return Ok(());
        }
        let boundary_blocks = &seq.block_table[..end_block];
        // Vision chunks: skip both the radix insert and the SSM snapshot
        // attach — the placeholder token stream is identical for distinct
        // images of the same prompt, so a future hit would resurrect the
        // prior image's state.
        if self.tokens_have_vision_pad(boundary_tokens) {
            self.ssm_snapshots.free(snap_id);
            return Ok(());
        }
        let boundary_disk = if seq.disk_block_ids.len() >= end_block {
            &seq.disk_block_ids[..end_block]
        } else {
            &[][..]
        };
        // Intermediate checkpoint inserts tree nodes as a placeholder for
        // the snapshot boundary — the final chunk's insert will bump
        // ref_counts for seq-ownership over the full tokens range. Passing
        // matched_tokens = end_token here makes is_seq_owned=false for every
        // block in this checkpoint, avoiding a double-bump that would leak
        // the cache's baseline ref by +1 per checkpointed block after the
        // eventual release().
        self.prefix_cache.insert(
            boundary_tokens,
            boundary_blocks,
            boundary_disk,
            bs,
            end_token,
        );
        if let Some(old) = self.prefix_cache.insert_intermediate_snapshot(
            boundary_tokens,
            boundary_blocks,
            boundary_disk,
            bs,
            snap_id,
            seq.session_hash,
            end_token,
        ) {
            self.ssm_snapshots.free(old);
        }
        tracing::info!(
            "Intermediate SSM checkpoint saved at token {} (snapshot_id {}, block {})",
            end_token,
            snap_id,
            end_block,
        );
        Ok(())
    }
}
