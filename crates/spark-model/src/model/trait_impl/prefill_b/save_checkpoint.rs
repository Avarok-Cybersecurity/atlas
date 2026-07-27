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
    pub(in crate::model) fn prefill_b_save_checkpoint(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        stream: u64,
    ) -> Result<()> {
        if !self.ssm_snapshots.is_enabled() {
            return Ok(());
        }
        let bs = kv_cache.block_size();
        let end_token = chunk_start + chunk_len;
        let end_block = end_token / bs;
        // Tail checkpoints (issue #15 follow-up): the last two block
        // boundaries below the prompt end bracket the next turn's
        // block-aligned radix match (divergence sits within the template's
        // generation-only suffix, < block_size tokens before `total`), so a
        // snapshot at each makes warm multi-turn restores work regardless of
        // --ssm-checkpoint-interval. The final chunk is split at these
        // boundaries by `prefill_chunk_dispatch`. Interval checkpoints
        // additionally fire at chunk boundaries that are interval-block
        // multiples (with full-size chunks that granularity is coarse — the
        // tail checkpoints + leaf carry the warm path).
        let tail = (tokens.len().saturating_sub(1) / bs) * bs;
        let is_prompt_tail = end_token == tail || (tail >= bs && end_token == tail - bs);
        // NOTE (2026-07-21, dgx2 SSM audit): `--ssm-checkpoint-interval` is a
        // FILTER over chunk boundaries, not a generator of them. This
        // function only runs at a chunk end, so the effective checkpoint
        // spacing is the CHUNK size, not the interval — the interval can only
        // suppress boundaries, never create one. The auto-clamp that used to
        // force the prefill budget down to `interval * block_size` was
        // removed deliberately (impl_a1.rs, issue #15, 2026-07-02) because it
        // forced micro-chunked prefill.
        //
        // Consequence to be aware of when reading a serve log: with
        // `--ssm-checkpoint-interval 32` (32 blocks = 512 tokens at bs=16)
        // and `--max-prefill-tokens 8192`, chunk ends land on blocks 512,
        // 1024, ... — every one of which is a multiple of 32 — so interval
        // checkpoints fire every 8192 tokens, NOT every 512. The warm path is
        // carried by the tail checkpoints and the leaf above, which is why
        // this is not currently a correctness problem.
        //
        // Making the interval a real generator (splitting chunks at interval
        // boundaries) is a behaviour change with a prefill-throughput cost
        // and is deliberately NOT made here; it needs its own measured A/B.
        let on_interval = self.ssm_checkpoint_interval > 0
            && end_block.is_multiple_of(self.ssm_checkpoint_interval);
        if end_block == 0 || !(is_prompt_tail || on_interval) {
            return Ok(());
        }
        // Stale-V cap (mirrors finalize_last): never checkpoint-cache a block
        // past the contiguous fully-written-KV prefix. If this boundary's
        // blocks aren't all KV-valid yet, skip the intermediate insert rather
        // than cache stale V.
        if seq.kv_valid_tokens / bs < end_block {
            tracing::debug!(
                "Skip intermediate checkpoint at block {end_block}: \
                 kv_valid_tokens={} only covers {} complete blocks",
                seq.kv_valid_tokens,
                seq.kv_valid_tokens / bs,
            );
            return Ok(());
        }
        if std::env::var("ATLAS_SSM_SAVE_DUMP").is_ok() {
            self.ssm_pool.debug_state_checksum(
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
                &format!("ckpt_save@{end_token}"),
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
                // Pool exhausted — try to reclaim from cache
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
        let acquired = self.prefix_cache.insert(
            boundary_tokens,
            boundary_blocks,
            boundary_disk,
            bs,
            end_token,
            seq.adapter_id,
        );
        super::super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
        if let Some(old) = self.prefix_cache.insert_intermediate_snapshot(
            boundary_tokens,
            boundary_blocks,
            boundary_disk,
            bs,
            snap_id,
            seq.session_hash,
            end_token,
            seq.adapter_id,
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
