// SPDX-License-Identifier: AGPL-3.0-only

//! Shared prefix-cache lookup: KV walk + SSM index, then optional pairing.
//!
//! `lookup` records a KV-only hit (pure-attention models). `lookup_paired`
//! is the hybrid/GDN serve entry: a hit is KV blocks AND a restorable SSM
//! snapshot at the same length. Unpaired KV is released and counted a miss.

use crate::prefix_cache::{PrefixCache, PrefixMatch};

use super::RadixTree;
use super::snapshot;

impl RadixTree {
    /// Walk + snapshot index + `inc_refs`. Does not touch hit/miss counters.
    pub(super) fn lookup_uncounted(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> PrefixMatch {
        let (matched_blocks, matched_disk_block_ids, matched_tokens) = {
            let mut inner = self.inner.lock();
            let (blocks, disk, matched) = inner.walk(tokens, block_size, adapter_id);
            if matched > 0 {
                inner.inc_refs(tokens, block_size, matched, adapter_id);
            }
            (blocks, disk, matched)
        };
        let mut ssm_snapshot = None;
        let mut ssm_snapshot_tokens = 0;
        let mut ssm_snapshot_tier_key = None;
        let mut ssm_snapshot_tier_tokens = 0;
        let mut ssm_snapshot_is_tail = false;
        if matched_tokens > 0 {
            let mut idx = self.snapshot_index.lock();
            if let Some(m) = idx.lookup_tiered(tokens, matched_tokens, session_hash, adapter_id) {
                ssm_snapshot_is_tail = m.is_tail;
                match m.loc {
                    snapshot::SnapLoc::Hbm(slot) => {
                        ssm_snapshot = Some(slot);
                        ssm_snapshot_tokens = m.token_count;
                    }
                    snapshot::SnapLoc::Tier(key) => {
                        ssm_snapshot_tier_key = Some(key);
                        ssm_snapshot_tier_tokens = m.token_count;
                    }
                }
            }
        }
        let matched_disk_block_ids = if matched_disk_block_ids.iter().all(|&id| id == u32::MAX) {
            Vec::new()
        } else {
            matched_disk_block_ids
        };
        PrefixMatch {
            matched_blocks,
            matched_disk_block_ids,
            matched_tokens,
            ssm_snapshot,
            ssm_snapshot_tokens,
            ssm_snapshot_tier_key,
            ssm_snapshot_tier_tokens,
            ssm_snapshot_is_tail,
        }
    }

    /// Couple a raw match for hybrid/GDN: keep only a restorable SSM length,
    /// or release the KV refs and return empty. Records hit/miss here.
    pub(super) fn pair_or_miss(
        &self,
        tokens: &[u32],
        block_size: usize,
        adapter_id: u64,
        mut m: PrefixMatch,
    ) -> PrefixMatch {
        match m.paired_ssm_tokens(block_size) {
            None => {
                if m.matched_tokens > 0 {
                    tracing::info!(
                        "Prefix cache miss: {} KV tokens ({} blocks) without a restorable \
                         SSM snapshot at that length — full prefill",
                        m.matched_tokens,
                        m.matched_blocks.len(),
                    );
                    self.release_matched(tokens, block_size, m.matched_tokens, adapter_id);
                }
                crate::prefix_cache::record_cache_miss();
                PrefixMatch::empty()
            }
            Some(paired) if paired < m.matched_tokens => {
                {
                    let mut inner = self.inner.lock();
                    inner.dec_refs(tokens, block_size, m.matched_tokens, adapter_id);
                    inner.inc_refs(tokens, block_size, paired, adapter_id);
                }
                m.trim_to_paired_len(paired, block_size);
                crate::prefix_cache::record_cache_hit(paired);
                m
            }
            Some(paired) => {
                crate::prefix_cache::record_cache_hit(paired);
                m
            }
        }
    }
}
