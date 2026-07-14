// SPDX-License-Identifier: AGPL-3.0-only

//! Radix tree prefix cache for KV block reuse.
//!
//! Token sequences are chunked at `block_size` granularity. Each node in
//! the tree corresponds to one KV cache block. Lookup walks the tree
//! matching block-aligned chunks, returning cached physical block indices.
//!
//! Thread-safe via `Mutex<RadixTreeInner>`.

use parking_lot::Mutex;

use crate::prefix_cache::{EvictedBlocks, PrefixCache, PrefixMatch};

mod inner;
mod snapshot;

#[cfg(test)]
mod tests;

use inner::RadixTreeInner;
use snapshot::SsmSnapshotIndex;

/// FNV-1a-ish stable hash for the first `count` tokens — used to key SSM
/// snapshots independently of the radix tree (allows the same prefix hash to be
/// reproduced across requests).
///
/// Task #24 (adapter-correct KV): `adapter_id` is folded in so two adapters that
/// share a token prefix key to DIFFERENT snapshot hashes (no cross-adapter SSM
/// restore). The fold is a strict no-op when `adapter_id == 0` (the base / no-
/// adapter sentinel), so base keying is BYTE-IDENTICAL to the pre-LoRA hash and
/// existing prefix-cache/snapshot hit rates are unchanged.
pub(crate) fn hash_token_prefix(tokens: &[u32], count: usize, adapter_id: u64) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a basis
    if adapter_id != 0 {
        h ^= adapter_id;
        h = h.wrapping_mul(0x100000001b3);
    }
    for &t in &tokens[..count] {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Thread-safe radix tree prefix cache.
///
/// SSM snapshots are stored in a separate `SsmSnapshotIndex`, decoupled from
/// tree node lifetime. This ensures snapshots survive KV cache eviction.
/// Lock ordering: acquire `inner` first (then release), then `snapshot_index`.
pub struct RadixTree {
    inner: Mutex<RadixTreeInner>,
    snapshot_index: Mutex<SsmSnapshotIndex>,
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixTree {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(RadixTreeInner::new()),
            snapshot_index: Mutex::new(SsmSnapshotIndex::new()),
        }
    }
}

impl PrefixCache for RadixTree {
    fn lookup(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> PrefixMatch {
        // Phase 1: walk tree (lock inner, then release)
        let (matched_blocks, matched_disk_block_ids, matched_tokens) = {
            let mut inner = self.inner.lock();
            let (blocks, disk, matched) = inner.walk(tokens, block_size, adapter_id);
            if matched > 0 {
                inner.inc_refs(tokens, block_size, matched, adapter_id);
                crate::prefix_cache::record_cache_hit(matched);
            } else {
                crate::prefix_cache::record_cache_miss();
            }
            (blocks, disk, matched)
        };
        // Phase 2: snapshot lookup (lock snapshot_index, inner NOT held).
        // Tier-aware: `lookup_tiered` returns the deepest anchor across resident
        // AND spilled entries. A resident hit populates `ssm_snapshot` (restore
        // directly); a spilled hit populates `ssm_snapshot_tier_key` (caller
        // faults it in). When nothing is spilled (ATLAS_SSM_TIER off) this is
        // byte-identical to the old resident-only lookup.
        let mut ssm_snapshot = None;
        let mut ssm_snapshot_tokens = 0;
        let mut ssm_snapshot_tier_key = None;
        let mut ssm_snapshot_tier_tokens = 0;
        if matched_tokens > 0 {
            let mut idx = self.snapshot_index.lock();
            if let Some(m) = idx.lookup_tiered(tokens, matched_tokens, session_hash, adapter_id) {
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
        // Filter disk_block_ids to MAX-free entries when HSS isn't in use, so
        // the caller can check `!matched_disk_block_ids.is_empty()` as the
        // HSS-engaged signal. When HSS *is* in use every entry should be a
        // valid disk_id (not MAX).
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
        }
    }

    fn peek_matched_tokens(&self, tokens: &[u32], block_size: usize, adapter_id: u64) -> usize {
        self.inner.lock().walk(tokens, block_size, adapter_id).2
    }

    fn insert(
        &self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        matched_tokens: usize,
        adapter_id: u64,
    ) -> Vec<u32> {
        self.inner.lock().insert(
            tokens,
            block_table,
            disk_block_ids,
            block_size,
            matched_tokens,
            adapter_id,
        )
    }

    fn insert_with_snapshot(
        &self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        snapshot_id: usize,
        session_hash: u64,
        matched_tokens: usize,
        adapter_id: u64,
    ) -> (Option<usize>, Vec<u32>) {
        // Phase 1: insert tree nodes (lock inner, then release)
        let newly_acquired = self.inner.lock().insert(
            tokens,
            block_table,
            disk_block_ids,
            block_size,
            matched_tokens,
            adapter_id,
        );
        // Phase 2: register snapshot in index (lock snapshot_index, inner NOT held)
        let prefix_hash = hash_token_prefix(tokens, tokens.len(), adapter_id);
        let mut idx = self.snapshot_index.lock();
        let displaced = idx.insert(prefix_hash, snapshot_id, session_hash, tokens.len());
        (displaced, newly_acquired)
    }

    fn insert_intermediate_snapshot(
        &self,
        tokens: &[u32],
        _block_table: &[u32],
        _disk_block_ids: &[u32],
        _block_size: usize,
        snapshot_id: usize,
        session_hash: u64,
        _matched_tokens: usize,
        adapter_id: u64,
    ) -> Option<usize> {
        // Intermediate snapshots go directly into the index with the correct
        // token boundary (tokens.len()). Tree nodes are already inserted by
        // a prior `insert()` call, which handled the ref_count bookkeeping.
        let prefix_hash = hash_token_prefix(tokens, tokens.len(), adapter_id);
        let mut idx = self.snapshot_index.lock();
        idx.insert(prefix_hash, snapshot_id, session_hash, tokens.len())
    }

    fn release(&self, tokens: &[u32], block_size: usize, adapter_id: u64) {
        self.inner.lock().dec_refs(tokens, block_size, adapter_id);
    }

    fn evict(&self, num_blocks: usize) -> EvictedBlocks {
        let (physical, disk) = self.inner.lock().evict(num_blocks);
        // Filter MAX sentinels out — the caller only needs disk_block_ids to
        // dec_disk_ref on, and MAX entries don't correspond to a live HSS ref.
        let disk_block_ids: Vec<u32> = disk.into_iter().filter(|&id| id != u32::MAX).collect();
        EvictedBlocks {
            physical,
            disk_block_ids,
        }
    }

    fn evict_snapshot_lru(&self) -> Option<usize> {
        self.snapshot_index.lock().evict_lru()
    }

    fn evict_snapshot_to_tier(&self) -> Option<(usize, u64)> {
        self.snapshot_index.lock().evict_to_tier()
    }

    fn promote_snapshot(&self, key: u64, new_slot: usize) -> bool {
        self.snapshot_index.lock().promote(key, new_slot)
    }

    fn snapshot_count(&self) -> usize {
        self.snapshot_index.lock().len()
    }

    fn stats(&self) -> (usize, usize) {
        let inner = self.inner.lock();
        let entries = inner.num_entries();
        (entries, entries)
    }
}
