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
mod lookup;
mod snapshot;
mod snapshot_insert;
mod snapshot_stats;
mod snapshot_tier;

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
        let m = self.lookup_uncounted(tokens, block_size, session_hash, adapter_id);
        if m.matched_tokens > 0 {
            crate::prefix_cache::record_cache_hit(m.matched_tokens);
        } else {
            crate::prefix_cache::record_cache_miss();
        }
        m
    }

    fn lookup_paired(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> PrefixMatch {
        let m = self.lookup_uncounted(tokens, block_size, session_hash, adapter_id);
        self.pair_or_miss(tokens, block_size, adapter_id, m)
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
    ) -> crate::prefix_cache::InsertAcquired {
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
    ) -> (Option<usize>, crate::prefix_cache::InsertAcquired) {
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

    fn insert_tail_snapshot(
        &self,
        tokens: &[u32],
        snapshot_id: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> Vec<usize> {
        // Index only. The tree nodes for [0, tokens.len()) are inserted by the
        // final chunk's `insert` (finalize_last); re-inserting the whole prefix
        // here cost ~0.9 s/turn for zero benefit.
        let prefix_hash = hash_token_prefix(tokens, tokens.len(), adapter_id);
        self.snapshot_index
            .lock()
            .insert_tail(prefix_hash, snapshot_id, session_hash, tokens.len())
    }

    fn insert_tail_sibling_snapshot(
        &self,
        tokens: &[u32],
        snapshot_id: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> Option<usize> {
        // Index only, like the tail (finalize_last's insert lays the tree nodes).
        let prefix_hash = hash_token_prefix(tokens, tokens.len(), adapter_id);
        self.snapshot_index.lock().insert_tail_sibling(
            prefix_hash,
            snapshot_id,
            session_hash,
            tokens.len(),
        )
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
        self.inner
            .lock()
            .dec_refs(tokens, block_size, tokens.len(), adapter_id);
    }

    fn release_matched(
        &self,
        tokens: &[u32],
        block_size: usize,
        matched_tokens: usize,
        adapter_id: u64,
    ) {
        self.inner
            .lock()
            .dec_refs(tokens, block_size, matched_tokens, adapter_id);
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

    fn evict_snapshot_to_tier(&self, min_tokens: usize) -> Option<crate::prefix_cache::TierEvict> {
        self.snapshot_index.lock().evict_to_tier(min_tokens)
    }

    fn promote_snapshot(&self, key: u64, new_slot: usize) -> bool {
        self.snapshot_index.lock().promote(key, new_slot)
    }

    fn forget_snapshot_tier_key(&self, key: u64) -> bool {
        self.snapshot_index.lock().forget_tiered(key)
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
