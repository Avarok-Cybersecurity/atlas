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

/// FNV-1a-ish stable hash for the first `count` tokens — used to key
/// snapshots independently of the radix tree (allows the same prefix
/// hash to be reproduced across requests).
pub(crate) fn hash_token_prefix(tokens: &[u32], count: usize) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a basis
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
    fn lookup(&self, tokens: &[u32], block_size: usize, session_hash: u64) -> PrefixMatch {
        // Phase 1: walk tree (lock inner, then release)
        let (matched_blocks, matched_disk_block_ids, matched_tokens) = {
            let mut inner = self.inner.lock();
            let (blocks, disk, matched) = inner.walk(tokens, block_size);
            if matched > 0 {
                inner.inc_refs(tokens, block_size, matched);
                crate::prefix_cache::record_cache_hit(matched);
            } else {
                crate::prefix_cache::record_cache_miss();
            }
            (blocks, disk, matched)
        };
        // Phase 2: snapshot lookup (lock snapshot_index, inner NOT held)
        let (ssm_snapshot, ssm_snapshot_tokens) = if matched_tokens > 0 {
            let mut idx = self.snapshot_index.lock();
            match idx.lookup(tokens, matched_tokens, session_hash) {
                Some((snap_id, tok_count)) => (Some(snap_id), tok_count),
                None => (None, 0),
            }
        } else {
            (None, 0)
        };
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
        }
    }

    fn insert(
        &self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        matched_tokens: usize,
    ) {
        self.inner.lock().insert(
            tokens,
            block_table,
            disk_block_ids,
            block_size,
            matched_tokens,
        );
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
    ) -> Option<usize> {
        // Phase 1: insert tree nodes (lock inner, then release)
        self.inner.lock().insert(
            tokens,
            block_table,
            disk_block_ids,
            block_size,
            matched_tokens,
        );
        // Phase 2: register snapshot in index (lock snapshot_index, inner NOT held)
        let prefix_hash = hash_token_prefix(tokens, tokens.len());
        let mut idx = self.snapshot_index.lock();
        idx.insert(prefix_hash, snapshot_id, session_hash, tokens.len())
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
    ) -> Option<usize> {
        // Intermediate snapshots go directly into the index with the correct
        // token boundary (tokens.len()). Tree nodes are already inserted by
        // a prior `insert()` call, which handled the ref_count bookkeeping.
        let prefix_hash = hash_token_prefix(tokens, tokens.len());
        let mut idx = self.snapshot_index.lock();
        idx.insert(prefix_hash, snapshot_id, session_hash, tokens.len())
    }

    fn release(&self, tokens: &[u32], block_size: usize) {
        self.inner.lock().dec_refs(tokens, block_size);
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

    fn snapshot_count(&self) -> usize {
        self.snapshot_index.lock().len()
    }

    fn stats(&self) -> (usize, usize) {
        let inner = self.inner.lock();
        let entries = inner.num_entries();
        (entries, entries)
    }
}
