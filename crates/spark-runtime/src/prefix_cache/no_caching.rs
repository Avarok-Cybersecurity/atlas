// SPDX-License-Identifier: AGPL-3.0-only

//! The no-op [`PrefixCache`] used when prefix caching is disabled.
//!
//! Split out of `prefix_cache.rs` to keep it under the repo's 500-LoC cap.

use super::{EvictedBlocks, InsertAcquired, PrefixCache, PrefixMatch};

/// No-op prefix cache (zero overhead when disabled).
pub struct NoPrefixCaching;

impl PrefixCache for NoPrefixCaching {
    fn is_active(&self) -> bool {
        false
    }

    fn lookup(
        &self,
        _tokens: &[u32],
        _block_size: usize,
        _session_hash: u64,
        _adapter_id: u64,
    ) -> PrefixMatch {
        PrefixMatch::empty()
    }

    fn insert(
        &self,
        _tokens: &[u32],
        _block_table: &[u32],
        _disk_block_ids: &[u32],
        _block_size: usize,
        _matched_tokens: usize,
        _adapter_id: u64,
    ) -> InsertAcquired {
        InsertAcquired::default()
    }

    fn insert_with_snapshot(
        &self,
        _tokens: &[u32],
        _block_table: &[u32],
        _disk_block_ids: &[u32],
        _block_size: usize,
        _snapshot_id: usize,
        _session_hash: u64,
        _matched_tokens: usize,
        _adapter_id: u64,
    ) -> (Option<usize>, InsertAcquired) {
        (None, InsertAcquired::default())
    }

    fn insert_intermediate_snapshot(
        &self,
        _tokens: &[u32],
        _block_table: &[u32],
        _disk_block_ids: &[u32],
        _block_size: usize,
        _snapshot_id: usize,
        _session_hash: u64,
        _matched_tokens: usize,
        _adapter_id: u64,
    ) -> Option<usize> {
        None
    }

    fn insert_tail_snapshot(
        &self,
        _tokens: &[u32],
        _snapshot_id: usize,
        _session_hash: u64,
        _adapter_id: u64,
    ) -> Vec<usize> {
        Vec::new()
    }

    fn insert_tail_sibling_snapshot(
        &self,
        _tokens: &[u32],
        _snapshot_id: usize,
        _session_hash: u64,
        _adapter_id: u64,
    ) -> Option<usize> {
        None
    }

    fn release(&self, _tokens: &[u32], _block_size: usize, _adapter_id: u64) {}

    fn release_matched(
        &self,
        _tokens: &[u32],
        _block_size: usize,
        _matched_tokens: usize,
        _adapter_id: u64,
    ) {
    }

    fn evict(&self, _num_blocks: usize) -> EvictedBlocks {
        EvictedBlocks::default()
    }

    fn evict_snapshot_lru(&self) -> Option<usize> {
        None
    }

    fn snapshot_count(&self) -> usize {
        0
    }

    fn stats(&self) -> (usize, usize) {
        (0, 0)
    }
}
