// SPDX-License-Identifier: AGPL-3.0-only

//! Radix tree prefix cache for KV block reuse.
//!
//! Token sequences are chunked at `block_size` granularity. Each node in
//! the tree corresponds to one KV cache block. Lookup walks the tree
//! matching block-aligned chunks, returning cached physical block indices.
//!
//! Thread-safe via `Mutex<RadixTreeInner>`.
//!
//! Port sprint levers, both resolved once per tree at construction from the
//! setters `port_apply` calls at startup (default off, so a tree built
//! without them behaves exactly as before):
//! - `radix_lru_index` ([`set_lru_index_enabled`]): `evict` takes its victim
//!   from an ordered index instead of scanning every node; same victim.
//! - `prefix_cow_tail` ([`set_cow_tail_enabled`]): [`PrefixCache::lookup_cow`]
//!   may end a match inside a block and names the source block for the
//!   caller to copy (issue #1193); plain `lookup` is unchanged.

use parking_lot::Mutex;

use crate::prefix_cache::{EvictedBlocks, PrefixCache, PrefixMatch};

mod inner;
mod partial_tail;
mod port_metrics;
mod snapshot;
mod snapshot_insert;
mod snapshot_session;
mod snapshot_stats;
mod snapshot_tier;

#[cfg(test)]
mod tests;

use inner::RadixTreeInner;
use snapshot::SsmSnapshotIndex;

pub use inner::lru_index::set_enabled as set_lru_index_enabled;
pub use partial_tail::set_cow_tail_enabled;
pub use port_metrics::{record_cow_copy, record_cow_fallback, render_prometheus};

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
    /// Whether `evict` feeds `atlas_prefix_evict_*` (the LRU index row is on,
    /// or `AVAROK_PREFIX_EVICT_TIMING=1`). Off: no clock read, zero series.
    evict_metrics: bool,
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixTree {
    pub fn new() -> Self {
        Self::from_inner(RadixTreeInner::new())
    }

    /// A tree with explicit levers instead of the published rows and
    /// `AVAROK_PREFIX_SUBBLOCK`: `lru_index` = row `radix_lru_index`,
    /// `cow_tail` = row `prefix_cow_tail` (false = block-aligned matches only;
    /// the legacy sharing arms are never armed here). For tests and embedders
    /// that pass their own configuration.
    pub fn with_levers(lru_index: bool, cow_tail: bool) -> Self {
        let mode = if cow_tail {
            partial_tail::SubBlockMode::Cow
        } else {
            partial_tail::SubBlockMode::Off
        };
        Self::from_inner(RadixTreeInner::new_with(lru_index, mode))
    }

    fn from_inner(inner: RadixTreeInner) -> Self {
        let evict_metrics = inner.use_lru || port_metrics::evict_timing_env();
        Self {
            inner: Mutex::new(inner),
            snapshot_index: Mutex::new(SsmSnapshotIndex::new()),
            evict_metrics,
        }
    }

    /// The lookup both trait entry points share. `cow` = the caller copies a
    /// sub-block tail (`lookup_cow`); it takes effect only when this tree was
    /// built in copy-on-write mode. With `cow == false` this is the lookup
    /// the tree always ran.
    fn lookup_with(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
        adapter_id: u64,
        cow: bool,
    ) -> PrefixMatch {
        // Phase 1: walk tree (lock inner, then release)
        let (matched_blocks, matched_disk_block_ids, matched_tokens, cow_tail) = {
            let mut inner = self.inner.lock();
            let (blocks, disk, matched, source) = if cow && inner.cow_tail {
                inner.walk_cow(tokens, block_size, adapter_id)
            } else {
                let (blocks, disk, matched) = inner.walk(tokens, block_size, adapter_id);
                (blocks, disk, matched, None)
            };
            if matched > 0 {
                // `inc_refs` takes refs on the FULL blocks only (it counts
                // `matched / block_size`); a copy source is never referenced
                // by the tree on the requester's behalf.
                inner.inc_refs(tokens, block_size, matched, adapter_id);
                if let Some(node) = source.and_then(|s| s.node) {
                    inner.touch(node);
                }
                crate::prefix_cache::record_cache_hit(matched);
            } else {
                crate::prefix_cache::record_cache_miss();
            }
            (blocks, disk, matched, source.map(|s| (s.block, s.rows)))
        };
        // Phase 2: snapshot lookup (lock snapshot_index, inner NOT held).
        // Tier-aware: `lookup_tiered` returns the deepest anchor across resident
        // AND spilled entries. A resident hit populates `ssm_snapshot` (restore
        // directly); a spilled hit populates `ssm_snapshot_tier_key` (caller
        // faults it in). When nothing is spilled (AVAROK_SSM_TIER off) this is
        // byte-identical to the old resident-only lookup. With a copy-on-write
        // tail `matched_tokens` includes the tail rows, so an exact repeat of a
        // non-block-aligned prompt can reach the leaf snapshot at prompt end.
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
            ssm_snapshot_is_tail,
            cow_tail,
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
        self.lookup_with(tokens, block_size, session_hash, adapter_id, false)
    }

    fn lookup_cow(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> PrefixMatch {
        self.lookup_with(tokens, block_size, session_hash, adapter_id, true)
    }

    fn peek_matched_tokens(&self, tokens: &[u32], block_size: usize, adapter_id: u64) -> usize {
        self.inner.lock().walk(tokens, block_size, adapter_id).2
    }

    fn peek_snapshot_tokens(
        &self,
        tokens: &[u32],
        max_tokens: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> usize {
        // The same acceptance as `lookup_tiered` (depth bound, session gate,
        // prefix hash), without its side effects.
        let hermetic = crate::hermetic_enabled();
        let limit = max_tokens.min(tokens.len());
        let idx = self.snapshot_index.lock();
        idx.entries
            .iter()
            .filter(|e| e.token_count > 0 && e.token_count <= limit)
            .filter(|e| !snapshot_session::session_gate_blocks(e, session_hash, hermetic))
            .filter(|e| hash_token_prefix(tokens, e.token_count, adapter_id) == e.prefix_hash)
            .map(|e| e.token_count)
            .max()
            .unwrap_or(0)
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
        let (physical, disk) = if self.evict_metrics {
            let mut inner = self.inner.lock();
            let before = inner.evicted_nodes_total;
            let t0 = std::time::Instant::now();
            let out = inner.evict(num_blocks);
            port_metrics::record_evict(t0.elapsed(), inner.evicted_nodes_total - before);
            out
        } else {
            self.inner.lock().evict(num_blocks)
        };
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

    fn take_orphaned_snapshot_tier_keys(&self) -> Vec<u64> {
        self.snapshot_index.lock().take_orphaned_tier_keys()
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
