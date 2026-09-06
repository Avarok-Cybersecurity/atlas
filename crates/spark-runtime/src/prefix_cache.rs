// SPDX-License-Identifier: AGPL-3.0-only

//! Prefix caching trait for KV block reuse (SDD).
//!
//! When multiple requests share a common prompt prefix, previously-computed
//! KV cache blocks can be reused instead of re-running prefill. The cache
//! is indexed by token sequences at block granularity via a radix tree.
//!
//! Two implementations:
//! - `NoPrefixCaching`: no-ops (zero overhead when disabled)
//! - `RadixTree` (see `crate::radix_tree`): full radix tree with LRU eviction

use std::sync::atomic::Ordering;

mod no_caching;
mod ssm_anchor;
mod tier_evict;
pub use no_caching::NoPrefixCaching;
pub use ssm_anchor::SsmAnchor;
pub use tier_evict::TierEvict;

// The three counters that lived here are fields of the single run mailbox,
// `crate::run_metrics::RunMetrics` — see that module for why one static and
// not none, and why it is cleared at run start.

pub fn record_cache_hit(matched_tokens: usize) {
    let m = crate::run_metrics::metrics();
    m.cache_hits.fetch_add(1, Ordering::Relaxed);
    m.cache_hit_tokens
        .fetch_add(matched_tokens as u64, Ordering::Relaxed);
}

pub fn record_cache_miss() {
    crate::run_metrics::metrics()
        .cache_misses
        .fetch_add(1, Ordering::Relaxed);
}

pub fn cache_hit_count() -> u64 {
    crate::run_metrics::metrics()
        .cache_hits
        .load(Ordering::Relaxed)
}
pub fn cache_miss_count() -> u64 {
    crate::run_metrics::metrics()
        .cache_misses
        .load(Ordering::Relaxed)
}
pub fn cache_hit_tokens_total() -> u64 {
    crate::run_metrics::metrics()
        .cache_hit_tokens
        .load(Ordering::Relaxed)
}

/// Result of evicting LRU cached blocks (Phase 6.1.e).
#[derive(Debug, Clone, Default)]
pub struct EvictedBlocks {
    /// Physical block indices freed (caller calls `PagedKvCache::free_block`).
    pub physical: Vec<u32>,
    /// Parallel disk-block IDs to release (caller calls
    /// `HighSpeedSwap::dec_disk_ref`). Empty when HSS isn't in use.
    pub disk_block_ids: Vec<u32>,
}

/// What an `insert` newly took ownership of, so the caller can take the
/// matching references.
///
/// Ownership rule: the cache holds exactly ONE reference on the physical block
/// stored in each radix node, taken when that node is CREATED and returned when
/// the node is evicted (`return_evicted_block`). The reference therefore has to
/// follow the block the NODE holds — not the block the inserting sequence had at
/// that position. Those two diverge whenever a node already exists for a token
/// chunk: `insert` keeps the node's original `block_idx`, while the sequence has
/// a different block there (it did not get that block from the cache — e.g. it
/// never matched, or it was restored from a swap file). Referencing the
/// sequence's block instead left the node's block with no reference at all, so
/// evicting that node decremented a ref belonging to a LIVE sequence: the block
/// went back on the free list while still in use and was handed out again, with
/// the second owner's teardown underflowing. Reporting the blocks here keeps
/// node lifetime and reference lifetime identical by construction.
#[derive(Debug, Clone, Default)]
pub struct InsertAcquired {
    /// Disk-block IDs the cache newly references (caller `inc_disk_ref`s each).
    pub disk_block_ids: Vec<u32>,
    /// Physical KV blocks stored in radix nodes CREATED by this insert; the
    /// caller `inc_ref`s each exactly once.
    pub blocks: Vec<u32>,
    /// Physical KV blocks this insert stopped storing — a `partial_suffix` slot
    /// that was overwritten or dropped. The caller `dec_ref`s each exactly once,
    /// releasing the reference taken when that slot was first filled.
    pub released_blocks: Vec<u32>,
}

impl EvictedBlocks {
    pub fn is_empty(&self) -> bool {
        self.physical.is_empty()
    }

    pub fn len(&self) -> usize {
        self.physical.len()
    }
}

/// Result of looking up a token sequence in the prefix cache.
#[derive(Debug, Clone)]
pub struct PrefixMatch {
    /// Physical KV cache block indices to reuse (in order).
    pub matched_blocks: Vec<u32>,
    /// `--high-speed-swap` disk-block IDs parallel to `matched_blocks`
    /// (Phase 6.1.e). Empty when HSS is not in use. Same length as
    /// `matched_blocks` when populated. Caller must `inc_disk_ref` each
    /// before treating them as live (the cache itself does not
    /// retain disk-side refs — see `RadixTree::lookup` for the bump).
    pub matched_disk_block_ids: Vec<u32>,
    /// Number of tokens matched (always block-aligned).
    pub matched_tokens: usize,
    /// SSM state snapshot ID at the deepest matched node (Marconi caching).
    /// When `Some`, the caller can restore SSM h_state + conv_state from
    /// this snapshot and skip SSM computation for the matched prefix.
    pub ssm_snapshot: Option<usize>,
    /// Number of tokens covered by `ssm_snapshot` (Marconi intermediate checkpoints).
    /// With leaf-only snapshots this equals `matched_tokens`. With intermediate
    /// checkpoints it may be less — the caller must recompute SSM state for
    /// tokens between `ssm_snapshot_tokens` and `matched_tokens`.
    pub ssm_snapshot_tokens: usize,
    /// Phase 1b spill tier: when the deepest anchor for this prefix is SPILLED
    /// (not resident in HBM), `ssm_snapshot` is `None` and this holds the tier
    /// key (prefix hash). The caller faults the bytes into a fresh snapshot slot
    /// (`SsmSnapshotPool::fault_in_slot`), `promote_snapshot`s the entry, then
    /// restores. `None` whenever nothing is tiered (i.e. `ATLAS_SSM_TIER` off) —
    /// so this field is inert on the default path.
    pub ssm_snapshot_tier_key: Option<u64>,
    /// Token depth covered by `ssm_snapshot_tier_key` (analogue of
    /// `ssm_snapshot_tokens` for a tiered anchor).
    pub ssm_snapshot_tier_tokens: usize,
    /// Whether the matched SSM snapshot is a TAIL (bleeds past the exact
    /// prefix). The restore site session-gates only tails; exact and
    /// is_tail_sibling snapshots are content-addressed (safe cross-session).
    pub ssm_snapshot_is_tail: bool,
}

impl PrefixMatch {
    /// Empty match (no cached prefix found).
    pub fn empty() -> Self {
        Self {
            matched_blocks: Vec::new(),
            matched_disk_block_ids: Vec::new(),
            matched_tokens: 0,
            ssm_snapshot: None,
            ssm_snapshot_tokens: 0,
            ssm_snapshot_tier_key: None,
            ssm_snapshot_tier_tokens: 0,
            ssm_snapshot_is_tail: false,
        }
    }

    /// Whether any prefix was matched.
    pub fn is_empty(&self) -> bool {
        self.matched_tokens == 0
    }
}

/// Trait for prefix caching strategies.
///
/// All methods take `&self` — implementations use interior mutability
/// (e.g., `Mutex`) for thread safety. This allows the prefix cache to be
/// shared between the model (prefill) and scheduler (free_sequence) without
/// requiring `&mut self`.
pub trait PrefixCache: Send + Sync {
    /// Whether this implementation is active (i.e., a real cache that
    /// actually inserts/holds refs). `NoPrefixCaching` returns false;
    /// `RadixTree` returns true. Callers use this to skip ref-bookkeeping
    /// that's only meaningful when the cache holds refs (e.g., the manual
    /// `kv_cache.inc_ref` in `cache_sequence` that pairs with eviction's
    /// `return_evicted_block`).
    fn is_active(&self) -> bool {
        true
    }

    /// Look up a token sequence and return cached KV blocks for the
    /// longest matching prefix (block-aligned).
    ///
    /// Increments ref_count on matched nodes so they survive eviction
    /// while the sequence is active. `session_hash` is used for SSM
    /// snapshot isolation (0 = legacy/no session tracking).
    ///
    /// Task #24: `adapter_id` keys the KV/prefix + SSM-snapshot cache so a
    /// request reuses ONLY blocks computed under the same adapter. `0` = base /
    /// no adapter, which keys byte-identically to the pre-LoRA token-only cache.
    fn lookup(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> PrefixMatch;

    /// Read-only longest-prefix probe: number of tokens (block-aligned)
    /// `lookup` would match, WITHOUT taking refs, touching LRU state, or
    /// counting a hit/miss. Used by the prefill tail-checkpoint split to
    /// detect conversation reuse before deciding to pay the extra pass.
    /// Task #24: keyed by `adapter_id` so a cross-adapter peek reports a miss.
    fn peek_matched_tokens(&self, _tokens: &[u32], _block_size: usize, _adapter_id: u64) -> usize {
        0
    }

    /// Marconi re-anchor: the deepest SSM snapshot for `tokens` whose depth is
    /// `<= max_tokens`, WITHOUT walking the KV radix or taking block refs.
    ///
    /// `lookup` hands back the deepest snapshot at or below the matched
    /// prefix. When the restore site declines that anchor — an exact
    /// full-prompt leaf is bypassed by default (see `prefill_b/prefix_lookup`)
    /// — the caller passes `max_tokens = prompt_len - 1` to fall back to the
    /// next-deepest anchor (an intermediate checkpoint) instead of recomputing
    /// the whole prefix. Same candidate filter as `lookup` (prefix-hash match,
    /// tail snapshots session-gated, adapter-keyed, resident or spilled).
    /// `SsmAnchor::NONE` when nothing qualifies — the default for caches that
    /// hold no snapshots.
    fn lookup_ssm_anchor(
        &self,
        _tokens: &[u32],
        _max_tokens: usize,
        _session_hash: u64,
        _adapter_id: u64,
    ) -> SsmAnchor {
        SsmAnchor::NONE
    }

    /// Insert a completed prefill's blocks into the cache.
    ///
    /// `block_table[i]` is the physical block for tokens
    /// `[i*block_size .. (i+1)*block_size]`.
    ///
    /// `disk_block_ids` parallels `block_table` for `--high-speed-swap`
    /// (Phase 6.1.e). Empty when HSS is not in use; same length as
    /// `block_table` when populated. The cache stores these alongside the
    /// physical block IDs and returns them in `EvictedBlocks` so the
    /// caller can `dec_disk_ref` the orchestrator's per-block refcount.
    ///
    /// **Disk-ref obligation (Issue #17 fix):** the returned vec lists every
    /// disk_block_id on which this insert call newly took an ownership ref
    /// (a node was created OR an existing node had its `disk_block_id`
    /// populated for the first time). The caller MUST `inc_disk_ref` each
    /// returned ID so the swap allocator's refcount matches the cache's
    /// reachability. Already-cached portions (matched-prefix entries, or
    /// blocks a prior intermediate insert already covered) are NOT in the
    /// returned vec — re-incing them would leak the cache's refcount.
    ///
    /// `matched_tokens` is the number of tokens the inserting sequence
    /// already acquired via `lookup()`'s `inc_refs` (0 for a cache-miss
    /// request). Tokens past this offset are "seq-owned" — the inserting
    /// sequence's eventual `release()` will decrement them — so `insert`
    /// must bump their ref_count to keep the cache's own reference alive
    /// after the release. See the release/lookup dance at the top of
    /// `radix_tree.rs`.
    fn insert(
        &self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        matched_tokens: usize,
        adapter_id: u64,
    ) -> InsertAcquired;

    /// Insert blocks with an SSM state snapshot registered in the snapshot index.
    ///
    /// The snapshot ID references a slot in an external `SsmSnapshotPool`.
    /// On future lookups matching this prefix, the snapshot ID is returned
    /// in `PrefixMatch::ssm_snapshot` so the caller can restore SSM state.
    /// `session_hash` tags the snapshot for session-scoped isolation.
    /// `matched_tokens` has the same semantics as in `insert`.
    /// Returns `(displaced_snapshot_id, newly_acquired_disk_ids)`. The
    /// disk-ref obligation matches `insert`: caller `inc_disk_ref`s each
    /// returned ID.
    #[allow(clippy::too_many_arguments)]
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
    ) -> (Option<usize>, InsertAcquired);

    /// Insert an SSM snapshot at an intermediate token boundary.
    ///
    /// `tokens` is the token sequence up to and including the snapshot point.
    /// `block_table` contains the physical block indices for those tokens.
    /// `session_hash` tags the snapshot for session-scoped isolation.
    /// `matched_tokens` has the same semantics as in `insert`.
    /// Returns the displaced snapshot ID if an existing entry was overwritten.
    #[allow(clippy::too_many_arguments)]
    fn insert_intermediate_snapshot(
        &self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        snapshot_id: usize,
        session_hash: u64,
        matched_tokens: usize,
        adapter_id: u64,
    ) -> Option<usize>;

    /// Register the per-session TAIL snapshot in the index WITHOUT touching the
    /// radix tree (the final chunk's `insert` covers those blocks). Supersedes
    /// this session's previous tail; returns displaced snapshot ids to free.
    fn insert_tail_snapshot(
        &self,
        tokens: &[u32],
        snapshot_id: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> Vec<usize>;

    /// Register the tail's EARLY sibling (`tb - bs`) in the index. Must be
    /// called after `insert_tail_snapshot` in the same finalize (the tail
    /// insert sweeps the session's previous tail + sibling). Returns a
    /// displaced snapshot id to free, if the prefix was already registered.
    fn insert_tail_sibling_snapshot(
        &self,
        tokens: &[u32],
        snapshot_id: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> Option<usize>;

    /// Release ref_counts on blocks that were acquired via `lookup`.
    ///
    /// Called when a sequence finishes. Decrements ref_count on cache
    /// nodes matching the token prefix, making them eligible for eviction.
    /// Task #24: `adapter_id` must match the one used at `lookup`/`insert`.
    fn release(&self, tokens: &[u32], block_size: usize, adapter_id: u64);

    /// Release exactly the block-aligned prefix acquired by one `lookup`.
    ///
    /// Batched-prefill admission may acquire several candidates, then reject
    /// the batch before any sequence owns their blocks. Releasing the full
    /// token slice in that case would be racy: another request could insert a
    /// longer matching suffix between acquisition and rollback. Implementors
    /// must therefore decrement no more than `matched_tokens`.
    fn release_matched(
        &self,
        tokens: &[u32],
        block_size: usize,
        matched_tokens: usize,
        adapter_id: u64,
    );

    /// Evict up to `num_blocks` cached blocks, returning their physical
    /// indices and parallel disk-block IDs (Phase 6.1.e).
    ///
    /// Picks LRU zero-ref leaf nodes. Returns fewer than requested if not
    /// enough evictable blocks exist. The caller is responsible for
    /// `dec_disk_ref`-ing every entry in `disk_block_ids` (releasing the
    /// cache's HSS-side refcount). When HSS isn't in use the `disk_block_ids`
    /// vec is empty.
    fn evict(&self, num_blocks: usize) -> EvictedBlocks;

    /// Evict the least-recently-used SSM snapshot from the snapshot index.
    /// Returns the snapshot ID so the caller can free it in `SsmSnapshotPool`.
    fn evict_snapshot_lru(&self) -> Option<usize>;

    /// Phase 1b spill tier: pick a spill victim (same policy as
    /// `evict_snapshot_lru`, HBM-resident only) and decide whether it is worth
    /// spilling — see [`TierEvict`]. A victim shallower than `min_tokens`
    /// cannot repay the spill's fixed cost, so its entry is dropped outright
    /// rather than left findable-but-empty. `min_tokens == 0` disables the
    /// gate. `None` when nothing resident remains; default `None` (no tier).
    fn evict_snapshot_to_tier(&self, min_tokens: usize) -> Option<TierEvict> {
        let _ = min_tokens;
        None
    }

    /// Phase 1b spill tier: after the caller faulted a spilled snapshot's bytes
    /// into `new_slot`, re-home its index entry to HBM. Returns `false` if the
    /// key is unknown. Default: `false`.
    fn promote_snapshot(&self, key: u64, new_slot: usize) -> bool {
        let _ = (key, new_slot);
        false
    }

    /// Phase 1b spill tier: the FAILED-fault-in twin of [`Self::promote_snapshot`].
    /// The caller's `store.get(key)` MISSED, so this entry is findable by
    /// `lookup_tiered` with no bytes behind it. Left in place, every warm turn
    /// on this prefix repeats the whole doomed cycle — spill a LIVE 66 MB
    /// victim D2H to free a slot, fault in, miss, free the slot — and then
    /// recomputes anyway; under `ATLAS_SSM_TIER_DISK_GB` that doomed spill
    /// evicts one MORE tier record, so the cap's own pressure re-amplifies
    /// itself. Dropping the entry degrades the prefix to a plain recompute
    /// ONCE.
    ///
    /// Only removes an entry that is still `tiered`: a resident entry's
    /// `snapshot_id` is a LIVE pool slot that only its owner may free, so a
    /// by-key remove of one would leak it. Returns whether an entry was
    /// dropped. Default: `false` (no tier).
    fn forget_snapshot_tier_key(&self, key: u64) -> bool {
        let _ = key;
        false
    }

    /// Number of SSM snapshots currently stored in the snapshot index.
    fn snapshot_count(&self) -> usize;

    /// (entries, cached_blocks) for logging.
    fn stats(&self) -> (usize, usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_match_empty() {
        let m = PrefixMatch::empty();
        assert!(m.is_empty());
        assert_eq!(m.matched_tokens, 0);
        assert!(m.matched_blocks.is_empty());
    }

    #[test]
    fn test_no_prefix_caching_is_noop() {
        let cache = NoPrefixCaching;
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let block_table = vec![0, 1];
        let disk_block_ids: Vec<u32> = vec![];

        let m = cache.lookup(&tokens, 4, 0, 0);
        assert!(m.is_empty());

        // These should not panic
        let new_acq = cache.insert(&tokens, &block_table, &disk_block_ids, 4, 0, 0);
        assert!(new_acq.disk_block_ids.is_empty());
        assert!(new_acq.blocks.is_empty());
        cache.release(&tokens, 4, 0);

        let evicted = cache.evict(10);
        assert!(evicted.is_empty());

        assert_eq!(cache.evict_snapshot_lru(), None);
        assert_eq!(cache.snapshot_count(), 0);

        assert_eq!(cache.stats(), (0, 0));
    }
}
