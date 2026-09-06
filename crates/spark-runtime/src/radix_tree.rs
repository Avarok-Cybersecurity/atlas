// SPDX-License-Identifier: AGPL-3.0-only

//! Radix tree prefix cache for KV block reuse.
//!
//! Token sequences are chunked at `block_size` granularity. Each node in
//! the tree corresponds to one KV cache block. Lookup walks the tree
//! matching block-aligned chunks, returning cached physical block indices.
//!
//! Thread-safe via `Mutex<RadixTreeInner>`.

use parking_lot::Mutex;

use crate::prefix_cache::{EvictedBlocks, PrefixCache, PrefixMatch, SsmAnchor};

mod inner;
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
    /// Allow a lookup to match FEWER tokens than a cached block holds
    /// (`inner::walk`'s two sub-block arms). Default OFF — see
    /// [`subblock_matching_from_env`].
    subblock_matching: bool,
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Is sub-block prefix matching enabled? OPT-IN (`ATLAS_PREFIX_SUBBLOCK=1`).
///
/// The two sub-block arms in [`inner::RadixTreeInner::walk`] return a
/// `matched_tokens` that is NOT block-aligned, by reusing a block whose KV was
/// computed for a LONGER key. Two things then go wrong, and both were measured
/// on qwen3.8-flash-next with `--enable-prefix-caching`:
///
/// 1. The tail of that block is a DIFFERENT continuation — the previous turn's
///    generated tokens — which the sequence's own new tokens are then written
///    over, inside a block the radix tree still shares with its original key.
/// 2. The model side takes `matched_tokens` as the KV write floor
///    (`forward_layers.rs`), so the boundary lands mid-block.
///
/// Symptom: three identical temp-0 requests returned three different
/// completions with it on, and six identical completions with it off (probe
/// `warmrepro.py`, 2026-09-02). It was on by default; it is now opt-in, and
/// `inner.rs`'s own comment has flagged the hazard since the arm was written.
///
/// Turning it off costs the warm-TTFT shortcut for the last partial block of a
/// prompt — at most `block_size - 1` tokens of re-prefill per warm turn.
pub fn subblock_matching_from_env() -> bool {
    subblock_matching_from_value(std::env::var("ATLAS_PREFIX_SUBBLOCK").ok().as_deref())
}

/// The polarity of [`subblock_matching_from_env`], over a plain value so a
/// test can pin it without mutating the process environment. Exact opt-in:
/// only the string "1" turns the arm back on.
pub fn subblock_matching_from_value(value: Option<&str>) -> bool {
    value == Some("1")
}

impl RadixTree {
    pub fn new() -> Self {
        let subblock_matching = subblock_matching_from_env();
        if subblock_matching {
            tracing::warn!(
                "Prefix cache: SUB-BLOCK matching ENABLED (ATLAS_PREFIX_SUBBLOCK=1). \
                 A lookup may reuse a block whose KV was computed for a longer key, \
                 and the sequence writes its own tokens into that shared block. \
                 Measured: warm temp-0 requests stop being reproducible. Diagnostic only."
            );
        } else {
            tracing::info!(
                "Prefix cache: sub-block matching off (default) — matches are \
                 block-aligned. ATLAS_PREFIX_SUBBLOCK=1 restores the old behaviour."
            );
        }
        Self {
            inner: Mutex::new(RadixTreeInner::new()),
            snapshot_index: Mutex::new(SsmSnapshotIndex::new()),
            subblock_matching,
        }
    }

    /// Construct with sub-block matching forced on or off, without touching the
    /// process environment — the constructor tests use to exercise both
    /// behaviours in one (parallel) test binary.
    pub fn with_subblock_matching(subblock_matching: bool) -> Self {
        Self {
            inner: Mutex::new(RadixTreeInner::new()),
            snapshot_index: Mutex::new(SsmSnapshotIndex::new()),
            subblock_matching,
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
            let (blocks, disk, matched) =
                inner.walk(tokens, block_size, adapter_id, self.subblock_matching);
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
        let anchor = if matched_tokens > 0 {
            self.lookup_ssm_anchor(tokens, matched_tokens, session_hash, adapter_id)
        } else {
            SsmAnchor::NONE
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
        let mut m = PrefixMatch {
            matched_blocks,
            matched_disk_block_ids,
            matched_tokens,
            ..PrefixMatch::empty()
        };
        m.set_ssm_anchor(anchor);
        m
    }

    fn lookup_ssm_anchor(
        &self,
        tokens: &[u32],
        max_tokens: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> SsmAnchor {
        let mut idx = self.snapshot_index.lock();
        match idx.lookup_tiered(tokens, max_tokens, session_hash, adapter_id) {
            Some(m) => {
                let mut anchor = SsmAnchor {
                    is_tail: m.is_tail,
                    ..SsmAnchor::NONE
                };
                match m.loc {
                    snapshot::SnapLoc::Hbm(slot) => {
                        anchor.snapshot = Some(slot);
                        anchor.snapshot_tokens = m.token_count;
                    }
                    snapshot::SnapLoc::Tier(key) => {
                        anchor.tier_key = Some(key);
                        anchor.tier_tokens = m.token_count;
                    }
                }
                anchor
            }
            None => SsmAnchor::NONE,
        }
    }

    fn peek_matched_tokens(&self, tokens: &[u32], block_size: usize, adapter_id: u64) -> usize {
        self.inner
            .lock()
            .walk(tokens, block_size, adapter_id, self.subblock_matching)
            .2
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
