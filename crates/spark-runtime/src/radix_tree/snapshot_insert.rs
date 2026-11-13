// SPDX-License-Identifier: AGPL-3.0-only

//! Snapshot-index insert paths (plain, tail, tail-sibling). Split from
//! `snapshot.rs` (file-size cap); same `SsmSnapshotIndex` impl.
//!
//! Also records the tier keys these paths ORPHAN (port sprint, lane L25): a
//! re-home of a spilled prefix to HBM, or the tail sweep dropping a spilled
//! entry, leaves the tier blob under that key unreachable. The index holds no
//! store handle, so it only remembers the key; spark-model drains the list
//! through `PrefixCache::take_orphaned_snapshot_tier_keys` and removes the
//! blobs from its store (row `ssm_store_lru`). Recording changes no decision
//! the index makes, and it only ever happens with the SSM tier on (the only
//! way an entry becomes `tiered`).

use super::snapshot::{SnapshotEntry, SsmSnapshotIndex};

/// Orphaned keys kept for the next drain. A model that never drains (row off)
/// holds at most this many `u64`s; the oldest are dropped first.
pub(super) const MAX_ORPHANED_TIER_KEYS: usize = 4096;

/// The slot an entry being displaced hands back for the caller to free — or
/// `None` if it holds no HBM slot.
///
/// A `tiered` entry's `snapshot_id` is STALE: `evict_to_tier` already returned
/// that slot to the caller (`TierEvict::Spill { slot, .. }`) and the caller
/// already freed it; the entry stayed in the index only as a fault-in record.
/// Handing the same id back a second time is a double-free into
/// `SsmSnapshotPool::free`, whose free list is a plain `Vec` push with no
/// membership check — so the slot is handed to TWO sequences, which then share
/// one GDN/conv state buffer. That is silent cross-stream corruption, not a
/// crash: the same class of fault `ssm_pool::claim_specific` exists to prevent.
///
/// Every other consumer of the index already honours this — `lookup` skips
/// tiered entries ("no HBM slot"), and both victim scans pass
/// `skip_tiered = true`. The insert paths were the one place that did not.
fn freeable_slot(entry: &SnapshotEntry) -> Option<usize> {
    (!entry.tiered).then_some(entry.snapshot_id)
}

impl SsmSnapshotIndex {
    /// Remember that the tier blob under `prefix_hash` lost its index entry
    /// (re-homed to HBM or dropped while `tiered`).
    fn note_orphan(&mut self, was_tiered: bool, prefix_hash: u64) {
        if !was_tiered {
            return;
        }
        if self.orphaned_tier_keys.len() >= MAX_ORPHANED_TIER_KEYS {
            self.orphaned_tier_keys.remove(0);
        }
        self.orphaned_tier_keys.push(prefix_hash);
    }

    /// Hand the recorded orphan keys to the caller (who removes their blobs)
    /// and forget them. A key whose prefix is `tiered` AGAIN is kept out: it
    /// was re-spilled after the orphaning, so its blob is live. Each key is
    /// returned once.
    pub(super) fn take_orphaned_tier_keys(&mut self) -> Vec<u64> {
        let mut keys = std::mem::take(&mut self.orphaned_tier_keys);
        keys.sort_unstable();
        keys.dedup();
        keys.retain(|&k| !self.entries.iter().any(|e| e.prefix_hash == k && e.tiered));
        keys
    }

    pub(super) fn insert(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
    ) -> Option<usize> {
        let rehomed = self
            .entries
            .iter()
            .any(|e| e.prefix_hash == prefix_hash && e.tiered);
        self.note_orphan(rehomed, prefix_hash);
        for entry in &mut self.entries {
            if entry.prefix_hash == prefix_hash {
                let old = freeable_slot(entry);
                entry.snapshot_id = snapshot_id;
                entry.session_hash = session_hash;
                entry.token_count = token_count;
                // A fresh HBM save re-homes the prefix: it is resident again.
                //
                // ⚠ The tier blob under this key is now UNREACHABLE AND NOT
                // RECLAIMED. An earlier version of this comment claimed it was
                // "left to the store's own budget"; that is false on three of
                // the four store arms and the claim is withdrawn:
                //
                //   * the DEFAULT arm is `MemBlobStore::new(0)` — `cap_bytes
                //     == 0` means UNBOUNDED (`selectors.rs`, `store.rs`), so
                //     there is no budget to leave it to;
                //   * a capped `MemBlobStore` evicts FIFO by INSERTION order,
                //     not by coldness, so an orphan is not "the coldest thing
                //     there" under any policy it implements;
                //   * `ArenaSnapshotStore`/`RdmaSnapshotStore` have NO eviction
                //     — slots return only via `remove`, so orphans permanently
                //     consume arena slots and eventually kill the tier.
                //
                // Only `UnifiedSnapshotStore` (default OFF) behaves as claimed.
                // This index layer holds no store handle, so it cannot remove
                // the blob itself: `note_orphan` (above, before this loop)
                // records the key and spark-model removes the blob when it
                // drains the list (row `ssm_store_lru`; port sprint, L25). With
                // the row off the leak is as before the sprint.
                entry.tiered = false;
                // A plain save re-homing this prefix is by definition NOT a
                // tail. Without this, an overwrite could re-home another
                // session's is_tail entry (new session_hash, is_tail still
                // set), breaching the <=1-leased-entry-per-session invariant
                // insert_tail's supersede sweep maintains.
                entry.is_tail = false;
                entry.is_tail_sibling = false;
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                return old;
            }
        }
        self.access_counter += 1;
        self.stats.saves += 1;
        self.entries.push(SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash,
            last_access: self.access_counter,
            tiered: false,
            is_tail: false,
            is_tail_sibling: false,
        });
        None
    }

    /// Insert the per-session TAIL snapshot, superseding this session's previous one.
    /// Returns displaced snapshot_ids for the caller to free.
    pub(super) fn insert_tail(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
    ) -> Vec<usize> {
        let mut displaced = Vec::new();
        if session_hash != 0 {
            let mut i = 0;
            while i < self.entries.len() {
                if (self.entries[i].is_tail || self.entries[i].is_tail_sibling)
                    && self.entries[i].session_hash == session_hash
                {
                    let gone = self.entries.swap_remove(i);
                    // A swept SPILLED entry leaves its blob behind (also when
                    // this insert saves the same prefix again: that save is
                    // resident, so the blob is stale either way).
                    self.note_orphan(gone.tiered, gone.prefix_hash);
                    displaced.extend(freeable_slot(&gone));
                } else {
                    i += 1;
                }
            }
        }
        let rehomed = self
            .entries
            .iter()
            .any(|e| e.prefix_hash == prefix_hash && e.tiered);
        self.note_orphan(rehomed, prefix_hash);
        for entry in &mut self.entries {
            if entry.prefix_hash == prefix_hash {
                displaced.extend(freeable_slot(entry));
                entry.snapshot_id = snapshot_id;
                entry.session_hash = session_hash;
                entry.token_count = token_count;
                // Re-homed to HBM, same as the plain `insert` path. This was
                // the ONE overwrite arm that never cleared the flag: an entry
                // left `tiered` while holding a live slot is skipped by
                // `lookup` and by both victim scans, so the slot is reachable
                // by nothing and freeable by nothing — a permanent leak of a
                // scarce snapshot-pool slot, on top of `lookup_tiered` then
                // faulting in bytes that no longer describe this entry.
                entry.tiered = false;
                entry.is_tail = true;
                entry.is_tail_sibling = false;
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                return displaced;
            }
        }
        self.access_counter += 1;
        self.entries.push(SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash,
            last_access: self.access_counter,
            tiered: false,
            is_tail: true,
            is_tail_sibling: false,
        });
        displaced
    }

    /// Insert the tail's EARLY sibling (`tb - bs`). MUST be called after
    /// [`Self::insert_tail`] within the same finalize — the tail insert's
    /// supersede sweep clears the session's previous tail AND sibling, so
    /// this insert never needs (and must not run) its own sweep.
    pub(super) fn insert_tail_sibling(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
    ) -> Option<usize> {
        let rehomed = self
            .entries
            .iter()
            .any(|e| e.prefix_hash == prefix_hash && e.tiered);
        self.note_orphan(rehomed, prefix_hash);
        for entry in &mut self.entries {
            if entry.prefix_hash == prefix_hash {
                let old = freeable_slot(entry);
                entry.snapshot_id = snapshot_id;
                entry.session_hash = session_hash;
                entry.token_count = token_count;
                entry.tiered = false;
                entry.is_tail = false;
                entry.is_tail_sibling = true;
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                return old;
            }
        }
        self.access_counter += 1;
        self.stats.saves += 1;
        self.entries.push(SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash,
            last_access: self.access_counter,
            tiered: false,
            is_tail: false,
            is_tail_sibling: true,
        });
        None
    }
}

#[cfg(test)]
mod orphan_tests {
    use super::*;

    /// An index with one entry for `key`, spilled to the tier.
    fn spilled(key: u64) -> SsmSnapshotIndex {
        let mut idx = SsmSnapshotIndex::new();
        assert_eq!(idx.insert(key, 3, 7, 160), None);
        idx.entries[0].tiered = true;
        idx
    }

    #[test]
    fn resident_overwrites_orphan_nothing() {
        let mut idx = SsmSnapshotIndex::new();
        idx.insert(10, 1, 7, 160);
        idx.insert(10, 2, 7, 160);
        idx.insert_tail_sibling(10, 4, 7, 160);
        idx.insert_tail(10, 5, 7, 160);
        assert!(idx.take_orphaned_tier_keys().is_empty());
    }

    #[test]
    fn every_rehome_path_records_the_stale_blob() {
        let mut a = spilled(10);
        a.insert(10, 4, 7, 160);
        assert_eq!(a.take_orphaned_tier_keys(), vec![10]);
        assert!(
            a.take_orphaned_tier_keys().is_empty(),
            "each key is handed out once"
        );

        let mut b = spilled(11);
        b.insert_tail(11, 4, 7, 160);
        assert_eq!(b.take_orphaned_tier_keys(), vec![11]);

        let mut c = spilled(12);
        c.insert_tail_sibling(12, 4, 7, 160);
        assert_eq!(c.take_orphaned_tier_keys(), vec![12]);
    }

    #[test]
    fn the_tail_sweep_records_a_dropped_spilled_tail() {
        let mut idx = SsmSnapshotIndex::new();
        idx.insert_tail(20, 1, 7, 160);
        idx.entries[0].tiered = true;
        // The session's next tail supersedes it: the spilled entry is swept.
        idx.insert_tail(21, 2, 7, 320);
        assert_eq!(idx.take_orphaned_tier_keys(), vec![20]);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn a_key_spilled_again_before_the_drain_is_kept() {
        let mut idx = spilled(30);
        idx.insert(30, 4, 7, 160);
        // Spilled again: the blob under 30 is the NEW snapshot's and must stay.
        idx.entries[0].tiered = true;
        assert!(idx.take_orphaned_tier_keys().is_empty());
    }

    #[test]
    fn the_record_is_bounded() {
        let mut idx = SsmSnapshotIndex::new();
        for k in 0..(MAX_ORPHANED_TIER_KEYS as u64 + 10) {
            idx.note_orphan(true, k);
        }
        let keys = idx.take_orphaned_tier_keys();
        assert_eq!(keys.len(), MAX_ORPHANED_TIER_KEYS);
        assert_eq!(keys[0], 10, "the oldest records went first");
    }
}
