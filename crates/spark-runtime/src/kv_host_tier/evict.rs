// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/distributed/storage_controllers/eviction_controller.py:160-195 (commit 3aefae1, Apache-2.0; watermark-triggered batch eviction that skips locked keys, 0.8 trigger and 0.2 ratio); no code copied.

//! Responsibility: host memory accounting for the tier. [`TierInner::place`] finds room for one
//! new block: when the used bytes would pass the high watermark it evicts a batch through the
//! policy down to `high_watermark - eviction_ratio` of the budget (synchronously, at write time:
//! there is no daemon thread), never past the hard budget, and when the arena still cannot place
//! a run (pool still pinning, fragmented) it evicts one more block at a time. Only READY entries
//! without leases are ever chosen. An evicted block is demoted to the cold store when one is
//! attached; a lookup that finds a demoted key promotes it back ([`TierInner::promote_cold`])
//! and the block becomes visible only after its whole image was read. Teardown helpers return
//! runs to the arena.

use std::collections::HashSet;

use super::arena::HostRun;
use super::fingerprint::BlockKey;
use super::index::Entry;
use super::{Shared, TierInner};

impl TierInner {
    /// A host run for one new block of `bytes`, evicting as needed. None when the block cannot
    /// be placed without touching staging or leased entries, or beyond the budget.
    pub(crate) fn place(&mut self, sh: &Shared, bytes: u64) -> Option<HostRun> {
        self.place_protected(sh, bytes, None)
    }

    /// [`Self::place`] that also never evicts a key in `protect` (the hits a lookup already
    /// counted while it promotes a later block from the cold store).
    pub(crate) fn place_protected(
        &mut self,
        sh: &Shared,
        bytes: u64,
        protect: Option<&HashSet<BlockKey>>,
    ) -> Option<HostRun> {
        if !self.make_room(sh, bytes, protect) {
            return None;
        }
        loop {
            if let Some(run) = sh.arena.alloc(bytes as usize) {
                return Some(run);
            }
            if self.evict_n(sh, 1, protect) == 0 {
                return None;
            }
        }
    }

    /// Watermark eviction ahead of adding `need` bytes. True when the bytes then fit the budget.
    fn make_room(&mut self, sh: &Shared, need: u64, protect: Option<&HashSet<BlockKey>>) -> bool {
        let budget = sh.cfg.budget_bytes;
        let used = self.index.used_bytes();
        let high = (budget as f64 * sh.opts.high_watermark) as u64;
        if used + need <= high {
            return true;
        }
        let low_frac = (sh.opts.high_watermark - sh.opts.eviction_ratio).max(0.0);
        let target = (budget as f64 * low_frac) as u64;
        let to_free = (used + need).saturating_sub(target);
        let bb = sh.block_bytes.max(1) as u64;
        let blocks = to_free.div_ceil(bb) as usize;
        if blocks > 0 {
            self.evict_n(sh, blocks, protect);
        }
        self.index.used_bytes() + need <= budget
    }

    /// Evict up to `n` blocks in policy order, never one in `protect`. Returns how many left.
    pub(crate) fn evict_n(
        &mut self,
        sh: &Shared,
        n: usize,
        protect: Option<&HashSet<BlockKey>>,
    ) -> usize {
        let index = &self.index;
        let victims = self.policy.victims(n, &|k: &BlockKey| {
            index.is_evictable(k) && !protect.is_some_and(|p| p.contains(k))
        });
        let mut evicted = 0;
        for k in victims {
            let Some(entry) = self.index.remove_evictable(&k) else {
                // The policy and the index disagree; the key is gone from the policy now.
                tracing::warn!(
                    "kv_host_tier: eviction victim {:x} was not evictable",
                    k.short()
                );
                continue;
            };
            self.demote(sh, &k, &entry);
            sh.arena.free(entry.run);
            sh.stats.add_eviction();
            evicted += 1;
        }
        evicted
    }

    /// Hand an evicted READY entry to the cold store, if one is attached.
    fn demote(&mut self, sh: &Shared, k: &BlockKey, entry: &Entry) {
        let Some(cold) = self.cold.clone() else {
            return;
        };
        // SAFETY: the run is a live arena allocation of at least `entry.bytes` bytes; the entry
        // is READY with no lease, so no DMA reads or writes it, and the tier lock is held.
        let kv =
            unsafe { std::slice::from_raw_parts(entry.run.ptr as *const u8, entry.bytes as usize) };
        match cold.demote_with_tokens(k, &entry.tokens, kv) {
            Ok(()) => {
                self.cold_keys.insert(*k);
                sh.stats.add_cold_demoted();
            }
            Err(e) => {
                if !self.warned_demote {
                    tracing::warn!(
                        "kv_host_tier: cold store demote failed ({e:#}); dropping evicted blocks"
                    );
                    self.warned_demote = true;
                }
            }
        }
    }

    /// Bring the cold copy of `k` back into host RAM as a READY entry (the store refuses it when
    /// its stored tokens differ from the probe's `tokens`). Making room never evicts a key in
    /// `protect`. True on success; a miss forgets the key.
    pub(crate) fn promote_cold(
        &mut self,
        sh: &Shared,
        k: &BlockKey,
        tokens: &[u32],
        protect: &HashSet<BlockKey>,
    ) -> bool {
        let Some(cold) = self.cold.clone() else {
            return false;
        };
        if sh.block_bytes == 0 {
            return false;
        }
        // Read into a scratch buffer first: a miss or a failed read must not cost an eviction.
        let mut buf = vec![0u8; sh.block_bytes];
        match cold.promote_with_tokens(k, tokens, &mut buf) {
            Ok(true) => {}
            Ok(false) => {
                self.cold_keys.remove(k);
                return false;
            }
            Err(e) => {
                tracing::warn!(
                    "kv_host_tier: cold store promote of {:x} failed: {e:#}",
                    k.short()
                );
                self.cold_keys.remove(k);
                return false;
            }
        }
        let bb = sh.block_bytes as u64;
        let Some(run) = self.place_protected(sh, bb, Some(protect)) else {
            return false;
        };
        // SAFETY: `run` was just allocated with at least `bb` bytes and no DMA references it yet.
        let dst = unsafe { std::slice::from_raw_parts_mut(run.ptr as *mut u8, sh.block_bytes) };
        dst.copy_from_slice(&buf);
        if let Err(run) = self.index.insert_ready(*k, tokens, run, bb) {
            sh.arena.free(run);
            return false;
        }
        self.policy.insert_batch(&[*k]);
        sh.stats.add_cold_promoted();
        true
    }

    /// Return every run to the arena (after a full quiesce: no DMA in flight).
    pub(crate) fn free_all(&mut self, sh: &Shared) {
        for (k, e) in self.index.drain_all() {
            self.policy.remove(&k);
            sh.arena.free(e.run);
        }
    }

    /// Return the runs no DMA can touch; the rest are leaked. Returns how many were leaked.
    pub(crate) fn free_quiescent(&mut self, sh: &Shared) -> usize {
        for (k, e) in self.index.drain_quiescent() {
            self.policy.remove(&k);
            sh.arena.free(e.run);
        }
        self.index.len()
    }
}
