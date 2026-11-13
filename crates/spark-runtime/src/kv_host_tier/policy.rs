// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/distributed/eviction_policy/lru.py:72-95 (commit 3aefae1, Apache-2.0; "later keys in a request should be evicted first, because of prefix match"); no code copied.
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the host tier's eviction order. [`Policy`] wraps the three choices of
//! `--kv-tier-policy`:
//!
//! * **LRU** ([`TierPolicy::Lru`]): least recently used first.
//! * **ARC** ([`TierPolicy::Arc`], `policy_arc.rs`): adaptive replacement with ghost lists.
//! * **Heat** ([`TierPolicy::Heat`]): fewest accesses first, every heat halved after each
//!   `heat_halve_every` accesses (KVMem's selection-frequency idea), recency as the tie-break.
//!
//! **Tail-first.** Keys arrive in prefix order (block 0 first). A block is useful only while
//! every block before it is still cached somewhere, so a request's LATER blocks must leave
//! first. Every policy therefore applies a batch in reverse: the last block of the batch gets
//! the oldest stamp, the first block the newest.
//!
//! The policy only orders READY entries. It never decides alone: [`Policy::victims`] skips every
//! key the caller's predicate rejects (staging, leased), so a block a DMA may touch is never
//! chosen.

use std::collections::{BTreeMap, HashMap};

use super::config::TierPolicy;
use super::fingerprint::BlockKey;

#[path = "policy_arc.rs"]
mod arc;

/// An ordered set by recency: stamps grow, the smallest stamp is the least recent.
#[derive(Default, Debug)]
pub(crate) struct Recency {
    stamp_of: HashMap<BlockKey, u64>,
    order: BTreeMap<u64, BlockKey>,
    next: u64,
}

impl Recency {
    pub(crate) fn len(&self) -> usize {
        self.stamp_of.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.stamp_of.is_empty()
    }

    pub(crate) fn contains(&self, k: &BlockKey) -> bool {
        self.stamp_of.contains_key(k)
    }

    /// Make `k` the most recent (inserting it when absent).
    pub(crate) fn push_mru(&mut self, k: BlockKey) {
        if let Some(old) = self.stamp_of.remove(&k) {
            self.order.remove(&old);
        }
        self.next += 1;
        self.stamp_of.insert(k, self.next);
        self.order.insert(self.next, k);
    }

    pub(crate) fn remove(&mut self, k: &BlockKey) -> bool {
        match self.stamp_of.remove(k) {
            Some(s) => {
                self.order.remove(&s);
                true
            }
            None => false,
        }
    }

    pub(crate) fn pop_lru(&mut self) -> Option<BlockKey> {
        let (_, k) = self.order.pop_first()?;
        self.stamp_of.remove(&k);
        Some(k)
    }

    /// The least recent key the predicate accepts.
    pub(crate) fn lru_where(&self, ok: &dyn Fn(&BlockKey) -> bool) -> Option<BlockKey> {
        self.order.values().find(|k| ok(k)).copied()
    }

    /// Keys from least to most recent.
    pub(crate) fn iter_lru(&self) -> impl Iterator<Item = &BlockKey> {
        self.order.values()
    }
}

/// Least recently used.
#[derive(Default, Debug)]
pub(crate) struct Lru {
    list: Recency,
}

impl Lru {
    fn access(&mut self, k: BlockKey) {
        self.list.push_mru(k);
    }

    fn victims(&mut self, want: usize, ok: &dyn Fn(&BlockKey) -> bool) -> Vec<BlockKey> {
        let picked: Vec<BlockKey> = self
            .list
            .iter_lru()
            .filter(|k| ok(k))
            .take(want)
            .copied()
            .collect();
        for k in &picked {
            self.list.remove(k);
        }
        picked
    }
}

#[derive(Clone, Copy, Debug)]
struct HeatEntry {
    heat: u64,
    /// Halving epoch at which `heat` was last written.
    epoch: u64,
    stamp: u64,
}

/// Access frequency with periodic halving; recency breaks ties.
#[derive(Debug)]
pub(crate) struct Heat {
    entries: HashMap<BlockKey, HeatEntry>,
    halve_every: u64,
    accesses: u64,
    epoch: u64,
    next_stamp: u64,
}

impl Heat {
    fn new(halve_every: u64) -> Self {
        Self {
            entries: HashMap::new(),
            halve_every: halve_every.max(1),
            accesses: 0,
            epoch: 0,
            next_stamp: 0,
        }
    }

    fn decayed(&self, e: &HeatEntry) -> u64 {
        let shift = self.epoch.saturating_sub(e.epoch);
        if shift >= 64 { 0 } else { e.heat >> shift }
    }

    /// Current heat of `k`, for tests.
    #[cfg(test)]
    pub(crate) fn heat_of(&self, k: &BlockKey) -> Option<u64> {
        self.entries.get(k).map(|e| self.decayed(e))
    }

    fn access(&mut self, k: BlockKey) {
        self.accesses += 1;
        if self.accesses.is_multiple_of(self.halve_every) {
            self.epoch += 1;
        }
        self.next_stamp += 1;
        let heat = self.entries.get(&k).map(|e| self.decayed(e)).unwrap_or(0) + 1;
        self.entries.insert(
            k,
            HeatEntry {
                heat,
                epoch: self.epoch,
                stamp: self.next_stamp,
            },
        );
    }

    fn victims(&mut self, want: usize, ok: &dyn Fn(&BlockKey) -> bool) -> Vec<BlockKey> {
        if want == 0 {
            return Vec::new();
        }
        let mut cands: Vec<(u64, u64, BlockKey)> = self
            .entries
            .iter()
            .filter(|(k, _)| ok(k))
            .map(|(k, e)| (self.decayed(e), e.stamp, *k))
            .collect();
        if cands.len() > want {
            cands.select_nth_unstable(want - 1);
            cands.truncate(want);
        }
        cands.sort_unstable();
        let picked: Vec<BlockKey> = cands.into_iter().map(|(_, _, k)| k).collect();
        for k in &picked {
            self.entries.remove(k);
        }
        picked
    }
}

/// The configured eviction policy.
#[derive(Debug)]
// One `Policy` exists per tier, so the size gap between variants costs nothing;
// boxing ARC would add a pointer chase to every touch.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Policy {
    Lru(Lru),
    Arc(arc::ArcPolicy),
    Heat(Heat),
}

impl Policy {
    /// `capacity_blocks` is how many blocks the budget holds (ARC sizes its ghost lists by it).
    pub(crate) fn new(kind: TierPolicy, capacity_blocks: usize, heat_halve_every: u64) -> Self {
        match kind {
            TierPolicy::Lru => Policy::Lru(Lru::default()),
            TierPolicy::Arc => Policy::Arc(arc::ArcPolicy::new(capacity_blocks)),
            TierPolicy::Heat => Policy::Heat(Heat::new(heat_halve_every)),
        }
    }

    #[cfg(test)]
    pub(crate) fn kind(&self) -> TierPolicy {
        match self {
            Policy::Lru(_) => TierPolicy::Lru,
            Policy::Arc(_) => TierPolicy::Arc,
            Policy::Heat(_) => TierPolicy::Heat,
        }
    }

    /// Resident keys the policy tracks, for tests.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        match self {
            Policy::Lru(p) => p.list.len(),
            Policy::Arc(p) => p.resident_len(),
            Policy::Heat(p) => p.entries.len(),
        }
    }

    /// Newly READY keys, in prefix order (applied tail-first).
    pub(crate) fn insert_batch(&mut self, keys: &[BlockKey]) {
        for k in keys.iter().rev() {
            match self {
                Policy::Lru(p) => p.access(*k),
                Policy::Arc(p) => p.admit(*k),
                Policy::Heat(p) => p.access(*k),
            }
        }
    }

    /// Keys just used by a restore, in prefix order (applied tail-first).
    pub(crate) fn touch_batch(&mut self, keys: &[BlockKey]) {
        for k in keys.iter().rev() {
            match self {
                Policy::Lru(p) => p.access(*k),
                Policy::Arc(p) => p.hit(*k),
                Policy::Heat(p) => p.access(*k),
            }
        }
    }

    /// `k` left the tier for a reason other than eviction (failed copy, teardown).
    pub(crate) fn remove(&mut self, k: &BlockKey) {
        match self {
            Policy::Lru(p) => {
                p.list.remove(k);
            }
            Policy::Arc(p) => p.forget(k),
            Policy::Heat(p) => {
                p.entries.remove(k);
            }
        }
    }

    /// Choose up to `want` victims in eviction order among keys `ok` accepts, and drop them from
    /// the resident order (ARC remembers them in its ghost lists).
    pub(crate) fn victims(&mut self, want: usize, ok: &dyn Fn(&BlockKey) -> bool) -> Vec<BlockKey> {
        match self {
            Policy::Lru(p) => p.victims(want, ok),
            Policy::Arc(p) => p.victims(want, ok),
            Policy::Heat(p) => p.victims(want, ok),
        }
    }

    /// The ARC state, for tests.
    #[cfg(test)]
    pub(crate) fn as_arc(&self) -> Option<&arc::ArcPolicy> {
        match self {
            Policy::Arc(p) => Some(p),
            _ => None,
        }
    }

    /// The heat state, for tests.
    #[cfg(test)]
    pub(crate) fn as_heat(&self) -> Option<&Heat> {
        match self {
            Policy::Heat(p) => Some(p),
            _ => None,
        }
    }
}
