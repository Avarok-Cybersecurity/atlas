// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/distributed/eviction_policy/isolated_lru.py and lru.py
// (commit 3aefae1, Apache-2.0): keep the eviction candidates in an ordered index so victim
// selection never holds or walks a whole-tree snapshot; no code copied.

//! Responsibility: the ordered victim index behind [`RadixTreeInner::evict`]
//! (row `radix_lru_index`, `AVAROK_RADIX_LRU_INDEX`), the victim rule both
//! eviction modes share, and the row's spark-runtime switch.
//!
//! # The rule (unchanged)
//!
//! A node is a CANDIDATE when it is not a root (`parent.is_some()`), at most
//! the cache's own reference holds it (`ref_count <= 1`), it has no children,
//! and it stores a block (`block_idx != u32::MAX`). The victim is the
//! candidate with the smallest `last_access`; the full scan breaks ties toward
//! the LOWEST node id, because it walks the arena in id order and replaces its
//! best only on a strict `<`.
//!
//! # Why the tuple order reproduces the scan
//!
//! `last_access` is NOT unique: one `inc_refs` or `insert` stamps every node on
//! its path with the same tick. The index is a `BTreeSet<(last_access, NodeId)>`
//! and `NodeId` IS the arena index the scan iterates, so `first()` is the
//! smallest `last_access` and, among equal ticks, the smallest id: exactly the
//! scan's pick. If `NodeId` ever stops being the arena index, this equivalence
//! breaks and the oracle test (`radix_tree/tests/lru_index.rs`) fails.
//!
//! # Maintenance
//!
//! Every write that can change one of the four predicate inputs or a
//! candidate's `last_access` calls [`RadixTreeInner::lru_refresh`] for the
//! node it touched: `inc_refs` and `dec_refs` (ref count and tick), `insert`
//! (tick and ref count of an existing node; a new child makes its parent a
//! non-leaf), `evict` (the victim leaves; its parent may become a leaf), and
//! the copy-on-write source touch. With the row off the index is not
//! allocated and every refresh is one `is_none` branch. Test builds always
//! maintain it and compare it with the scan on every pick, so every radix test
//! runs both modes.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{NodeId, RadixTreeInner};

/// The row value, published once at startup by `port_apply`
/// (`spark_model::layers::ops::port_rows::radix_lru_index::row()`). Default
/// off: a tree built without the setter scans, exactly as before.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Publish the resolved `radix_lru_index` row. Read once per tree, at
/// construction; call it before the model (and its prefix cache) is built.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// The published row value.
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Candidates ordered by `(last_access, id)`, plus each indexed node's key so
/// an update can remove the old entry without searching.
#[derive(Clone, Debug, Default)]
pub(in crate::radix_tree) struct LruIndex {
    set: BTreeSet<(u64, NodeId)>,
    keys: Vec<Option<u64>>,
}

impl LruIndex {
    /// Make node `id`'s entry `key` (`None` = not a candidate).
    pub(in crate::radix_tree) fn update(&mut self, id: NodeId, key: Option<u64>) {
        if self.keys.len() <= id {
            self.keys.resize(id + 1, None);
        }
        let old = self.keys[id];
        if old == key {
            return;
        }
        if let Some(t) = old {
            self.set.remove(&(t, id));
        }
        if let Some(t) = key {
            self.set.insert((t, id));
        }
        self.keys[id] = key;
    }

    /// The least recently used candidate (lowest id among equal ticks).
    pub(in crate::radix_tree) fn first(&self) -> Option<NodeId> {
        self.set.first().map(|&(_, id)| id)
    }

    /// The tick node `id` is indexed under, if it is indexed.
    pub(in crate::radix_tree) fn key_of(&self, id: NodeId) -> Option<u64> {
        self.keys.get(id).copied().flatten()
    }

    /// Number of indexed candidates.
    #[cfg(test)]
    pub(in crate::radix_tree) fn len(&self) -> usize {
        self.set.len()
    }

    fn clear(&mut self) {
        self.set.clear();
        self.keys.clear();
    }

    /// Test hook: plant an entry the tree does not justify.
    #[cfg(test)]
    pub(in crate::radix_tree) fn plant_stale(&mut self, id: NodeId, key: u64) {
        self.update(id, Some(key));
    }
}

impl RadixTreeInner {
    /// The victim predicate, shared by the scan and the index.
    pub(in crate::radix_tree) fn evictable(&self, id: NodeId) -> bool {
        let n = &self.nodes[id];
        n.parent.is_some() && n.ref_count <= 1 && n.children.is_empty() && n.block_idx != u32::MAX
    }

    /// Re-derive node `id`'s index entry after a write to it. No-op without
    /// an index.
    pub(in crate::radix_tree) fn lru_refresh(&mut self, id: NodeId) {
        if self.lru.is_none() {
            return;
        }
        let key = self.lru_key(id);
        if let Some(lru) = self.lru.as_mut() {
            lru.update(id, key);
        }
    }

    /// Node `id`'s index key: its tick when it is a candidate.
    fn lru_key(&self, id: NodeId) -> Option<u64> {
        self.evictable(id).then_some(self.nodes[id].last_access)
    }

    /// The full scan: every node in id order, best replaced only on a strict
    /// `<`. Kept as the row-off path and as the oracle.
    pub(in crate::radix_tree) fn scan_victim(&self) -> Option<NodeId> {
        let mut best: Option<(NodeId, u64)> = None;
        for (id, node) in self.nodes.iter().enumerate() {
            // Roots (one per adapter_id) carry `parent == None` and
            // `block_idx == u32::MAX`; the block_idx guard below already
            // excludes them, but skip explicitly for clarity.
            if node.parent.is_none() {
                continue;
            }
            if node.ref_count <= 1 && node.children.is_empty() && node.block_idx != u32::MAX {
                match best {
                    None => best = Some((id, node.last_access)),
                    Some((_, best_access)) if node.last_access < best_access => {
                        best = Some((id, node.last_access));
                    }
                    _ => {}
                }
            }
        }
        best.map(|(id, _)| id)
    }

    /// The next victim under this tree's mode. In test builds, whenever the
    /// index exists, the index and the scan must agree on every pick and the
    /// index must equal a fresh rebuild.
    pub(in crate::radix_tree) fn pick_victim(&mut self) -> Option<NodeId> {
        #[cfg(test)]
        self.lru_cross_check();
        if self.use_lru && self.lru.is_some() {
            self.index_victim(cfg!(any(test, debug_assertions)))
        } else {
            self.scan_victim()
        }
    }

    /// The index's victim, checked against the predicate (an O(1) test on
    /// every evict). A stale entry would evict a node a sequence still uses,
    /// which corrupts KV, so it is never trusted: `strict` (test and debug
    /// builds) panics; otherwise the index is rebuilt from the arena, the
    /// event is logged, and the pick is retried on the rebuilt index.
    pub(in crate::radix_tree) fn index_victim(&mut self, strict: bool) -> Option<NodeId> {
        let id = self.lru.as_ref()?.first()?;
        let key = self.lru.as_ref().and_then(|l| l.key_of(id));
        if self.evictable(id) && key == Some(self.nodes[id].last_access) {
            return Some(id);
        }
        assert!(
            !strict,
            "radix LRU index holds a stale entry for node {id} (key {key:?}); the \
             candidate does not satisfy the victim predicate"
        );
        tracing::error!(
            "radix LRU index held a stale entry for node {id} (key {key:?}); rebuilding the \
             index from the tree (victim selection is unchanged, this costs one full scan)"
        );
        self.lru_rebuild();
        let id = self.lru.as_ref()?.first()?;
        debug_assert!(self.evictable(id));
        Some(id)
    }

    /// Rebuild the index from the arena.
    pub(in crate::radix_tree) fn lru_rebuild(&mut self) {
        let Some(mut lru) = self.lru.take() else {
            return;
        };
        lru.clear();
        for id in 0..self.nodes.len() {
            lru.update(id, self.lru_key(id));
        }
        self.lru = Some(lru);
    }

    /// Test check before every pick: with an index present (always, in test
    /// builds), it equals a fresh rebuild and its victim is the scan's.
    #[cfg(test)]
    fn lru_cross_check(&self) {
        if self.lru.is_none() {
            return;
        }
        self.lru_assert_consistent();
        let indexed = self.lru.as_ref().and_then(LruIndex::first);
        assert_eq!(
            indexed,
            self.scan_victim(),
            "radix LRU index and full scan disagree on the victim"
        );
    }

    /// Test check: the maintained index equals a fresh rebuild.
    #[cfg(test)]
    pub(in crate::radix_tree) fn lru_assert_consistent(&self) {
        let Some(lru) = self.lru.as_ref() else {
            return;
        };
        let want: BTreeSet<(u64, NodeId)> = (0..self.nodes.len())
            .filter(|&id| self.evictable(id))
            .map(|id| (self.nodes[id].last_access, id))
            .collect();
        assert_eq!(
            lru.set, want,
            "radix LRU index drifted from the tree (a transition did not refresh it)"
        );
    }

    /// Test hook: force a tick, as many same-tick ties as a test wants.
    #[cfg(test)]
    pub(in crate::radix_tree) fn test_set_last_access(&mut self, id: NodeId, tick: u64) {
        self.nodes[id].last_access = tick;
        self.lru_refresh(id);
    }

    /// Test view: node `id`'s tick.
    #[cfg(test)]
    pub(in crate::radix_tree) fn test_tick(&self, id: NodeId) -> u64 {
        self.nodes[id].last_access
    }

    /// Test view: ids of nodes that store a block (live, non-root).
    #[cfg(test)]
    pub(in crate::radix_tree) fn test_live_nodes(&self) -> Vec<NodeId> {
        (0..self.nodes.len())
            .filter(|&id| self.nodes[id].block_idx != u32::MAX)
            .collect()
    }

    /// Test view: the whole tree as one canonical line per arena slot, so two
    /// trees fed the same operations can be compared exactly.
    #[cfg(test)]
    pub(in crate::radix_tree) fn test_dump(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(id, n)| {
                let mut kids: Vec<NodeId> = n.children.values().copied().collect();
                kids.sort_unstable();
                format!(
                    "{id}: blk={} disk={} ref={} tick={} parent={:?} key={:?} kids={kids:?} \
                     partial={:?} ctx={:#x}",
                    n.block_idx,
                    n.disk_block_id,
                    n.ref_count,
                    n.last_access,
                    n.parent,
                    n.parent_key,
                    n.partial_suffix,
                    n.context_hash,
                )
            })
            .collect();
        out.push(format!("free={:?}", self.free_nodes));
        out.push(format!("ticks={}", self.access_counter));
        out
    }
}
