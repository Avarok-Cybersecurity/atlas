// SPDX-License-Identifier: AGPL-3.0-only
//
// Placement + eviction bookkeeping for the KV cache tier cascade (T1 = local
// pinned LPDDR write-back cache in front of the peer/SSD backing). Pure logic,
// no I/O and no CUDA, so it unit-tests on the metal/skip build with no hardware.
//
// LRU by default: the `StorageBackend` trait passes no predictor score to the
// backend, so a predictor-scored T1 eviction would need an out-of-trait side
// channel (deferred). Groups are keyed by `GroupKey` — write_from_host / read
// are per-group and each group is independently addressed.

use std::collections::{HashMap, VecDeque};

use crate::group::GroupKey;

/// Result of planning a T1 write: the slot to write `key` into, and — if a
/// resident group had to be evicted to make room — the victim to flush DOWN to
/// the backing tier first (its bytes still occupy `slot` until overwritten).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WritePlan {
    pub slot: u32,
    pub flush_victim: Option<(GroupKey, u32)>,
}

/// A fixed-capacity LRU set of resident groups over `cap_slots` byte slots.
pub struct SlotCache {
    cap_slots: u32,
    lookup: HashMap<GroupKey, u32>,
    slot_key: Vec<Option<GroupKey>>,
    free: VecDeque<u32>,
    /// LRU order: front = least-recently-used (eviction target), back = MRU.
    lru: VecDeque<u32>,
}

impl SlotCache {
    pub fn new(cap_slots: u32) -> Self {
        assert!(cap_slots > 0, "SlotCache needs at least one slot");
        Self {
            cap_slots,
            lookup: HashMap::new(),
            slot_key: vec![None; cap_slots as usize],
            free: (0..cap_slots).collect(),
            lru: VecDeque::with_capacity(cap_slots as usize),
        }
    }

    pub fn capacity(&self) -> u32 {
        self.cap_slots
    }

    fn move_to_back(&mut self, slot: u32) {
        if let Some(pos) = self.lru.iter().position(|&s| s == slot) {
            self.lru.remove(pos);
        }
        self.lru.push_back(slot);
    }

    /// Bump `slot` to most-recently-used (call on every hit).
    pub fn touch(&mut self, slot: u32) {
        self.move_to_back(slot);
    }

    /// Plan a write of `key`. Overwrite-in-place if resident (no victim); else a
    /// free slot; else evict the LRU slot and report its group to flush down.
    pub fn plan_write(&mut self, key: GroupKey) -> WritePlan {
        if let Some(&slot) = self.lookup.get(&key) {
            self.move_to_back(slot);
            return WritePlan {
                slot,
                flush_victim: None,
            };
        }
        if let Some(slot) = self.free.pop_front() {
            self.install(key, slot);
            return WritePlan {
                slot,
                flush_victim: None,
            };
        }
        // Full → evict the LRU tail's group (front of the deque).
        let victim_slot = self.lru.pop_front().expect("full cache has an LRU entry");
        let victim_key = self.slot_key[victim_slot as usize]
            .take()
            .expect("occupied slot has a key");
        self.lookup.remove(&victim_key);
        self.install(key, victim_slot);
        WritePlan {
            slot: victim_slot,
            flush_victim: Some((victim_key, victim_slot)),
        }
    }

    fn install(&mut self, key: GroupKey, slot: u32) {
        self.lookup.insert(key, slot);
        self.slot_key[slot as usize] = Some(key);
        self.lru.push_back(slot);
    }

    /// Partition request keys into T1 hits `(request_index, slot)` and misses
    /// `(request_index)`. Read-only — the caller `touch`es the hits after the
    /// copy so a failed copy doesn't perturb LRU order.
    pub fn plan_read(&self, keys: &[GroupKey]) -> (Vec<(usize, u32)>, Vec<usize>) {
        let mut hits = Vec::new();
        let mut misses = Vec::new();
        for (i, k) in keys.iter().enumerate() {
            match self.lookup.get(k) {
                Some(&slot) => hits.push((i, slot)),
                None => misses.push(i),
            }
        }
        (hits, misses)
    }

    /// Every resident `(key, slot)` — used to flush the whole cache on drop.
    pub fn residents(&self) -> Vec<(GroupKey, u32)> {
        self.lookup.iter().map(|(&k, &s)| (k, s)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::KvKind;

    fn k(block: u32) -> GroupKey {
        GroupKey::new(0, block, 0, KvKind::K)
    }

    #[test]
    fn free_slots_then_overwrite_in_place() {
        let mut c = SlotCache::new(3);
        let p0 = c.plan_write(k(0));
        assert_eq!(p0.flush_victim, None);
        let p1 = c.plan_write(k(1));
        assert_ne!(p1.slot, p0.slot);
        // Re-write k(0): same slot, no victim.
        let p0b = c.plan_write(k(0));
        assert_eq!(p0b.slot, p0.slot);
        assert_eq!(p0b.flush_victim, None);
    }

    #[test]
    fn fill_then_evict_lru_tail() {
        let mut c = SlotCache::new(2);
        let s0 = c.plan_write(k(0)).slot;
        let _s1 = c.plan_write(k(1)).slot;
        // Touch k(0) so k(1) becomes LRU.
        let (hits, _) = c.plan_read(&[k(0)]);
        c.touch(hits[0].1);
        // Write k(2) → evicts k(1) (LRU), reusing its slot.
        let p2 = c.plan_write(k(2));
        assert_eq!(p2.flush_victim.map(|(vk, _)| vk), Some(k(1)));
        // k(2) reuses k(1)'s (evicted) slot, not k(0)'s still-resident slot.
        assert_ne!(p2.slot, s0);
    }

    #[test]
    fn hit_miss_partition() {
        let mut c = SlotCache::new(4);
        c.plan_write(k(0));
        c.plan_write(k(2));
        let (hits, misses) = c.plan_read(&[k(0), k(1), k(2), k(3)]);
        let hit_idx: Vec<usize> = hits.iter().map(|(i, _)| *i).collect();
        assert_eq!(hit_idx, vec![0, 2]);
        assert_eq!(misses, vec![1, 3]);
    }

    #[test]
    fn residents_lists_all_live_groups() {
        let mut c = SlotCache::new(3);
        c.plan_write(k(0));
        c.plan_write(k(1));
        let mut r: Vec<GroupKey> = c.residents().into_iter().map(|(k, _)| k).collect();
        r.sort_by_key(|g| g.block);
        assert_eq!(r, vec![k(0), k(1)]);
    }

    #[test]
    fn evicted_group_is_no_longer_a_hit() {
        let mut c = SlotCache::new(1);
        c.plan_write(k(0));
        let p = c.plan_write(k(1)); // evicts k(0)
        assert_eq!(p.flush_victim.map(|(vk, _)| vk), Some(k(0)));
        let (hits, misses) = c.plan_read(&[k(0), k(1)]);
        assert_eq!(hits.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![1]);
        assert_eq!(misses, vec![0]);
    }
}
