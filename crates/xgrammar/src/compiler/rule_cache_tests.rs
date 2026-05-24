// SPDX-License-Identifier: AGPL-3.0-only
//
// Unit tests for the cross-grammar `RuleLevelCache` (Tier 2).
//
// These exercise the cache in isolation — hit/miss, structural keying,
// LRU eviction, the memory budget and `get_or_compute`. End-to-end
// reuse across two real grammar compiles is covered in
// `tests/compile_tests.rs`.

use std::sync::Arc;

use super::*;
use crate::compiler::mask::{AdaptiveTokenMask, StoreType};

/// A mask with `n` accepted indices — `memory_size()` is `4*n` bytes,
/// so tests can dial the cache's memory accounting precisely.
fn mask(n: usize) -> Arc<AdaptiveTokenMask> {
    let mut m = AdaptiveTokenMask::empty(StoreType::Accepted, 0);
    m.accepted_indices = (0..n as i32).collect();
    Arc::new(m)
}

fn key(hash: u64, node: i32) -> RuleMaskKey {
    RuleMaskKey {
        fsm_hash: hash,
        fsm_new_node_id: node,
        state_cnt: 3,
        edge_cnt: 5,
    }
}

#[test]
fn miss_then_hit() {
    let cache = RuleLevelCache::new(UNLIMITED_SIZE);
    let k = key(0xABCD, 1);
    assert!(cache.get(&k).is_none(), "fresh cache must miss");

    let m = mask(4);
    assert!(cache.add(k, Arc::clone(&m)), "first add must store");

    let hit = cache.get(&k).expect("must hit after add");
    assert!(
        Arc::ptr_eq(&hit, &m),
        "hit must return the exact cached Arc"
    );
}

#[test]
fn duplicate_add_is_rejected() {
    let cache = RuleLevelCache::new(UNLIMITED_SIZE);
    let k = key(1, 0);
    assert!(cache.add(k, mask(2)));
    // A second add of the same key is a no-op (upstream returns false).
    assert!(!cache.add(k, mask(99)));
    assert_eq!(cache.len(), 1);
    // The original entry is untouched.
    assert_eq!(cache.get(&k).unwrap().accepted_indices.len(), 2);
}

#[test]
fn distinct_keys_do_not_collide() {
    // Structurally different rules differ in at least one key field;
    // none of these four must alias another's mask.
    let cache = RuleLevelCache::new(UNLIMITED_SIZE);
    let keys = [
        RuleMaskKey {
            fsm_hash: 1,
            fsm_new_node_id: 0,
            state_cnt: 2,
            edge_cnt: 2,
        },
        RuleMaskKey {
            fsm_hash: 2,
            fsm_new_node_id: 0,
            state_cnt: 2,
            edge_cnt: 2,
        },
        RuleMaskKey {
            fsm_hash: 1,
            fsm_new_node_id: 1,
            state_cnt: 2,
            edge_cnt: 2,
        },
        RuleMaskKey {
            fsm_hash: 1,
            fsm_new_node_id: 0,
            state_cnt: 9,
            edge_cnt: 2,
        },
    ];
    for (i, k) in keys.iter().enumerate() {
        assert!(cache.add(*k, mask(i + 1)));
    }
    assert_eq!(cache.len(), 4);
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(cache.get(k).unwrap().accepted_indices.len(), i + 1);
    }
}

#[test]
fn same_structural_key_reuses_across_grammars() {
    // Two different "grammars" each produce a rule whose FSM hashes to
    // the same structural key — the second reuses the first's mask.
    let cache = RuleLevelCache::new(UNLIMITED_SIZE);
    let shared = key(0xF00D, 7);

    let computed = cache.get_or_compute(shared, || {
        let mut m = AdaptiveTokenMask::empty(StoreType::Accepted, 0);
        m.accepted_indices = vec![10, 20, 30];
        m
    });
    assert_eq!(computed.accepted_indices, vec![10, 20, 30]);

    // Second request, structurally identical rule: `compute` MUST NOT
    // run again — assert via a closure that would panic if invoked.
    let reused = cache.get_or_compute(shared, || panic!("must reuse, not recompute"));
    assert!(Arc::ptr_eq(&computed, &reused));
}

#[test]
fn get_or_compute_inserts_on_miss() {
    let cache = RuleLevelCache::new(UNLIMITED_SIZE);
    assert!(cache.is_empty());
    let k = key(5, 5);
    let _ = cache.get_or_compute(k, || AdaptiveTokenMask::empty(StoreType::Accepted, 0));
    assert_eq!(
        cache.len(),
        1,
        "get_or_compute must insert the computed mask"
    );
    assert!(cache.get(&k).is_some());
}

#[test]
fn memory_size_tracks_entries() {
    let cache = RuleLevelCache::new(UNLIMITED_SIZE);
    assert_eq!(cache.memory_size(), 0);
    cache.add(key(1, 1), mask(4)); // 16 bytes
    cache.add(key(2, 2), mask(6)); // 24 bytes
    assert_eq!(cache.memory_size(), 16 + 24);
}

#[test]
fn lru_evicts_least_recently_used() {
    // Budget fits exactly two 16-byte masks. Adding a third evicts the
    // oldest — unless it was refreshed by a `get`.
    let cache = RuleLevelCache::new(32);
    let (k1, k2, k3) = (key(1, 1), key(2, 2), key(3, 3));
    assert!(cache.add(k1, mask(4)));
    assert!(cache.add(k2, mask(4)));
    assert_eq!(cache.len(), 2);

    // Touch k1 so k2 becomes the least-recently-used entry.
    assert!(cache.get(&k1).is_some());

    assert!(cache.add(k3, mask(4)));
    assert_eq!(cache.len(), 2, "budget holds only two entries");
    assert!(cache.get(&k1).is_some(), "k1 was refreshed — must survive");
    assert!(cache.get(&k3).is_some(), "k3 is newest — must be present");
    assert!(cache.get(&k2).is_none(), "k2 was LRU — must be evicted");
}

#[test]
fn oversized_mask_is_not_cached() {
    // A mask larger than the whole budget can never be stored.
    let cache = RuleLevelCache::new(8);
    assert!(
        !cache.add(key(1, 1), mask(100)),
        "100*4 bytes > 8-byte budget"
    );
    assert!(cache.is_empty());
    // get_or_compute still returns the computed value even when the
    // insert is rejected.
    let m = cache.get_or_compute(key(1, 1), || {
        let mut x = AdaptiveTokenMask::empty(StoreType::Accepted, 0);
        x.accepted_indices = (0..100).collect();
        x
    });
    assert_eq!(m.accepted_indices.len(), 100);
    assert!(cache.is_empty(), "rejected insert leaves the cache empty");
}

#[test]
fn clear_empties_the_cache() {
    let cache = RuleLevelCache::new(UNLIMITED_SIZE);
    cache.add(key(1, 1), mask(4));
    cache.add(key(2, 2), mask(4));
    assert_eq!(cache.len(), 2);
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.memory_size(), 0);
    assert!(cache.get(&key(1, 1)).is_none());
}

#[test]
fn unlimited_cache_never_evicts() {
    let cache = RuleLevelCache::new(UNLIMITED_SIZE);
    for i in 0..256 {
        assert!(cache.add(key(i, i as i32), mask(50)));
    }
    assert_eq!(cache.len(), 256, "UNLIMITED_SIZE must never evict");
    assert_eq!(cache.max_size(), UNLIMITED_SIZE);
}

#[test]
fn lru_eviction_indices_stay_consistent() {
    // Stress the index-fix-up paths: many adds with eviction, then a
    // touch of a survivor, then more adds. Every survivor must remain
    // retrievable with the correct mask.
    let cache = RuleLevelCache::new(48); // holds three 16-byte masks
    for i in 0..3 {
        cache.add(key(100 + i, i as i32), mask(4));
    }
    // Touch the middle entry, then add two more — forces removes at
    // index 0 with subsequent index shifts.
    assert!(cache.get(&key(101, 1)).is_some());
    cache.add(key(200, 9), mask(4));
    cache.add(key(201, 10), mask(4));
    assert_eq!(cache.len(), 3);
    // Whatever survived must still map to a 4-index mask.
    for k in [
        key(100, 0),
        key(101, 1),
        key(102, 2),
        key(200, 9),
        key(201, 10),
    ] {
        if let Some(m) = cache.get(&k) {
            assert_eq!(m.accepted_indices.len(), 4);
        }
    }
    // The most recent two are guaranteed present.
    assert!(cache.get(&key(200, 9)).is_some());
    assert!(cache.get(&key(201, 10)).is_some());
}
