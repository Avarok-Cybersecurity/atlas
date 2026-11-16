// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: tests of the index lifecycle bookkeeping: row matching by
//! block table and by id, the gap and rollback rules, invalidation by a write
//! into a covered block, release of stale sequences, and the index cache
//! (prefix lookup, copy-on-write sharing, least-recently-used eviction).

use std::sync::Arc;

use spark_runtime::gpu::DevicePtr;

use super::arena::Region;
use super::lifecycle::*;
use super::state::{IndexLayout, LayerIndex};

fn layout() -> IndexLayout {
    IndexLayout {
        kv_heads: 1,
        head_dim: 8,
        cap_centroids: 8,
        cap_tokens: 64,
    }
}

fn entry(layer: u32, blocks: &[u32], end: u32, off: usize) -> Arc<IndexEntry> {
    Arc::new(IndexEntry {
        key: IndexKey {
            model: 1,
            attn_layer: layer,
            params: 2,
            blocks: blocks.to_vec(),
            end,
        },
        index: LayerIndex::at(DevicePtr(0x10000 + off as u64), layout()),
        region: Region {
            offset: off,
            bytes: 256,
        },
        fingerprint: Fingerprint {
            pos: end - 1,
            bytes: vec![1, 2, 3],
        },
        build_secs: 0.0,
    })
}

#[test]
fn rows_match_the_sequence_whose_table_they_extend() {
    let mut t = SeqTable::default();
    let a = t.sight(None, &[10, 11, 12], 40, 1, &[]);
    assert!(matches!(a, Sighting::New(_)));
    let b = t.sight(None, &[20, 21], 30, 1, &[a.index()]);
    assert!(matches!(b, Sighting::New(_)));
    // Next step: A grew a block, B did not.
    let a2 = t.sight(None, &[10, 11, 12, 13], 50, 2, &[]);
    assert_eq!(a2, Sighting::Continuing(a.index()));
    let b2 = t.sight(None, &[20, 21], 31, 2, &[a2.index()]);
    assert_eq!(b2, Sighting::Continuing(b.index()));
    // A different sequence sharing A's first blocks is not A.
    let c = t.sight(None, &[10, 11, 99], 40, 2, &[a2.index(), b2.index()]);
    assert!(matches!(c, Sighting::New(_)));
    assert_eq!(t.len(), 3);
}

#[test]
fn ids_win_over_tables_and_one_row_per_entry() {
    let mut t = SeqTable::default();
    let a = t.sight(Some(7), &[1, 2], 20, 1, &[]);
    let same = t.sight(Some(7), &[5, 6], 21, 2, &[]);
    assert_eq!(same, Sighting::Continuing(a.index()), "the id decides");
    let other = t.sight(Some(8), &[5, 6, 7], 40, 2, &[a.index()]);
    assert!(
        matches!(other, Sighting::New(_)),
        "an id never takes another id's entry"
    );
    assert!(t.remove_id(7).is_some());
    assert!(t.remove_id(7).is_none());
}

#[test]
fn gaps_and_deep_rollbacks_are_returning() {
    let mut t = SeqTable::default();
    let a = t.sight(None, &[1, 2, 3], 40, 1, &[]);
    let r = t.sight(None, &[1, 2, 3], 41, 1 + GAP_STEPS + 1, &[]);
    assert_eq!(r, Sighting::Returning(a.index()));
    let c = t.sight(None, &[1, 2, 3], 42, 2 + GAP_STEPS + 1, &[]);
    assert_eq!(c, Sighting::Continuing(a.index()));
    let back = t.sight(None, &[1, 2, 3], 10, 3 + GAP_STEPS + 1, &[]);
    assert_eq!(
        back,
        Sighting::Returning(a.index()),
        "rolled back further than a verify"
    );
    t.get_mut(a.index()).layers.insert(0, LayerSlot::default());
    t.mark_suspect_except(&[]);
    let s = t.sight(None, &[1, 2, 3], 11, 4 + GAP_STEPS + 1, &[]);
    assert_eq!(
        s,
        Sighting::Returning(a.index()),
        "an unmapped write makes it suspect"
    );
}

#[test]
fn a_write_into_a_covered_block_ends_that_sequence() {
    let mut t = SeqTable::default();
    let a = t.sight(None, &[1, 2, 3, 4], 60, 1, &[]).index();
    t.get_mut(a).covered.extend([1, 2]);
    let a_uid = t.get(a).uid;
    let b = t.sight(None, &[9, 8], 20, 1, &[a]).index();
    let b_uid = t.get(b).uid;
    assert!(
        t.invalidate_blocks(&[2], Some(a_uid)).is_empty(),
        "its own write is fine"
    );
    let gone = t.invalidate_blocks(&[2], Some(b_uid));
    assert_eq!(gone.len(), 1);
    assert_eq!(gone[0].uid, a_uid);
    assert!(t.by_uid(a_uid).is_none());
    assert!(t.by_uid(b_uid).is_some());
}

#[test]
fn stale_and_surplus_sequences_are_swept() {
    let mut t = SeqTable::default();
    t.sight(None, &[1], 10, 1, &[]);
    t.sight(None, &[2], 10, 1 + STALE_STEPS, &[]);
    let gone = t.sweep(2 + STALE_STEPS);
    assert_eq!(gone.len(), 1);
    assert_eq!(gone[0].table, vec![1]);
    for i in 0..(MAX_LIVE as u32 + 3) {
        t.sight(None, &[100 + i], 10, 3 + STALE_STEPS + u64::from(i), &[]);
    }
    let gone = t.sweep(3 + STALE_STEPS + MAX_LIVE as u64 + 3);
    assert!(t.len() <= MAX_LIVE);
    assert!(!gone.is_empty());
}

#[test]
fn idle_eviction_spares_rows_of_the_current_step() {
    let mut t = SeqTable::default();
    let a = t.sight(None, &[1], 10, 5, &[]).index();
    t.get_mut(a).layers.insert(0, LayerSlot::default());
    assert!(t.evict_idle(5).is_none(), "seen this step");
    let e = t.evict_idle(6).unwrap();
    assert_eq!(e.table, vec![1]);
}

#[test]
fn the_cache_serves_the_longest_matching_prefix() {
    let mut s = IndexStore::default();
    assert!(s.insert(entry(0, &[1, 2], 32, 0)).is_none());
    assert!(s.insert(entry(0, &[1, 2, 3, 4], 64, 256)).is_none());
    s.insert(entry(1, &[1, 2, 3, 4], 64, 512));
    let c = s.candidates(1, 0, 2, &[1, 2, 3, 4, 5], 80);
    assert_eq!(
        c.iter().map(|e| e.key.end).collect::<Vec<_>>(),
        vec![64, 32]
    );
    assert!(
        s.candidates(1, 0, 2, &[1, 9, 3, 4], 80).is_empty(),
        "a different block breaks the chain"
    );
    assert_eq!(
        s.candidates(1, 0, 2, &[1, 2, 3, 4], 48).len(),
        1,
        "an entry past the wanted end is skipped"
    );
    assert!(
        s.candidates(1, 0, 3, &[1, 2, 3, 4], 80).is_empty(),
        "other build parameters"
    );
    assert!(
        s.candidates(9, 0, 2, &[1, 2, 3, 4], 80).is_empty(),
        "other model"
    );
    assert_eq!(s.bytes(), 768);
}

#[test]
fn shared_entries_are_not_evicted_and_slots_release_cleanly() {
    let mut s = IndexStore::default();
    let held = entry(0, &[1], 16, 0);
    s.insert(held.clone());
    s.insert(entry(0, &[2], 16, 256));
    // The held one is skipped; the free one goes.
    assert_eq!(
        s.evict_one(),
        Some(Region {
            offset: 256,
            bytes: 256
        })
    );
    assert!(
        s.evict_one().is_none(),
        "the remaining entry is held by a slot"
    );
    let slot = LayerSlot {
        base: Some(held),
        own: Some((
            LayerIndex::at(DevicePtr(0x9000), layout()),
            Region {
                offset: 1024,
                bytes: 256,
            },
        )),
        fingerprint: None,
    };
    // The store still holds the base, so only the private region is freed.
    assert_eq!(
        slot.release(),
        vec![Region {
            offset: 1024,
            bytes: 256
        }]
    );
    assert_eq!(
        s.evict_one(),
        Some(Region {
            offset: 0,
            bytes: 256
        })
    );
}

#[test]
fn invalidation_and_discard_free_unshared_entries() {
    let mut s = IndexStore::default();
    s.insert(entry(0, &[1, 2], 32, 0));
    s.insert(entry(0, &[3, 4], 32, 256));
    assert_eq!(
        s.invalidate_blocks(&[4]),
        vec![Region {
            offset: 256,
            bytes: 256
        }]
    );
    assert_eq!(s.len(), 1);
    let c = s.candidates(1, 0, 2, &[1, 2], 32).pop().unwrap();
    assert_eq!(
        s.discard(&c),
        Some(Region {
            offset: 0,
            bytes: 256
        })
    );
    assert!(s.is_empty());
    // Re-inserting an equal key hands back the replaced entry's region.
    s.insert(entry(0, &[5], 16, 512));
    assert_eq!(
        s.insert(entry(0, &[5], 16, 768)),
        Some(Region {
            offset: 512,
            bytes: 256
        })
    );
}
