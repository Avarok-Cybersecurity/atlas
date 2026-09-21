// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//! CPU-only tests for the expert cache policy over an in-memory source: the
//! arena is a `Vec<u8>` and the "device" address is the host address, which is
//! exactly the GB10 unified-memory case with the alias removed. Byte identity
//! against the real shards is `deepseek_v41_stream_oracle_test.rs`.

use std::sync::Mutex;

use anyhow::{Result, bail};

use super::expert_lru::ExpertLru;
use super::expert_stream::{ExpertSource, SlotLayout};
use crate::gpu::DevicePtr;

const LAYOUT: SlotLayout = SlotLayout {
    gate_off: 0,
    up_off: 16,
    down_off: 32,
    gate_bytes: 16,
    up_bytes: 16,
    down_bytes: 24,
    bytes: 56,
};

/// Deterministic bytes for (layer, expert, byte index).
fn expected(layer: u32, expert: u32, i: usize) -> u8 {
    (layer as usize * 131 + expert as usize * 17 + i * 7 + 3) as u8
}

struct MemSource {
    experts: usize,
    /// Keys whose read must fail.
    poison: Vec<(u32, u32)>,
    reads: Mutex<Vec<(u32, u32)>>,
}

impl MemSource {
    fn new(experts: usize) -> Self {
        MemSource {
            experts,
            poison: Vec::new(),
            reads: Mutex::new(Vec::new()),
        }
    }
}

impl ExpertSource for MemSource {
    fn slot_layout(&self) -> SlotLayout {
        LAYOUT
    }
    fn num_experts(&self) -> usize {
        self.experts
    }
    fn read_expert(&self, layer: u32, expert: u32, dst: &mut [u8]) -> Result<()> {
        assert_eq!(dst.len(), LAYOUT.bytes);
        self.reads.lock().unwrap().push((layer, expert));
        if self.poison.contains(&(layer, expert)) {
            bail!("poisoned expert ({layer}, {expert})");
        }
        for (i, b) in dst.iter_mut().enumerate() {
            *b = expected(layer, expert, i);
        }
        Ok(())
    }
}

struct Arena(Vec<u8>);

impl Arena {
    fn new(slots: usize) -> Self {
        Arena(vec![0u8; slots * LAYOUT.bytes])
    }
    fn lru(&mut self) -> ExpertLru {
        let p = self.0.as_mut_ptr();
        ExpertLru::new(p, DevicePtr(p as u64), self.0.len(), LAYOUT).unwrap()
    }
}

fn slot_bytes(s: &super::expert_lru::ExpertSlot) -> &[u8] {
    // SAFETY: test arena outlives the slot; a slot is LAYOUT.bytes long.
    unsafe { std::slice::from_raw_parts(s.host, LAYOUT.bytes) }
}

fn check_slot(s: &super::expert_lru::ExpertSlot, layer: u32, expert: u32) {
    let b = slot_bytes(s);
    for (i, &v) in b.iter().enumerate() {
        assert_eq!(v, expected(layer, expert, i), "({layer},{expert}) byte {i}");
    }
    // the three device pointers are the host address plus the layout offsets
    assert_eq!(s.gate.0, s.host as u64 + LAYOUT.gate_off as u64);
    assert_eq!(s.up.0, s.host as u64 + LAYOUT.up_off as u64);
    assert_eq!(s.down.0, s.host as u64 + LAYOUT.down_off as u64);
}

#[test]
fn slot_layout_is_contiguous() {
    let l = SlotLayout::new(3, 5, 7);
    assert_eq!((l.gate_off, l.up_off, l.down_off, l.bytes), (0, 3, 8, 15));
}

#[test]
fn miss_then_hit_and_bytes_land_in_the_slot() {
    let src = MemSource::new(8);
    let mut arena = Arena::new(4);
    let mut lru = arena.lru();
    assert_eq!(lru.n_slots(), 4);
    let (s, hit) = lru.fetch(&src, 3, 5).unwrap();
    assert!(!hit);
    check_slot(&s, 3, 5);
    let (s2, hit) = lru.fetch(&src, 3, 5).unwrap();
    assert!(hit);
    assert_eq!(s, s2);
    let st = lru.stats();
    assert_eq!(
        (st.hits, st.misses, st.evictions, st.bytes_read),
        (1, 1, 0, LAYOUT.bytes as u64)
    );
    assert_eq!(src.reads.lock().unwrap().len(), 1);
}

#[test]
fn evicts_least_recently_used_and_respects_touch_order() {
    let src = MemSource::new(8);
    let mut arena = Arena::new(3);
    let mut lru = arena.lru();
    for e in [0, 1, 2] {
        lru.fetch(&src, 0, e).unwrap();
    }
    assert_eq!(lru.resident(), 3);
    // touch 0 so 1 becomes the LRU
    lru.begin_token();
    lru.fetch(&src, 0, 0).unwrap();
    lru.begin_token();
    let (s, hit) = lru.fetch(&src, 0, 3).unwrap();
    assert!(!hit);
    check_slot(&s, 0, 3);
    assert!(!lru.contains(0, 1), "1 was the LRU and should be gone");
    assert!(lru.contains(0, 0) && lru.contains(0, 2) && lru.contains(0, 3));
    // next victim is 2
    lru.begin_token();
    lru.fetch(&src, 0, 4).unwrap();
    assert!(!lru.contains(0, 2));
    assert!(lru.contains(0, 0) && lru.contains(0, 3) && lru.contains(0, 4));
    // every resident slot still holds its own bytes
    for e in [0, 3, 4] {
        let (s, hit) = lru.fetch(&src, 0, e).unwrap();
        assert!(hit);
        check_slot(&s, 0, e);
    }
    assert_eq!(lru.stats().evictions, 2);
}

#[test]
fn a_token_cannot_evict_its_own_working_set() {
    let src = MemSource::new(8);
    let mut arena = Arena::new(2);
    let mut lru = arena.lru();
    lru.begin_token();
    let slots = lru.fetch_many(&src, &[(0, 0), (0, 1)], 1).unwrap();
    check_slot(&slots[0], 0, 0);
    check_slot(&slots[1], 0, 1);
    // a third expert in the same epoch has no unpinned victim
    let err = lru.fetch(&src, 0, 2).unwrap_err();
    assert!(err.to_string().contains("smaller than one token"), "{err}");
    assert_eq!(lru.resident(), 2, "the failed fetch mapped nothing");
    // next token: the oldest becomes evictable again
    lru.begin_token();
    let (s, hit) = lru.fetch(&src, 0, 2).unwrap();
    assert!(!hit);
    check_slot(&s, 0, 2);
    assert!(!lru.contains(0, 0));
}

#[test]
fn fetch_many_reads_misses_in_parallel_and_dedups() {
    let src = MemSource::new(64);
    let mut arena = Arena::new(32);
    let mut lru = arena.lru();
    let keys: Vec<(u32, u32)> = (0..16u32).map(|i| (i % 4, i * 3 % 64)).collect();
    let mut with_dup = keys.clone();
    with_dup.push(keys[5]);
    lru.begin_token();
    let slots = lru.fetch_many(&src, &with_dup, 4).unwrap();
    assert_eq!(slots.len(), 17);
    for (s, &(l, e)) in slots.iter().zip(&with_dup) {
        check_slot(s, l, e);
    }
    assert_eq!(
        slots[5], slots[16],
        "the duplicate key resolves to the same slot"
    );
    let st = lru.stats();
    assert_eq!((st.misses, st.hits), (16, 1));
    assert_eq!(st.bytes_read, 16 * LAYOUT.bytes as u64);
    assert_eq!(
        src.reads.lock().unwrap().len(),
        16,
        "one read per distinct key"
    );
    // second token: all hits, no reads
    lru.begin_token();
    let again = lru.fetch_many(&src, &keys, 4).unwrap();
    assert_eq!(again, slots[..16].to_vec());
    assert_eq!(lru.stats().hits, 17);
    assert_eq!(src.reads.lock().unwrap().len(), 16);
}

#[test]
fn a_failed_read_leaves_nothing_mapped() {
    let mut src = MemSource::new(8);
    src.poison.push((1, 1));
    let mut arena = Arena::new(4);
    let mut lru = arena.lru();
    assert!(lru.fetch(&src, 1, 1).is_err());
    assert!(!lru.contains(1, 1));
    assert_eq!(lru.resident(), 0);
    lru.begin_token();
    let err = lru
        .fetch_many(&src, &[(1, 0), (1, 1), (1, 2)], 3)
        .unwrap_err();
    assert!(
        err.to_string().contains("1 of 3 expert reads failed"),
        "{err}"
    );
    assert!(lru.contains(1, 0) && lru.contains(1, 2) && !lru.contains(1, 1));
    // the good ones are intact and the freed slot is reusable
    let (s, hit) = lru.fetch(&src, 1, 0).unwrap();
    assert!(hit);
    check_slot(&s, 1, 0);
    let (s, hit) = lru.fetch(&src, 1, 3).unwrap();
    assert!(!hit);
    check_slot(&s, 1, 3);
    assert_eq!(lru.stats().evictions, 0);
}

#[test]
fn pool_reads_misses_urgently_predictions_in_the_background_and_waits_for_the_keys_only() {
    let src = std::sync::Arc::new(MemSource::new(64));
    let mut arena = Arena::new(32);
    let mut lru = arena.lru();
    lru.set_pool(src.clone(), 4);
    let keys: Vec<(u32, u32)> = (0..6u32).map(|e| (1, e)).collect();
    let pred: Vec<(u32, u32)> = (10..14u32).map(|e| (2, e)).collect();
    lru.begin_token();
    let slots = lru.fetch_many_prefetching(&keys, &pred).unwrap();
    for (s, &(l, e)) in slots.iter().zip(&keys) {
        check_slot(s, l, e);
    }
    let st = lru.stats();
    assert_eq!((st.misses, st.hits, st.prefetched), (6, 0, 4));
    // the next layer of the same token asks for two of the predictions: hits,
    // waited on if still landing
    let got = lru
        .fetch_many_prefetching(&[(2, 10), (2, 11)], &[])
        .unwrap();
    check_slot(&got[0], 2, 10);
    check_slot(&got[1], 2, 11);
    let st = lru.stats();
    assert_eq!((st.misses, st.hits), (6, 2));
    // the two predictions nobody asked for are demoted at the next token and
    // go before anything else, even the empty slots (a real token is ~50 ms,
    // the reads have landed; here we wait for them)
    lru.wait_in_flight();
    lru.begin_token();
    assert_eq!(lru.stats().prefetch_unused, 2);
    assert!(lru.contains(2, 12) && lru.contains(2, 13));
    let many: Vec<(u32, u32)> = (0..24u32).map(|e| (3, e)).collect();
    let got = lru.fetch_many_prefetching(&many, &[]).unwrap();
    for (s, &(l, e)) in got.iter().zip(&many) {
        check_slot(s, l, e);
    }
    assert!(!lru.contains(2, 12) && !lru.contains(2, 13));
    assert!(lru.contains(1, 0) && lru.contains(2, 10) && lru.contains(2, 11));
    assert_eq!(lru.resident(), 32);
}

#[test]
fn pool_failures_unmap_a_failed_key_now_and_a_failed_prediction_at_the_next_token() {
    let mut m = MemSource::new(64);
    m.poison = vec![(1, 3), (2, 12)];
    let src = std::sync::Arc::new(m);
    let mut arena = Arena::new(16);
    let mut lru = arena.lru();
    lru.set_pool(src.clone(), 2);
    lru.begin_token();
    let r = lru.fetch_many_prefetching(&[(1, 2), (1, 3)], &[(2, 12), (2, 13)]);
    assert!(r.is_err(), "a poisoned key fails the fetch");
    assert!(!lru.contains(1, 3), "the failed key is unmapped");
    assert!(lru.contains(1, 2), "the good key stays");
    // the poisoned prediction is unmapped once its ticket is reaped
    lru.wait_in_flight();
    lru.begin_token();
    assert!(!lru.contains(2, 12));
    assert!(lru.contains(2, 13));
    let s = lru.fetch_many_prefetching(&[(2, 13)], &[]).unwrap();
    check_slot(&s[0], 2, 13);
}
