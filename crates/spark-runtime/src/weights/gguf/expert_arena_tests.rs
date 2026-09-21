// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The device arena on the mock backend: misses go through the staging ring
//! into their device slots byte for byte, in batches when the ring is
//! smaller than the token's misses, and the plain fetches refuse the cache.

use std::sync::Mutex;

use anyhow::{Result, bail};

use super::expert_arena::ExpertArena;
use super::expert_lru::{ExpertLru, ExpertSlot};
use super::expert_stream::{ExpertSource, SlotLayout};
use crate::gpu::mock::MockGpuBackend;
use crate::gpu::{DevicePtr, GpuBackend};

const LAYOUT: SlotLayout = SlotLayout {
    gate_off: 0,
    up_off: 16,
    down_off: 32,
    gate_bytes: 16,
    up_bytes: 16,
    down_bytes: 24,
    bytes: 56,
};

fn expected(layer: u32, expert: u32, i: usize) -> u8 {
    (layer as usize * 131 + expert as usize * 17 + i * 7 + 3) as u8
}

struct MemSource {
    poison: Vec<(u32, u32)>,
    reads: Mutex<usize>,
}

impl ExpertSource for MemSource {
    fn slot_layout(&self) -> SlotLayout {
        LAYOUT
    }
    fn num_experts(&self) -> usize {
        64
    }
    fn read_expert(&self, layer: u32, expert: u32, dst: &mut [u8]) -> Result<()> {
        self.read_expert_range(layer, expert, 0, dst)
    }
    fn read_expert_range(&self, layer: u32, expert: u32, off: usize, dst: &mut [u8]) -> Result<()> {
        *self.reads.lock().unwrap() += 1;
        if self.poison.contains(&(layer, expert)) {
            bail!("poisoned expert ({layer}, {expert})");
        }
        for (i, b) in dst.iter_mut().enumerate() {
            *b = expected(layer, expert, off + i);
        }
        Ok(())
    }
}

fn device_bytes(gpu: &dyn GpuBackend, s: &ExpertSlot) -> Vec<u8> {
    assert!(s.host.is_null(), "a device slot has no host alias");
    let mut v = vec![0u8; LAYOUT.bytes];
    gpu.copy_d2h(DevicePtr(s.gate.0 - LAYOUT.gate_off as u64), &mut v)
        .unwrap();
    v
}

fn check(gpu: &dyn GpuBackend, s: &ExpertSlot, layer: u32, expert: u32) {
    let got = device_bytes(gpu, s);
    let want: Vec<u8> = (0..LAYOUT.bytes)
        .map(|i| expected(layer, expert, i))
        .collect();
    assert_eq!(got, want, "device slot of ({layer}, {expert})");
}

#[test]
fn staged_misses_land_in_device_slots_through_a_small_ring() {
    let gpu = MockGpuBackend::default();
    let g: &dyn GpuBackend = &gpu;
    // 6 slots, a 2-slot ring (one slot a half): five misses = five batches
    let (arena, mut lru) = ExpertArena::alloc(g, 6 * LAYOUT.bytes, LAYOUT, true, 2).unwrap();
    assert!(matches!(arena, ExpertArena::Device(_)));
    assert!(lru.is_device());
    let src = MemSource {
        poison: Vec::new(),
        reads: Mutex::new(0),
    };
    lru.begin_token();
    let keys: Vec<(u32, u32)> = (0..5).map(|e| (7, e)).collect();
    let slots = lru.fetch_many_on(g, 0, &src, &keys, &[], 3).unwrap();
    assert_eq!(slots.len(), 5);
    for (s, &(l, e)) in slots.iter().zip(&keys) {
        check(g, s, l, e);
    }
    assert_eq!(lru.stats().misses, 5);
    assert_eq!(lru.stats().bytes_read, 5 * LAYOUT.bytes as u64);
    // a second token: two hits, two new experts; the eviction (6 slots, 7
    // distinct experts) refills a slot through the ring correctly
    lru.begin_token();
    let keys2 = [(7, 1), (7, 4), (7, 5), (7, 6)];
    let slots2 = lru.fetch_many_on(g, 0, &src, &keys2, &[], 3).unwrap();
    for (s, &(l, e)) in slots2.iter().zip(&keys2) {
        check(g, s, l, e);
    }
    assert_eq!(lru.stats().hits, 2);
    assert_eq!(lru.stats().misses, 7);
    assert_eq!(lru.stats().evictions, 1);
    // every resident expert still reads back as itself
    for e in 0..7u32 {
        if lru.contains(7, e) {
            lru.begin_token();
            let s = lru.fetch_many_on(g, 0, &src, &[(7, e)], &[], 1).unwrap();
            check(g, &s[0], 7, e);
        }
    }
    arena.free(g).unwrap();
}

#[test]
fn staged_read_failure_unmaps_only_the_failed_slot() {
    let gpu = MockGpuBackend::default();
    let g: &dyn GpuBackend = &gpu;
    let (arena, mut lru) = ExpertArena::alloc(g, 8 * LAYOUT.bytes, LAYOUT, true, 4).unwrap();
    let src = MemSource {
        poison: vec![(2, 3)],
        reads: Mutex::new(0),
    };
    lru.begin_token();
    let err = lru
        .fetch_many_on(g, 0, &src, &[(2, 1), (2, 3), (2, 5)], &[], 2)
        .unwrap_err();
    assert!(
        err.to_string().contains("1 of 3 expert reads failed"),
        "{err:#}"
    );
    assert!(!lru.contains(2, 3));
    assert!(lru.contains(2, 1) && lru.contains(2, 5));
    lru.begin_token();
    let s = lru
        .fetch_many_on(g, 0, &src, &[(2, 1), (2, 5)], &[], 2)
        .unwrap();
    check(g, &s[0], 2, 1);
    check(g, &s[1], 2, 5);
    assert_eq!(lru.stats().hits, 2);
    arena.free(g).unwrap();
}

#[test]
fn plain_fetches_refuse_a_device_cache_and_the_pinned_path_is_unchanged() {
    let gpu = MockGpuBackend::default();
    let g: &dyn GpuBackend = &gpu;
    let src = MemSource {
        poison: Vec::new(),
        reads: Mutex::new(0),
    };
    let (arena, mut lru) = ExpertArena::alloc(g, 4 * LAYOUT.bytes, LAYOUT, true, 2).unwrap();
    assert!(lru.fetch(&src, 1, 1).is_err());
    assert!(lru.fetch_many(&src, &[(1, 1)], 2).is_err());
    arena.free(g).unwrap();
    // a page-locked arena (a plain buffer here: the mock has no host alias):
    // `fetch_many_on` is the scoped-thread gather, the slots have a host side
    let mut buf = vec![0u8; 4 * LAYOUT.bytes];
    let p = buf.as_mut_ptr();
    let mut lru = ExpertLru::new(p, DevicePtr(p as u64), buf.len(), LAYOUT).unwrap();
    assert!(!lru.is_device());
    lru.begin_token();
    let s = lru
        .fetch_many_on(g, 0, &src, &[(1, 1), (1, 2)], &[], 2)
        .unwrap();
    assert!(!s[0].host.is_null());
    let got = unsafe { std::slice::from_raw_parts(s[1].host, LAYOUT.bytes) };
    let want: Vec<u8> = (0..LAYOUT.bytes).map(|i| expected(1, 2, i)).collect();
    assert_eq!(got.to_vec(), want, "pinned slot of (1, 2)");
    drop(lru);
}
