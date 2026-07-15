// SPDX-License-Identifier: AGPL-3.0-only

// Unit tests for the Phase-1 SSM snapshot spill/fault-in primitives. Exercise
// spill_slot / fault_in_slot / spill_blob_bytes / acquire_or_spill_slot against
// a `MockGpuBackend` + host-RAM `MemBlobStore`, so the otherwise-dead fault-in
// primitives get coverage before the Phase-1b serving wiring lands.

use super::*;
use crate::model::ssm_tier::{MemBlobStore, SnapshotBlobStore};
use spark_runtime::gpu::mock::MockGpuBackend;

/// Build a small Marconi-only pool (no decode-rollback region).
fn pool(gpu: &dyn GpuBackend, slots: usize, layers: usize) -> SsmSnapshotPool {
    SsmSnapshotPool::new(
        slots, /*h_bytes*/ 32, /*conv_bytes*/ 16, layers, /*decode_ring*/ 0,
        /*decode_max_seqs*/ 0, /*hidden_bytes*/ 8, gpu,
    )
    .unwrap()
}

/// Fill slot `s`'s per-layer (h,conv) device chunks with a pattern unique
/// per (layer, field) so a mis-scatter would be caught.
fn write_pattern(p: &SsmSnapshotPool, gpu: &dyn GpuBackend, s: usize) {
    for i in 0..p.num_ssm_layers {
        let h = vec![(0x10 + i) as u8; p.h_bytes];
        let c = vec![(0x80 + i) as u8; p.conv_bytes];
        gpu.copy_h2d(&h, p.h_snapshots[i].offset(s * p.h_bytes))
            .unwrap();
        gpu.copy_h2d(&c, p.conv_snapshots[i].offset(s * p.conv_bytes))
            .unwrap();
    }
}

fn read_slot(p: &SsmSnapshotPool, gpu: &dyn GpuBackend, s: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut hs = Vec::new();
    let mut cs = Vec::new();
    for i in 0..p.num_ssm_layers {
        let mut h = vec![0u8; p.h_bytes];
        let mut c = vec![0u8; p.conv_bytes];
        gpu.copy_d2h(p.h_snapshots[i].offset(s * p.h_bytes), &mut h)
            .unwrap();
        gpu.copy_d2h(p.conv_snapshots[i].offset(s * p.conv_bytes), &mut c)
            .unwrap();
        hs.push(h);
        cs.push(c);
    }
    (hs, cs)
}

/// The headline invariant: spill a slot's scattered state to the tier, then
/// fault it back into a DIFFERENT slot — the recurrent state is bit-for-bit
/// preserved. This is "spill-not-drop" proven end-to-end at the pool layer.
#[test]
fn spill_then_fault_in_preserves_bytes() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, /*slots*/ 4, /*layers*/ 3);
    let store = MemBlobStore::new(0);
    let key = 0xABCD_1234;

    write_pattern(&p, &gpu, /*src*/ 1);
    let want = read_slot(&p, &gpu, 1);

    assert!(p.spill_slot(1, key, &store, &gpu, 0).unwrap());
    assert_eq!(store.len(), 1);
    assert_eq!(store.bytes_resident(), p.spill_blob_bytes());

    // Fault into slot 2 (which is still zeroed) and compare to slot 1.
    assert!(p.fault_in_slot(2, key, &store, &gpu, 0).unwrap());
    let got = read_slot(&p, &gpu, 2);
    assert_eq!(
        got, want,
        "faulted-in slot must equal the spilled slot bit-for-bit"
    );
}

/// Faulting an absent key is a clean miss (caller recomputes), not an error.
#[test]
fn fault_in_absent_key_is_miss() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 4, 2);
    let store = MemBlobStore::new(0);
    assert!(!p.fault_in_slot(0, /*absent*/ 999, &store, &gpu, 0).unwrap());
}

/// Blob size accounts for every layer's h+conv.
#[test]
fn spill_blob_bytes_matches_layout() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 2, 5);
    assert_eq!(p.spill_blob_bytes(), 5 * (32 + 16));
}

/// Full-pool fault-in: when no slot is free, `acquire_or_spill_slot` spills a
/// resident victim (to the tier, keeping it faultable) and hands back its
/// freed slot — so a warm tiered hit isn't lost to a busy pool.
#[test]
fn acquire_or_spill_frees_a_slot_under_full_pool() {
    use spark_runtime::prefix_cache::PrefixCache;
    use spark_runtime::radix_tree::RadixTree;

    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, /*slots*/ 2, /*layers*/ 2);
    let store = MemBlobStore::new(0);
    let tree = RadixTree::new();

    // Register two resident snapshots (slots 0 and 1) for two prefixes, then
    // drain the free list so the pool is full.
    let toks_a: Vec<u32> = (0..16).collect();
    let toks_b: Vec<u32> = (100..116).collect();
    tree.insert_with_snapshot(
        &toks_a,
        &[10],
        &[],
        16,
        /*slot*/ 0,
        /*sess*/ 7,
        0,
        0,
    );
    tree.insert_with_snapshot(
        &toks_b,
        &[20],
        &[],
        16,
        /*slot*/ 1,
        /*sess*/ 9,
        0,
        0,
    );
    assert!(p.try_pop_free_slot().is_some());
    assert!(p.try_pop_free_slot().is_some());
    assert_eq!(p.try_pop_free_slot(), None, "pool is now full");

    // Acquire must spill a victim and return its slot.
    let slot = p
        .acquire_or_spill_slot(&tree, &store, &gpu)
        .expect("a resident victim exists to spill");
    assert!(slot == 0 || slot == 1);
    assert_eq!(
        store.len(),
        1,
        "the evicted victim was spilled, not dropped"
    );
    // The other snapshot stays resident (drop path can still free it).
    assert!(tree.evict_snapshot_lru().is_some());
}

/// The integration invariant: the tier is keyed by prefix, INDEPENDENT of
/// HBM slot lifecycle. Spill snapshot A from slot 0, recycle slot 0 for a
/// different snapshot B, spill B under its own key, then fault BOTH back —
/// each must recover its own bytes. This is exactly what the Phase-1b
/// serving wiring creates: `evict_to_tier` frees a slot that `save` then
/// reuses, and a later warm turn faults the spilled key into a fresh slot.
#[test]
fn tier_survives_slot_recycling() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, /*slots*/ 3, /*layers*/ 2);
    let store = MemBlobStore::new(0);
    let (key_a, key_b) = (0xAAAA, 0xBBBB);

    // Snapshot A lives in slot 0; spill it.
    write_pattern(&p, &gpu, 0);
    let want_a = read_slot(&p, &gpu, 0);
    assert!(p.spill_slot(0, key_a, &store, &gpu, 0).unwrap());

    // Recycle slot 0 for a DIFFERENT snapshot B (distinct bytes), spill it.
    for i in 0..p.num_ssm_layers {
        let h = vec![0xEE; p.h_bytes];
        let c = vec![0xDD; p.conv_bytes];
        gpu.copy_h2d(&h, p.h_snapshots[i].offset(0)).unwrap();
        gpu.copy_h2d(&c, p.conv_snapshots[i].offset(0)).unwrap();
    }
    let want_b = read_slot(&p, &gpu, 0);
    assert_ne!(want_a, want_b, "B must differ from A for the test to bite");
    assert!(p.spill_slot(0, key_b, &store, &gpu, 0).unwrap());
    assert_eq!(store.len(), 2);

    // Fault each key into fresh slots — bytes recovered independently.
    assert!(p.fault_in_slot(1, key_a, &store, &gpu, 0).unwrap());
    assert!(p.fault_in_slot(2, key_b, &store, &gpu, 0).unwrap());
    assert_eq!(
        read_slot(&p, &gpu, 1),
        want_a,
        "key A recovered after slot recycle"
    );
    assert_eq!(read_slot(&p, &gpu, 2), want_b, "key B recovered");
}
