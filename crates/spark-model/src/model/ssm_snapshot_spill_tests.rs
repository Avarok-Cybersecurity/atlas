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

/// THE regression test for the measured defect: the gather must be
/// `2 × layers` ASYNC enqueues followed by exactly ONE trailing stream sync,
/// never one blocking `copy_d2h` (= one full stream drain) per chunk. The old
/// shape moved 66,846,720 B in ~400 ms (~165 MB/s) purely because of the 60
/// drains; the bytes were correct throughout, so only the shape can catch it.
#[test]
fn spill_issues_exactly_one_trailing_sync() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, /*slots*/ 2, /*layers*/ 5);
    let store = MemBlobStore::new(0);
    write_pattern(&p, &gpu, 0);

    let syncs_before = gpu.sync_count();
    let d2h_before = gpu.d2h_blocking_count();
    assert!(p.spill_slot(0, 0x11, &store, &gpu, 0).unwrap());

    assert_eq!(
        gpu.d2h_async_count(),
        2 * p.num_ssm_layers,
        "every (h,conv) chunk must be an async enqueue"
    );
    assert_eq!(
        gpu.d2h_blocking_count() - d2h_before,
        0,
        "a blocking copy_d2h in the gather is the defect itself"
    );
    assert_eq!(
        gpu.sync_count() - syncs_before,
        2,
        "exactly two drains: the leading save-drain and ONE trailing commit"
    );
}

/// The staging buffer is allocated once and reused — not re-allocated (and
/// re-zeroed, and re-page-faulted) per eviction, which was 66 MB of pure waste
/// on every spill AND every fault-in.
#[test]
fn staging_buffer_allocated_once() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 3, 4);
    let store = MemBlobStore::new(0);
    write_pattern(&p, &gpu, 0);
    write_pattern(&p, &gpu, 1);

    assert!(p.spill_slot(0, 0xA, &store, &gpu, 0).unwrap());
    assert!(p.spill_slot(1, 0xB, &store, &gpu, 0).unwrap());
    assert!(p.fault_in_slot(2, 0xA, &store, &gpu, 0).unwrap());
    assert_eq!(
        gpu.host_pinned_alloc_count(),
        1,
        "one buffer, shared by spill and fault-in, for the model's lifetime"
    );
    p.free_staging(&gpu);
}

/// The buffer is deliberately NOT zeroed between uses (the zero-fill was part
/// of the measured cost), so the gather must overwrite every byte. Spill
/// pattern A, then a DIFFERENT pattern B from another slot: B's stored blob
/// must contain none of A.
#[test]
fn staging_reuse_leaves_no_stale_bytes() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 2, 3);
    let store = MemBlobStore::new(0);

    // Slot 0 = pattern A (0x10.., 0x80..), slot 1 = a distinct constant.
    write_pattern(&p, &gpu, 0);
    for i in 0..p.num_ssm_layers {
        gpu.copy_h2d(&vec![0x5A; p.h_bytes], p.h_snapshots[i].offset(p.h_bytes))
            .unwrap();
        gpu.copy_h2d(
            &vec![0x5A; p.conv_bytes],
            p.conv_snapshots[i].offset(p.conv_bytes),
        )
        .unwrap();
    }
    assert!(p.spill_slot(0, 0xA, &store, &gpu, 0).unwrap());
    assert!(p.spill_slot(1, 0xB, &store, &gpu, 0).unwrap());

    let mut got = vec![0u8; p.spill_blob_bytes()];
    assert!(store.get(0xB, &mut got).unwrap());
    assert!(
        got.iter().all(|&b| b == 0x5A),
        "B's blob must be wholly B's bytes — any A byte means an unwritten hole"
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
    // drain the free list so the pool is full. The prefixes are DEEP (2048
    // tokens) so the spill-side cost gate (`ATLAS_SSM_SPILL_MIN_TOKENS`,
    // default 1024) takes its Spill arm — the shallow/Drop arm is covered by
    // `shallow_victim_is_dropped_but_still_yields_a_slot` below.
    let toks_a: Vec<u32> = (0..2048).collect();
    let toks_b: Vec<u32> = (100_000..102_048).collect();
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

/// The spill-side cost gate in the acquire path: a SHALLOW victim is dropped
/// rather than spilled — nothing reaches the tier — but the caller still gets
/// its slot, so "a reclaim always yields a slot" survives the gate.
#[test]
fn shallow_victim_is_dropped_but_still_yields_a_slot() {
    use spark_runtime::prefix_cache::PrefixCache;
    use spark_runtime::radix_tree::RadixTree;

    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, /*slots*/ 1, /*layers*/ 2);
    let store = MemBlobStore::new(0);
    let tree = RadixTree::new();

    // 16 tokens — far below the default 1024-token spill gate.
    let toks: Vec<u32> = (0..16).collect();
    tree.insert_with_snapshot(
        &toks,
        &[10],
        &[],
        16,
        /*slot*/ 0,
        /*sess*/ 7,
        0,
        0,
    );
    assert!(p.try_pop_free_slot().is_some());
    assert_eq!(p.try_pop_free_slot(), None, "pool is now full");

    let slot = p
        .acquire_or_spill_slot(&tree, &store, &gpu)
        .expect("the gate must still free a slot");
    assert_eq!(slot, 0);
    assert_eq!(store.len(), 0, "a ~45ms spill cannot repay 16 tokens");
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

/// **FOLLOW-UP 1 — the stale tier-key thrash. Expected to FAIL until a tier
/// miss retires the key.**
///
/// A cap eviction (`ATLAS_SSM_TIER_DISK_GB`) drops a blob, but nothing tells
/// the prefix cache: the index entry stays `tiered` and keeps handing out the
/// same dead `ssm_snapshot_tier_key` on every warm lookup. So every warm turn
/// on that prefix repeats the whole failed cycle — spill a LIVE snapshot D2H
/// to free a slot (which, under the cap, evicts yet another tier record),
/// fault, miss, free the slot — and then recomputes anyway. Self-amplifying:
/// the cap's own pressure manufactures more cap pressure.
///
/// The property: a dropped blob must cost ONE failed fault-in and then degrade
/// to plain recompute, forever. This drives the production cycle
/// (`SsmSnapshotPool::fault_in_for_key`, which `try_fault_in_ssm_snapshot`
/// delegates to) against a `MockGpuBackend`, a real `RadixTree` and a
/// one-blob-capped `MemBlobStore` — no GPU, no container.
#[test]
fn tier_miss_retires_the_key_instead_of_thrashing() {
    use std::sync::atomic::Ordering;

    use spark_runtime::prefix_cache::{PrefixCache, TierEvict};
    use spark_runtime::radix_tree::RadixTree;

    const BLK: usize = 16;
    /// Every prefix must clear `ATLAS_SSM_SPILL_MIN_TOKENS` (default 1024) so
    /// victim selection takes the Spill arm — the Drop arm would remove the
    /// entry and there would be no stale key to thrash on.
    const DEEP: u32 = 2048;

    /// A deep prefix plus a disjoint block table, so each `base` is its own
    /// radix branch.
    fn seq(base: u32) -> (Vec<u32>, Vec<u32>) {
        let toks: Vec<u32> = (base..base + DEEP).collect();
        let first_blk = base / BLK as u32;
        let blocks: Vec<u32> = (first_blk..first_blk + DEEP / BLK as u32).collect();
        (toks, blocks)
    }

    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, /*slots*/ 2, /*layers*/ 2);
    let blob = p.spill_blob_bytes();
    // Cap = exactly ONE blob: the smallest honest model of a full
    // ATLAS_SSM_TIER_DISK_GB, where every new record drops the oldest.
    let store = MemBlobStore::new(blob);
    let tree = RadixTree::new();

    // 1. The warm session's anchor, resident in slot 0.
    let (warm, warm_blocks) = seq(0);
    tree.insert_with_snapshot(
        &warm,
        &warm_blocks,
        &[],
        BLK,
        /*slot*/ 0,
        /*sess*/ 7,
        0,
        0,
    );
    assert_eq!(p.try_pop_free_slot(), Some(0));

    // 2. Pool pressure evicts it — SPILLED, so the index entry stays findable
    //    and its bytes live in the tier.
    let TierEvict::Spill { slot, key, .. } = tree.evict_snapshot_to_tier(1024).unwrap() else {
        panic!("a {DEEP}-token victim must spill, not drop");
    };
    assert!(p.spill_slot(slot, key, &store, &gpu, 0).unwrap());
    p.free(slot);

    // 3. One more record arrives (any other session's spill) and the cap
    //    FIFO-drops the warm anchor's blob. The prefix cache is never told:
    //    the entry is still `tiered` and still carries `key`. THE STALE KEY.
    store.put(0xDEAD_BEEF, &vec![0u8; blob]).unwrap();
    let mut probe = vec![0u8; blob];
    assert!(
        !store.get(key, &mut probe).unwrap(),
        "the cap must have dropped the warm anchor's blob for this test to bite"
    );

    // 4. Refill the pool with other sessions' live snapshots. The steady state
    //    under cap pressure is a FULL pool — that is what makes each doomed
    //    retry cost a real (66 MB in production) live-snapshot spill.
    for (i, base) in [200_000u32, 300_000].into_iter().enumerate() {
        let (t, b) = seq(base);
        let s = p.try_pop_free_slot().expect("pool has 2 slots");
        tree.insert_with_snapshot(&t, &b, &[], BLK, s, /*sess*/ 100 + i as u64, 0, 0);
    }
    assert_eq!(p.try_pop_free_slot(), None, "the pool must be full");

    // Baseline the counters AFTER the setup: step 3's `store.get` probe is
    // itself a miss, and step 2 + the unrelated record are puts.
    let puts_before = store.stats.puts.load(Ordering::Relaxed);
    let misses_before = store.stats.get_misses.load(Ordering::Relaxed);

    // 5. Four warm turns on the SAME prefix. Turn 0 legitimately tries the
    //    tier — a miss is only discoverable by trying. Turns 1-3 must not:
    //    the key was already proven dead on turn 0.
    let mut tier_attempts = 0usize;
    for turn in 0..4u32 {
        let m = tree.lookup(&warm, BLK, /*sess*/ 7, 0);
        tree.release(&warm, BLK, 0);
        let Some(k) = m.ssm_snapshot_tier_key else {
            continue;
        };
        tier_attempts += 1;
        assert!(
            p.fault_in_for_key(
                &tree,
                &store,
                &gpu,
                k,
                /*sess*/ 7,
                m.ssm_snapshot_tier_tokens,
                0
            )
            .is_none(),
            "the blob is gone — every fault-in attempt must miss"
        );
        // The turn now recomputes the prefix and saves its own snapshot into
        // the slot the failed cycle freed, so the pool is full again next
        // turn. Without this the pool stays one slot short and later retries
        // would not spill a victim, hiding the amplification.
        let s = p
            .try_pop_free_slot()
            .expect("the failed fault-in returned its slot");
        let (t, b) = seq(500_000 + turn * DEEP);
        tree.insert_with_snapshot(&t, &b, &[], BLK, s, /*sess*/ 7, 0, 0);
    }

    assert_eq!(
        tier_attempts, 1,
        "a dropped blob must cost ONE failed fault-in and then degrade to plain \
         recompute — re-offering the same dead key on every warm turn IS the thrash"
    );
    assert_eq!(
        store.stats.get_misses.load(Ordering::Relaxed) - misses_before,
        1,
        "one miss proves the blob is gone; every further miss is re-discovering it"
    );
    assert_eq!(
        store.stats.puts.load(Ordering::Relaxed) - puts_before,
        1,
        "each doomed retry spills a LIVE snapshot D2H to free a slot it then throws \
         away — and under the cap that spill evicts yet another tier record"
    );
    assert_eq!(
        tree.lookup(&warm, BLK, /*sess*/ 7, 0).ssm_snapshot_tier_key,
        None,
        "after the miss the anchor must stop advertising a tier key"
    );
}

/// A store whose `get` always ERRORS (transport/IO failure), while its blobs
/// stay perfectly intact. Models the trap the reap must not fall into:
/// `Residency` restores `disk_lru` and returns `Err` on a failed record read,
/// leaving the record on disk AND still mapped.
struct ErrOnGetStore {
    inner: MemBlobStore,
    gets: std::sync::atomic::AtomicUsize,
    removes: std::sync::atomic::AtomicUsize,
}

impl ErrOnGetStore {
    fn new() -> Self {
        Self {
            inner: MemBlobStore::new(0),
            gets: std::sync::atomic::AtomicUsize::new(0),
            removes: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl SnapshotBlobStore for ErrOnGetStore {
    fn put(&self, key: u64, bytes: &[u8]) -> anyhow::Result<bool> {
        self.inner.put(key, bytes)
    }
    fn get(&self, _key: u64, _out: &mut [u8]) -> anyhow::Result<bool> {
        self.gets.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        anyhow::bail!("simulated tier read failure — the bytes are still there")
    }
    fn remove(&self, key: u64) {
        self.removes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.remove(key);
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn bytes_resident(&self) -> usize {
        self.inner.bytes_resident()
    }
}

/// **The error-vs-miss asymmetry** — the guard that keeps the reap from
/// destroying a LIVE snapshot. A failed read is not evidence of absence: the
/// blob is still on disk and still mapped, and the next turn would have read it
/// successfully. So an `Err` must return the slot and RETAIN the key (cost: one
/// wasted cycle), where a miss retires it (cost of not retiring: forever).
#[test]
fn tier_error_retains_the_key() {
    use std::sync::atomic::Ordering;

    use spark_runtime::prefix_cache::{PrefixCache, TierEvict};
    use spark_runtime::radix_tree::RadixTree;

    const BLK: usize = 16;
    const DEEP: u32 = 2048;

    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, /*slots*/ 2, /*layers*/ 2);
    let store = ErrOnGetStore::new();
    let tree = RadixTree::new();

    // A deep anchor in slot 0, spilled to the tier: its bytes are genuinely
    // present, only the reads fail.
    let warm: Vec<u32> = (0..DEEP).collect();
    let warm_blocks: Vec<u32> = (0..DEEP / BLK as u32).collect();
    tree.insert_with_snapshot(
        &warm,
        &warm_blocks,
        &[],
        BLK,
        /*slot*/ 0,
        /*sess*/ 7,
        0,
        0,
    );
    assert_eq!(p.try_pop_free_slot(), Some(0));
    let TierEvict::Spill { slot, key, .. } = tree.evict_snapshot_to_tier(1024).unwrap() else {
        panic!("a {DEEP}-token victim must spill, not drop");
    };
    assert!(p.spill_slot(slot, key, &store, &gpu, 0).unwrap());
    p.free(slot);
    assert_eq!(store.len(), 1, "the blob is present throughout this test");

    let m = tree.lookup(&warm, BLK, /*sess*/ 7, 0);
    tree.release(&warm, BLK, 0);
    let k = m.ssm_snapshot_tier_key.expect("the anchor is tiered");
    let free_before = p.free_slots.lock().len();

    assert!(
        p.fault_in_for_key(
            &tree,
            &store,
            &gpu,
            k,
            /*sess*/ 7,
            m.ssm_snapshot_tier_tokens,
            0
        )
        .is_none(),
        "a failed read restores nothing this turn"
    );

    assert_eq!(store.gets.load(Ordering::Relaxed), 1, "exactly one attempt");
    assert_eq!(
        p.free_slots.lock().len(),
        free_before,
        "the slot the failed fault-in took must go back on the free list"
    );
    assert_eq!(
        store.removes.load(Ordering::Relaxed),
        0,
        "reaping on an error would delete a live 66MB snapshot to save one retry"
    );
    assert_eq!(
        tree.lookup(&warm, BLK, /*sess*/ 7, 0).ssm_snapshot_tier_key,
        Some(k),
        "an error is not evidence of absence — the key must survive to be retried"
    );
}

/// The spill-side twin (follow-up 1b): when the tier REFUSES a blob,
/// `evict_to_tier` has already marked the entry `tiered` holding nothing — a
/// stale key manufactured eagerly. Retiring it there means the next warm turn
/// never even offers a tier key, so it costs ZERO fault-in attempts (and zero
/// live-snapshot spills) rather than one.
#[test]
fn spill_refusal_retires_the_entry_immediately() {
    use spark_runtime::prefix_cache::PrefixCache;
    use spark_runtime::radix_tree::RadixTree;

    const BLK: usize = 16;
    const DEEP: u32 = 2048;

    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, /*slots*/ 1, /*layers*/ 2);
    // Cap smaller than one blob: `MemBlobStore::put` refuses outright (a blob
    // larger than the whole cap can never fit).
    let store = MemBlobStore::new(p.spill_blob_bytes() - 1);
    let tree = RadixTree::new();

    let warm: Vec<u32> = (0..DEEP).collect();
    let warm_blocks: Vec<u32> = (0..DEEP / BLK as u32).collect();
    tree.insert_with_snapshot(
        &warm,
        &warm_blocks,
        &[],
        BLK,
        /*slot*/ 0,
        /*sess*/ 7,
        0,
        0,
    );
    assert_eq!(p.try_pop_free_slot(), Some(0));
    assert_eq!(p.try_pop_free_slot(), None, "the pool is full");

    // A fault-in for some other prefix drives the acquire path, which spills
    // our victim — and the tier refuses it.
    let slot = p
        .acquire_or_spill_slot(&tree, &store, &gpu)
        .expect("the victim's slot is freed regardless");
    assert_eq!(slot, 0);
    assert_eq!(store.len(), 0, "the tier took no bytes");

    let m = tree.lookup(&warm, BLK, /*sess*/ 7, 0);
    tree.release(&warm, BLK, 0);
    assert_eq!(
        m.ssm_snapshot_tier_key, None,
        "a refused spill must not leave a findable-but-empty entry"
    );
    assert_eq!(m.ssm_snapshot, None, "and no resident slot either");
}
