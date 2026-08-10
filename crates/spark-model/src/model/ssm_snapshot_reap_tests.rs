// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for tier-key REAPING: a fault-in miss (or a refused spill) must
//! retire the index entry so a capped disk cannot thrash, while an ERROR must
//! retain it (an error is not evidence of absence).
//!
//! Split out of `ssm_snapshot_spill_tests.rs` to keep both files under the
//! repo's 500-LoC cap.

use super::*;
use crate::model::ssm_tier::{MemBlobStore, SnapshotBlobStore};
use spark_runtime::gpu::mock::MockGpuBackend;

/// Build a small Marconi-only pool (no decode-rollback region).
///
/// Duplicated from `ssm_snapshot_spill_tests.rs`: the two files are separate
/// `#[path]` test modules (split for the 500-LoC cap), so neither can see the
/// other's private helpers.
fn pool(gpu: &dyn GpuBackend, slots: usize, layers: usize) -> SsmSnapshotPool {
    SsmSnapshotPool::new(
        slots, /*h_bytes*/ 32, /*conv_bytes*/ 16, layers, /*decode_ring*/ 0,
        /*decode_max_seqs*/ 0, /*hidden_bytes*/ 8, gpu,
    )
    .unwrap()
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
