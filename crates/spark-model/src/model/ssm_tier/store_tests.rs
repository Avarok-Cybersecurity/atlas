// SPDX-License-Identifier: AGPL-3.0-only

//! MemBlobStore contract tests (moved from the pre-split ssm_tier.rs).

use super::*;

#[test]
fn put_get_round_trip() {
    let s = MemBlobStore::new(0);
    assert!(s.put(42, &[1, 2, 3, 4]).unwrap());
    let mut out = [0u8; 4];
    assert!(s.get(42, &mut out).unwrap());
    assert_eq!(out, [1, 2, 3, 4]);
    assert_eq!(s.len(), 1);
    assert_eq!(s.bytes_resident(), 4);
}

#[test]
fn get_absent_is_miss_not_error() {
    let s = MemBlobStore::new(0);
    let mut out = [0xa5u8; 4];
    assert!(!s.get(7, &mut out).unwrap());
    assert_eq!(out, [0xa5; 4], "a miss must not modify caller memory");
    assert_eq!(s.stats.get_misses.load(Ordering::Relaxed), 1);
}

#[test]
fn wrong_size_get_refused() {
    let s = MemBlobStore::new(0);
    s.put(1, &[9; 8]).unwrap();
    let mut short = [0xa5u8; 7];
    let mut long = [0x5au8; 9];
    assert!(!s.get(1, &mut short).unwrap(), "reject a short output");
    assert!(!s.get(1, &mut long).unwrap(), "reject a long output");
    assert_eq!(short, [0xa5; 7], "a refusal must not modify output");
    assert_eq!(long, [0x5a; 9], "a refusal must not modify output");
    assert_eq!(s.stats.get_misses.load(Ordering::Relaxed), 2);
}

#[test]
fn overwrite_reclaims_bytes() {
    let s = MemBlobStore::new(0);
    s.put(1, &[0x11; 10]).unwrap();
    s.put(1, &[0x22; 3]).unwrap();
    assert_eq!(s.len(), 1);
    assert_eq!(
        s.bytes_resident(),
        3,
        "old blob bytes reclaimed on overwrite"
    );
    let mut out = [0u8; 3];
    assert!(s.get(1, &mut out).unwrap());
    assert_eq!(out, [0x22; 3], "the replacement payload must be visible");
}

#[test]
fn growing_oldest_overwrite_evicts_another_blob_without_exceeding_cap() {
    let s = MemBlobStore::new(10);
    assert!(s.put(1, &[0x11; 4]).unwrap());
    assert!(s.put(2, &[0x22; 4]).unwrap());

    assert!(s.put(1, &[0x33; 8]).unwrap());

    assert_eq!(s.len(), 1);
    assert_eq!(s.bytes_resident(), 8);
    assert_eq!(s.stats.evictions.load(Ordering::Relaxed), 1);
    let mut replacement = [0u8; 8];
    assert!(s.get(1, &mut replacement).unwrap());
    assert_eq!(replacement, [0x33; 8]);
    let mut evicted = [0xa5u8; 4];
    assert!(!s.get(2, &mut evicted).unwrap());
    assert_eq!(evicted, [0xa5; 4]);
}

#[test]
fn cap_evicts_fifo() {
    let s = MemBlobStore::new(10);
    s.put(1, &[0x11; 4]).unwrap(); // 4
    s.put(2, &[0x22; 4]).unwrap(); // 8
    s.put(3, &[0x33; 4]).unwrap(); // 12 before FIFO eviction
    assert_eq!(s.len(), 2);
    assert_eq!(s.bytes_resident(), 8);
    assert_eq!(s.stats.evictions.load(Ordering::Relaxed), 1);
    let mut out = [0xa5u8; 4];
    assert!(!s.get(1, &mut out).unwrap(), "oldest evicted");
    assert_eq!(out, [0xa5; 4]);
    assert!(s.get(2, &mut out).unwrap(), "middle blob retained");
    assert_eq!(out, [0x22; 4]);
    assert!(s.get(3, &mut out).unwrap(), "newest blob retained");
    assert_eq!(out, [0x33; 4]);
}

#[test]
fn blob_larger_than_cap_refused() {
    let s = MemBlobStore::new(4);
    assert!(s.put(1, &[0x11; 4]).unwrap());
    assert!(
        !s.put(1, &[0x22; 8]).unwrap(),
        "over-cap replacement refused, not partial"
    );
    assert_eq!(s.len(), 1);
    assert_eq!(s.bytes_resident(), 4);
    assert_eq!(s.stats.put_rejects.load(Ordering::Relaxed), 1);
    assert_eq!(s.stats.puts.load(Ordering::Relaxed), 1);
    let mut out = [0u8; 4];
    assert!(s.get(1, &mut out).unwrap());
    assert_eq!(out, [0x11; 4], "refusal must preserve the old value");
}

// ── Port sprint (lane L25): LRU order and the pool-backed arm ──

#[test]
fn the_default_order_is_fifo_with_the_row_off() {
    // Every target declares `ssm_store_lru = false`; the environment override is unset in the
    // test process, so `new` builds exactly the pre-sprint store.
    if std::env::var_os("AVAROK_SSM_STORE_LRU").is_none() {
        assert_eq!(MemBlobStore::new(0).policy(), BlobEviction::Fifo);
    }
}

#[test]
fn fifo_ignores_reads() {
    let s = MemBlobStore::with_policy(8, BlobEviction::Fifo);
    s.put(1, &[1; 4]).unwrap();
    s.put(2, &[2; 4]).unwrap();
    let mut out = [0u8; 4];
    assert!(s.get(1, &mut out).unwrap());
    s.put(3, &[3; 4]).unwrap();
    assert!(
        !s.get(1, &mut out).unwrap(),
        "FIFO evicts the oldest insert even after a read"
    );
    assert!(s.get(2, &mut out).unwrap());
}

#[test]
fn lru_touch_on_get_keeps_the_hot_blob() {
    let s = MemBlobStore::with_policy(8, BlobEviction::Lru);
    s.put(1, &[1; 4]).unwrap();
    s.put(2, &[2; 4]).unwrap();
    let mut out = [0u8; 4];
    assert!(
        s.get(1, &mut out).unwrap(),
        "touch 1: now 2 is the least recent"
    );
    s.put(3, &[3; 4]).unwrap();
    assert!(s.get(1, &mut out).unwrap(), "the touched blob survives");
    assert_eq!(out, [1; 4]);
    assert!(
        !s.get(2, &mut out).unwrap(),
        "the least recent one was evicted"
    );
    assert!(s.get(3, &mut out).unwrap());
    assert_eq!(s.stats.evictions.load(Ordering::Relaxed), 1);
}

#[test]
fn lru_overwrite_makes_the_key_most_recent() {
    let s = MemBlobStore::with_policy(12, BlobEviction::Lru);
    s.put(1, &[1; 4]).unwrap();
    s.put(2, &[2; 4]).unwrap();
    s.put(3, &[3; 4]).unwrap();
    s.put(1, &[9; 4]).unwrap(); // 1 is now the most recent; 2 the least
    s.put(4, &[4; 4]).unwrap();
    let mut out = [0u8; 4];
    assert!(!s.get(2, &mut out).unwrap());
    assert!(s.get(1, &mut out).unwrap());
    assert_eq!(out, [9; 4]);
    assert_eq!(s.bytes_resident(), 12);
}

#[test]
fn lru_misses_do_not_reorder() {
    let s = MemBlobStore::with_policy(8, BlobEviction::Lru);
    s.put(1, &[1; 4]).unwrap();
    s.put(2, &[2; 4]).unwrap();
    let mut wrong = [0u8; 3];
    assert!(!s.get(1, &mut wrong).unwrap(), "a size mismatch is a miss");
    s.put(3, &[3; 4]).unwrap();
    let mut out = [0u8; 4];
    assert!(!s.get(1, &mut out).unwrap(), "a miss did not touch 1");
}

#[test]
fn an_orphan_nobody_reads_is_the_first_lru_victim() {
    // The blob a re-homed snapshot leaves behind is never read again. Under LRU it ages out
    // ahead of every blob that is still being faulted in.
    let s = MemBlobStore::with_policy(12, BlobEviction::Lru);
    s.put(100, &[0xee; 4]).unwrap(); // the orphan
    s.put(1, &[1; 4]).unwrap();
    s.put(2, &[2; 4]).unwrap();
    let mut out = [0u8; 4];
    for _ in 0..3 {
        assert!(s.get(1, &mut out).unwrap());
        assert!(s.get(2, &mut out).unwrap());
    }
    s.put(3, &[3; 4]).unwrap();
    assert!(!s.get(100, &mut out).unwrap());
    assert!(s.get(1, &mut out).unwrap() && s.get(2, &mut out).unwrap());
}

#[test]
fn removing_an_absent_key_changes_nothing() {
    for policy in [BlobEviction::Fifo, BlobEviction::Lru] {
        let s = MemBlobStore::with_policy(0, policy);
        s.put(1, &[1; 4]).unwrap();
        s.remove(2);
        s.remove(1);
        s.remove(1);
        assert_eq!(s.len(), 0);
        assert_eq!(s.bytes_resident(), 0);
    }
}

fn pool(chunks: usize, chunk: usize) -> std::sync::Arc<spark_runtime::pinned_pool::PinnedPool> {
    let gpu: std::sync::Arc<dyn spark_runtime::gpu::GpuBackend> =
        std::sync::Arc::new(spark_runtime::gpu::mock::MockGpuBackend::new());
    spark_runtime::pinned_pool::PinnedPool::new(gpu, chunks * chunk, chunks * chunk, chunk).unwrap()
}

#[test]
fn a_pool_backed_store_keeps_its_bytes_in_the_pool() {
    let p = pool(2, 16 * 4096);
    let s = MemBlobStore::pool_backed(0, BlobEviction::Lru, std::sync::Arc::clone(&p));
    let blob: Vec<u8> = (0..10_000u32).map(|i| (i * 7) as u8).collect();
    assert!(s.put(5, &blob).unwrap());
    assert_eq!(s.pooled_blobs(), 1);
    assert_eq!(p.stats().runs, 1);
    let mut out = vec![0u8; blob.len()];
    assert!(s.get(5, &mut out).unwrap());
    assert_eq!(out, blob);
    // Overwrite: the old run goes back before the new one is placed.
    assert!(s.put(5, &blob[..5000]).unwrap());
    assert_eq!(p.stats().runs, 1);
    s.remove(5);
    assert_eq!(p.stats().runs, 0);
    assert_eq!(p.stats().used, 0);
}

#[test]
fn a_blob_the_pool_cannot_place_falls_back_to_the_heap() {
    let p = pool(1, 4 * 4096);
    let s = MemBlobStore::pool_backed(0, BlobEviction::Fifo, std::sync::Arc::clone(&p));
    let big = vec![0x42u8; 5 * 4096]; // larger than one pool chunk
    assert!(s.put(1, &big).unwrap());
    assert_eq!(s.pooled_blobs(), 0);
    assert_eq!(s.stats.pool_fallbacks.load(Ordering::Relaxed), 1);
    let mut out = vec![0u8; big.len()];
    assert!(s.get(1, &mut out).unwrap());
    assert_eq!(out, big);
}

#[test]
fn eviction_and_drop_return_every_pool_run() {
    let p = pool(2, 8 * 4096);
    {
        let s = MemBlobStore::pool_backed(3 * 4096, BlobEviction::Lru, std::sync::Arc::clone(&p));
        for k in 0..6u64 {
            assert!(s.put(k, &[k as u8; 4096]).unwrap());
        }
        assert_eq!(s.len(), 3);
        assert_eq!(p.stats().runs, 3, "evicted blobs released their runs");
    }
    assert_eq!(p.stats().runs, 0, "dropping the store released the rest");
}
