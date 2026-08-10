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
    let mut out = [0u8; 4];
    assert!(!s.get(7, &mut out).unwrap());
    assert_eq!(s.stats.get_misses.load(Ordering::Relaxed), 1);
}

#[test]
fn wrong_size_get_refused() {
    let s = MemBlobStore::new(0);
    s.put(1, &[9; 8]).unwrap();
    let mut out = [0u8; 4]; // mismatched
    assert!(
        !s.get(1, &mut out).unwrap(),
        "never scatter a wrong-sized blob"
    );
}

#[test]
fn overwrite_reclaims_bytes() {
    let s = MemBlobStore::new(0);
    s.put(1, &[0; 10]).unwrap();
    s.put(1, &[0; 3]).unwrap();
    assert_eq!(s.len(), 1);
    assert_eq!(
        s.bytes_resident(),
        3,
        "old blob bytes reclaimed on overwrite"
    );
}

#[test]
fn cap_evicts_fifo() {
    let s = MemBlobStore::new(10);
    s.put(1, &[0; 4]).unwrap(); // 4
    s.put(2, &[0; 4]).unwrap(); // 8
    s.put(3, &[0; 4]).unwrap(); // would be 12 > 10 → evict key 1 (oldest)
    assert!(s.bytes_resident() <= 10);
    let mut o = [0u8; 4];
    assert!(!s.get(1, &mut o).unwrap(), "oldest evicted");
    assert!(s.get(3, &mut o).unwrap(), "newest resident");
    assert!(s.stats.evictions.load(Ordering::Relaxed) >= 1);
}

#[test]
fn blob_larger_than_cap_refused() {
    let s = MemBlobStore::new(4);
    assert!(
        !s.put(1, &[0; 8]).unwrap(),
        "over-cap blob refused, not partial"
    );
    assert_eq!(s.len(), 0);
    assert_eq!(s.stats.put_rejects.load(Ordering::Relaxed), 1);
}
