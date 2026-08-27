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
