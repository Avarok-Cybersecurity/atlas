// SPDX-License-Identifier: AGPL-3.0-only

//! RdmaSnapshotStore (over MockSnapshotTransport) + FileSnapshotArena tests
//! (moved from the pre-split ssm_tier.rs).

use super::super::{FileSnapshotArena, MockSnapshotTransport};
use super::*;

// ── Phase 4b: RdmaSnapshotStore (over MockSnapshotTransport) ──────────
// Fixed blob size (BLOB) + finite arena (SLOTS) so full-arena / slot-reuse
// paths are exercised. These mirror the MemBlobStore contract tests.
const BLOB: usize = 4;
fn rdma_store(slots: usize) -> RdmaSnapshotStore {
    let t = Box::new(MockSnapshotTransport::new(slots * BLOB));
    RdmaSnapshotStore::new(t, BLOB, slots)
}

#[test]
fn rdma_put_get_round_trip_bit_identical() {
    let s = rdma_store(4);
    assert!(s.put(42, &[1, 2, 3, 4]).unwrap());
    let mut out = [0u8; BLOB];
    assert!(s.get(42, &mut out).unwrap());
    assert_eq!(out, [1, 2, 3, 4], "spill->arena->fault is bit-identical");
    assert_eq!(s.len(), 1);
    assert_eq!(s.bytes_resident(), BLOB);
}

#[test]
fn rdma_get_absent_is_miss_not_error() {
    let s = rdma_store(4);
    let mut out = [0u8; BLOB];
    assert!(!s.get(7, &mut out).unwrap());
    assert_eq!(s.stats.get_misses.load(Ordering::Relaxed), 1);
}

#[test]
fn rdma_wrong_size_get_refused_out_untouched() {
    let s = rdma_store(4);
    s.put(1, &[9; BLOB]).unwrap();
    let mut out = [0u8; BLOB + 4]; // mismatched
    assert!(
        !s.get(1, &mut out).unwrap(),
        "never scatter a wrong-sized blob"
    );
    assert_eq!(out, [0u8; BLOB + 4], "out left untouched on refusal");
}

#[test]
fn rdma_wrong_size_put_refused() {
    let s = rdma_store(4);
    assert!(
        !s.put(1, &[0; BLOB + 1]).unwrap(),
        "off-size blob refused, not corrupt"
    );
    assert_eq!(s.len(), 0);
}

#[test]
fn rdma_full_arena_put_returns_false_not_err() {
    let s = rdma_store(2);
    assert!(s.put(1, &[1; BLOB]).unwrap());
    assert!(s.put(2, &[2; BLOB]).unwrap());
    // Third key, no free slot → graceful Ok(false), NOT Err, NOT overwrite.
    assert!(!s.put(3, &[3; BLOB]).unwrap());
    assert_eq!(s.len(), 2);
    assert_eq!(s.stats.put_rejects.load(Ordering::Relaxed), 1);
    // The two resident keys are intact.
    let mut o = [0u8; BLOB];
    assert!(s.get(1, &mut o).unwrap() && o == [1; BLOB]);
    assert!(s.get(2, &mut o).unwrap() && o == [2; BLOB]);
}

#[test]
fn rdma_remove_frees_slot_for_reuse() {
    let s = rdma_store(1); // single slot
    assert!(s.put(1, &[1; BLOB]).unwrap());
    assert!(!s.put(2, &[2; BLOB]).unwrap(), "arena full");
    s.remove(1);
    assert!(s.put(2, &[2; BLOB]).unwrap(), "freed slot reused");
    let mut o = [0u8; BLOB];
    assert!(s.get(2, &mut o).unwrap() && o == [2; BLOB]);
    assert!(!s.get(1, &mut o).unwrap(), "removed key gone");
    assert_eq!(s.len(), 1);
}

#[test]
fn rdma_overwrite_in_place_no_slot_leak() {
    let s = rdma_store(1); // single slot forces in-place overwrite
    assert!(s.put(1, &[1; BLOB]).unwrap());
    assert!(
        s.put(1, &[2; BLOB]).unwrap(),
        "overwrite reuses the same slot"
    );
    assert_eq!(s.len(), 1);
    assert_eq!(s.bytes_resident(), BLOB);
    let mut o = [0u8; BLOB];
    assert!(s.get(1, &mut o).unwrap());
    assert_eq!(o, [2; BLOB], "reads the overwritten value");
}

// ── Decode rolling tier: FileSnapshotArena (local NVMe) ───────────────
#[test]
fn file_arena_round_trip_bit_identical() {
    let dir = std::env::temp_dir().join(format!("atlas-decode-test-{}", std::process::id()));
    let dir = dir.to_str().unwrap();
    let store = ArenaSnapshotStore::new(
        Box::new(FileSnapshotArena::create(dir, 4 * BLOB as u64).unwrap()),
        BLOB,
        4,
    );
    assert!(store.put(0xDEAD, &[9, 8, 7, 6]).unwrap());
    let mut out = [0u8; BLOB];
    assert!(store.get(0xDEAD, &mut out).unwrap());
    assert_eq!(
        out,
        [9, 8, 7, 6],
        "file spill->arena->fault is bit-identical"
    );
    // Slot recycle: distinct key reuses a fresh slot, both recover.
    assert!(store.put(0xBEEF, &[1, 2, 3, 4]).unwrap());
    let mut o2 = [0u8; BLOB];
    assert!(store.get(0xBEEF, &mut o2).unwrap() && o2 == [1, 2, 3, 4]);
    assert!(store.get(0xDEAD, &mut out).unwrap() && out == [9, 8, 7, 6]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn file_arena_write_past_capacity_errs_not_corrupts() {
    let dir = std::env::temp_dir().join(format!("atlas-decode-cap-{}", std::process::id()));
    let dir = dir.to_str().unwrap();
    let arena = FileSnapshotArena::create(dir, BLOB as u64).unwrap();
    assert!(arena.write_blob(0, &[1; BLOB]).is_ok());
    assert!(
        arena.write_blob(1, &[1; BLOB]).is_err(),
        "over-capacity write refused"
    );
    let _ = std::fs::remove_dir_all(dir);
}
