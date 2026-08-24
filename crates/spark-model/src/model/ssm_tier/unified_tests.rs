// SPDX-License-Identifier: AGPL-3.0-only

//! §4 unification (`ATLAS_SSM_TIER_UNIFIED`) store contract tests.

use super::super::MockSnapshotTransport;
use super::*;

// Fixed blob size for the in-process arena and swap fixtures.
const BLOB: usize = 4;

#[test]
fn unified_flag_accepts_only_documented_truthy_values() {
    for on in ["1", "true", "on", "yes", " 1 ", " true ", " on ", " yes "] {
        assert!(unified_flag_truthy(Some(on)), "{on:?} must engage the flag");
    }
    for off in ["", "0", "false", "off", "no", "TRUE", "2"] {
        assert!(!unified_flag_truthy(Some(off)), "{off:?} must stay off");
    }
    assert!(!unified_flag_truthy(None), "unset = default OFF");
}

fn unified_store(slots: usize) -> UnifiedSnapshotStore {
    UnifiedSnapshotStore::new(
        Box::new(atlas_tier::VecSlotArena::new(BLOB, slots)),
        Box::new(atlas_tier::MemSwapStore::new(BLOB)),
        BLOB,
    )
    .unwrap()
}

fn unified_store_capped(slots: usize, max_disk: usize) -> UnifiedSnapshotStore {
    UnifiedSnapshotStore::new_capped(
        Box::new(atlas_tier::VecSlotArena::new(BLOB, slots)),
        Box::new(atlas_tier::MemSwapStore::new(BLOB)),
        BLOB,
        max_disk,
    )
    .unwrap()
}

// ─────────────── ATLAS_SSM_TIER_DISK_GB: the bounded Marconi tier ───────────

/// A cap bounds the disk tier but NEVER converts into a reject: the store's
/// "never full" contract is what keeps the bounded-tier warn in
/// `reclaim_from_cache` ("SSM spill tier refused a blob") unreached — the cap's
/// error channel is a later clean MISS, not a refused PUT.
#[test]
fn capped_store_bounds_disk_but_never_rejects() {
    const CAP: usize = 3;
    let s = unified_store_capped(2, CAP);
    for k in 0..64u64 {
        assert!(
            s.put(k, &[k as u8; BLOB]).unwrap(),
            "put {k} must never be refused by a CAPPED tier either"
        );
        assert!(
            s.disk_records() <= CAP,
            "on-disk records bounded by the cap (k={k})"
        );
    }
    assert_eq!(s.disk_records(), CAP, "the requested disk cap is usable");
    assert_eq!(s.disk_evictions(), 59, "only overflow is dropped");
    assert_eq!(s.len(), 2 + CAP, "hot and disk capacities are both usable");
    assert_eq!(s.stats.put_rejects.load(Ordering::Relaxed), 0);
}

/// A dropped snapshot must degrade EXACTLY like a tier-disabled turn:
/// `Ok(false)`, not `Err` and not torn bytes — the contract
/// `try_fault_in_ssm_snapshot` relies on to free the slot and recompute.
#[test]
fn capped_store_miss_is_clean() {
    let s = unified_store_capped(1, 2);
    for k in 0..16u64 {
        assert!(s.put(k, &[k as u8; BLOB]).unwrap());
    }
    assert_eq!(s.disk_records(), 2);
    assert_eq!(s.disk_evictions(), 13);
    assert_eq!(s.len(), 3);
    let mut o = [0xAAu8; BLOB];
    assert!(
        !s.get(0, &mut o).unwrap(),
        "the coldest key was dropped at the cap → clean miss"
    );
    assert_eq!(o, [0xAAu8; BLOB], "out untouched on a miss (no torn bytes)");
    // The survivors are still byte-identical — a cap drops, it doesn't corrupt.
    assert!(s.get(15, &mut o).unwrap());
    assert_eq!(o, [15u8; BLOB]);
}

/// THE DECODE GUARD — the tripwire for the constructor split.
///
/// `UnifiedSnapshotStore::new` must stay UNCAPPED. A decode rollback target
/// that misses is a CORRUPT restore, not a recompute (unlike Marconi's
/// miss→recompute), so `build_decode_tier_store` relies on this constructor
/// dropping nothing BY CONSTRUCTION rather than by arena sizing. If someone
/// later routes `new` through a cap, this test fails first.
#[test]
fn uncapped_new_never_drops() {
    let s = unified_store(2);
    for k in 0..256u64 {
        assert!(s.put(k, &[k as u8; BLOB]).unwrap());
    }
    assert_eq!(
        s.disk_evictions(),
        0,
        "the decode tier's constructor must never drop a blob — a dropped \
         rollback target is a corrupt restore"
    );
    assert_eq!(s.len(), 256, "every key still tracked");
    let mut o = [0u8; BLOB];
    for k in 0..256u64 {
        assert!(s.get(k, &mut o).unwrap(), "key {k} still present");
        assert_eq!(o, [k as u8; BLOB], "key {k} byte-identical");
    }
}

/// The GiB→records conversion (env-free core). Holo-3.1-35B blob geometry:
/// 66,846,720 B = 16,320 × 4 KiB.
#[test]
fn ssm_tier_disk_slots_conversion_and_strictness() {
    const HOLO_BLOB: usize = 66_846_720;
    assert_eq!(
        disk_slots_from(None, HOLO_BLOB).unwrap(),
        0,
        "unset ⇒ unbounded"
    );
    assert_eq!(disk_slots_from(Some(""), HOLO_BLOB).unwrap(), 0);
    assert_eq!(
        disk_slots_from(Some("0"), HOLO_BLOB).unwrap(),
        0,
        "0 is the explicit unbounded sentinel"
    );
    // 32 GiB / 66,846,720 B = 514 records; one is reserved for the fault-in
    // transient, so the WORST-CASE file (514 records) is still ≤ 32 GiB.
    assert_eq!(disk_slots_from(Some("32"), HOLO_BLOB).unwrap(), 513);
    assert_eq!(disk_slots_from(Some(" 32 "), HOLO_BLOB).unwrap(), 513);
    assert!(
        (513 + 1) as u64 * HOLO_BLOB as u64 <= 32 * (1u64 << 30),
        "worst-case swap file must fit the operator's budget"
    );
    // Strict (PCND): a typo must never mean "unbounded".
    for bad in ["abc", "-1", "32GB", "nan", "inf", "1e400"] {
        assert!(
            disk_slots_from(Some(bad), HOLO_BLOB).is_err(),
            "{bad:?} must be a config error, not a silent unbounded tier"
        );
    }
    // A budget too small for two snapshots is an error, NOT 0 — that would
    // collide with the unbounded sentinel and invert the operator's intent.
    let tiny = disk_slots_from(Some("0.0001"), HOLO_BLOB);
    assert!(
        tiny.is_err(),
        "under-sized budget must fail fast, got {tiny:?}"
    );
    assert!(
        disk_slots_from(Some("1"), 0).is_err(),
        "a positive budget requires nonzero snapshot geometry"
    );
}

/// LRU (not FIFO): touching the oldest-inserted key protects it — the
/// spill victim is the least-recently-USED key. (A capped MemBlobStore
/// would evict key 1 here; RdmaSnapshotStore would refuse key 3 outright.)
#[test]
fn unified_store_victim_is_lru_not_fifo_and_not_a_reject() {
    let s = unified_store(2);
    assert!(s.put(1, &[1; BLOB]).unwrap());
    assert!(s.put(2, &[2; BLOB]).unwrap());
    let mut o = [0u8; BLOB];
    assert!(s.get(1, &mut o).unwrap()); // touch 1 → 2 is now coldest
    assert!(s.put(3, &[3; BLOB]).unwrap(), "no drop-on-full");
    assert_eq!(s.bytes_resident(), 2 * BLOB, "two hot slots resident");
    // The hot-again key SURVIVED IN THE HOT TIER: getting key 1 is a
    // resident hit (no disk fault), where FIFO would have evicted it as
    // oldest-inserted; key 2 was the LRU spill victim and faults back.
    let faults0 = s.inner.lock().stats().faults_from_disk;
    assert!(s.get(1, &mut o).unwrap(), "hot-again key survives");
    assert_eq!(o, [1u8; BLOB]);
    assert_eq!(
        s.inner.lock().stats().faults_from_disk,
        faults0,
        "key 1 was still RESIDENT — the LRU victim was key 2, not the FIFO-oldest"
    );
    assert!(
        s.get(2, &mut o).unwrap(),
        "spilled key faults back, never dropped"
    );
    assert_eq!(o, [2u8; BLOB]);
    assert_eq!(
        s.inner.lock().stats().faults_from_disk,
        faults0 + 1,
        "key 2 came back via a disk fault"
    );
    assert!(s.get(3, &mut o).unwrap());
    assert_eq!(o, [3u8; BLOB]);
}

#[test]
fn unified_store_wrong_size_refused_gracefully() {
    let s = unified_store(2);
    assert!(s.put(1, &[7; BLOB]).unwrap());
    assert!(!s.put(1, &[8; BLOB - 1]).unwrap(), "short put refused");
    assert!(!s.put(1, &[9; BLOB + 1]).unwrap(), "long put refused");
    assert_eq!(s.stats.put_rejects.load(Ordering::Relaxed), 2);
    let mut short = [0xa5u8; BLOB - 1];
    let mut long = [0x5au8; BLOB + 1];
    assert!(!s.get(1, &mut short).unwrap(), "short get refused");
    assert!(!s.get(1, &mut long).unwrap(), "long get refused");
    assert_eq!(short, [0xa5; BLOB - 1]);
    assert_eq!(long, [0x5a; BLOB + 1]);
    assert_eq!(s.stats.get_misses.load(Ordering::Relaxed), 2);
    let mut exact = [0u8; BLOB];
    assert!(s.get(1, &mut exact).unwrap());
    assert_eq!(exact, [7; BLOB], "refused replacements preserve the value");
}

#[test]
fn unified_store_remove_releases_resident_key() {
    let s = unified_store(2);
    assert!(s.put(1, &[1; BLOB]).unwrap());
    s.remove(1);
    let mut o = [0u8; BLOB];
    assert!(!s.get(1, &mut o).unwrap());
    assert_eq!(s.len(), 0);
    assert_eq!(s.bytes_resident(), 0);
}

/// Unified over the SAME transport geometry the bounded RDMA store uses:
/// where `RdmaSnapshotStore` returns Ok(false) at slot 5, the unified wrap
/// keeps accepting (LRU spill to the swap tier) — the live §4 bug arm.
#[test]
fn unified_over_transport_never_drops_where_bounded_store_did() {
    const SLOTS: usize = 4;
    let hot = Box::new(TransportSlotArena {
        transport: Box::new(MockSnapshotTransport::new(SLOTS * BLOB)),
        slot_bytes: BLOB,
        num_slots: SLOTS,
    });
    let s = UnifiedSnapshotStore::new(hot, Box::new(atlas_tier::MemSwapStore::new(BLOB)), BLOB)
        .unwrap();
    let mut o = [0u8; BLOB];
    for k in 0..16u64 {
        assert!(
            s.put(k, &[k as u8; BLOB]).unwrap(),
            "arena-full put {k} accepted"
        );
    }
    for k in 0..16u64 {
        assert!(s.get(k, &mut o).unwrap(), "key {k} recoverable");
        assert_eq!(o, [k as u8; BLOB]);
    }
}
