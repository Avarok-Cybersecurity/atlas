// SPDX-License-Identifier: AGPL-3.0-only

//! Public-API integration tests for the [`Residency`] policy core over the
//! host-RAM reference impls ([`VecSlotArena`] + [`MemSwapStore`]), moved out
//! of `src/lib.rs` per the repo's tests-in-their-own-file convention. Living
//! here also pins the crate's public surface.

use atlas_tier::{MemSwapStore, Residency, SlotArena, SwapStore, VecSlotArena};

const B: usize = 8; // tiny blob for tests

fn blob(tag: u8) -> Vec<u8> {
    vec![tag; B]
}

/// Client-side helper: alloc → write bytes into the arena slot → commit.
fn put(r: &mut Residency<VecSlotArena, MemSwapStore>, key: u64, tag: u8) {
    let slot = r.alloc(key).unwrap();
    r.arena_mut().write_slot(slot, &blob(tag)).unwrap();
    r.commit(key).unwrap();
}
fn get(r: &mut Residency<VecSlotArena, MemSwapStore>, key: u64) -> Option<Vec<u8>> {
    r.locate(key).unwrap().map(|slot| {
        let mut out = vec![0u8; B];
        r.arena().read_slot(slot, &mut out).unwrap();
        out
    })
}

fn residency(slots: usize) -> Residency<VecSlotArena, MemSwapStore> {
    Residency::new(VecSlotArena::new(B, slots), MemSwapStore::new(B)).unwrap()
}

fn residency_capped(slots: usize, max_disk: usize) -> Residency<VecSlotArena, MemSwapStore> {
    Residency::new_capped(VecSlotArena::new(B, slots), MemSwapStore::new(B), max_disk).unwrap()
}

/// Disk cap: the swap tier is BOUNDED — beyond RAM slots + `max_disk`, the
/// COLDEST on-disk snapshot is dropped (a later GET misses → recompute), so
/// the swap file never grows past the cap. This is the operator's 50 GB
/// sanity limit at the paging layer.
#[test]
fn disk_cap_bounds_swap_and_drops_coldest() {
    let mut r = residency_capped(2, 3); // 2 RAM + 3 disk = 5 total capacity
    for k in 0..10u64 {
        put(&mut r, k, k as u8);
    }
    assert_eq!(r.stats().spills_to_disk, 8);
    assert_eq!(r.stats().disk_evictions, 5, "only overflow is dropped");
    assert_eq!(r.disk_count(), 3, "the configured disk capacity is usable");
    assert_eq!(r.total_keys(), 2 + 3, "RAM and disk capacities are full");
    // Coldest keys were dropped → clean miss (checked first: a miss doesn't
    // perturb residency).
    assert_eq!(
        get(&mut r, 0),
        None,
        "oldest key evicted from the capped disk"
    );
    assert_eq!(get(&mut r, 1), None);
    // The hottest keys survive (resident) and are byte-identical.
    assert_eq!(get(&mut r, 9).as_deref(), Some(&blob(9)[..]));
    assert_eq!(get(&mut r, 8).as_deref(), Some(&blob(8)[..]));
}

/// The SWAP-FILE bound an operator actually budgets against. `disk_count` is
/// the live-record count, but the file never shrinks (`DirectSwapFile` doesn't
/// override `discard_record`), so its size is `disk_high_water × blob_bytes`.
/// `locate`'s OnDisk arm pulls the faulting key out of `disk_lru` while its
/// record is still live, so `make_disk_room` counts one fewer than exist and
/// the high-water mark can reach `cap + 1` — never more. That "+1" is exactly
/// what the caller-side conversion subtracts before handing a cap down, so a
/// budget is never overshot.
#[test]
fn disk_high_water_never_exceeds_cap_plus_one() {
    const CAP: usize = 4;
    let mut r = residency_capped(2, CAP);
    for k in 0..40u64 {
        put(&mut r, k, k as u8);
        // Interleave GETs of (likely) on-disk keys to exercise the transient
        // where a faulting record is live but out of disk_lru.
        for probe in k.saturating_sub(3)..k {
            let _ = get(&mut r, probe);
        }
        assert!(
            r.disk_count() <= CAP,
            "live on-disk records must stay within the cap (k={k}, {} > {CAP})",
            r.disk_count()
        );
        assert!(
            r.disk_high_water() <= CAP + 1,
            "swap file must never exceed cap+1 records (k={k}, {} > {})",
            r.disk_high_water(),
            CAP + 1
        );
    }
    assert_eq!(r.max_disk_slots(), CAP);
    assert!(
        r.stats().disk_evictions > 0,
        "the cap must actually have been load-bearing in this run"
    );
}

/// The one correctness property a cap could genuinely break: a fault-in must
/// never drop the very record it is faulting. The GET below triggers
/// `acquire_slot` → `evict_coldest_to_disk` → `make_disk_room` INSIDE the same
/// call, with the disk already at cap — so the faulting key is the coldest
/// candidate by insertion order. `locate` self-pins it (pulling it out of
/// `disk_lru` before acquiring a slot), so the victim is some other key and the
/// blob comes back byte-identical. This is the achievable form of "a record
/// something is actively using is never dropped".
#[test]
fn capped_fault_in_never_self_evicts() {
    const CAP: usize = 2;
    let mut r = residency_capped(1, CAP); // 1 hot slot ⇒ every put spills
    for k in 0..4u64 {
        put(&mut r, k, k as u8);
    }
    assert_eq!(r.disk_count(), CAP, "disk sitting exactly at the cap");
    let evictions_before = r.stats().disk_evictions;
    // Key 1 is on disk AND is the coldest on-disk entry — without the self-pin
    // at locate's OnDisk arm, the nested make_disk_room would see disk_lru at
    // the cap and drop key 1 itself, then read the record it just discarded.
    assert_eq!(
        get(&mut r, 1),
        Some(blob(1)),
        "the faulting record must survive its own fault-in, byte-identical"
    );
    assert_eq!(
        r.stats().disk_evictions,
        evictions_before,
        "the self-pin freed the cap slot, so the fault dropped NOTHING — least \
         of all its own record"
    );
    assert_eq!(
        get(&mut r, 2),
        Some(blob(2)),
        "the bystander on-disk key spilled during the fault is intact too"
    );
    assert!(
        r.disk_high_water() <= CAP + 1,
        "the transient extra live record is the documented +1, not unbounded growth"
    );
}

/// THE headline invariant: put far more keys than the arena holds; the
/// coldest spill to the disk tier and fault back BYTE-IDENTICAL. "Infinite
/// depth, never dropped" proven at the paging layer.
#[test]
fn infinite_depth_spill_and_fault_byte_identical() {
    let mut r = residency(4); // 4 slots
    for k in 0..64u64 {
        put(&mut r, k, k as u8);
    }
    assert!(
        r.stats().spills_to_disk >= 60,
        "most keys must have spilled to disk"
    );
    assert_eq!(r.resident_count(), 4, "only 4 slots resident at once");
    assert_eq!(r.total_keys(), 64, "all 64 keys tracked — nothing dropped");
    // Every key faults back to its exact bytes.
    for k in 0..64u64 {
        assert_eq!(
            get(&mut r, k).as_deref(),
            Some(&blob(k as u8)[..]),
            "key {k}"
        );
    }
    assert!(r.stats().faults_from_disk > 0);
}

/// THE eviction-pin guarantee (WS-A GET→RDMA-read race): a read-pinned key
/// is never chosen as an eviction victim, even when it is the LRU-coldest —
/// a concurrent allocation spills the next-coldest unpinned key instead, so the
/// client's in-flight one-sided RDMA READ is never torn by slot reuse.
#[test]
fn read_pin_survives_concurrent_eviction() {
    let mut r = residency(2);
    put(&mut r, 0, 0); // resident: [0] (coldest)
    put(&mut r, 1, 1); // resident: [0,1]
    // Client A GETs key 0 (the coldest) and begins its RDMA read → pin it.
    assert!(r.locate(0).unwrap().is_some());
    r.pin_read(0);
    assert_eq!(r.read_pin_count(0), 1);
    let faults_before = r.stats().faults_from_disk;

    // Client B allocates a new key → arena full → must evict. Key 0 is coldest
    // but pinned, so key 1 is spilled instead.
    put(&mut r, 2, 2);
    assert_eq!(r.stats().spills_to_disk, 1, "exactly one eviction");
    assert_eq!(
        get(&mut r, 1),
        Some(blob(1)),
        "the UNPINNED key 1 was the victim"
    );
    // Key 0 is still resident (byte-intact) and never touched disk.
    assert_eq!(
        get(&mut r, 0),
        Some(blob(0)),
        "pinned key 0 survived intact"
    );
    assert_eq!(
        r.stats().faults_from_disk,
        faults_before + 1,
        "only key 1 faulted back; key 0 never spilled"
    );
}

/// Ref-counted: concurrent readers each add a pin; the key stays protected
/// until the LAST reader releases, then rejoins the LRU (no double-insert).
#[test]
fn refcounted_read_pins_release_to_evictable() {
    let mut r = residency(2);
    put(&mut r, 0, 0);
    put(&mut r, 1, 1);
    r.pin_read(0);
    r.pin_read(0); // second concurrent reader of key 0
    assert_eq!(r.read_pin_count(0), 2);
    r.unpin_read(0);
    assert_eq!(r.read_pin_count(0), 1, "still one reader → still pinned");
    // Force an eviction: key 0 is still protected → key 1 spills.
    put(&mut r, 2, 2);
    assert_eq!(
        get(&mut r, 1),
        Some(blob(1)),
        "key 1 evicted while key 0 still pinned"
    );
    // Last reader releases → key 0 rejoins the LRU exactly once and is now
    // an eligible victim again.
    r.unpin_read(0);
    assert_eq!(r.read_pin_count(0), 0);
    assert_eq!(
        r.resident_count(),
        2,
        "keys 0 and 2 resident; no LRU double-insert"
    );
    put(&mut r, 3, 3); // evicts the now-unpinned coldest (key 0)
    put(&mut r, 4, 4);
    assert_eq!(
        get(&mut r, 0),
        Some(blob(0)),
        "unpinned key 0 spilled+faulted byte-identical"
    );
}

#[test]
fn resident_hit_does_not_touch_disk() {
    let mut r = residency(4);
    for k in 0..3u64 {
        put(&mut r, k, k as u8);
    }
    let spills_before = r.stats().spills_to_disk;
    assert_eq!(get(&mut r, 1), Some(blob(1)));
    assert_eq!(
        r.stats().spills_to_disk,
        spills_before,
        "resident hit spills nothing"
    );
    assert!(r.stats().resident_hits >= 1);
}

#[test]
fn lru_evicts_coldest_first() {
    let mut r = residency(2);
    put(&mut r, 10, 10); // resident: [10]
    put(&mut r, 11, 11); // resident: [10,11]
    get(&mut r, 10); // touch 10 → [11,10]; 11 now coldest
    put(&mut r, 12, 12); // arena full → evict coldest (11) to disk
    // 11 must be on disk, 10 & 12 resident.
    assert_eq!(get(&mut r, 11), Some(blob(11))); // faults back correctly
    assert!(r.stats().faults_from_disk >= 1);
}

#[test]
fn overwrite_in_place_reuses_slot_no_leak() {
    let mut r = residency(2);
    put(&mut r, 5, 100);
    put(&mut r, 5, 200); // rewrite same key
    assert_eq!(get(&mut r, 5), Some(blob(200)));
    assert_eq!(r.total_keys(), 1, "no phantom duplicate");
    assert_eq!(r.resident_count(), 1);
}

#[test]
fn overwrite_spilled_key_reclaims_disk() {
    let mut r = residency(1); // force spilling
    put(&mut r, 1, 1);
    put(&mut r, 2, 2); // spills key 1 to disk
    assert_eq!(r.disk_high_water(), 1);
    put(&mut r, 1, 99); // rewrite the SPILLED key 1
    assert_eq!(
        r.disk_high_water(),
        2,
        "rewriting secures a slot before reclaiming the old record"
    );
    put(&mut r, 3, 3); // must reuse the reclaimed record, not grow another
    assert_eq!(
        r.disk_high_water(),
        2,
        "the overwritten key's old disk record is reusable"
    );
    assert_eq!(get(&mut r, 1), Some(blob(99)));
    assert_eq!(get(&mut r, 2), Some(blob(2)));
    assert_eq!(get(&mut r, 3), Some(blob(3)));
}

#[test]
fn remove_frees_resources() {
    let mut r = residency(2);
    put(&mut r, 1, 1);
    put(&mut r, 2, 2);
    put(&mut r, 3, 3); // 1 spills to disk
    r.remove(1); // remove a spilled key
    r.remove(2); // remove a resident key
    assert_eq!(get(&mut r, 1), None, "removed key is a clean miss");
    assert_eq!(get(&mut r, 2), None);
    assert_eq!(get(&mut r, 3), Some(blob(3)));
    assert_eq!(r.total_keys(), 1);

    let spills = r.stats().spills_to_disk;
    let high_water = r.disk_high_water();
    put(&mut r, 4, 4);
    assert_eq!(
        r.stats().spills_to_disk,
        spills,
        "the removed resident key releases its arena slot"
    );
    put(&mut r, 5, 5);
    assert_eq!(
        r.disk_high_water(),
        high_water,
        "the removed spilled key releases its disk record for reuse"
    );
    assert_eq!(get(&mut r, 4), Some(blob(4)));
    assert_eq!(get(&mut r, 5), Some(blob(5)));
}

#[test]
fn unknown_key_is_clean_miss() {
    let mut r = residency(2);
    assert_eq!(r.locate(0xdead).unwrap(), None);
    assert_eq!(r.stats().get_miss, 1);
}

#[test]
fn reserved_slot_pinned_during_put() {
    // A slot handed out by alloc must not be evictable before commit.
    let mut r = residency(1);
    let slot = r.alloc(1).unwrap();
    // Second alloc with the only slot reserved-and-uncommitted must error,
    // not silently evict the in-flight PUT.
    let err = r.alloc(2);
    assert!(err.is_err(), "must not evict an uncommitted reserved slot");
    // Finish the first PUT and it all works.
    r.arena_mut().write_slot(slot, &blob(1)).unwrap();
    r.commit(1).unwrap();
    assert_eq!(get(&mut r, 1), Some(blob(1)));
}

#[test]
fn size_mismatch_rejected() {
    let bad = Residency::new(VecSlotArena::new(8, 2), MemSwapStore::new(16));
    assert!(
        bad.is_err(),
        "arena/swap size mismatch must be rejected at construction"
    );
}

// ───────────── one-shot helpers + boxed composition ─────────────

/// `put_blob`/`get_blob` never reject and round-trip bytes exactly like the
/// two-phase alloc/commit path (the in-process consumer contract).
#[test]
fn put_get_blob_helpers_never_reject_and_roundtrip() {
    let mut r = residency(2);
    for k in 0..32u64 {
        r.put_blob(k, &blob(k as u8)).unwrap();
    }
    assert_eq!(r.total_keys(), 32, "never-reject: every key tracked");
    assert_eq!(r.resident_count(), 2);
    let mut out = vec![0u8; B];
    for k in 0..32u64 {
        assert!(r.get_blob(k, &mut out).unwrap(), "key {k} present");
        assert_eq!(out, blob(k as u8), "key {k} byte-identical");
    }
    assert!(
        !r.get_blob(999, &mut out).unwrap(),
        "unknown key is a clean miss"
    );
    // Size mismatches are hard errors, never silent corruption.
    assert!(r.put_blob(1, &[0u8; B + 1]).is_err());
    let mut short = vec![0u8; B - 1];
    assert!(r.get_blob(1, &mut short).is_err());
}

/// `Residency<Box<dyn SlotArena>, Box<dyn SwapStore>>` composes (runtime
/// arena/swap selection — what the unified SSM store uses).
#[test]
fn boxed_trait_objects_compose() {
    let arena: Box<dyn SlotArena> = Box::new(VecSlotArena::new(B, 2));
    let swap: Box<dyn SwapStore> = Box::new(MemSwapStore::new(B));
    let mut r = Residency::new(arena, swap).unwrap();
    for k in 0..16u64 {
        r.put_blob(k, &blob(k as u8)).unwrap();
    }
    let mut out = vec![0u8; B];
    for k in 0..16u64 {
        assert!(r.get_blob(k, &mut out).unwrap());
        assert_eq!(out, blob(k as u8));
    }
}
