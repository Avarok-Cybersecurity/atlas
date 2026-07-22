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
    assert!(
        r.stats().disk_evictions >= 5,
        "coldest disk snaps must be dropped at cap"
    );
    assert!(
        r.total_keys() <= 2 + 3,
        "total tracked keys bounded by RAM + disk cap"
    );
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
    put(&mut r, 1, 99); // rewrite the SPILLED key 1
    assert_eq!(get(&mut r, 1), Some(blob(99)));
    assert_eq!(get(&mut r, 2), Some(blob(2)));
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

// ───────────────────── fault-injection: mid-op I/O failure rollbacks ─────────
//
// The reference impls never fail, so the trickiest invariant in the policy core
// — that a blob-move (spill/fault/put) that errors HALFWAY leaves the page table
// consistent — is otherwise untested. These fakes fail one operation on demand.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail};

/// `VecSlotArena` that fails the next `write_slot` when armed (via `arena_mut`).
struct FaultyArena {
    inner: VecSlotArena,
    fail_next_write: bool,
}
impl FaultyArena {
    fn new(slots: usize) -> Self {
        Self {
            inner: VecSlotArena::new(B, slots),
            fail_next_write: false,
        }
    }
}
impl SlotArena for FaultyArena {
    fn slot_bytes(&self) -> usize {
        self.inner.slot_bytes()
    }
    fn num_slots(&self) -> usize {
        self.inner.num_slots()
    }
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        self.inner.read_slot(slot, out)
    }
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        if self.fail_next_write {
            self.fail_next_write = false;
            bail!("injected write_slot failure");
        }
        self.inner.write_slot(slot, bytes)
    }
}

/// Shared arm-once toggles for [`FaultySwap`] (the store is moved into the
/// `Residency`, so the test keeps a clone to arm faults after construction).
#[derive(Default)]
struct SwapFaults {
    fail_write: AtomicBool,
    fail_read: AtomicBool,
}
struct FaultySwap {
    inner: MemSwapStore,
    faults: Arc<SwapFaults>,
}
impl FaultySwap {
    fn new() -> (Self, Arc<SwapFaults>) {
        let faults = Arc::new(SwapFaults::default());
        (
            Self {
                inner: MemSwapStore::new(B),
                faults: Arc::clone(&faults),
            },
            faults,
        )
    }
}
impl SwapStore for FaultySwap {
    fn record_bytes(&self) -> usize {
        self.inner.record_bytes()
    }
    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()> {
        if self.faults.fail_write.swap(false, Ordering::SeqCst) {
            bail!("injected write_record failure");
        }
        self.inner.write_record(disk_slot, bytes)
    }
    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()> {
        if self.faults.fail_read.swap(false, Ordering::SeqCst) {
            bail!("injected read_record failure");
        }
        self.inner.read_record(disk_slot, out)
    }
    fn discard_record(&mut self, disk_slot: usize) {
        self.inner.discard_record(disk_slot);
    }
}

/// `put_blob` whose arena WRITE fails must roll the reservation back — the slot
/// is reclaimed (not stranded `Reserved`), the key is absent (a GET misses
/// cleanly), and prior keys survive.
#[test]
fn put_blob_rolls_back_reservation_on_arena_write_failure() {
    let (swap, _f) = FaultySwap::new();
    let mut r = Residency::new(FaultyArena::new(2), swap).unwrap();
    r.put_blob(1, &blob(1)).unwrap();

    r.arena_mut().fail_next_write = true;
    assert!(r.put_blob(2, &blob(2)).is_err(), "write failure propagates");

    let mut out = vec![0u8; B];
    assert!(
        !r.get_blob(2, &mut out).unwrap(),
        "rolled-back key 2 misses cleanly (not a torn Reserved slot)"
    );
    assert_eq!(r.total_keys(), 1, "no stranded key-2 entry");
    // The freed slot is reusable and key 1 is intact.
    r.put_blob(3, &blob(3)).unwrap();
    assert!(
        r.get_blob(1, &mut out).unwrap() && out == blob(1),
        "key 1 intact"
    );
    assert!(
        r.get_blob(3, &mut out).unwrap() && out == blob(3),
        "key 3 reuses the reclaimed slot"
    );
}

/// A spill whose swap WRITE fails must roll back: the victim stays resident and
/// intact, no spill is counted, and the disk-slot pool isn't leaked (a later
/// successful spill reuses it).
#[test]
fn spill_rolls_back_on_swap_write_failure_victim_stays_resident() {
    let (swap, faults) = FaultySwap::new();
    let mut r = Residency::new(FaultyArena::new(1), swap).unwrap(); // 1 slot → new key evicts
    r.put_blob(10, &blob(10)).unwrap();

    faults.fail_write.store(true, Ordering::SeqCst);
    assert!(
        r.put_blob(11, &blob(11)).is_err(),
        "spill write failure propagates"
    );
    assert_eq!(r.stats().spills_to_disk, 0, "failed spill is not counted");
    assert_eq!(r.total_keys(), 1, "failed put left no key 11");

    let mut out = vec![0u8; B];
    assert!(
        r.get_blob(10, &mut out).unwrap() && out == blob(10),
        "victim 10 stayed resident and intact"
    );
    assert!(!r.get_blob(11, &mut out).unwrap(), "key 11 never landed");
    // A subsequent spill succeeds — the disk-slot pool wasn't corrupted.
    r.put_blob(12, &blob(12)).unwrap();
    assert_eq!(r.stats().spills_to_disk, 1);
    assert!(
        r.get_blob(10, &mut out).unwrap() && out == blob(10),
        "10 faults back from disk byte-identical"
    );
}

/// A fault-in whose swap READ fails must leave the key `OnDisk` (re-pinned) and
/// return its scratched slot — the error propagates but a retry succeeds, and a
/// bystander key spilled during the failed fault's `acquire_slot` is intact.
#[test]
fn fault_in_read_failure_keeps_key_on_disk_and_frees_slot() {
    let (swap, faults) = FaultySwap::new();
    let mut r = Residency::new(FaultyArena::new(1), swap).unwrap();
    r.put_blob(20, &blob(20)).unwrap();
    r.put_blob(21, &blob(21)).unwrap(); // evicts 20 to disk
    assert_eq!(r.stats().spills_to_disk, 1);

    faults.fail_read.store(true, Ordering::SeqCst);
    let mut out = vec![0u8; B];
    assert!(
        r.get_blob(20, &mut out).is_err(),
        "fault-in read failure propagates"
    );

    // 20 is still OnDisk; a retry (fault auto-cleared) faults it in cleanly.
    assert!(
        r.get_blob(20, &mut out).unwrap() && out == blob(20),
        "20 faults back on retry"
    );
    assert!(
        r.get_blob(21, &mut out).unwrap() && out == blob(21),
        "bystander 21 (spilled during the failed fault) is intact"
    );
}
