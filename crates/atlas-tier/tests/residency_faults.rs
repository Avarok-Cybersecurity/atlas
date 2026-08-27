// SPDX-License-Identifier: AGPL-3.0-only

//! Fault-injection integration tests for [`Residency`]: what happens when a
//! blob-move (spill / fault-in / put) fails HALFWAY.
//!
//! Split out of `residency.rs` to keep both files under the repo's 500-LoC cap.
//! The reference impls (`VecSlotArena` + `MemSwapStore`) never fail, so the
//! trickiest invariant in the policy core — that a half-completed move leaves
//! the page table consistent — is untestable without the fakes defined here.

use atlas_tier::{MemSwapStore, Residency, SlotArena, SwapStore, VecSlotArena};

const B: usize = 8; // tiny blob for tests

fn blob(tag: u8) -> Vec<u8> {
    vec![tag; B]
}

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

/// THE no-aliasing invariant: a disk record must be owned by AT MOST ONE key.
///
/// `alloc`'s `OnDisk` arm frees the key's disk record BEFORE `acquire_slot` has
/// succeeded, but leaves `map[key] = OnDisk(that record)` in place. If the
/// spill inside `acquire_slot` then fails (canonically ENOSPC on the swap file),
/// the error propagates with the record on the free list AND still claimed by
/// the key's map entry. The next spill hands that same record to a DIFFERENT
/// key, and a later GET of the first key reads the second key's blob and
/// reports `Ok(true)` — silent cross-request corruption, not an error. In
/// production these blobs are whole SSM snapshots, so this restores one
/// request's recurrent state into another request's sequence.
#[test]
fn failed_reput_of_spilled_key_must_not_alias_its_disk_record() {
    let (swap, faults) = FaultySwap::new();
    let mut r = Residency::new(FaultyArena::new(1), swap).unwrap(); // 1 hot slot ⇒ every new key spills
    r.put_blob(1, &blob(1)).unwrap();
    r.put_blob(2, &blob(2)).unwrap(); // spills key 1 → disk record 0
    assert_eq!(r.stats().spills_to_disk, 1, "key 1 is on disk");

    // Re-PUT the SPILLED key 1 with the swap file full: alloc frees record 0,
    // then the spill of key 2 inside acquire_slot hits ENOSPC and errors.
    faults.fail_write.store(true, Ordering::SeqCst);
    assert!(
        r.put_blob(1, &blob(99)).is_err(),
        "ENOSPC during the re-PUT of a spilled key propagates"
    );

    // One more successful spill hands the (wrongly) freed record to key 2.
    r.put_blob(3, &blob(3)).unwrap();

    let mut out = vec![0u8; B];
    let hit = r.get_blob(1, &mut out).unwrap();
    assert!(
        hit && out == blob(1),
        "the failed re-PUT must preserve key 1's old record, got hit={hit} {out:?}"
    );
}

/// The same defect in its double-free form: because the failed re-PUT leaves
/// `map[key] = OnDisk(d)` intact, RETRYING the re-PUT (the natural response to
/// a transient ENOSPC) takes the `OnDisk` arm a second time and pushes `d` onto
/// `free_disk` AGAIN. Two later spills then hand the one record to two keys.
#[test]
fn retried_reput_must_not_double_free_the_same_disk_record() {
    let (swap, faults) = FaultySwap::new();
    let mut r = Residency::new(FaultyArena::new(1), swap).unwrap();
    r.put_blob(1, &blob(1)).unwrap();
    r.put_blob(2, &blob(2)).unwrap(); // spills key 1 → disk record 0

    faults.fail_write.store(true, Ordering::SeqCst);
    assert!(
        r.put_blob(1, &blob(99)).is_err(),
        "first attempt hits ENOSPC"
    );
    r.put_blob(1, &blob(99)).unwrap(); // retry succeeds — and re-frees record 0
    r.put_blob(4, &blob(4)).unwrap(); // consumes the duplicate free-list entry

    let mut out = vec![0u8; B];
    let hit = r.get_blob(2, &mut out).unwrap();
    assert!(
        hit && out == blob(2),
        "key 2 must remain present with its own bytes after retry, got hit={hit} {out:?}"
    );
}
