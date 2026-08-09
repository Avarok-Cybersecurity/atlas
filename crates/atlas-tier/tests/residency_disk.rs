// SPDX-License-Identifier: AGPL-3.0-only

//! Public-API integration tests for the DISK half of [`Residency`] — when the
//! cold tier is actually written, and whether the buffer it is written from is
//! O_DIRECT-ready. Split from `residency.rs` (500-LoC cap).
//!
//! Both tests exist because of one live confusion: eight successful `put`s
//! against a real O_DIRECT-backed tier left the swap file at 0 bytes. That is
//! EXPECTED — the hot arena had absorbed all eight — but nothing in the code
//! said so, and nothing proved writes would land once it filled.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;

use atlas_tier::{MemSwapStore, Residency, SwapStore, VecSlotArena};

const B: usize = 8; // tiny blob for tests

fn blob(tag: u8) -> Vec<u8> {
    vec![tag; B]
}

fn residency(slots: usize) -> Residency<VecSlotArena, MemSwapStore> {
    Residency::new(VecSlotArena::new(B, slots), MemSwapStore::new(B)).unwrap()
}

// ───────────────── O_DIRECT readiness of the shared scratch ─────────────────

/// TRIPWIRE for the O_DIRECT bounce: `DirectSwapFile` checks `ptr & 0xfff == 0`
/// and, when it fails, stages the WHOLE record through an internal aligned
/// buffer — an extra full-record memcpy on every disk write AND every fault-in.
/// The residency's scratch used to be a plain `Vec`, which glibc essentially
/// never returns page-aligned, so the disk tier silently paid double. Nothing
/// observable changes when this regresses; only bandwidth does.
#[test]
fn scratch_is_page_aligned() {
    let r = residency(2);
    assert_eq!(
        r.scratch_addr() & 0xfff,
        0,
        "residency scratch must be 4 KiB-aligned or every O_DIRECT record bounces"
    );
    // Also true for a record-sized (non-tiny) blob, the production shape.
    let big = Residency::new(VecSlotArena::new(4096, 2), MemSwapStore::new(4096)).unwrap();
    assert_eq!(big.scratch_addr() & 0xfff, 0);
}

/// Counts `write_record` calls so a test can prove WHEN the disk tier is
/// actually touched (a real file's size would answer the same question, but
/// tmpfs CI cannot open O_DIRECT).
struct SpySwap {
    inner: MemSwapStore,
    writes: Arc<AtomicUsize>,
}
impl SwapStore for SpySwap {
    fn record_bytes(&self) -> usize {
        self.inner.record_bytes()
    }
    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        self.inner.write_record(disk_slot, bytes)
    }
    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()> {
        self.inner.read_record(disk_slot, out)
    }
    fn discard_record(&mut self, disk_slot: usize) {
        self.inner.discard_record(disk_slot)
    }
}

/// The answer to "8 successful puts but the swap file is still 0 bytes": the
/// disk tier is written ONLY when the hot arena has no free slot. With N slots,
/// the first N distinct keys are absorbed in RAM and write NOTHING; the first
/// disk write is put N+1. Re-putting the SAME key never writes either.
#[test]
fn disk_write_lands_only_when_hot_arena_full() {
    let writes = Arc::new(AtomicUsize::new(0));
    let spy = SpySwap {
        inner: MemSwapStore::new(B),
        writes: Arc::clone(&writes),
    };
    let mut r = Residency::new(VecSlotArena::new(B, 1), spy).unwrap();

    r.put_blob(1, &blob(1)).unwrap();
    assert_eq!(
        writes.load(Ordering::SeqCst),
        0,
        "the 1-slot hot arena absorbs the first key — a 0-byte swap file here is EXPECTED"
    );
    r.put_blob(1, &blob(9)).unwrap();
    assert_eq!(
        writes.load(Ordering::SeqCst),
        0,
        "re-PUT of a live key overwrites in place, it does not spill"
    );
    r.put_blob(2, &blob(2)).unwrap();
    assert_eq!(
        writes.load(Ordering::SeqCst),
        1,
        "the FIRST key beyond the arena is what finally writes disk record 0"
    );
    assert_eq!(r.stats().spills_to_disk, 1);
    assert_eq!(r.disk_high_water(), 1, "exactly one record exists on disk");
}
