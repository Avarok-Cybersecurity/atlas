// SPDX-License-Identifier: AGPL-3.0-only

// ─────────────────────────── real peer-mmap arena ───────────────────────────

use anyhow::{Result, bail};

use super::SlotArena;

/// `SlotArena` over the peer's RDMA-registered `mmap` region (a raw base ptr).
/// The peer memcpys between an arena slot and the disk swap on spill/fault; the
/// client one-sided-RDMAs into/out of the same slots. The base VA is stable and
/// registered ONCE per rail — this NEVER re-registers (no MR churn).
///
/// SAFETY: `base` must point at a live mapping of at least `num_slots *
/// slot_bytes` bytes, page-aligned (mmap guarantees this), outliving the arena.
/// (Peer-specific — deliberately NOT lifted into atlas-tier: the lifted crate
/// carries no unsafe raw-pointer arena types.)
pub struct MmapSlotArena {
    base: *mut u8,
    slot_bytes: usize,
    num_slots: usize,
}
unsafe impl Send for MmapSlotArena {}

impl MmapSlotArena {
    /// # Safety
    /// `base` must be a valid, writable mapping of `>= num_slots*slot_bytes`
    /// bytes that outlives this arena.
    pub unsafe fn new(base: *mut u8, slot_bytes: usize, num_slots: usize) -> Self {
        Self {
            base,
            slot_bytes,
            num_slots,
        }
    }
    fn slot_ptr(&self, slot: usize) -> *mut u8 {
        // slot < num_slots enforced by callers (residency free-list).
        unsafe { self.base.add(slot * self.slot_bytes) }
    }
}

impl SlotArena for MmapSlotArena {
    fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    fn num_slots(&self) -> usize {
        self.num_slots
    }
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        if slot >= self.num_slots || out.len() != self.slot_bytes {
            bail!("read_slot({slot}) out of range / size mismatch");
        }
        unsafe {
            std::ptr::copy_nonoverlapping(self.slot_ptr(slot), out.as_mut_ptr(), self.slot_bytes)
        };
        Ok(())
    }
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        if slot >= self.num_slots || bytes.len() != self.slot_bytes {
            bail!("write_slot({slot}) out of range / size mismatch");
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.slot_ptr(slot), self.slot_bytes)
        };
        Ok(())
    }
}

#[cfg(test)]
#[path = "mmap_arena_tests.rs"]
mod tests;
