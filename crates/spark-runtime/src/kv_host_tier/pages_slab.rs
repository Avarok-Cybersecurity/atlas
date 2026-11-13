// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the window desk's host memory (port plan lane L36): a [`HostArena`] of
//! fixed-size page slots carved from large slabs, each slab one anonymous mapping page-locked
//! through `GpuBackend::host_register` (so page copies are truly asynchronous DMA).
//!
//! Every page of the desk is one KV block, so a fixed slot size is exact and allocation is a pop
//! from a free list. The arena cannot reach the backend from `HostArena::alloc` (the trait takes
//! no backend), so it grows only through [`SlabArena::reserve`], which the desk calls with its
//! backend before a demote batch; `alloc` serves what is reserved and answers `None` when full,
//! which the page API reports as a refused page (the block stays on the GPU).
//!
//! A slab whose registration was refused (a WSL2 or `ulimit -l` cap) still serves, as pageable
//! memory (correct, slower), with one log line. Teardown: [`SlabArena::release_all`] unregisters
//! and unmaps every slab once no run is out. A slab still registered when the arena is dropped is
//! leaked on purpose: unmapping it would leave its address recorded as page-locked in
//! `pinned_hosts`, and a later mapping at that address would inherit the record.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use parking_lot::Mutex;

use crate::gpu::GpuBackend;
use crate::kv_host_tier::{HostArena, HostRun};

/// Slot sizes are rounded up to this many bytes (keeps every page 256-byte aligned).
const SLOT_ALIGN: usize = 256;

struct Slab {
    map: memmap2::MmapMut,
    base: usize,
    registered: bool,
}

struct SlabInner {
    slabs: Vec<Slab>,
    /// Free `(slab, slot)` pairs.
    free: Vec<(usize, usize)>,
    /// Run id -> `(slab, slot)`.
    live: HashMap<u64, (usize, usize)>,
    next_id: u64,
    warned_register: bool,
}

/// Fixed-slot host arena over registered slabs (see the module doc).
pub struct SlabArena {
    slot_bytes: usize,
    slab_slots: usize,
    budget_bytes: u64,
    inner: Mutex<SlabInner>,
}

impl SlabArena {
    /// Slots of at least `slot_bytes` (rounded up to 256), slabs of about `slab_bytes` (at least
    /// one slot), at most `budget_bytes` reserved in all.
    pub fn new(slot_bytes: usize, slab_bytes: usize, budget_bytes: u64) -> Self {
        let slot = slot_bytes.max(1).div_ceil(SLOT_ALIGN) * SLOT_ALIGN;
        Self {
            slot_bytes: slot,
            slab_slots: (slab_bytes / slot).max(1),
            budget_bytes,
            inner: Mutex::new(SlabInner {
                slabs: Vec::new(),
                free: Vec::new(),
                live: HashMap::new(),
                next_id: 1,
                warned_register: false,
            }),
        }
    }

    /// Bytes of one slot (what `alloc` hands out).
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Bytes reserved in slabs now.
    pub fn reserved_bytes(&self) -> u64 {
        let inner = self.inner.lock();
        (inner.slabs.len() * self.slab_slots * self.slot_bytes) as u64
    }

    /// Slots free now.
    pub fn free_slots(&self) -> usize {
        self.inner.lock().free.len()
    }

    /// Runs handed out and not yet freed.
    pub fn live_runs(&self) -> usize {
        self.inner.lock().live.len()
    }

    fn slab_bytes(&self) -> usize {
        self.slab_slots * self.slot_bytes
    }

    /// Grow until at least `slots` slots are free or the budget is reached. Returns the free
    /// slots. A failed mapping stops the growth with one log line; nothing already served moves.
    pub fn reserve(&self, gpu: &dyn GpuBackend, slots: usize) -> usize {
        let mut inner = self.inner.lock();
        while inner.free.len() < slots {
            let have = (inner.slabs.len() * self.slab_bytes()) as u64;
            if have + self.slab_bytes() as u64 > self.budget_bytes {
                break;
            }
            match self.map_slab(gpu, &mut inner) {
                Ok(()) => {}
                Err(e) => {
                    tracing::warn!("kv window desk: host slab reservation stopped: {e:#}");
                    break;
                }
            }
        }
        inner.free.len()
    }

    fn map_slab(&self, gpu: &dyn GpuBackend, inner: &mut SlabInner) -> Result<()> {
        let bytes = self.slab_bytes();
        let mut map = memmap2::MmapMut::map_anon(bytes)
            .with_context(|| format!("mapping a {bytes}-byte host slab"))?;
        let base = map.as_mut_ptr() as usize;
        let registered = match gpu.host_register(base, bytes) {
            Ok(()) => true,
            Err(e) => {
                if !inner.warned_register {
                    inner.warned_register = true;
                    tracing::warn!(
                        "kv window desk: page-locking a {bytes}-byte host slab failed ({e:#}); \
                         serving pageable memory (page copies stay correct but bounce through the driver)"
                    );
                }
                false
            }
        };
        let idx = inner.slabs.len();
        inner.slabs.push(Slab {
            map,
            base,
            registered,
        });
        // Hand out low slots first (pop from the end).
        inner
            .free
            .extend((0..self.slab_slots).rev().map(|s| (idx, s)));
        Ok(())
    }

    /// Unregister and unmap every slab. Refused while a run is out (a copy may still use it).
    pub fn release_all(&self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut inner = self.inner.lock();
        if !inner.live.is_empty() {
            bail!(
                "kv window desk: {} host page(s) are still out; the slabs stay mapped",
                inner.live.len()
            );
        }
        for slab in inner.slabs.drain(..) {
            let unlocked = !slab.registered
                || match gpu.host_unregister(slab.base) {
                    Ok(()) => true,
                    Err(e) => {
                        tracing::warn!(
                            "kv window desk: unregistering a host slab failed ({e:#}); it stays mapped"
                        );
                        false
                    }
                };
            if unlocked {
                drop(slab.map);
            } else {
                std::mem::forget(slab.map);
            }
        }
        inner.free.clear();
        Ok(())
    }
}

impl HostArena for SlabArena {
    fn alloc(&self, bytes: usize) -> Option<HostRun> {
        if bytes == 0 || bytes > self.slot_bytes {
            return None;
        }
        let mut inner = self.inner.lock();
        let (slab, slot) = inner.free.pop()?;
        let id = inner.next_id;
        inner.next_id += 1;
        inner.live.insert(id, (slab, slot));
        Some(HostRun {
            ptr: inner.slabs[slab].base + slot * self.slot_bytes,
            bytes: self.slot_bytes,
            id,
        })
    }

    fn free(&self, run: HostRun) {
        let mut inner = self.inner.lock();
        match inner.live.remove(&run.id) {
            Some(at) => inner.free.push(at),
            None => tracing::warn!("kv window desk: freeing unknown host page {}", run.id),
        }
    }
}

impl Drop for SlabArena {
    fn drop(&mut self) {
        let inner = self.inner.get_mut();
        for slab in inner.slabs.drain(..) {
            if slab.registered {
                // See the module doc: never unmap an address still recorded as page-locked.
                std::mem::forget(slab.map);
            }
        }
    }
}
