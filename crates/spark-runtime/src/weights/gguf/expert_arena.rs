// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The expert cache's slots in DEVICE memory, filled through a page-locked
//! staging ring.
//!
//! The GB10's expert GEMV reads the same bytes at 215 GB/s from `cudaMalloc`
//! memory and at 176 GB/s from the page-locked arena the GPU reads in place
//! (`kquant_mmvq_q2_k_experts_w8` at the real shapes, 09-19: same L2 hit
//! rate, same occupancy, same long-scoreboard stall share; only the path
//! differs). A decode step reads 40 layers x 6 experts = 2.93 GB of them, so
//! the address the bytes live at is worth 3.2 ms of a 44 ms step.
//!
//! The CPU cannot write `cudaMalloc` memory on GB10 (SIGSEGV), so a miss is
//! read by the reader threads into a slot of a small page-locked ring and
//! then copied device-to-device-speed by the copy engine (`cuMemcpyHtoDAsync`,
//! 217 us for one 12.22 MiB expert) INTO its device slot on the compute
//! stream, ahead of the token's launches: stream order is the whole
//! synchronisation, no launch can read a slot before its copy has landed,
//! and a victim's new copy is enqueued after every launch that read the old
//! bytes. The ring is two halves used alternately; a half is rewritten only
//! after the event recorded behind its last copies has completed, so the
//! copy engine never reads a ring slot the readers are refilling. A prefill
//! layer's hundreds of misses go through in half-ring batches with the
//! previous half's copies in flight.
//!
//! [`ExpertArena`] is the owner's handle for either kind of arena;
//! [`ExpertLru::fetch_many_on`] is the one fetch that works on every kind.

use anyhow::{Result, bail, ensure};

use super::expert_lru::{ExpertLru, ExpertSlot, PinnedArena, SlotPtr};
use super::expert_stream::{ExpertSource, SlotLayout};
use crate::gpu::{DevicePtr, GpuBackend};

/// One device arena plus its page-locked staging ring.
pub struct DeviceArena {
    dev: DevicePtr,
    bytes: usize,
    ring: *mut u8,
    ring_bytes: usize,
    events: [u64; 2],
}

// SAFETY: raw addresses of memory the owner keeps alive; access is serialised
// by the `ExpertLru` that uses them.
unsafe impl Send for DeviceArena {}
unsafe impl Sync for DeviceArena {}

/// The staging state an [`ExpertLru`] over a [`DeviceArena`] carries.
pub struct Staging {
    pub(super) ring: *mut u8,
    /// Slots in the ring (even, >= 2); a half is `ring_slots / 2`.
    pub(super) ring_slots: usize,
    /// One per half, recorded on the compute stream behind the half's copies.
    pub(super) events: [u64; 2],
    /// The event was recorded since the half was last waited for.
    pub(super) armed: [bool; 2],
    /// The half the next batch of misses goes through.
    pub(super) half: usize,
}

// SAFETY: as for `DeviceArena`.
unsafe impl Send for Staging {}

impl DeviceArena {
    /// `bytes` of device memory (off the allocation ledger) and a ring of
    /// `ring_slots` page-locked expert slots (rounded up to an even count).
    pub fn alloc(
        gpu: &dyn GpuBackend,
        bytes: usize,
        layout: SlotLayout,
        ring_slots: usize,
    ) -> Result<(Self, ExpertLru)> {
        let ring_slots = ring_slots.max(2).next_multiple_of(2);
        let dev = gpu.alloc_arena(bytes)?;
        let ring_bytes = ring_slots * layout.bytes;
        let ring = match gpu.alloc_host_pinned(ring_bytes) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.free_arena(dev);
                return Err(e);
            }
        };
        let events = [gpu.create_event()?, gpu.create_event()?];
        let mut lru = ExpertLru::new(std::ptr::null_mut(), dev, bytes, layout)?;
        lru.staging = Some(Staging {
            ring,
            ring_slots,
            events,
            armed: [false; 2],
            half: 0,
        });
        Ok((
            DeviceArena {
                dev,
                bytes,
                ring,
                ring_bytes,
                events,
            },
            lru,
        ))
    }

    pub fn dev(&self) -> DevicePtr {
        self.dev
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for e in self.events {
            gpu.destroy_event(e)?;
        }
        gpu.free_host_pinned(self.ring, self.ring_bytes)?;
        gpu.free_arena(self.dev)
    }
}

/// Either arena, as the runtime that keeps it alive holds it.
pub enum ExpertArena {
    /// Page-locked host memory the GPU reads in place.
    Pinned(PinnedArena),
    /// Device memory behind the staging ring.
    Device(DeviceArena),
}

impl ExpertArena {
    /// The arena of `bytes` and the cache over it: device slots with a
    /// `ring_slots` staging ring when `device`, the page-locked arena otherwise.
    pub fn alloc(
        gpu: &dyn GpuBackend,
        bytes: usize,
        layout: SlotLayout,
        device: bool,
        ring_slots: usize,
    ) -> Result<(Self, ExpertLru)> {
        if device {
            let (a, lru) = DeviceArena::alloc(gpu, bytes, layout, ring_slots)?;
            return Ok((ExpertArena::Device(a), lru));
        }
        let a = PinnedArena::alloc(gpu, bytes)?;
        let lru = ExpertLru::new(a.host(), a.dev(), a.bytes(), layout)?;
        Ok((ExpertArena::Pinned(a), lru))
    }

    pub fn bytes(&self) -> usize {
        match self {
            ExpertArena::Pinned(a) => a.bytes(),
            ExpertArena::Device(a) => a.bytes(),
        }
    }

    /// One line for the load log.
    pub fn describe(&self) -> String {
        let gib = self.bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
        match self {
            ExpertArena::Pinned(_) => format!("{gib:.1} GiB page-locked, read in place"),
            ExpertArena::Device(a) => format!(
                "{gib:.1} GiB device memory behind a {:.0} MiB page-locked staging ring",
                a.ring_bytes as f64 / 1048576.0
            ),
        }
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        match self {
            ExpertArena::Pinned(a) => a.free(gpu),
            ExpertArena::Device(a) => a.free(gpu),
        }
    }
}

/// One byte range of a missed expert as a reader thread sees it: `(slot,
/// key, ring slot within the half, byte offset, byte length)`.
type StagedRange = (u32, (u32, u32), usize, usize, usize);

impl ExpertLru {
    /// The slots are device memory behind a staging ring.
    pub fn is_device(&self) -> bool {
        self.staging.is_some()
    }

    /// One token's gather on whichever path this cache was built for: the
    /// staging ring (device slots), the reader pool (`predicted` experts
    /// started in the background), or scoped reader threads. The copies of a
    /// staged fetch are enqueued on `stream` ahead of the caller's launches.
    pub fn fetch_many_on<S: ExpertSource + ?Sized>(
        &mut self,
        gpu: &dyn GpuBackend,
        stream: u64,
        src: &S,
        keys: &[(u32, u32)],
        predicted: &[(u32, u32)],
        threads: usize,
    ) -> Result<Vec<ExpertSlot>> {
        if self.staging.is_some() {
            self.fetch_many_staged(gpu, stream, src, keys, threads)
        } else if self.has_pool() {
            self.fetch_many_prefetching(keys, predicted)
        } else {
            self.fetch_many(src, keys, threads)
        }
    }

    /// The staged gather: misses read into the ring by up to `threads`
    /// scoped threads (each miss cut into byte ranges as `fetch_many` does),
    /// then one H2D copy per miss into its device slot on `stream`, in
    /// half-ring batches.
    fn fetch_many_staged<S: ExpertSource + ?Sized>(
        &mut self,
        gpu: &dyn GpuBackend,
        stream: u64,
        src: &S,
        keys: &[(u32, u32)],
        threads: usize,
    ) -> Result<Vec<ExpertSlot>> {
        let (ring, ring_slots, events) = match &self.staging {
            Some(s) => (s.ring, s.ring_slots, s.events),
            None => bail!("fetch_many_staged on a page-locked expert cache"),
        };
        let evictions0 = self.stats.evictions;
        let mut out: Vec<u32> = Vec::with_capacity(keys.len());
        let mut misses: Vec<(u32, (u32, u32))> = Vec::new();
        for &key in keys {
            if let Some(&i) = self.map.get(&key) {
                self.touch(i);
                self.stats.hits += 1;
                out.push(i);
                continue;
            }
            let i = self.assign(key)?;
            misses.push((i, key));
            out.push(i);
        }
        let bytes = self.layout().bytes;
        let half_slots = ring_slots / 2;
        let threads = threads.max(1);
        let mut failures: Vec<(u32, anyhow::Error)> = Vec::new();
        for batch in misses.chunks(half_slots) {
            let h = {
                let st = self.staging.as_mut().expect("staging");
                let h = st.half;
                st.half ^= 1;
                h
            };
            // the half's previous copies must have landed before its ring
            // slots are rewritten
            if self.staging.as_ref().is_some_and(|s| s.armed[h]) {
                gpu.event_synchronize(events[h])?;
                self.staging.as_mut().expect("staging").armed[h] = false;
            }
            // SAFETY: `h * half_slots * bytes + half_slots * bytes <= ring bytes`.
            let half_base = SlotPtr(unsafe { ring.add(h * half_slots * bytes) });
            // the reads: every miss cut into `parts` ranges spread over the
            // threads (>= 1 MiB a range), exactly as `fetch_many` does
            let parts = (threads / batch.len())
                .clamp(1, threads)
                .min((bytes / (1 << 20)).max(1));
            let per_part = bytes.div_ceil(parts);
            let work: Vec<StagedRange> = batch
                .iter()
                .enumerate()
                .flat_map(|(j, &(i, k))| {
                    (0..parts).map(move |p| {
                        let off = p * per_part;
                        (i, k, j, off, per_part.min(bytes - off))
                    })
                })
                .filter(|w| w.4 > 0)
                .collect();
            let per = work.len().div_ceil(threads.min(work.len()));
            let results: Vec<Vec<(u32, anyhow::Error)>> = std::thread::scope(|s| {
                let handles: Vec<_> = work
                    .chunks(per)
                    .map(|chunk| {
                        s.spawn(move || {
                            let mut errs = Vec::new();
                            for &(i, (l, e), j, off, len) in chunk {
                                // SAFETY: ring slot `j` of this half is this
                                // miss's alone and its ranges are disjoint.
                                let dst = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        half_base.get().add(j * bytes + off),
                                        len,
                                    )
                                };
                                if let Err(err) = src.read_expert_range(l, e, off, dst) {
                                    errs.push((i, err));
                                }
                            }
                            errs
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("expert reader thread panicked"))
                    .collect()
            });
            let mut seen = std::collections::HashSet::new();
            let failed_now: Vec<(u32, anyhow::Error)> = results
                .into_iter()
                .flatten()
                .filter(|(i, _)| seen.insert(*i))
                .collect();
            // the copies, one per miss that read whole, stream-ordered ahead
            // of this token's launches; the ring stays valid until the event
            for (j, &(i, _)) in batch.iter().enumerate() {
                if seen.contains(&i) {
                    continue;
                }
                // SAFETY: as above; the readers have joined.
                let src_bytes =
                    unsafe { std::slice::from_raw_parts(half_base.get().add(j * bytes), bytes) };
                let dst = DevicePtr(self.slot(i).gate.0 - self.layout().gate_off as u64);
                gpu.copy_h2d_async_retained(src_bytes, dst, stream)?;
            }
            gpu.record_event(events[h], stream)?;
            self.staging.as_mut().expect("staging").armed[h] = true;
            failures.extend(failed_now);
        }
        let n_ok = misses.len() - failures.len();
        self.stats.misses += n_ok as u64;
        self.stats.bytes_read += (n_ok * bytes) as u64;
        if self.trace.is_some() {
            let ev = self.stats.evictions - evictions0;
            self.trace_line(keys, misses.len(), ev, 0);
        }
        if let Some((_, first)) = failures.first() {
            let msg = format!(
                "{} of {} expert reads failed; first: {first:#}",
                failures.len(),
                misses.len()
            );
            for (i, _) in &failures {
                self.unmap(*i);
            }
            bail!(msg);
        }
        ensure!(out.len() == keys.len(), "staged fetch lost a key");
        Ok(out.into_iter().map(|i| self.slot(i)).collect())
    }
}
