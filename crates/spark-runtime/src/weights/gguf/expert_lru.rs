// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The device-visible expert cache: fixed-size slots in one page-locked host
//! arena, addressed by the GPU through its `cuMemHostGetDevicePointer` alias
//! (on GB10 the two addresses name the same unified memory), filled by `pread`
//! and evicted least-recently-used.
//!
//! A slot holds one routed expert in [`SlotLayout`](super::expert_stream::SlotLayout)
//! order (`gate | up | down`, 12.22 MiB on the Q2_K checkpoint). A token
//! touches 6 x 40 = 240 of them; [`ExpertLru::begin_token`] opens an epoch and
//! every slot fetched inside it is pinned against eviction until the next
//! epoch, so one token's working set can never evict itself. Misses inside
//! [`ExpertLru::fetch_many`] are read on a scoped thread pool (positional reads
//! share no file offset), which is what turns 240 x 12 MiB of NVMe traffic into
//! one parallel gather instead of a serial walk.
//!
//! The arena is NOT owned here: the caller allocates it (see [`PinnedArena`])
//! and keeps it alive for the cache's lifetime, and the unit tests back it with
//! a plain `Vec<u8>`. The cache never reads or writes the arena itself except
//! through [`ExpertSource::read_expert`](super::expert_stream::ExpertSource).

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail, ensure};

use super::expert_prefetch::{ReaderPool, Ticket};
use super::expert_stream::{ExpertSource, SlotLayout};
use crate::gpu::{DevicePtr, GpuBackend};

/// One page-locked host arena with its device alias.
pub struct PinnedArena {
    host: *mut u8,
    dev: DevicePtr,
    bytes: usize,
}

// SAFETY: a page-locked region with no thread affinity; the owner serialises access.
unsafe impl Send for PinnedArena {}
unsafe impl Sync for PinnedArena {}

impl PinnedArena {
    /// `alloc_host_pinned` zero-fills, so a large arena costs one memset at
    /// load. The device alias is what the kernels are handed.
    pub fn alloc(gpu: &dyn GpuBackend, bytes: usize) -> Result<Self> {
        let host = gpu.alloc_host_pinned(bytes)?;
        let dev = match gpu.host_ptr_to_device(host) {
            Ok(d) => d,
            Err(e) => {
                let _ = gpu.free_host_pinned(host, bytes);
                return Err(e);
            }
        };
        Ok(PinnedArena { host, dev, bytes })
    }

    pub fn host(&self) -> *mut u8 {
        self.host
    }

    pub fn dev(&self) -> DevicePtr {
        self.dev
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free_host_pinned(self.host, self.bytes)
    }
}

/// Where one cached expert's three projections are, as the GPU sees them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertSlot {
    pub gate: DevicePtr,
    pub up: DevicePtr,
    pub down: DevicePtr,
    /// Host address of the slot (for oracles and CPU-side reads).
    pub host: *const u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LruStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub bytes_read: u64,
    /// Hits on a slot whose read was still in flight (a prediction that
    /// arrived in time to be waited on rather than missed).
    pub waited: u64,
    /// Experts read on a prediction, and of those, never asked for.
    pub prefetched: u64,
    pub prefetch_unused: u64,
}

const NONE: u32 = u32::MAX;

pub(super) struct SlotMeta {
    pub(super) key: Option<(u32, u32)>,
    prev: u32,
    next: u32,
    /// Epoch of the last fetch; equal to the current epoch = pinned.
    pub(super) epoch: u64,
    /// The read filling this slot, while it is in flight (pool only).
    pub(super) ticket: Option<Arc<Ticket>>,
    /// Filled by a prediction and not yet asked for by a real fetch.
    pub(super) speculative: bool,
}

/// A raw slot address that may cross into a scoped reader thread: every miss
/// in one `fetch_many` owns a distinct slot, so the regions never overlap.
#[derive(Clone, Copy)]
pub(super) struct SlotPtr(pub(super) *mut u8);

impl SlotPtr {
    /// The address, read through `self` so a closure captures the `Send`
    /// wrapper and not the raw pointer.
    pub(super) fn get(self) -> *mut u8 {
        self.0
    }
}
// SAFETY: see above; the pointer is only dereferenced as a disjoint `&mut [u8]`
// by the one thread whose chunk names it (or, split into byte ranges, by
// several threads on disjoint ranges of it).
unsafe impl Send for SlotPtr {}
unsafe impl Sync for SlotPtr {}

/// One byte range of a missed expert's slot, as handed to a reader thread by
/// `fetch_many`: `(slot index, (layer, expert) key, slot host pointer, byte
/// offset of the range within the slot, byte length of the range)`.
type MissRange = (u32, (u32, u32), SlotPtr, usize, usize);

pub struct ExpertLru {
    host: *mut u8,
    dev: u64,
    layout: SlotLayout,
    n_slots: usize,
    pub(super) meta: Vec<SlotMeta>,
    pub(super) map: HashMap<(u32, u32), u32>,
    /// Most recently used.
    head: u32,
    /// Least recently used.
    tail: u32,
    pub(super) epoch: u64,
    pub(super) stats: LruStats,
    /// The persistent reader pool and the source it reads from (see
    /// `set_pool` in `expert_lru_pool.rs`); without it, misses are read on
    /// scoped threads inside `fetch_many`.
    pub(super) pool: Option<(Arc<dyn ExpertSource + Send + Sync>, ReaderPool)>,
    /// Slots with a ticket that may still be in flight.
    pub(super) in_flight: Vec<u32>,
    /// `ATLAS_DS41_ROUTE_TRACE`: one line per `fetch_many` (see `set_trace`).
    pub(super) trace: Option<std::io::BufWriter<std::fs::File>>,
    pub(super) t0: std::time::Instant,
}

// SAFETY: the raw arena pointers are addresses into memory the caller owns
// and keeps alive; the cache holds no thread-affine state.
unsafe impl Send for ExpertLru {}

impl ExpertLru {
    /// `bytes / layout.bytes` slots over the arena at `host` (device alias
    /// `dev`). Every slot starts empty and at the LRU end.
    pub fn new(host: *mut u8, dev: DevicePtr, bytes: usize, layout: SlotLayout) -> Result<Self> {
        ensure!(layout.bytes > 0, "expert slot layout has zero bytes");
        let n_slots = bytes / layout.bytes;
        ensure!(
            n_slots >= 1,
            "arena of {bytes} bytes holds no {}-byte expert slot",
            layout.bytes
        );
        ensure!(n_slots < NONE as usize, "too many slots");
        let mut meta = Vec::with_capacity(n_slots);
        for i in 0..n_slots as u32 {
            meta.push(SlotMeta {
                key: None,
                prev: if i == 0 { NONE } else { i - 1 },
                next: if i as usize + 1 == n_slots {
                    NONE
                } else {
                    i + 1
                },
                epoch: 0,
                ticket: None,
                speculative: false,
            });
        }
        Ok(ExpertLru {
            host,
            dev: dev.0,
            layout,
            n_slots,
            meta,
            map: HashMap::new(),
            head: 0,
            tail: n_slots as u32 - 1,
            epoch: 1,
            stats: LruStats::default(),
            pool: None,
            in_flight: Vec::new(),
            trace: None,
            t0: std::time::Instant::now(),
        })
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    pub fn layout(&self) -> SlotLayout {
        self.layout
    }

    /// Experts currently cached.
    pub fn resident(&self) -> usize {
        self.map.len()
    }

    pub fn contains(&self, layer: u32, expert: u32) -> bool {
        self.map.contains_key(&(layer, expert))
    }

    pub fn stats(&self) -> LruStats {
        self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = LruStats::default();
    }

    /// Open a new epoch: slots fetched from now on are pinned until the next
    /// `begin_token`; everything fetched earlier becomes evictable again.
    pub fn begin_token(&mut self) {
        self.epoch += 1;
        if self.pool.is_some() {
            self.reap();
        }
    }

    pub fn slot(&self, i: u32) -> ExpertSlot {
        let base = i as usize * self.layout.bytes;
        let l = self.layout;
        ExpertSlot {
            gate: DevicePtr(self.dev + (base + l.gate_off) as u64),
            up: DevicePtr(self.dev + (base + l.up_off) as u64),
            down: DevicePtr(self.dev + (base + l.down_off) as u64),
            // SAFETY: `base + bytes <= n_slots * bytes <= arena bytes`.
            host: unsafe { self.host.add(base) as *const u8 },
        }
    }

    pub(super) fn slot_ptr(&self, i: u32) -> SlotPtr {
        // SAFETY: as in `slot`.
        SlotPtr(unsafe { self.host.add(i as usize * self.layout.bytes) })
    }

    fn unlink(&mut self, i: u32) {
        let (p, n) = (self.meta[i as usize].prev, self.meta[i as usize].next);
        if p == NONE {
            self.head = n;
        } else {
            self.meta[p as usize].next = n;
        }
        if n == NONE {
            self.tail = p;
        } else {
            self.meta[n as usize].prev = p;
        }
    }

    fn push_front(&mut self, i: u32) {
        let m = &mut self.meta[i as usize];
        m.prev = NONE;
        m.next = self.head;
        if self.head != NONE {
            self.meta[self.head as usize].prev = i;
        }
        self.head = i;
        if self.tail == NONE {
            self.tail = i;
        }
    }

    pub(super) fn touch(&mut self, i: u32) {
        self.unlink(i);
        self.push_front(i);
        self.meta[i as usize].epoch = self.epoch;
    }

    /// Move `i` to the LRU end without unmapping it (a prediction that was
    /// not used: the next victim, but still a hit if it is asked for).
    pub(super) fn demote(&mut self, i: u32) {
        self.unlink(i);
        let m = &mut self.meta[i as usize];
        m.next = NONE;
        m.prev = self.tail;
        if self.tail != NONE {
            self.meta[self.tail as usize].next = i;
        }
        self.tail = i;
        if self.head == NONE {
            self.head = i;
        }
    }

    /// Take the least recently used slot that is not pinned in this epoch,
    /// dropping whatever it held.
    fn take_victim(&mut self) -> Result<u32> {
        let mut i = self.tail;
        while i != NONE {
            // never a slot pinned in this epoch, never one a read is filling
            if self.meta[i as usize].epoch != self.epoch && self.meta[i as usize].ticket.is_none() {
                if let Some(k) = self.meta[i as usize].key.take() {
                    self.map.remove(&k);
                    self.stats.evictions += 1;
                }
                return Ok(i);
            }
            i = self.meta[i as usize].prev;
        }
        bail!(
            "expert cache of {} slots is smaller than one token's working set ({} pinned)",
            self.n_slots,
            self.map.len()
        )
    }

    /// Assign a slot for `key` and map it (the bytes are not read yet).
    pub(super) fn assign(&mut self, key: (u32, u32)) -> Result<u32> {
        let i = self.take_victim()?;
        self.meta[i as usize].key = Some(key);
        self.map.insert(key, i);
        self.touch(i);
        Ok(i)
    }

    pub(super) fn unmap(&mut self, i: u32) {
        if let Some(k) = self.meta[i as usize].key.take() {
            self.map.remove(&k);
        }
        self.meta[i as usize].epoch = 0;
        self.meta[i as usize].ticket = None;
        self.meta[i as usize].speculative = false;
        self.unlink(i);
        // back to the LRU end so it is the next victim
        let m = &mut self.meta[i as usize];
        m.next = NONE;
        m.prev = self.tail;
        if self.tail != NONE {
            self.meta[self.tail as usize].next = i;
        }
        self.tail = i;
        if self.head == NONE {
            self.head = i;
        }
    }

    /// One expert. Returns the slot and whether it was already resident.
    pub fn fetch<S: ExpertSource + ?Sized>(
        &mut self,
        src: &S,
        layer: u32,
        expert: u32,
    ) -> Result<(ExpertSlot, bool)> {
        let key = (layer, expert);
        if let Some(&i) = self.map.get(&key) {
            self.touch(i);
            self.stats.hits += 1;
            return Ok((self.slot(i), true));
        }
        let i = self.assign(key)?;
        let p = self.slot_ptr(i);
        // SAFETY: slot `i` is exactly `layout.bytes` long and mapped to no other key.
        let dst = unsafe { std::slice::from_raw_parts_mut(p.0, self.layout.bytes) };
        if let Err(e) = src.read_expert(layer, expert, dst) {
            self.unmap(i);
            return Err(e);
        }
        self.stats.misses += 1;
        self.stats.bytes_read += self.layout.bytes as u64;
        Ok((self.slot(i), false))
    }

    /// One token's gather: every key resolved to a slot, misses read on up to
    /// `threads` scoped threads. Duplicate keys are one read. Slots are pinned
    /// for the current epoch, so call [`Self::begin_token`] first.
    pub fn fetch_many<S: ExpertSource + ?Sized>(
        &mut self,
        src: &S,
        keys: &[(u32, u32)],
        threads: usize,
    ) -> Result<Vec<ExpertSlot>> {
        if self.pool.is_some() {
            return self.fetch_many_prefetching(keys, &[]);
        }
        let mut out = Vec::with_capacity(keys.len());
        let mut misses: Vec<(u32, (u32, u32))> = Vec::new();
        let evictions0 = self.stats.evictions;
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
        let bytes = self.layout.bytes;
        let mut failures: Vec<(u32, anyhow::Error)> = Vec::new();
        if threads <= 1 {
            for &(i, (l, e)) in &misses {
                let p = self.slot_ptr(i);
                // SAFETY: as in `fetch`.
                let dst = unsafe { std::slice::from_raw_parts_mut(p.0, bytes) };
                if let Err(err) = src.read_expert(l, e, dst) {
                    failures.push((i, err));
                }
            }
        } else if !misses.is_empty() {
            // A layer's decode step misses one or two experts: one thread
            // per expert leaves the other readers idle, so every miss is
            // cut into `parts` byte ranges and the ranges are spread over
            // the threads (>= 1 MiB a range, up to `threads` per expert).
            let parts = (threads / misses.len())
                .clamp(1, threads)
                .min((bytes / (1 << 20)).max(1));
            let per_part = bytes.div_ceil(parts);
            let work: Vec<MissRange> = misses
                .iter()
                .flat_map(|&(i, k)| {
                    let p = self.slot_ptr(i);
                    (0..parts).map(move |j| {
                        let off = j * per_part;
                        (i, k, p, off, per_part.min(bytes - off))
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
                            for (i, (l, e), p, off, len) in chunk {
                                // SAFETY: each miss owns a distinct slot and
                                // the ranges of one slot are disjoint.
                                let dst =
                                    unsafe { std::slice::from_raw_parts_mut(p.0.add(*off), *len) };
                                if let Err(err) = src.read_expert_range(*l, *e, *off, dst) {
                                    errs.push((*i, err));
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
            // one failure per slot: a slot whose ranges failed twice is
            // unmapped once
            let mut seen = std::collections::HashSet::new();
            failures = results
                .into_iter()
                .flatten()
                .filter(|(i, _)| seen.insert(*i))
                .collect();
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
        Ok(out.into_iter().map(|i| self.slot(i)).collect())
    }
}
