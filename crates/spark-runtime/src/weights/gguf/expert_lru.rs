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

use anyhow::{Result, bail, ensure};

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
}

const NONE: u32 = u32::MAX;

struct SlotMeta {
    key: Option<(u32, u32)>,
    prev: u32,
    next: u32,
    /// Epoch of the last fetch; equal to the current epoch = pinned.
    epoch: u64,
}

/// A raw slot address that may cross into a scoped reader thread: every miss
/// in one `fetch_many` owns a distinct slot, so the regions never overlap.
struct SlotPtr(*mut u8);
// SAFETY: see above; the pointer is only dereferenced as a disjoint `&mut [u8]`
// by the one thread whose chunk names it.
unsafe impl Send for SlotPtr {}
unsafe impl Sync for SlotPtr {}

pub struct ExpertLru {
    host: *mut u8,
    dev: u64,
    layout: SlotLayout,
    n_slots: usize,
    meta: Vec<SlotMeta>,
    map: HashMap<(u32, u32), u32>,
    /// Most recently used.
    head: u32,
    /// Least recently used.
    tail: u32,
    epoch: u64,
    stats: LruStats,
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

    fn slot_ptr(&self, i: u32) -> SlotPtr {
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

    fn touch(&mut self, i: u32) {
        self.unlink(i);
        self.push_front(i);
        self.meta[i as usize].epoch = self.epoch;
    }

    /// Take the least recently used slot that is not pinned in this epoch,
    /// dropping whatever it held.
    fn take_victim(&mut self) -> Result<u32> {
        let mut i = self.tail;
        while i != NONE {
            if self.meta[i as usize].epoch != self.epoch {
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
    fn assign(&mut self, key: (u32, u32)) -> Result<u32> {
        let i = self.take_victim()?;
        self.meta[i as usize].key = Some(key);
        self.map.insert(key, i);
        self.touch(i);
        Ok(i)
    }

    fn unmap(&mut self, i: u32) {
        if let Some(k) = self.meta[i as usize].key.take() {
            self.map.remove(&k);
        }
        self.meta[i as usize].epoch = 0;
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
        let mut out = Vec::with_capacity(keys.len());
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
        let bytes = self.layout.bytes;
        let mut failures: Vec<(u32, anyhow::Error)> = Vec::new();
        if misses.len() <= 1 || threads <= 1 {
            for &(i, (l, e)) in &misses {
                let p = self.slot_ptr(i);
                // SAFETY: as in `fetch`.
                let dst = unsafe { std::slice::from_raw_parts_mut(p.0, bytes) };
                if let Err(err) = src.read_expert(l, e, dst) {
                    failures.push((i, err));
                }
            }
        } else {
            let work: Vec<(u32, (u32, u32), SlotPtr)> = misses
                .iter()
                .map(|&(i, k)| (i, k, self.slot_ptr(i)))
                .collect();
            let per = work.len().div_ceil(threads.min(work.len()));
            let results: Vec<Vec<(u32, anyhow::Error)>> = std::thread::scope(|s| {
                let handles: Vec<_> = work
                    .chunks(per)
                    .map(|chunk| {
                        s.spawn(move || {
                            let mut errs = Vec::new();
                            for (i, (l, e), p) in chunk {
                                // SAFETY: each miss owns a distinct slot; regions are disjoint.
                                let dst = unsafe { std::slice::from_raw_parts_mut(p.0, bytes) };
                                if let Err(err) = src.read_expert(*l, *e, dst) {
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
            failures = results.into_iter().flatten().collect();
        }
        let n_ok = misses.len() - failures.len();
        self.stats.misses += n_ok as u64;
        self.stats.bytes_read += (n_ok * bytes) as u64;
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
