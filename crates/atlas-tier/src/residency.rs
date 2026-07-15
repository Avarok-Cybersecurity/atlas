// SPDX-License-Identifier: AGPL-3.0-only

//! [`Residency`] — the policy core: the page table over a bounded hot
//! [`SlotArena`] and an unbounded cold [`SwapStore`].

use std::collections::{HashMap, VecDeque};

use anyhow::{Result, bail};

use crate::traits::{SlotArena, SwapStats, SwapStore};

/// Where a key's blob currently lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Loc {
    /// Arena slot handed out for an in-flight PUT; pinned (not evictable) until
    /// `commit`. Holds the caller's about-to-be-written bytes.
    Reserved(usize),
    /// Live in an arena slot (RDMA-readable now).
    Resident(usize),
    /// Spilled to a disk record; a GET faults it back into a slot.
    OnDisk(usize),
}

/// The page table: `key → Loc` over a bounded [`SlotArena`] (hot) backed by an
/// unbounded [`SwapStore`] (cold), with LRU eviction of resident slots to disk.
pub struct Residency<A: SlotArena, S: SwapStore> {
    arena: A,
    swap: S,
    blob_bytes: usize,
    map: HashMap<u64, Loc>,
    /// Free arena slot indices (LIFO reuse).
    free_slots: Vec<usize>,
    /// Resident keys, front = coldest (LRU eviction victim). Reserved keys are
    /// NOT in here (pinned).
    lru: VecDeque<u64>,
    /// On-disk keys, front = coldest — the disk-cap eviction victim. Every
    /// `OnDisk` entry is exactly once in here (a bounded two-level LRU:
    /// RAM `lru` above disk `disk_lru`).
    disk_lru: VecDeque<u64>,
    /// Max simultaneous on-disk records (the disk cap / blob_bytes). 0 =
    /// unbounded. When full, the coldest on-disk snapshot is dropped to make
    /// room — a later GET for it misses and the model recomputes (correct
    /// degradation, keeps the swap file bounded).
    max_disk_slots: usize,
    /// Free disk record indices (reused before growing the high-water mark).
    free_disk: Vec<usize>,
    next_disk: usize,
    /// Reusable scratch for a single blob move (spill/fault), sized once.
    scratch: Vec<u8>,
    /// Read-pins: `key → active reader count`. A GET hands the client an arena
    /// offset it then one-sided-RDMA-READs; the peer drops the residency lock
    /// before that read, so a concurrent allocation on another connection could pick
    /// the slot as an eviction victim and reuse it mid-read (torn restore). A
    /// pinned key is held OUT of `lru` (like a `Reserved` slot) so
    /// `evict_coldest_to_disk` can never choose it. Ref-counted for concurrent
    /// readers of the same key. Invariant: `key ∈ lru ⟺ Resident AND unpinned`.
    read_pins: HashMap<u64, u32>,
    stats: SwapStats,
}

impl<A: SlotArena, S: SwapStore> Residency<A, S> {
    /// Unbounded disk tier (no cap). Prefer [`Residency::new_capped`] in
    /// production.
    pub fn new(arena: A, swap: S) -> Result<Self> {
        Self::new_capped(arena, swap, 0)
    }

    /// `max_disk_slots` bounds the on-disk record count (0 = unbounded). When
    /// full, spilling evicts the coldest on-disk snapshot (dropped → later GET
    /// misses → recompute), keeping the swap file at ≤ `max_disk_slots` records.
    pub fn new_capped(arena: A, swap: S, max_disk_slots: usize) -> Result<Self> {
        let blob_bytes = arena.slot_bytes();
        if blob_bytes == 0 {
            bail!("Residency: slot_bytes must be > 0");
        }
        if swap.record_bytes() != blob_bytes {
            bail!(
                "Residency: arena slot ({}) and swap record ({}) sizes differ",
                blob_bytes,
                swap.record_bytes()
            );
        }
        let n = arena.num_slots();
        if n == 0 {
            bail!("Residency: arena must have >= 1 slot");
        }
        Ok(Self {
            arena,
            swap,
            blob_bytes,
            map: HashMap::new(),
            free_slots: (0..n).rev().collect(),
            lru: VecDeque::new(),
            disk_lru: VecDeque::new(),
            max_disk_slots,
            free_disk: Vec::new(),
            next_disk: 0,
            scratch: vec![0u8; blob_bytes],
            read_pins: HashMap::new(),
            stats: SwapStats::default(),
        })
    }

    pub fn blob_bytes(&self) -> usize {
        self.blob_bytes
    }
    pub fn stats(&self) -> &SwapStats {
        &self.stats
    }
    pub fn resident_count(&self) -> usize {
        self.lru.len()
    }
    pub fn total_keys(&self) -> usize {
        self.map.len()
    }

    /// Direct arena access for the data plane. The peer writes the slots its
    /// clients one-sided-RDMA into/out of; in-process consumers should prefer
    /// [`Residency::put_blob`] / [`Residency::get_blob`].
    pub fn arena(&self) -> &A {
        &self.arena
    }
    pub fn arena_mut(&mut self) -> &mut A {
        &mut self.arena
    }

    /// Byte offset of an arena slot (what the client RDMA-reads/writes).
    pub fn slot_offset(&self, slot: usize) -> u64 {
        (slot as u64) * (self.blob_bytes as u64)
    }

    // ─────────────────────────── control-plane ops ───────────────────────────

    /// PUT step 1 — reserve an arena slot for `key`. Evicts the coldest resident
    /// slot to disk if the arena is full (never rejects). The caller then
    /// RDMA-WRITEs the blob into `slot_offset(slot)` and calls `commit(key)`.
    /// Re-PUT of a live key reuses its current slot (idempotent overwrite).
    pub fn alloc(&mut self, key: u64) -> Result<usize> {
        self.stats.puts += 1;
        // Overwrite-in-place: a key already resident/reserved keeps its slot.
        match self.map.get(&key).copied() {
            Some(Loc::Resident(slot)) => {
                self.lru_remove(key); // pin during the rewrite
                self.map.insert(key, Loc::Reserved(slot));
                return Ok(slot);
            }
            Some(Loc::Reserved(slot)) => return Ok(slot),
            Some(Loc::OnDisk(disk_slot)) => {
                // Rewriting a spilled key: reclaim its disk record, give a slot.
                self.disk_lru_remove(key);
                self.free_disk.push(disk_slot);
                self.swap.discard_record(disk_slot);
            }
            None => {}
        }
        let slot = self.acquire_slot()?;
        self.map.insert(key, Loc::Reserved(slot));
        Ok(slot)
    }

    /// PUT step 2 — the client's RDMA-WRITE into the reserved slot has landed;
    /// mark `key` resident (and hottest in the LRU).
    pub fn commit(&mut self, key: u64) -> Result<()> {
        match self.map.get(&key).copied() {
            Some(Loc::Reserved(slot)) => {
                self.map.insert(key, Loc::Resident(slot));
                // Maintain the `in-lru ⟺ Resident AND unpinned` invariant: if a
                // reader pinned this key while the re-PUT was in flight, leave it
                // out of the LRU — `unpin_read` re-adds it when the last reader
                // releases.
                if !self.read_pins.contains_key(&key) {
                    self.lru.push_back(key); // hottest
                }
                Ok(())
            }
            Some(Loc::Resident(_)) => Ok(()), // already committed — idempotent
            _ => bail!("commit({key:#x}): no reserved slot (alloc not called / evicted)"),
        }
    }

    /// GET — ensure `key` is resident and return its arena slot (offset via
    /// `slot_offset`). Faults from disk into a slot if it was spilled (evicting
    /// a victim to make room). `Ok(None)` = unknown key (caller recomputes).
    pub fn locate(&mut self, key: u64) -> Result<Option<usize>> {
        self.stats.gets += 1;
        match self.map.get(&key).copied() {
            Some(Loc::Resident(slot)) => {
                self.stats.resident_hits += 1;
                // A concurrently-pinned key is held out of `lru`; touching it
                // would re-insert it (breaking the invariant + making it an
                // eviction victim while still being read). The caller pins right
                // after this returns, so unpinned hits get refreshed here and
                // pinned ones stay out.
                if !self.read_pins.contains_key(&key) {
                    self.lru_touch(key);
                }
                Ok(Some(slot))
            }
            Some(Loc::Reserved(slot)) => {
                // A GET racing an uncommitted PUT: the bytes are (being) written
                // by the same caller; hand back the slot.
                Ok(Some(slot))
            }
            Some(Loc::OnDisk(disk_slot)) => {
                // Pin against `acquire_slot`'s spill+make_disk_room evicting THIS
                // key (it is still OnDisk until the fault below completes).
                self.disk_lru_remove(key);
                let slot = match self.acquire_slot() {
                    Ok(s) => s,
                    Err(e) => {
                        self.disk_lru.push_front(key); // un-pin (still on disk)
                        return Err(e);
                    }
                };
                // scratch is exclusive to one move at a time (control loop is
                // single-threaded per connection); read disk → arena slot.
                let mut buf = std::mem::take(&mut self.scratch);
                let r = self.swap.read_record(disk_slot, &mut buf);
                if r.is_ok() {
                    r.and_then(|_| self.arena.write_slot(slot, &buf))?;
                } else {
                    self.scratch = buf;
                    self.free_slots.push(slot);
                    self.disk_lru.push_front(key); // still on disk; re-pin (cold)
                    return r.map(|_| None);
                }
                self.scratch = buf;
                self.free_disk.push(disk_slot);
                self.swap.discard_record(disk_slot);
                self.map.insert(key, Loc::Resident(slot));
                self.lru.push_back(key);
                self.stats.faults_from_disk += 1;
                Ok(Some(slot))
            }
            None => {
                self.stats.get_miss += 1;
                Ok(None)
            }
        }
    }

    /// Drop `key` entirely, reclaiming its arena slot or disk record.
    pub fn remove(&mut self, key: u64) {
        match self.map.remove(&key) {
            Some(Loc::Resident(slot)) | Some(Loc::Reserved(slot)) => {
                self.lru_remove(key);
                self.free_slots.push(slot);
            }
            Some(Loc::OnDisk(disk_slot)) => {
                self.disk_lru_remove(key);
                self.free_disk.push(disk_slot);
                self.swap.discard_record(disk_slot);
            }
            None => {}
        }
    }

    /// Read-pin `key` so its resident slot cannot be chosen as an eviction
    /// victim while a client's one-sided RDMA READ of it is in flight (the
    /// GET→RDMA-read race: the peer replies with the offset and drops the lock
    /// before the client reads, so a concurrent allocation could otherwise
    /// spill+reuse the slot). Ref-counted for concurrent readers; the first pin
    /// removes the key from `lru` (like a `Reserved` slot). No-op unless the key
    /// is currently `Resident`.
    pub fn pin_read(&mut self, key: u64) {
        if !matches!(self.map.get(&key), Some(Loc::Resident(_))) {
            return;
        }
        let n = self.read_pins.get(&key).copied().unwrap_or(0);
        if n == 0 {
            self.lru_remove(key); // exclude from eviction victims while read
        }
        self.read_pins.insert(key, n + 1);
    }

    /// Release one read-pin taken by [`Residency::pin_read`]. When the last
    /// reader releases, the key rejoins `lru` as hottest (it was just read).
    /// No-op if the key holds no pin. Robust to the key having been removed
    /// while pinned: only re-adds to `lru` if still `Resident` and not already
    /// present.
    pub fn unpin_read(&mut self, key: u64) {
        let Some(n) = self.read_pins.get_mut(&key) else {
            return;
        };
        *n -= 1;
        if *n == 0 {
            self.read_pins.remove(&key);
            if matches!(self.map.get(&key), Some(Loc::Resident(_))) && !self.lru.contains(&key) {
                self.lru.push_back(key); // hottest — just accessed
            }
        }
    }

    /// Active read-pin count (test/introspection).
    pub fn read_pin_count(&self, key: u64) -> u32 {
        self.read_pins.get(&key).copied().unwrap_or(0)
    }

    // ───────────────────── in-process one-shot helpers ─────────────────────

    /// One-shot in-process PUT: reserve a slot, copy `bytes` into it, commit.
    /// Same NEVER-reject chain as the two-phase peer path (alloc → spill the
    /// coldest resident → drop the coldest on-disk key when capped). For
    /// consumers whose data plane is a memcpy rather than a client RDMA-WRITE.
    pub fn put_blob(&mut self, key: u64, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.blob_bytes {
            bail!(
                "put_blob({key:#x}): {} bytes, expected {}",
                bytes.len(),
                self.blob_bytes
            );
        }
        let slot = self.alloc(key)?;
        if let Err(e) = self.arena.write_slot(slot, bytes) {
            // Roll back the reservation so the slot is not stranded Reserved
            // (a later GET must miss cleanly, never read a torn slot).
            self.remove(key);
            return Err(e);
        }
        self.commit(key)
    }

    /// One-shot in-process GET: fault `key` in (if spilled) and copy its blob
    /// into `out`. `Ok(false)` = unknown key (caller recomputes).
    pub fn get_blob(&mut self, key: u64, out: &mut [u8]) -> Result<bool> {
        if out.len() != self.blob_bytes {
            bail!(
                "get_blob({key:#x}): {} bytes, expected {}",
                out.len(),
                self.blob_bytes
            );
        }
        match self.locate(key)? {
            Some(slot) => {
                self.arena.read_slot(slot, out)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    // ─────────────────────────── internals ───────────────────────────

    /// A free arena slot, spilling the coldest resident slot to disk if none.
    fn acquire_slot(&mut self) -> Result<usize> {
        if let Some(s) = self.free_slots.pop() {
            return Ok(s);
        }
        self.evict_coldest_to_disk()
    }

    /// Spill the LRU-coldest RESIDENT key to a disk record and return its freed
    /// arena slot. Reserved (pinned) keys are never victims.
    fn evict_coldest_to_disk(&mut self) -> Result<usize> {
        let Some(victim) = self.lru.pop_front() else {
            bail!(
                "Residency: arena exhausted — all {} slots reserved (uncommitted \
                 PUTs) or read-pinned (in-flight RDMA READs)",
                self.arena.num_slots()
            );
        };
        let slot = match self.map.get(&victim).copied() {
            Some(Loc::Resident(slot)) => slot,
            other => bail!("LRU/map desync: victim {victim:#x} is {other:?}, expected Resident"),
        };
        // Bound the disk tier: drop the coldest on-disk snapshot(s) if at cap
        // BEFORE claiming a disk slot for this spill.
        self.make_disk_room();
        let disk_slot = self.alloc_disk_slot();
        let mut buf = std::mem::take(&mut self.scratch);
        let res = self
            .arena
            .read_slot(slot, &mut buf)
            .and_then(|_| self.swap.write_record(disk_slot, &buf));
        self.scratch = buf;
        if let Err(e) = res {
            // Roll back: victim stays resident, disk slot returns to the pool.
            self.free_disk.push(disk_slot);
            self.lru.push_front(victim);
            return Err(e);
        }
        self.map.insert(victim, Loc::OnDisk(disk_slot));
        self.disk_lru.push_back(victim); // warmest on-disk entry
        self.stats.spills_to_disk += 1;
        Ok(slot)
    }

    /// Evict the coldest on-disk snapshot(s) until there is room for one more
    /// under `max_disk_slots` (no-op when unbounded). A dropped snapshot's key
    /// leaves the map entirely → a later GET misses → the model recomputes.
    fn make_disk_room(&mut self) {
        if self.max_disk_slots == 0 {
            return;
        }
        while self.disk_lru.len() >= self.max_disk_slots {
            let Some(cold) = self.disk_lru.pop_front() else {
                break;
            };
            if let Some(Loc::OnDisk(ds)) = self.map.remove(&cold) {
                self.free_disk.push(ds);
                self.swap.discard_record(ds);
                self.stats.disk_evictions += 1;
            }
        }
    }

    fn alloc_disk_slot(&mut self) -> usize {
        if let Some(d) = self.free_disk.pop() {
            d
        } else {
            let d = self.next_disk;
            self.next_disk += 1;
            d
        }
    }

    fn disk_lru_remove(&mut self, key: u64) {
        if let Some(pos) = self.disk_lru.iter().position(|&k| k == key) {
            self.disk_lru.remove(pos);
        }
    }

    fn lru_touch(&mut self, key: u64) {
        self.lru_remove(key);
        self.lru.push_back(key);
    }

    fn lru_remove(&mut self, key: u64) {
        if let Some(pos) = self.lru.iter().position(|&k| k == key) {
            self.lru.remove(pos);
        }
    }
}
