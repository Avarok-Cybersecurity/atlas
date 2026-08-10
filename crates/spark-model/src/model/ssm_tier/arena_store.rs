// SPDX-License-Identifier: AGPL-3.0-only

//! The two remote stores: the peer-owned paging store and the client-side
//! fixed-slot arena store (RDMA or local file, via [`SnapshotTransport`]).

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::Ordering;

use anyhow::Result;
use parking_lot::Mutex;

use super::{BlobStoreStats, PagingTransport, SnapshotBlobStore, SnapshotTransport};

/// WS-A paging-mode store: the PEER owns residency + an NVMe swap file, so this
/// store just forwards PUT/GET/REMOVE over the arena's control channel. Unlike
/// [`RdmaSnapshotStore`] (client-side fixed-slot allocator, `Ok(false)` when the
/// arena fills), a paging PUT **never rejects** — the peer spills the coldest
/// slot to disk. This is the "infinite depth" tier + shared across clients (one
/// peer-owned map) that the bounded RDMA/host-RAM stores can't give.
pub(crate) struct PagingSnapshotStore {
    /// Keyed paging seam: the RDMA arena in production, a shared in-process
    /// mock peer in the cross-model isolation tests.
    arena: Box<dyn PagingTransport>,
    blob_bytes: usize,
    /// Per-model namespace folded into every key so a SHARED peer never serves
    /// one model's SSM state to another (the prefix_hash is model-independent —
    /// same tokens collide across models, but their recurrent state differs).
    /// `NonZeroU64`: the old ns=0 passthrough (which silently cross-served
    /// state between models) is unrepresentable — the namespace is either an
    /// explicit override or the config-derived model fingerprint.
    namespace: NonZeroU64,
}

impl PagingSnapshotStore {
    pub(crate) fn new(
        arena: Box<dyn PagingTransport>,
        blob_bytes: usize,
        namespace: NonZeroU64,
    ) -> Self {
        Self {
            arena,
            blob_bytes,
            namespace,
        }
    }

    /// Fold the namespace into a key (splitmix64 on `key ^ ns`). Deterministic
    /// (same model+key → same wire key → cache hit) with negligible cross-model
    /// collision, same 64-bit contract as `prefix_hash` itself.
    fn wire(&self, key: u64) -> u64 {
        // SSOT: this WAS a third transcription of the splitmix64 constants. It is
        // exactly `mix64(key, ns)` — same operands, same order — so routing it
        // through the one definition in `atlas_tier::hash` is VALUE-PRESERVING:
        // no key rotates, no FP_VERSION bump. Proven by the golden pin in
        // `paging_isolation_tests::wire_key_is_mix64_of_key_and_ns`.
        atlas_tier::hash::mix64(key, self.namespace.get())
    }
}

impl SnapshotBlobStore for PagingSnapshotStore {
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool> {
        if bytes.len() != self.blob_bytes {
            anyhow::bail!(
                "paging put: {} != blob_bytes {}",
                bytes.len(),
                self.blob_bytes
            );
        }
        self.arena.paging_put(self.wire(key), bytes)?;
        Ok(true) // never full — the peer spills to NVMe
    }
    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        if out.len() != self.blob_bytes {
            anyhow::bail!(
                "paging get: {} != blob_bytes {}",
                out.len(),
                self.blob_bytes
            );
        }
        self.arena.paging_get(self.wire(key), out)
    }
    fn remove(&self, key: u64) {
        if let Err(e) = self.arena.paging_remove(self.wire(key)) {
            tracing::debug!("paging remove {key:#x} failed: {e:#}");
        }
    }
    // Residency lives on the peer (RAM cache + NVMe); the client doesn't track it.
    fn len(&self) -> usize {
        0
    }
    fn bytes_resident(&self) -> usize {
        0
    }
}

/// RDMA snapshot spill tier: a `SnapshotBlobStore` over a remote byte arena.
/// Because every snapshot blob is the SAME fixed size (`spill_blob_bytes()`),
/// the allocator is a trivial **fixed-slot arena**: slot `i` lives at offset
/// `i * blob_bytes`; a free-list of slot indices + a `key → slot` map track
/// residency. A full arena makes `put` return `Ok(false)` — the graceful-drop
/// contract `reclaim_from_cache` already handles (free the pool slot; the entry
/// misses into recompute). All map/free-list mutation is `Mutex`-guarded; the
/// (blocking) transport op runs outside the lock and rolls the allocator back on
/// failure so a half-written slot is never left mapped (never read as garbage).
#[allow(dead_code)] // constructed by the Inc-3 gate wiring (impl_a1 selector)
pub(crate) struct RdmaSnapshotStore {
    transport: Box<dyn SnapshotTransport>,
    blob_bytes: usize,
    inner: Mutex<RdmaInner>,
    pub stats: BlobStoreStats,
}

struct RdmaInner {
    /// key → slot index (byte offset = slot × blob_bytes).
    map: HashMap<u64, usize>,
    /// Free slot indices (LIFO reuse).
    free: Vec<usize>,
}

#[allow(dead_code)]
impl RdmaSnapshotStore {
    /// Build a store over `transport` with `arena_slots` fixed slots of
    /// `blob_bytes` each. The transport's arena must cover
    /// `arena_slots × blob_bytes` bytes.
    pub(crate) fn new(
        transport: Box<dyn SnapshotTransport>,
        blob_bytes: usize,
        arena_slots: usize,
    ) -> Self {
        let free: Vec<usize> = (0..arena_slots).rev().collect();
        Self {
            transport,
            blob_bytes,
            inner: Mutex::new(RdmaInner {
                map: HashMap::new(),
                free,
            }),
            stats: BlobStoreStats::default(),
        }
    }

    #[inline]
    fn offset_of(&self, slot: usize) -> u64 {
        (slot * self.blob_bytes) as u64
    }
}

impl SnapshotBlobStore for RdmaSnapshotStore {
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool> {
        // Fixed-slot arena: only the snapshot blob size fits. A size mismatch is
        // a caller bug — refuse gracefully rather than corrupt a slot.
        if bytes.len() != self.blob_bytes {
            self.stats.put_rejects.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        // Pick the slot under the lock, but DON'T commit a new mapping until the
        // write succeeds (so a failed write never leaves a garbage slot mapped).
        let (slot, was_new) = {
            let mut g = self.inner.lock();
            match g.map.get(&key) {
                Some(&slot) => (slot, false), // overwrite in place
                None => {
                    let Some(slot) = g.free.pop() else {
                        // Arena full → graceful drop (entry misses → recompute).
                        self.stats.put_rejects.fetch_add(1, Ordering::Relaxed);
                        return Ok(false);
                    };
                    (slot, true)
                }
            }
        };
        match self.transport.write_blob(self.offset_of(slot), bytes) {
            Ok(()) => {
                if was_new {
                    self.inner.lock().map.insert(key, slot);
                }
                self.stats.puts.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }
            Err(e) => {
                // Roll back: a new slot returns to the free-list (nothing
                // mapped); an overwrite drops the mapping AND frees the slot —
                // its bytes may be half-overwritten, so a later `get` must miss
                // (recompute), never read a corrupted slot.
                let mut g = self.inner.lock();
                if was_new {
                    g.free.push(slot);
                } else if let Some(s) = g.map.remove(&key) {
                    g.free.push(s);
                }
                Err(e)
            }
        }
    }

    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        // Defensive: never scatter a wrong-sized blob into a slot.
        if out.len() != self.blob_bytes {
            self.stats.get_misses.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        let slot = match self.inner.lock().map.get(&key) {
            Some(&slot) => slot,
            None => {
                self.stats.get_misses.fetch_add(1, Ordering::Relaxed);
                return Ok(false);
            }
        };
        self.transport.read_blob(self.offset_of(slot), out)?;
        self.stats.get_hits.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    fn remove(&self, key: u64) {
        let mut g = self.inner.lock();
        if let Some(slot) = g.map.remove(&key) {
            g.free.push(slot);
        }
    }

    fn len(&self) -> usize {
        self.inner.lock().map.len()
    }

    fn bytes_resident(&self) -> usize {
        self.inner.lock().map.len() * self.blob_bytes
    }
}

/// The fixed-slot offset-addressed arena store ([`RdmaSnapshotStore`]) is
/// transport-agnostic — it runs equally over the RDMA arena or a local file
/// (see [`super::FileSnapshotArena`]). `ArenaSnapshotStore` is the
/// transport-neutral name the decode rolling tier selects; `RdmaSnapshotStore`
/// remains as an alias for the existing Marconi call sites.
pub(crate) type ArenaSnapshotStore = RdmaSnapshotStore;

#[cfg(test)]
#[path = "arena_store_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "paging_isolation_tests.rs"]
mod paging_isolation_tests;
