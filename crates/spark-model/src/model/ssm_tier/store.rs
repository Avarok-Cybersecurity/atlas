// SPDX-License-Identifier: AGPL-3.0-only
// LRU arm and pool backing: design after LMCache lmcache/v1/memory_allocators/ (page runs in a
// pinned pool) and its LRU cache policy (touch on hit, evict the least recent), commit 3aefae1,
// Apache-2.0; no code copied. Changes: one keyed blob store with FIFO (unchanged default) and
// LRU-on-get orders over one stamp index, by Atlas contributors.

//! Responsibility: the [`SnapshotBlobStore`] seam and the host-RAM reference store.
//!
//! [`MemBlobStore`] keeps one blob per key under an optional byte cap. Its eviction order is
//! FIFO by first insertion (the pre-sprint behavior, and still the default) or, with the
//! `ssm_store_lru` row on, least recently used: a `get` hit and a `put` both make the key the
//! most recent, so a cold orphan is the first thing a full store gives up. Blobs live in heap
//! `Vec`s, or in runs of a [`PinnedPool`] when the store is built with one (page-locked, so a
//! fault-in reads from pinned memory; a blob the pool cannot place falls back to the heap).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::pinned_pool::{PageRun, PinnedPool};

/// A keyed byte-blob store backing the SSM spill tier. One blob == one
/// snapshot's full `(h,conv)×layers` state, keyed by its prefix hash (the same
/// stable identity the [`crate::traits`] snapshot index keys on).
///
/// Implementations must be cheap to share (`Send + Sync`) — the tier is
/// consulted from the scheduler thread on eviction and on prefix lookup.
pub(crate) trait SnapshotBlobStore: Send + Sync {
    /// Store `bytes` under `key`, replacing any prior value. Returns `false`
    /// when the tier is full and refused the write — the caller then falls back
    /// to a plain drop (correct degradation, never a hard error).
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool>;

    /// Copy the blob for `key` into `out` (which must be sized to the blob).
    /// Returns `false` if `key` is absent (caller recomputes) or the length
    /// mismatches (defensive: never scatter a wrong-sized blob into a slot).
    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool>;

    /// Drop the blob for `key` if present (e.g. when its prefix is invalidated).
    fn remove(&self, key: u64);

    /// Resident blob count.
    fn len(&self) -> usize;

    /// Total resident bytes — for budget enforcement and telemetry.
    fn bytes_resident(&self) -> usize;
}

/// Aggregate spill-tier telemetry (mirrors the Phase-0 snapshot stats).
#[derive(Default)]
pub(crate) struct BlobStoreStats {
    pub puts: AtomicUsize,
    pub put_rejects: AtomicUsize,
    pub get_hits: AtomicUsize,
    pub get_misses: AtomicUsize,
    pub evictions: AtomicUsize,
    /// Pool-backed store only: blobs the pool could not place, kept on the heap instead.
    pub pool_fallbacks: AtomicUsize,
}

/// Which resident blob a full [`MemBlobStore`] gives up first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlobEviction {
    /// Oldest by first insertion; an overwrite keeps its position (the pre-sprint order).
    Fifo,
    /// Least recently put or read.
    Lru,
}

impl BlobEviction {
    /// The order the `ssm_store_lru` row selects: `Lru` when it is on, else `Fifo`.
    pub(crate) fn from_row() -> Self {
        if crate::layers::ops::port_rows::ssm_store_lru::row() {
            Self::Lru
        } else {
            Self::Fifo
        }
    }
}

/// Where one blob's bytes live.
enum Backing {
    Heap(Vec<u8>),
    Pool(PageRun),
}

struct Blob {
    backing: Backing,
    len: usize,
    /// Position in `MemInner::order`.
    stamp: u64,
}

/// In-memory host-RAM spill tier. On GB10 (unified LPDDR) this is a real T1
/// tier, not a test stand-in: spilling here frees a scarce pinned snapshot-pool
/// slot while the bytes remain in abundant UMA. Bounded by `cap_bytes` (`0`
/// means unbounded) with FIFO or LRU eviction (see the module doc) so a runaway
/// session can't exhaust host memory.
pub(crate) struct MemBlobStore {
    inner: Mutex<MemInner>,
    bytes: AtomicUsize,
    cap_bytes: usize,
    policy: BlobEviction,
    pool: Option<Arc<PinnedPool>>,
    pub stats: BlobStoreStats,
}

struct MemInner {
    map: HashMap<u64, Blob>,
    /// Eviction order: stamp -> key, oldest first. FIFO stamps a key on first insert only;
    /// LRU restamps it on every put and every get hit.
    order: BTreeMap<u64, u64>,
    next_stamp: u64,
}

impl MemInner {
    fn stamp(&mut self) -> u64 {
        self.next_stamp += 1;
        self.next_stamp
    }

    /// Make `key` (resident) the most recent.
    fn touch(&mut self, key: u64) {
        let stamp = self.stamp();
        if let Some(b) = self.map.get_mut(&key) {
            self.order.remove(&b.stamp);
            b.stamp = stamp;
            self.order.insert(stamp, key);
        }
    }
}

impl MemBlobStore {
    /// `cap_bytes == 0` → unbounded. Eviction order from the `ssm_store_lru` row (FIFO with the
    /// row off, exactly the pre-sprint store).
    pub(crate) fn new(cap_bytes: usize) -> Self {
        Self::with_policy(cap_bytes, BlobEviction::from_row())
    }

    /// A heap-backed store with an explicit eviction order.
    pub(crate) fn with_policy(cap_bytes: usize, policy: BlobEviction) -> Self {
        Self {
            inner: Mutex::new(MemInner {
                map: HashMap::new(),
                order: BTreeMap::new(),
                next_stamp: 0,
            }),
            bytes: AtomicUsize::new(0),
            cap_bytes,
            policy,
            pool: None,
            stats: BlobStoreStats::default(),
        }
    }

    /// A store whose blobs live in `pool`'s page runs (heap only for a blob the pool cannot
    /// place: larger than one pool chunk, or the pool full). `cap_bytes` still bounds the
    /// resident bytes, heap and pool together. The host KV tier's store selection (lane L26b,
    /// `ssm_tier/selectors.rs`) builds it with the tier's pool; nothing else does.
    #[allow(dead_code)]
    pub(crate) fn pool_backed(
        cap_bytes: usize,
        policy: BlobEviction,
        pool: Arc<PinnedPool>,
    ) -> Self {
        // Not struct-update syntax: the store implements `Drop`.
        let mut s = Self::with_policy(cap_bytes, policy);
        s.pool = Some(pool);
        s
    }

    #[cfg(test)]
    pub(crate) fn policy(&self) -> BlobEviction {
        self.policy
    }

    /// Blobs whose bytes are in the pinned pool (0 for a heap store).
    #[cfg(test)]
    pub(crate) fn pooled_blobs(&self) -> usize {
        let g = self.inner.lock();
        g.map
            .values()
            .filter(|b| matches!(b.backing, Backing::Pool(_)))
            .count()
    }

    /// Place `bytes` in the pool when there is one and it has room, else on the heap.
    fn back(&self, bytes: &[u8]) -> Backing {
        if let Some(pool) = &self.pool {
            if let Some(run) = pool.alloc_run(bytes.len()) {
                match pool.copy_in(&run, 0, bytes) {
                    Ok(()) => return Backing::Pool(run),
                    Err(e) => {
                        tracing::warn!("SSM blob store: pool write failed ({e:#}); using the heap");
                        pool.free(run);
                    }
                }
            }
            self.stats.pool_fallbacks.fetch_add(1, Ordering::Relaxed);
        }
        Backing::Heap(bytes.to_vec())
    }

    fn release(&self, backing: Backing) {
        if let (Backing::Pool(run), Some(pool)) = (backing, &self.pool) {
            pool.free(run);
        }
    }

    /// Remove `key`'s blob from `g` and the order, returning its bytes' backing.
    fn take(g: &mut MemInner, key: u64) -> Option<Blob> {
        let b = g.map.remove(&key)?;
        g.order.remove(&b.stamp);
        Some(b)
    }
}

impl SnapshotBlobStore for MemBlobStore {
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool> {
        // A single blob larger than the whole cap can never fit — refuse rather
        // than evict everything and still fail.
        if self.cap_bytes != 0 && bytes.len() > self.cap_bytes {
            self.stats.put_rejects.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        let mut g = self.inner.lock();
        // From here the put always succeeds (a pool that cannot place the blob falls back to the
        // heap), so an old value under `key` can be released now.
        let mut freed = Vec::new();
        // Overwrite in place: reclaim the old blob's bytes first. FIFO keeps the key's position;
        // LRU makes it the most recent. The key's `order` entry stays while the map entry is out.
        let stamp = match g.map.remove(&key) {
            Some(old) => {
                self.bytes.fetch_sub(old.len, Ordering::Relaxed);
                freed.push(old.backing);
                match self.policy {
                    BlobEviction::Fifo => old.stamp,
                    BlobEviction::Lru => {
                        g.order.remove(&old.stamp);
                        let s = g.stamp();
                        g.order.insert(s, key);
                        s
                    }
                }
            }
            None => {
                let s = g.stamp();
                g.order.insert(s, key);
                s
            }
        };
        // Evict oldest until the new blob fits under the cap.
        if self.cap_bytes != 0 {
            while self.bytes.load(Ordering::Relaxed) + bytes.len() > self.cap_bytes {
                // An overwrite keeps the key resident, but the value being replaced is no
                // longer part of the resident-byte count. Skip that key and evict the oldest
                // *other* blob.
                let Some(victim) = g.order.values().copied().find(|&k| k != key) else {
                    break;
                };
                if let Some(v) = Self::take(&mut g, victim) {
                    self.bytes.fetch_sub(v.len, Ordering::Relaxed);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                    freed.push(v.backing);
                }
            }
        }
        // Old pool runs go back before the new blob is placed, so it can reuse their pages.
        for b in freed {
            self.release(b);
        }
        let backing = self.back(bytes);
        self.bytes.fetch_add(bytes.len(), Ordering::Relaxed);
        g.map.insert(
            key,
            Blob {
                backing,
                len: bytes.len(),
                stamp,
            },
        );
        self.stats.puts.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        let mut g = self.inner.lock();
        let copied = match g.map.get(&key) {
            Some(b) if b.len == out.len() => match &b.backing {
                Backing::Heap(v) => {
                    out.copy_from_slice(v);
                    true
                }
                Backing::Pool(run) => match &self.pool {
                    Some(pool) => pool.copy_out(run, 0, out).is_ok(),
                    None => false,
                },
            },
            _ => false,
        };
        if !copied {
            self.stats.get_misses.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        if self.policy == BlobEviction::Lru {
            g.touch(key);
        }
        self.stats.get_hits.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    fn remove(&self, key: u64) {
        let taken = {
            let mut g = self.inner.lock();
            Self::take(&mut g, key)
        };
        if let Some(b) = taken {
            self.bytes.fetch_sub(b.len, Ordering::Relaxed);
            self.release(b.backing);
        }
    }

    fn len(&self) -> usize {
        self.inner.lock().map.len()
    }

    fn bytes_resident(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }
}

impl Drop for MemBlobStore {
    fn drop(&mut self) {
        if self.pool.is_none() {
            return;
        }
        let blobs: Vec<Blob> = self.inner.get_mut().map.drain().map(|(_, b)| b).collect();
        for b in blobs {
            self.release(b.backing);
        }
    }
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
