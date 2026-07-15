// SPDX-License-Identifier: AGPL-3.0-only

//! TIERED-CACHE-CONSOLIDATION §4 fix, step 3: the `ATLAS_SSM_TIER_UNIFIED`
//! flag and the [`UnifiedSnapshotStore`] it routes the spill stores through.

use std::sync::atomic::Ordering;

use anyhow::Result;
use parking_lot::Mutex;

use super::{BlobStoreStats, SnapshotBlobStore, SnapshotTransport};

/// Opt-in truthy parse for `ATLAS_SSM_TIER_UNIFIED` (style-matching
/// `ATLAS_HSS_COALESCE_WRITE_RUNS` in spark-storage/high_speed_swap.rs).
fn unified_flag_truthy(v: Option<&str>) -> bool {
    matches!(
        v.map(str::trim),
        Some("1") | Some("true") | Some("on") | Some("yes")
    )
}

/// TIERED-CACHE-CONSOLIDATION §4 fix, step 3: whether the client-side SSM
/// spill stores route through the ONE lifted policy core
/// ([`atlas_tier::Residency`] — two-level LRU, never rejects) instead of the
/// per-store policies (MemBlobStore FIFO, RdmaSnapshotStore drop-on-full).
/// DEFAULT OFF ⇒ the selectors construct exactly today's stores, byte- and
/// behavior-identical.
///
/// ⚠ **BEFORE FLIPPING THIS DEFAULT ON**, three flag-ON-only defects found by the
/// step-3 adversarial review must be fixed. None affect the default path; all three
/// are latent the moment the flag is engaged in production:
///
/// 1. **Lock held across transport I/O.** The RDMA arm's `put` path holds the
///    `UnifiedSnapshotStore` mutex across a victim evict (remote READ of a ~64 MB
///    blob, ~5–7 ms) *plus* the new blob's remote WRITE. Today's `RdmaSnapshotStore`
///    does not. Split the residency map ops from the byte moves — the core already
///    exposes the two-phase `alloc`/`commit` API needed to run transport I/O outside
///    the lock.
/// 2. **Silent downgrade to unbounded host RAM.** If `UnifiedSnapshotStore::new`
///    fails *after* a successful peer connect, the RDMA arm falls back to
///    `MemBlobStore::new(0)` (unbounded host RAM) with only a warn, abandoning the
///    connected arena. It should fall through to the legacy `RdmaSnapshotStore`
///    instead — the arena is already connected.
/// 3. **Swap files leak.** Flag-ON swap files (`atlas-ssm-{tag}.{pid}.swap`,
///    `atlas-decode-ring.{pid}.swap`) are per-PID and never unlinked, and the disk
///    tier grows unbounded by design. Unlink same-tag stale files on create, or open
///    with `O_TMPFILE`.
///
/// Coverage gap to close alongside: the flag-ON **RDMA** and **decode-NVMe** selector
/// arms are only component-tested, never exercised through `build_tier_store` /
/// `build_decode_tier_store` with the env set (the host-RAM arm is).
pub(crate) fn ssm_tier_unified() -> bool {
    unified_flag_truthy(std::env::var("ATLAS_SSM_TIER_UNIFIED").ok().as_deref())
}

// ─────────────────────────────────────────────────────────────────────────
// §4 unification (TIERED-CACHE-CONSOLIDATION step 3) — ATLAS_SSM_TIER_UNIFIED
//
// The SAME logical tier historically got a DIFFERENT eviction policy per
// backing store: MemBlobStore evicts FIFO by insertion order (latent — the
// production cap is always 0), RdmaSnapshotStore drops-on-full with no recency
// at all (live), while the peer's paging Residency does two-level LRU and
// never rejects. FIFO/drop-on-full defeat the HBM pool's session-aware victim
// selection: the carefully chosen victim spills into a tier that re-picks its
// own victim by insertion order — or silently discards it.
//
// Flag ON routes the client-side spill stores through the ONE policy core
// lifted from the peer (`atlas_tier::Residency`: LRU over a hot arena, spill
// to a swap tier, NEVER reject, uncapped disk ⇒ nothing ever dropped). Flag
// OFF (default) constructs exactly today's stores — byte/behavior-identical.
// The gather/scatter of the ~60 per-layer device regions stays ABOVE this
// boundary in SsmSnapshotPool::{spill_slot,fault_in_slot}; the store only ever
// moves ONE contiguous host blob, so no scatter-capable SwapStore is needed
// and the StorageBackend refusals above remain true.
// ─────────────────────────────────────────────────────────────────────────

/// Adapts a [`SnapshotTransport`] (flat offset-addressed remote/file arena) to
/// the [`atlas_tier::SlotArena`] hot-tier seam: slot `i` lives at offset
/// `i × slot_bytes` — the same fixed-slot geometry [`super::RdmaSnapshotStore`]
/// uses, so the peer arena layout is unchanged under the flag.
pub(super) struct TransportSlotArena {
    pub(super) transport: Box<dyn SnapshotTransport>,
    pub(super) slot_bytes: usize,
    pub(super) num_slots: usize,
}

impl atlas_tier::SlotArena for TransportSlotArena {
    fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    fn num_slots(&self) -> usize {
        self.num_slots
    }
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        if slot >= self.num_slots || out.len() != self.slot_bytes {
            anyhow::bail!("TransportSlotArena::read_slot({slot}) out of range / size mismatch");
        }
        self.transport
            .read_blob((slot * self.slot_bytes) as u64, out)
    }
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        if slot >= self.num_slots || bytes.len() != self.slot_bytes {
            anyhow::bail!("TransportSlotArena::write_slot({slot}) out of range / size mismatch");
        }
        self.transport
            .write_blob((slot * self.slot_bytes) as u64, bytes)
    }
}

/// The flag-ON [`SnapshotBlobStore`]: a `Mutex`-shared [`atlas_tier::Residency`]
/// (the peer's exact paging core, in-process). PUT never returns `Ok(false)`
/// for a right-sized blob — a full hot arena LRU-spills its coldest resident
/// into the swap tier, and the uncapped disk (`max_disk_slots = 0`) means
/// nothing is ever dropped, which also satisfies the decode tier's HARD
/// non-dropping requirement BY CONSTRUCTION rather than by sizing. The Mutex
/// is held across the byte move — the same tradeoff the peer's
/// `run_paging_loop_shared` documents (map op + one blob memcpy per call).
pub(crate) struct UnifiedSnapshotStore {
    inner: Mutex<
        atlas_tier::Residency<Box<dyn atlas_tier::SlotArena>, Box<dyn atlas_tier::SwapStore>>,
    >,
    blob_bytes: usize,
    pub stats: BlobStoreStats,
}

impl UnifiedSnapshotStore {
    pub(super) fn new(
        arena: Box<dyn atlas_tier::SlotArena>,
        swap: Box<dyn atlas_tier::SwapStore>,
        blob_bytes: usize,
    ) -> Result<Self> {
        // Uncapped disk tier: keys are NEVER dropped (a capped disk would let
        // make_disk_room silently discard live decode rollback targets).
        let residency = atlas_tier::Residency::new(arena, swap)?;
        Ok(Self {
            inner: Mutex::new(residency),
            blob_bytes,
            stats: BlobStoreStats::default(),
        })
    }
}

impl SnapshotBlobStore for UnifiedSnapshotStore {
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool> {
        // Fixed-size tier: an off-size blob is a caller bug — refuse gracefully
        // (same contract as RdmaSnapshotStore), never corrupt a slot.
        if bytes.len() != self.blob_bytes {
            self.stats.put_rejects.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        self.inner.lock().put_blob(key, bytes)?;
        self.stats.puts.fetch_add(1, Ordering::Relaxed);
        Ok(true) // never full — the residency spills, it doesn't reject
    }

    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        // Defensive: never scatter a wrong-sized blob into a slot.
        if out.len() != self.blob_bytes {
            self.stats.get_misses.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        let hit = self.inner.lock().get_blob(key, out)?;
        if hit {
            self.stats.get_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.get_misses.fetch_add(1, Ordering::Relaxed);
        }
        Ok(hit)
    }

    fn remove(&self, key: u64) {
        self.inner.lock().remove(key);
    }

    fn len(&self) -> usize {
        self.inner.lock().total_keys()
    }

    fn bytes_resident(&self) -> usize {
        // Hot (RAM-arena) bytes; swapped records live in the swap tier.
        self.inner.lock().resident_count() * self.blob_bytes
    }
}

/// The unified stores' swap tier. `ATLAS_SSM_TIER_SWAP_DIR` selects the lifted
/// O_DIRECT NVMe swap file (needs a 4 KiB-multiple blob — the O_DIRECT
/// stride); otherwise (or on any setup failure) host-RAM records — still
/// LRU-ordered and never-reject, just RAM-resident like today's stores.
pub(super) fn build_unified_swap(blob_bytes: usize, tag: &str) -> Box<dyn atlas_tier::SwapStore> {
    if let Some(dir) = std::env::var("ATLAS_SSM_TIER_SWAP_DIR")
        .ok()
        .filter(|s| !s.is_empty())
    {
        if blob_bytes > 0 && blob_bytes.is_multiple_of(4096) {
            let make = || -> Result<atlas_tier::DirectSwapFile> {
                std::fs::create_dir_all(&dir)?;
                let path = std::path::Path::new(&dir)
                    .join(format!("atlas-ssm-{tag}.{}.swap", std::process::id()));
                atlas_tier::DirectSwapFile::create(&path, blob_bytes)
            };
            match make() {
                Ok(f) => {
                    tracing::info!("unified SSM tier ({tag}): O_DIRECT swap file in {dir}");
                    return Box::new(f);
                }
                Err(e) => tracing::info!(
                    "unified SSM tier ({tag}): swap dir {dir} unusable ({e:#}); \
                     using host-RAM swap"
                ),
            }
        } else {
            tracing::info!(
                "unified SSM tier ({tag}): blob_bytes {blob_bytes} is not a 4 KiB multiple \
                 (O_DIRECT stride); using host-RAM swap"
            );
        }
    }
    Box::new(atlas_tier::MemSwapStore::new(blob_bytes))
}

/// Hot-arena slot count for the unified stores (`ATLAS_SSM_TIER_SLOTS`,
/// default 64). The hot arena is allocated up front at `slots × blob_bytes`.
pub(super) fn unified_hot_slots() -> usize {
    std::env::var("ATLAS_SSM_TIER_SLOTS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(64)
        .max(1)
}

#[cfg(test)]
#[path = "unified_tests.rs"]
mod tests;
