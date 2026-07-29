// SPDX-License-Identifier: AGPL-3.0-only

//! TIERED-CACHE-CONSOLIDATION §4 fix, step 3: the `ATLAS_SSM_TIER_UNIFIED`
//! flag and the [`UnifiedSnapshotStore`] it routes the spill stores through.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, anyhow};
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
/// into the swap tier. Whether a spilled blob can then be *dropped* is a
/// per-consumer decision made at CONSTRUCTION, and that choice is the tier's
/// only safety knob:
///
/// * [`UnifiedSnapshotStore::new`] — uncapped disk (`max_disk_slots = 0`):
///   nothing is ever dropped, which satisfies the decode rolling tier's HARD
///   non-dropping requirement BY CONSTRUCTION rather than by sizing.
/// * [`UnifiedSnapshotStore::new_capped`] — bounded disk for the Marconi/prefix
///   tier, where a dropped blob degrades to a clean miss → recompute
///   (`ssm_snapshot_spill.rs` `fault_in_slot`: "Returns `false` if the tier has
///   no blob for `key` (caller recomputes) — the correct miss degradation").
///
/// The Mutex is held across the byte move — the same tradeoff the peer's
/// `run_paging_loop_shared` documents (map op + one blob memcpy per call).
pub(crate) struct UnifiedSnapshotStore {
    inner: Mutex<
        atlas_tier::Residency<Box<dyn atlas_tier::SlotArena>, Box<dyn atlas_tier::SwapStore>>,
    >,
    blob_bytes: usize,
    /// Mirror of the residency's cap (0 = unbounded), read without the lock for
    /// the one-shot cap-engaged warn below.
    max_disk_slots: usize,
    /// One-shot latch for the "cap engaged" WARN: the operator needs to learn
    /// ONCE that the budget started dropping snapshots (it is under-sized for
    /// the workload), not once per steady-state eviction.
    cap_engaged: AtomicBool,
    /// One-shot latch for the "disk tier engaged" INFO — the moment the swap
    /// file stops being 0 bytes. Without it, a 0-byte swap file is ambiguous
    /// between EXPECTED (the hot arena has absorbed every put so far) and
    /// BROKEN, and this tier has already burned a debugging session on exactly
    /// that ambiguity. Latched in the consumer rather than in `Residency`
    /// because `atlas-tier`'s dependency budget is `anyhow` (+libc) — no
    /// `tracing` — by design.
    disk_engaged: AtomicBool,
    pub stats: BlobStoreStats,
}

impl UnifiedSnapshotStore {
    /// UNCAPPED disk tier — **the decode rolling tier's constructor**.
    ///
    /// Keys are NEVER dropped here, because a capped disk would let
    /// `make_disk_room` silently discard live decode rollback targets: a decode
    /// restore that misses is a CORRUPT restore, not a recompute (see
    /// `SsmSnapshotPool::restore_decode`). The non-dropping guarantee is
    /// structural — it is this constructor, not a sizing argument — so the
    /// decode arm of `build_decode_tier_store` must keep calling `new`, never
    /// `new_capped`. `uncapped_new_never_drops` in the tests is the tripwire.
    pub(super) fn new(
        arena: Box<dyn atlas_tier::SlotArena>,
        swap: Box<dyn atlas_tier::SwapStore>,
        blob_bytes: usize,
    ) -> Result<Self> {
        Self::new_capped(arena, swap, blob_bytes, 0)
    }

    /// BOUNDED disk tier (`max_disk_slots` records; 0 = unbounded ⇒ exactly
    /// [`UnifiedSnapshotStore::new`]).
    ///
    /// Safe ONLY for the Marconi/prefix tier, whose keys are prefix hashes: a
    /// dropped blob makes `try_fault_in_ssm_snapshot` return `None`, the warm
    /// turn recomputes, and that is precisely the shipped tier-disabled path.
    /// The decode rolling ring never reaches any `Residency` — `save_decode`
    /// and `restore_decode` take no `store` and copy D2D within a separate GPU
    /// region — so the hazard the uncapped constructor guards cannot occur on a
    /// store built here. Capping is what keeps the O_DIRECT swap file inside an
    /// operator's partition instead of growing without bound.
    pub(super) fn new_capped(
        arena: Box<dyn atlas_tier::SlotArena>,
        swap: Box<dyn atlas_tier::SwapStore>,
        blob_bytes: usize,
        max_disk_slots: usize,
    ) -> Result<Self> {
        let residency = atlas_tier::Residency::new_capped(arena, swap, max_disk_slots)?;
        Ok(Self {
            inner: Mutex::new(residency),
            blob_bytes,
            max_disk_slots,
            cap_engaged: AtomicBool::new(false),
            disk_engaged: AtomicBool::new(false),
            stats: BlobStoreStats::default(),
        })
    }

    /// Live on-disk records (tier-stats reporting / tests).
    pub(crate) fn disk_records(&self) -> usize {
        self.inner.lock().disk_count()
    }

    /// Snapshots dropped so far because the disk cap was hit (always 0 on an
    /// uncapped store).
    pub(crate) fn disk_evictions(&self) -> u64 {
        self.inner.lock().stats().disk_evictions
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
        let (disk_evictions, disk_records, spills, hot_slots) = {
            let mut r = self.inner.lock();
            r.put_blob(key, bytes)?;
            let n = r.arena().num_slots();
            (
                r.stats().disk_evictions,
                r.disk_count(),
                r.stats().spills_to_disk,
                n,
            )
        };
        self.stats.puts.fetch_add(1, Ordering::Relaxed);
        // First byte ever written to the swap file. Until this fires, a 0-byte
        // swap file is EXPECTED, not a bug: `Residency` only writes a record
        // when the hot arena has no free slot, so the first `hot_slots` puts of
        // distinct keys are absorbed in RAM.
        if spills > 0 && !self.disk_engaged.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "SSM tier disk tier ENGAGED: hot arena full ({hot_slots} slots); first record \
                 written — the swap file is no longer 0 bytes ({disk_records} records on disk)"
            );
        }
        // The moment the cap becomes load-bearing is the one an operator must
        // see: from here on, warm turns for dropped prefixes recompute instead
        // of faulting in. Latched, so steady-state eviction never floods.
        if self.max_disk_slots > 0
            && disk_evictions > 0
            && !self.cap_engaged.swap(true, Ordering::Relaxed)
        {
            tracing::warn!(
                "SSM tier disk cap engaged: dropping coldest snapshots (cap {} records = \
                 {:.2} GiB, {disk_records} on disk); warm turns for dropped prefixes \
                 recompute instead of faulting in",
                self.max_disk_slots,
                gib(self.max_disk_slots, self.blob_bytes),
            );
        }
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

/// What [`build_unified_swap`] actually ended up on. Returned rather than
/// inferred because the O_DIRECT arm degrades SILENTLY (an info log) on a
/// non-4 KiB blob or any create failure — without this, a "disk cap 32 GiB"
/// log line would be a lie about host RAM, which is how a box gets killed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SwapBacking {
    ODirect,
    HostRam,
}

/// The unified stores' swap tier. `ATLAS_SSM_TIER_SWAP_DIR` selects the lifted
/// O_DIRECT NVMe swap file (needs a 4 KiB-multiple blob — the O_DIRECT
/// stride); otherwise (or on any setup failure) host-RAM records — still
/// LRU-ordered and never-reject, just RAM-resident like today's stores.
pub(super) fn build_unified_swap(
    blob_bytes: usize,
    tag: &str,
) -> (Box<dyn atlas_tier::SwapStore>, SwapBacking) {
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
                    return (Box::new(f), SwapBacking::ODirect);
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
    (
        Box::new(atlas_tier::MemSwapStore::new(blob_bytes)),
        SwapBacking::HostRam,
    )
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

/// `records × blob_bytes` as GiB, for the operator-facing budget logs.
fn gib(records: usize, blob_bytes: usize) -> f64 {
    (records as f64) * (blob_bytes as f64) / (1024.0 * 1024.0 * 1024.0)
}

pub(super) const DISK_GB_VAR: &str = "ATLAS_SSM_TIER_DISK_GB";

/// Whether the operator asked for a disk budget at all (used to warn when the
/// budget lands on an arm that cannot honor it).
pub(super) fn disk_gb_requested() -> bool {
    std::env::var_os(DISK_GB_VAR).is_some_and(|v| !v.is_empty())
}

/// On-disk record cap for the unified Marconi tier from
/// [`DISK_GB_VAR`] (GiB, fractional accepted). `0` = unbounded = the default,
/// so an unset var reproduces today's behavior byte-for-byte.
pub(super) fn ssm_tier_disk_slots(blob_bytes: usize) -> Result<usize> {
    disk_slots_from(std::env::var(DISK_GB_VAR).ok().as_deref(), blob_bytes)
}

/// Env-free core of [`ssm_tier_disk_slots`], so the conversion is unit-testable
/// like `resolve_arena_bytes_from` (spark-storage/kv_paging/ns.rs).
///
/// STRICT parse (PCND), deliberately unlike this module's lenient
/// `unified_hot_slots`: a bad slot count only mis-sizes an arena, but a typo
/// here would mean "unbounded — fill the whole partition", i.e. exactly the
/// failure this budget exists to prevent. Same reasoning as the strict
/// `ATLAS_SSM_DECODE_TIER` arm in selectors.rs.
///
/// The core caps RECORDS, not bytes (`Residency::new_capped`), and validates
/// that the swap record stride equals the arena slot, so
/// `bytes = records × blob_bytes` exactly.
pub(super) fn disk_slots_from(raw: Option<&str>, blob_bytes: usize) -> Result<usize> {
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(0); // unset ⇒ unbounded ⇒ byte-identical to a pre-cap build
    };
    let gb: f64 = raw
        .parse()
        .map_err(|e| anyhow!("{DISK_GB_VAR}={raw:?} is not a number: {e}"))?;
    if !gb.is_finite() || gb < 0.0 {
        return Err(anyhow!(
            "{DISK_GB_VAR}={raw:?} must be a finite value >= 0 (0 = unbounded)"
        ));
    }
    if gb == 0.0 {
        return Ok(0); // the explicit unbounded sentinel (as in Residency)
    }
    if blob_bytes == 0 {
        return Err(anyhow!(
            "{DISK_GB_VAR}={raw:?} with blob_bytes 0 — no snapshot geometry to size against"
        ));
    }
    let cap_bytes = (gb * (1u64 << 30) as f64) as u64;
    let records = (cap_bytes / blob_bytes as u64) as usize;
    // Refuse rather than floor to 1 (the `.max(1)` that `carve_disk_slots` in
    // spark-storage uses is a TRAP here): `records - 1` reaching 0 would collide
    // with the unbounded sentinel, so a too-small budget would silently produce
    // the exact OPPOSITE of what the operator asked for.
    if records < 2 {
        return Err(anyhow!(
            "{DISK_GB_VAR}={raw:?} ({cap_bytes} B) cannot hold two SSM snapshots of \
             {blob_bytes} B — raise the budget or unset it for an unbounded tier"
        ));
    }
    // Budget ONE record of headroom: `Residency::locate`'s OnDisk arm pulls the
    // faulting key out of `disk_lru` while its record is still live, so
    // `make_disk_room` counts one fewer than exist and the swap file can reach
    // `max_disk_slots + 1` records. Subtracting here keeps the WORST-CASE file
    // ≤ the operator's budget, which is the entire point of a bounded disk.
    Ok(records - 1)
}

/// The UNIFIED tier's construction log, with the disk budget stated in both
/// directions. `arm` is the backing phrase ("over RDMA peer …", "in host RAM").
/// Uncapped emits the pre-cap line verbatim, so the default path logs exactly
/// as it did before.
pub(super) fn log_unified_tier(
    arm: &str,
    hot_slots: usize,
    blob_bytes: usize,
    max_disk_slots: usize,
    backing: SwapBacking,
) {
    // State the disk-write threshold explicitly. The hot arena absorbs the
    // first `hot_slots` distinct keys, so a swap file that is still 0 bytes
    // after a handful of spills is EXPECTED — an ambiguity that has already
    // cost one debugging session. The "disk tier ENGAGED" line in
    // `UnifiedSnapshotStore::put` marks the moment it stops being 0.
    let first_disk = hot_slots + 1;
    if max_disk_slots == 0 {
        tracing::info!(
            "SSM spill tier = UNIFIED residency {arm} ({hot_slots} hot slots × \
             {blob_bytes} B, LRU spill, never rejects); disk writes begin at spill \
             #{first_disk} of distinct keys (until then the swap file is 0 bytes BY DESIGN)"
        );
        return;
    }
    let steady = gib(max_disk_slots, blob_bytes);
    let worst = gib(max_disk_slots + 1, blob_bytes);
    match backing {
        SwapBacking::ODirect => tracing::info!(
            "SSM spill tier = UNIFIED residency {arm} ({hot_slots} hot slots × {blob_bytes} B); \
             disk cap {DISK_GB_VAR} → {max_disk_slots} records × {blob_bytes} B = {steady:.2} GiB \
             steady, {worst:.2} GiB worst case (one extra record is live during a fault-in); \
             bounding: O_DIRECT swap file; disk writes begin at spill #{first_disk} of \
             distinct keys (until then the swap file is 0 bytes BY DESIGN)"
        ),
        // A "disk" budget silently governing host RAM is how a box gets killed:
        // the swap tier here is anonymous memory, not the operator's partition.
        SwapBacking::HostRam => tracing::warn!(
            "SSM spill tier = UNIFIED residency {arm} ({hot_slots} hot slots × {blob_bytes} B); \
             disk cap {DISK_GB_VAR} → {max_disk_slots} records × {blob_bytes} B = {steady:.2} GiB \
             steady, {worst:.2} GiB worst case; bounding: host-RAM swap (O_DIRECT unavailable — \
             this budget caps RAM, not disk; set ATLAS_SSM_TIER_SWAP_DIR)"
        ),
    }
}

#[cfg(test)]
#[path = "unified_tests.rs"]
mod tests;
