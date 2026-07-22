// SPDX-License-Identifier: AGPL-3.0-only
//
// Process-global SHARED paging arenas (the cross-connection warm cache).
// This is peer POLICY with no home in `atlas-tier`/`atlas-rdma`: the
// (kind, blob_bytes) arena registry, the shared-vs-per-kind disk-cap carve,
// and the anonymous `Mmap` RAII the RDMA-registered arenas live in
// (deliberately NOT lifted to `atlas-tier`, whose charter excludes unsafe
// raw-pointer arena types — see `snapshot_swap/mmap_arena.rs`).
//
// All paging connections of one (kind, shape) reg_mr the SAME arena and drive
// ONE residency, so a snapshot PUT by one client is GET-able by another (same
// namespace — the client folds a per-model id into the key). Arena + blob
// geometry are fixed by the FIRST paging client of a shape; later clients must
// match blob_bytes. The arena, swap file and blade reservation live for the
// daemon's lifetime.

use anyhow::{Context, Result, bail};
use atlas_tier::{DirectSwapFile, Residency};

use super::server_impl::RdmaConfig;
use crate::snapshot_swap::MmapSlotArena;

/// A page-aligned anonymous mapping, unmapped on drop.
pub(super) struct Mmap {
    pub(super) addr: *mut libc::c_void,
    pub(super) len: usize,
}

impl Mmap {
    pub(super) fn anon(len: usize) -> Result<Self> {
        // SAFETY: standard anonymous private mapping of `len` bytes.
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            bail!(
                "mmap anon {len} failed: {}",
                std::io::Error::last_os_error()
            );
        }
        Ok(Self { addr, len })
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // SAFETY: addr/len from a successful mmap, unmapped once.
        unsafe { libc::munmap(self.addr, self.len) };
    }
}

/// One (kind, shape) shared arena: the anon mapping every same-shape paging
/// connection registers, plus the ONE residency that owns its slots.
pub(super) struct SharedPaging {
    pub(super) arena: Mmap,
    pub(super) residency: std::sync::Mutex<Residency<MmapSlotArena, DirectSwapFile>>,
    _reservation: crate::blade_cap::Reservation,
}
// SAFETY: `arena.addr` is a stable mapping; every mutable access to the
// residency (and thus the arena bytes) is serialized through its Mutex.
unsafe impl Send for SharedPaging {}
unsafe impl Sync for SharedPaging {}

/// Item 8: a REGISTRY of paging arenas keyed by (kind, blob_bytes) so ONE
/// peer serves per-(kind, shape) arenas. Distinct shapes coexist (different
/// fixed-slot geometries); same-shape clients share one arena (namespaced
/// keys). The disk cap is a single hard-ceiling budget carved across entries.
#[derive(Default)]
struct PagingRegistry {
    arenas: std::collections::HashMap<(u8, usize), std::sync::Arc<SharedPaging>>,
    /// Remaining disk-cap budget (bytes); the first (SSM) entry claims the
    /// remainder, honoring the hard `swap_cap_bytes` ceiling across arenas.
    remaining_cap: u64,
    cap_init: bool,
    legacy_cleaned: bool,
}

static SHARED_PAGING: std::sync::LazyLock<std::sync::Mutex<PagingRegistry>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(PagingRegistry::default()));

/// Decide a paging arena's disk-cap slot count and the updated shared-cap
/// remainder. Precedence: (1) per-kind override → this kind's OWN fixed
/// budget (0 = unbounded for it), leaving the shared remainder untouched so
/// it can't starve other kinds; (2) shared `swap_cap` ceiling → claim the
/// current remainder (≥1-record floor); (3) both unset (0) → unbounded.
/// Pure so the precedence is unit-tested without RDMA/mmap.
fn carve_disk_slots(
    per_kind_cap: Option<u64>,
    shared_cap: u64,
    shared_remaining: u64,
    blob_bytes: u64,
) -> (usize, u64) {
    let bb = blob_bytes.max(1);
    match per_kind_cap {
        Some(0) => (0, shared_remaining), // this kind explicitly unbounded
        Some(cap) => (((cap / bb) as usize).max(1), shared_remaining),
        None if shared_cap == 0 => (0, shared_remaining), // unbounded
        None => {
            let recs = (shared_remaining / bb) as usize;
            (
                recs.max(1),
                shared_remaining.saturating_sub(recs as u64 * bb),
            )
        }
    }
}

/// Get (first client of a (kind, blob) creates) that shape's shared arena.
/// Charges the blade ledger per arena; carves the disk cap from a shared
/// ceiling. The registry lock guards only the map + budget — residency ops
/// run under each entry's own Mutex, never this lock.
pub(super) fn get_or_init_shared_paging(
    rdma: &RdmaConfig,
    kind: u8,
    arena_bytes: usize,
    blob_bytes: usize,
    ledger: &std::sync::Arc<crate::blade_cap::CommitLedger>,
) -> Result<std::sync::Arc<SharedPaging>> {
    let mut reg = SHARED_PAGING.lock().expect("paging registry poisoned");
    let key = (kind, blob_bytes);
    if let Some(sh) = reg.arenas.get(&key) {
        return Ok(sh.clone()); // same (kind, shape) → share
    }
    let swap_dir = rdma
        .swap_dir
        .as_ref()
        .context("paging client but peer has no --swap-dir configured")?;
    std::fs::create_dir_all(swap_dir).ok();
    // One-time: init the shared disk budget + remove the pre-registry fixed
    // swap file (verify gap: orphaned atlas-snap-shared.swap on upgrade).
    if !reg.cap_init {
        reg.remaining_cap = rdma.swap_cap_bytes;
        reg.cap_init = true;
    }
    if !reg.legacy_cleaned {
        let _ = std::fs::remove_file(swap_dir.join("atlas-snap-shared.swap"));
        reg.legacy_cleaned = true;
    }
    let reservation = ledger
        .try_reserve(arena_bytes as u64)
        .context("paging blade cap")?;
    let arena = Mmap::anon(arena_bytes)?;
    let num_slots = arena_bytes / blob_bytes;
    // Disk-cap sizing (per-kind override → shared-ceiling carve → unbounded).
    let (max_disk_slots, new_remaining) = carve_disk_slots(
        rdma.per_kind_swap_cap_bytes.get(&kind).copied(),
        rdma.swap_cap_bytes,
        reg.remaining_cap,
        blob_bytes as u64,
    );
    reg.remaining_cap = new_remaining;
    let swap_path = swap_dir.join(format!("atlas-snap-{kind}-{blob_bytes}.swap"));
    let swap = DirectSwapFile::create(&swap_path, blob_bytes)?;
    // SAFETY: the Mmap is owned by SharedPaging (held by the registry Arc), so
    // its base VA outlives every MmapSlotArena view of it.
    let slot_arena = unsafe { MmapSlotArena::new(arena.addr as *mut u8, blob_bytes, num_slots) };
    let residency = Residency::new_capped(slot_arena, swap, max_disk_slots)?;
    tracing::info!(
        "cache-peer paging arena kind={kind} shape={blob_bytes}B: {num_slots} slots RAM \
         ({:.1} GiB) + NVMe swap {} (disk cap {} records; budget {:.0} GiB left)",
        arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        swap_path.display(),
        if max_disk_slots == 0 {
            "unbounded".to_string()
        } else {
            max_disk_slots.to_string()
        },
        reg.remaining_cap as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    let sh = std::sync::Arc::new(SharedPaging {
        arena,
        residency: std::sync::Mutex::new(residency),
        _reservation: reservation,
    });
    reg.arenas.insert(key, sh.clone());
    Ok(sh)
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
