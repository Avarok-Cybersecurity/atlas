// SPDX-License-Identifier: AGPL-3.0-only

//! Which host buffers are page-locked, and therefore which `copy_h2d_async`
//! calls are genuinely asynchronous.
//!
//! `cuMemcpyHtoDAsync_v2` behaves in two completely different ways depending on
//! the SOURCE, and nothing in its signature says which one you get:
//!
//! * **Pageable source** — the driver copies into its own staging buffer before
//!   returning. The call is async with respect to the GPU but synchronous with
//!   respect to the caller's memory, so dropping the source immediately is safe.
//! * **Page-locked source** — the DMA engine reads the caller's pages directly,
//!   after the call returns. Dropping or rewriting the source is a
//!   use-after-free / torn transfer.
//!
//! Almost every `copy_h2d_async` call site in Atlas hands over a stack array or
//! a local `Vec` that dies on the next line. Those are sound today purely
//! because they are pageable. Nothing recorded that dependency, so pinning any
//! one of those buffers — a normal, desirable optimisation, and one this tree
//! has already started doing for the SSM spill staging — would have turned a
//! whole class of call sites unsound at once, with no compile error and no
//! runtime complaint.
//!
//! This registry is what makes that fail loudly instead. Every page-locked
//! allocation Atlas makes is recorded here; the CUDA backend consults it on the
//! `copy_h2d_async` path and, for a pinned source, adds the synchronisation the
//! pageable path was getting from the driver for free. The cost of pinning a
//! buffer that a call site then drops is a stalled stream and a one-time
//! warning, not corruption.
//!
//! Scope: this tracks what goes through
//! [`crate::gpu::GpuBackend::alloc_host_pinned`] and
//! [`crate::gpu::GpuBackend::host_register`], the two doors to page-locked
//! memory in this workspace. Memory page-locked by any other route (a raw
//! `cuMemHostRegister` outside the backend) would not be seen; any such call must
//! register here too.
//!
//! **Tier-owned regions.** A region page-locked through `host_register` belongs to
//! a host-memory tier (the KV host tier and its pinned pool) that orders every
//! reuse of its pages behind a completed copy event, so the implicit
//! synchronisation above would only stall its DMA. Such regions are recorded with
//! [`register_tier`]: they still answer [`is_pinned`] truthfully, but
//! [`needs_transient_sync`] (the question the copy path asks) says no for them.
//! With no tier registration live, `needs_transient_sync` equals `is_pinned`.

use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// One live page-locked host region.
#[derive(Clone, Copy)]
struct Region {
    base: usize,
    len: usize,
    /// Registered through `host_register` (see the module doc).
    tier: bool,
}

/// Each live page-locked host region, kept sorted by base.
///
/// A `Vec` and a binary search rather than anything cleverer: the population is
/// the model's metadata staging blob, the SSM spill blob and the verify readback
/// blob (three entries, allocated at model load and held for the process
/// lifetime) plus, with a host tier on, its pinned chunks (tens).
static PINNED: RwLock<Vec<Region>> = RwLock::new(Vec::new());

// A poisoned lock means a previous holder panicked. Losing the registry must not
// take the process down: the consequence is a missed detection, so recover and
// carry on rather than propagate.
fn write() -> RwLockWriteGuard<'static, Vec<Region>> {
    match PINNED.write() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

fn read() -> RwLockReadGuard<'static, Vec<Region>> {
    match PINNED.read() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

fn insert(ptr: *const u8, bytes: usize, tier: bool) {
    if ptr.is_null() || bytes == 0 {
        return;
    }
    let base = ptr as usize;
    let mut g = write();
    match g.binary_search_by_key(&base, |r| r.base) {
        Ok(i) => {
            g[i].len = bytes;
            g[i].tier = tier;
        }
        Err(i) => g.insert(
            i,
            Region {
                base,
                len: bytes,
                tier,
            },
        ),
    }
}

/// Record a page-locked region. Idempotent for a repeated identical base.
pub fn register(ptr: *const u8, bytes: usize) {
    insert(ptr, bytes, false);
}

/// Record a TIER-OWNED page-locked region (see the module doc). Idempotent for a
/// repeated identical base; the latest registration's length and kind win.
pub fn register_tier(ptr: *const u8, bytes: usize) {
    insert(ptr, bytes, true);
}

/// Forget a region. Called from `free_host_pinned` and `host_unregister`, so a
/// freed-then-reused address is not misreported as still pinned.
pub fn unregister(ptr: *const u8) {
    if ptr.is_null() {
        return;
    }
    let base = ptr as usize;
    let mut g = write();
    if let Ok(i) = g.binary_search_by_key(&base, |r| r.base) {
        g.remove(i);
    }
}

/// The live region that contains all of `src`, if any.
fn containing(src: &[u8]) -> Option<Region> {
    if src.is_empty() {
        return None;
    }
    let start = src.as_ptr() as usize;
    let g = read();
    // Largest base <= start, then a containment test. Sub-slices of a pinned
    // blob (every real caller packs fields into one region and copies a prefix)
    // must match, which is why this is a range test and not a base lookup.
    let r = match g.binary_search_by_key(&start, |r| r.base) {
        Ok(i) => g[i],
        Err(0) => return None,
        Err(i) => g[i - 1],
    };
    (start >= r.base && start + src.len() <= r.base + r.len).then_some(r)
}

/// Does `src` lie inside a live page-locked region?
///
/// `true` means an H2D copy from it is genuinely asynchronous and the caller's
/// bytes are read after the enqueue returns.
pub fn is_pinned(src: &[u8]) -> bool {
    containing(src).is_some()
}

/// Does `src` lie inside a live tier-owned region?
pub fn is_tier_owned(src: &[u8]) -> bool {
    containing(src).is_some_and(|r| r.tier)
}

/// Does any `(address, bytes)` range of `ranges` start inside a live tier-owned region? One read
/// lock for the whole walk. Zero-byte ranges are skipped. Tier-owned regions are the ones locked
/// with `cuMemHostRegister` (`GpuBackend::host_register`) over ordinary memory, which
/// `cuMemcpyBatchAsync` must not touch on a driver that faults on it (see
/// `cuda_backend::gpu_impl_batch_copy`).
pub fn any_tier_owned(ranges: impl IntoIterator<Item = (usize, usize)>) -> bool {
    let g = read();
    if !g.iter().any(|r| r.tier) {
        return false;
    }
    ranges
        .into_iter()
        .filter(|&(_, bytes)| bytes > 0)
        .any(|(start, _)| {
            let r = match g.binary_search_by_key(&start, |r| r.base) {
                Ok(i) => g[i],
                Err(0) => return false,
                Err(i) => g[i - 1],
            };
            r.tier && start >= r.base && start < r.base + r.len
        })
}

/// Must a transient (`copy_h2d_async`) copy from `src` wait for the DMA before
/// returning? True for a page-locked source that no tier owns.
pub fn needs_transient_sync(src: &[u8]) -> bool {
    containing(src).is_some_and(|r| !r.tier)
}

/// Length of the live region based EXACTLY at `ptr`, if there is one.
pub fn region_len(ptr: *const u8) -> Option<usize> {
    let base = ptr as usize;
    let g = read();
    g.binary_search_by_key(&base, |r| r.base)
        .ok()
        .map(|i| g[i].len)
}

/// Does `[ptr, ptr + bytes)` intersect any live region?
pub fn overlaps(ptr: *const u8, bytes: usize) -> bool {
    if bytes == 0 {
        return false;
    }
    let start = ptr as usize;
    let end = start.saturating_add(bytes);
    read()
        .iter()
        .any(|r| start < r.base.saturating_add(r.len) && r.base < end)
}

/// Live region count, for tests and teardown assertions.
pub fn live_count() -> usize {
    read().len()
}

/// Live tier-owned region count.
pub fn tier_count() -> usize {
    read().iter().filter(|r| r.tier).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The registry is process-global and other test modules (the backend
    // extension tests, the pinned pool's) register their own buffers in
    // parallel, so these tests assert only about the addresses they own, never
    // about global counts.

    #[test]
    fn a_pageable_buffer_is_not_pinned() {
        let v = vec![0u8; 128];
        assert!(!is_pinned(&v));
        assert!(!needs_transient_sync(&v));
    }

    #[test]
    fn a_registered_region_and_its_subslices_are_pinned() {
        let mut v = vec![0u8; 256];
        register(v.as_ptr(), v.len());

        assert!(is_pinned(&v), "the whole region");
        // The shape every real caller uses: pack fields into the blob, copy a
        // prefix. A base-address-only lookup would miss these.
        assert!(is_pinned(&v[..64]), "prefix");
        assert!(is_pinned(&v[64..128]), "interior");
        assert!(is_pinned(&v[255..]), "last byte");
        assert!(needs_transient_sync(&v[..64]), "plain pinned sources sync");
        assert!(!is_tier_owned(&v), "a plain registration is not tier-owned");

        unregister(v.as_ptr());
        assert!(!is_pinned(&v), "unregistered again");
        v[0] = 1;
        assert_eq!(v[0], 1);
    }

    #[test]
    fn a_neighbouring_pageable_buffer_is_not_caught() {
        let pinned = vec![0u8; 128];
        let other = vec![0u8; 128];
        register(pinned.as_ptr(), pinned.len());
        assert!(is_pinned(&pinned));
        assert!(!is_pinned(&other), "a different allocation must not match");
        unregister(pinned.as_ptr());
    }

    #[test]
    fn empty_and_null_are_handled() {
        let v = [0u8; 16];
        register(std::ptr::null(), 16);
        register(v.as_ptr(), 0);
        assert_eq!(region_len(std::ptr::null()), None, "null is not a region");
        assert_eq!(region_len(v.as_ptr()), None, "zero bytes is not a region");
        unregister(std::ptr::null());
        assert!(!is_pinned(&[]));
        assert!(!overlaps(v.as_ptr(), 0));
    }

    #[test]
    fn re_registering_the_same_base_updates_the_length() {
        let v = vec![0u8; 256];
        register(v.as_ptr(), 64);
        register(v.as_ptr(), 256);
        assert_eq!(region_len(v.as_ptr()), Some(256), "one entry, updated");
        assert!(is_pinned(&v[..256]), "the updated length is in effect");
        unregister(v.as_ptr());
        assert_eq!(region_len(v.as_ptr()), None, "and it is gone again");
    }

    #[test]
    fn a_tier_region_is_pinned_but_needs_no_transient_sync() {
        let v = vec![0u8; 512];
        register_tier(v.as_ptr(), v.len());
        assert!(is_pinned(&v[100..200]));
        assert!(is_tier_owned(&v[100..200]));
        assert!(
            !needs_transient_sync(&v[100..200]),
            "tier DMA never takes the implicit sync"
        );
        assert!(tier_count() >= 1);
        unregister(v.as_ptr());
        assert!(!is_tier_owned(&v));
        assert_eq!(region_len(v.as_ptr()), None);
    }

    #[test]
    fn any_tier_owned_sees_only_tier_regions_and_skips_empty_ranges() {
        let tier = vec![0u8; 4096];
        let plain = vec![0u8; 4096];
        let pageable = vec![0u8; 4096];
        let (t, p, g) = (
            tier.as_ptr() as usize,
            plain.as_ptr() as usize,
            pageable.as_ptr() as usize,
        );
        register_tier(tier.as_ptr(), tier.len());
        register(plain.as_ptr(), plain.len());
        assert!(any_tier_owned([(t, 4096)]), "start inside a tier region");
        assert!(
            any_tier_owned([(g, 64), (t + 4000, 64)]),
            "one tier range is enough"
        );
        assert!(
            !any_tier_owned([(p, 64)]),
            "a plain page-locked region is not tier-owned"
        );
        assert!(
            !any_tier_owned([(g, 64)]),
            "pageable memory is not tier-owned"
        );
        assert!(!any_tier_owned([(t, 0)]), "zero-byte ranges are skipped");
        assert!(!any_tier_owned(std::iter::empty()));
        unregister(tier.as_ptr());
        unregister(plain.as_ptr());
        assert!(!any_tier_owned([(t, 64)]), "gone after unregister");
    }

    #[test]
    fn overlap_is_a_range_test() {
        let v = vec![0u8; 1024];
        let base = v.as_ptr();
        register_tier(base.wrapping_add(256), 256);
        assert!(overlaps(base, 300), "covers the start");
        assert!(overlaps(base.wrapping_add(500), 100), "covers the end");
        assert!(overlaps(base.wrapping_add(300), 10), "inside");
        assert!(!overlaps(base, 256), "ends exactly at the base");
        assert!(
            !overlaps(base.wrapping_add(512), 256),
            "starts exactly at the end"
        );
        unregister(base.wrapping_add(256));
        assert!(!overlaps(base.wrapping_add(300), 10));
    }

    #[test]
    fn a_slice_that_runs_past_its_region_is_not_pinned() {
        let v = vec![0u8; 256];
        register(v.as_ptr(), 128);
        assert!(is_pinned(&v[..128]));
        assert!(!is_pinned(&v[64..192]), "straddles the region's end");
        unregister(v.as_ptr());
    }
}
