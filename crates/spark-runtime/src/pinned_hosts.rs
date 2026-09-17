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
//! Scope and honesty about it: this tracks what goes through
//! [`crate::gpu::GpuBackend::alloc_host_pinned`], which is the only door to
//! page-locked memory in this workspace today. Memory page-locked by some other
//! route — a raw `cuMemHostRegister` on an existing arena, which
//! `model/ssm_snapshot_spill.rs` explicitly contemplates — would not be seen.
//! Any such call must register here too.

use std::sync::RwLock;

/// `(base, len)` of each live page-locked host region, kept sorted by base.
///
/// A `Vec` and a linear scan rather than anything cleverer: the population is
/// the model's metadata staging blob, the SSM spill blob and the verify readback
/// blob — three entries, allocated at model load and held for the process
/// lifetime. A tree would be more code and slower.
static PINNED: RwLock<Vec<(usize, usize)>> = RwLock::new(Vec::new());

/// Record a page-locked region. Idempotent for a repeated identical base.
pub fn register(ptr: *const u8, bytes: usize) {
    if ptr.is_null() || bytes == 0 {
        return;
    }
    let base = ptr as usize;
    let mut g = match PINNED.write() {
        Ok(g) => g,
        // A poisoned lock means a previous holder panicked. Losing the registry
        // must not take the process down — the consequence is a missed
        // detection, so recover and carry on rather than propagate.
        Err(e) => e.into_inner(),
    };
    match g.binary_search_by_key(&base, |&(b, _)| b) {
        Ok(i) => g[i].1 = bytes,
        Err(i) => g.insert(i, (base, bytes)),
    }
}

/// Forget a region. Called from `free_host_pinned`, so a freed-then-reused
/// address is not misreported as still pinned.
pub fn unregister(ptr: *const u8) {
    if ptr.is_null() {
        return;
    }
    let base = ptr as usize;
    let mut g = match PINNED.write() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    if let Ok(i) = g.binary_search_by_key(&base, |&(b, _)| b) {
        g.remove(i);
    }
}

/// Does `src` lie inside a live page-locked region?
///
/// `true` means an H2D copy from it is genuinely asynchronous and the caller's
/// bytes are read after the enqueue returns.
pub fn is_pinned(src: &[u8]) -> bool {
    if src.is_empty() {
        return false;
    }
    let start = src.as_ptr() as usize;
    let g = match PINNED.read() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    // Largest base <= start, then a containment test. Sub-slices of a pinned
    // blob (every real caller packs fields into one region and copies a prefix)
    // must match, which is why this is a range test and not a base lookup.
    match g.binary_search_by_key(&start, |&(b, _)| b) {
        Ok(i) => start + src.len() <= g[i].0 + g[i].1,
        Err(0) => false,
        Err(i) => {
            let (b, len) = g[i - 1];
            start < b + len && start + src.len() <= b + len
        }
    }
}

/// Live region count, for tests and teardown assertions.
pub fn live_count() -> usize {
    match PINNED.read() {
        Ok(g) => g.len(),
        Err(e) => e.into_inner().len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercises the registry over real allocations. Uses one shared lock so
    /// the tests in this module do not observe each other's registrations —
    /// `PINNED` is process-global by design.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn a_pageable_buffer_is_not_pinned() {
        let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let v = vec![0u8; 128];
        assert!(!is_pinned(&v));
    }

    #[test]
    fn a_registered_region_and_its_subslices_are_pinned() {
        let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut v = vec![0u8; 256];
        register(v.as_ptr(), v.len());

        assert!(is_pinned(&v), "the whole region");
        // The shape every real caller uses: pack fields into the blob, copy a
        // prefix. A base-address-only lookup would miss these.
        assert!(is_pinned(&v[..64]), "prefix");
        assert!(is_pinned(&v[64..128]), "interior");
        assert!(is_pinned(&v[255..]), "last byte");

        unregister(v.as_ptr());
        assert!(!is_pinned(&v), "unregistered again");
        v[0] = 1;
        assert_eq!(v[0], 1);
    }

    #[test]
    fn a_neighbouring_pageable_buffer_is_not_caught() {
        let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let pinned = vec![0u8; 128];
        let other = vec![0u8; 128];
        register(pinned.as_ptr(), pinned.len());
        assert!(is_pinned(&pinned));
        assert!(!is_pinned(&other), "a different allocation must not match");
        unregister(pinned.as_ptr());
    }

    #[test]
    fn empty_and_null_are_handled() {
        let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let before = live_count();
        register(std::ptr::null(), 16);
        register(0x1000 as *const u8, 0);
        assert_eq!(live_count(), before, "neither is a real region");
        unregister(std::ptr::null());
        assert!(!is_pinned(&[]));
    }

    #[test]
    fn re_registering_the_same_base_updates_the_length() {
        let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let v = vec![0u8; 256];
        let before = live_count();
        register(v.as_ptr(), 64);
        register(v.as_ptr(), 256);
        assert_eq!(live_count(), before + 1, "one entry, not two");
        assert!(is_pinned(&v[..256]), "the updated length is in effect");
        unregister(v.as_ptr());
    }
}
