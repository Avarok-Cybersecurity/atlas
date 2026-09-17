// SPDX-License-Identifier: AGPL-3.0-only

//! Release-on-consume: freeing a checkpoint tensor the moment the loader has
//! finished turning it into a layer-owned allocation.
//!
//! **WHY.** On a 32 GB discrete board the store's dead weight is the difference
//! between loading and not. Measured on an AMD Radeon AI PRO R9700 (gfx1201,
//! 31.9 GB, SCALE 1.7.1) with `unsloth/Qwen3.8-27B-NVFP4`, 2026-09-17: the
//! whole 21.81 GiB checkpoint uploads, and the serve then dies in model build
//! at layer 28 of 64 with `cuMemAlloc_v2 failed: status 2, requested
//! 167772160 bytes` while the ledger holds 33.73 GB live. **9.94 GiB of the
//! store is DEAD at that moment** — FP8 attention, GDN and tail-MLP
//! projections that have already been dequantised to BF16 and requantised to
//! NVFP4, whose only consumer copied them. Full accounting, with the ledger
//! sweep reproduced site by site from the shapes:
//! `docs/porting/r9700-residency.md`.
//!
//! **Why it is not on everywhere.** On GB10 the same 9.94 GiB is a rounding
//! error against 121 GB of unified memory, and keeping the store intact
//! removes a class of use-after-free with no diagnostic: a store tensor freed
//! while some arm nobody thought about still aliases it does not fault, it
//! reads whatever the allocator handed out next. `WeightStore::free_matching`
//! already says this in its own words ("the caller owns the 'is it dead?'
//! question"), and `ModelWeightLoader::prune_after_load` already exists for
//! the cases where the bytes are worth the proof obligation. So the default is
//! a compile-time cfg: ON for SCALE/AMD targets, OFF everywhere else, and an
//! NVIDIA build with the variable unset executes exactly what it executed
//! before this module existed.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;

/// Whether loaders should free a store tensor as soon as they have fully
/// converted it into a layer-owned allocation.
///
/// `ATLAS_LOAD_RELEASE_SOURCES=1` forces on, `=0` forces off, unset takes
/// `cfg!(atlas_scale)`. Resolved once: a serve never rewrites its own
/// environment, and a loader that asked twice and got two answers would free a
/// tensor a later layer still reads.
///
/// EXPLICIT `1`/`0`, not presence, because the interesting operator action
/// here is turning it OFF on an AMD board to bisect a suspected
/// use-after-free, and `ATLAS_LOAD_RELEASE_SOURCES=0` meaning "on" would make
/// that impossible.
pub fn release_sources_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        decide(
            std::env::var("ATLAS_LOAD_RELEASE_SOURCES").ok().as_deref(),
            cfg!(atlas_scale),
        )
    })
}

/// The decision on its own, so CPU-only CI can pin the table without touching
/// the process environment (the `OnceLock` above cannot be toggled per test).
pub fn decide(env: Option<&str>, is_scale: bool) -> bool {
    match env {
        Some("1") => true,
        Some("0") => false,
        // A value that is neither: treat as unset rather than guess at an
        // intent. An operator who typed `=true` gets the target's default and
        // the serve log's residency line tells them which one they got.
        _ => is_scale,
    }
}

/// The names whose device allocation a loader has released, and how many bytes
/// went with them.
///
/// Kept BESIDE the tensor map rather than removing the entry, because
/// `WeightStore::get` takes `&self` (every loader is handed `&WeightStore`) and
/// a `HashMap` cannot lose a key through a shared reference. Holding the entry
/// and refusing to hand out its pointer is strictly better than removing it
/// anyway: a read after release is a *named* error rather than the same
/// "not found in store" a typo produces.
#[derive(Default)]
pub(crate) struct ReleasedSet {
    // parking_lot: no poisoning, so a panic in a loader cannot wedge teardown.
    names: Mutex<HashMap<String, usize>>,
    /// Lock-free "is this set empty" for the hot read path. `WeightStore::get`
    /// is called once per tensor per loader, not per token, but it is called
    /// with the load stream busy and there is no reason to take a lock on a
    /// target where nothing is ever released.
    count: AtomicUsize,
    bytes: AtomicUsize,
}

impl ReleasedSet {
    /// True if `name`'s allocation is gone. Cheap when nothing was released.
    pub(crate) fn contains(&self, name: &str) -> bool {
        self.count.load(Ordering::Relaxed) != 0 && self.names.lock().contains_key(name)
    }

    /// Record a release. Returns false if this name was already recorded, in
    /// which case the CALLER MUST NOT free the pointer a second time.
    pub(crate) fn mark(&self, name: &str, bytes: usize) -> bool {
        let mut names = self.names.lock();
        if names.contains_key(name) {
            return false;
        }
        names.insert(name.to_owned(), bytes);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        true
    }

    /// Total bytes released on consume.
    pub(crate) fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }

    /// How many tensors were released on consume.
    pub(crate) fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_follows_the_target_and_the_knob_overrides_it() {
        // Unset: the target decides.
        assert!(decide(None, true), "SCALE/AMD defaults ON");
        assert!(!decide(None, false), "NVIDIA defaults OFF");
        // Explicit wins on both targets, in both directions.
        assert!(decide(Some("1"), false));
        assert!(!decide(Some("0"), true));
        // Anything else is "unset", never a silent yes.
        assert!(!decide(Some("true"), false));
        assert!(decide(Some(""), true));
    }

    #[test]
    fn an_empty_set_answers_without_taking_the_lock() {
        let s = ReleasedSet::default();
        assert!(!s.contains("anything"));
        assert_eq!(s.len(), 0);
        assert_eq!(s.bytes(), 0);
    }

    #[test]
    fn mark_is_idempotent_so_a_double_release_cannot_double_free() {
        let s = ReleasedSet::default();
        assert!(s.mark("a.weight", 1024), "first release is recorded");
        assert!(
            !s.mark("a.weight", 1024),
            "a second release must be refused — the pointer is already gone"
        );
        assert_eq!(s.len(), 1);
        assert_eq!(s.bytes(), 1024, "the refused release must not be counted");
        assert!(s.contains("a.weight"));
        assert!(!s.contains("b.weight"));
    }

    #[test]
    fn bytes_accumulate_across_names() {
        let s = ReleasedSet::default();
        s.mark("a", 100);
        s.mark("b", 250);
        assert_eq!(s.len(), 2);
        assert_eq!(s.bytes(), 350);
    }
}
