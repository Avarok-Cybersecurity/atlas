// SPDX-License-Identifier: AGPL-3.0-only

//! Model teardown — ordered, fallible release of state that owns device memory.
//!
//! # Why there are no caches here
//!
//! An earlier version of this module offered generation-checked statics
//! (`Scoped`, `ScopedFlag`, `ScopedMap`) as a safe home for state derived from
//! the loaded model. They are gone, and the reasoning is worth keeping:
//!
//! **A checked static is still a static.** It is a dependency the signature
//! does not declare, it cannot be varied in a test without mutating the
//! process, and a site that forgets it fails at *runtime* — if it is ever read
//! at all. Propagating the value instead makes the same question a
//! *compile-time* one: add a field, and every construction site that forgot it
//! stops building.
//!
//! In practice a carrier almost always already exists and the static was
//! bypassing it — `ForwardContext` reaches every dispatch site in the model,
//! `&dyn Model` reaches the scheduler, and a backend owns its own device
//! handles. Where no carrier exists, the answer is to add one, not to reach for
//! a guarded global.
//!
//! # The statics that legitimately remain
//!
//! What stays is state derived from the *process* or the *device* rather than
//! from the checkpoint. Every such site carries a comment arguing its case;
//! these are the categories, so a reader can tell at a glance whether a static
//! they have found is accounted for or is a straggler.
//!
//! 1. **The CUDA host** ([`crate::cuda_host`]). One primary context per device
//!    per process, enforced by the driver; outliving every model is the entire
//!    point, since not recreating it is what in-process swapping buys. The full
//!    argument is on the declaration.
//!
//! 2. **Log-once latches** (`std::sync::Once`, `*_LOGGED`, `*_WARNED`). These
//!    hold no value — a `Once` is a latch, not data — so they cannot produce a
//!    wrong answer. Their only cross-model effect is suppressing a duplicate
//!    log line for a route the previous model also took. Threading a logging
//!    concern through every kernel-dispatch signature to restore one INFO line
//!    is not a trade worth making.
//!
//! 3. **One-shot diagnostic latches** (`*_DUMP_DONE`, `*_DIAG_DONE`). Same
//!    shape, and live only when an `ATLAS_DUMP_*` variable is set: they gate a
//!    debug capture whose intent is "one sample per process", not "one per
//!    model". A stale latch suppresses a duplicate dump; it cannot corrupt one.
//!
//! 4. **Compile-time tables and descriptors.** Immutable data with no runtime
//!    state — lookup tables are `const` where the language allows it, and the
//!    plugin/benchmark descriptors are `static` only because they are reached
//!    as `&'static` and need a stable address.
//!
//! 5. **Process lifecycle** (the TUI's terminal guard, shutdown flags, log
//!    ring). These describe the *process's* relationship to its terminal and
//!    its exit, which no model has any bearing on.
//!
//! Anything not in one of those five is a straggler and should be scoped.
//!
//! # What this module does provide
//!
//! [`ModelResource`] and [`Teardown`] give an ordered, fallible release path,
//! which `Drop` cannot: it is neither ordered across independent
//! values nor able to report a failure, and on GB10 unified memory frees must
//! happen at a quiescent point in a controlled order.

// The `Generation` epoch counter that lived here is gone. It existed to
// invalidate the generation-checked statics described above; with those
// deleted its only reader was its own test, and a monotonic counter kept alive
// for a hypothetical future user is the same process global this module argues
// against. Teardown ordering — the real problem it was reaching for — is
// expressed by the traits below, which need no epoch.

/// State that owns device memory and must be released in a defined order.
///
/// `Drop` is the wrong contract for this and the reason is specific: on GB10
/// unified memory a device free posts in-band TLB invalidations that corrupt
/// *neighbouring* allocations when interleaved with other allocation traffic.
/// That constrains **when** frees happen, not whether — teardown, where nothing
/// else is allocating and the streams are synchronised, is the safe case, and
/// the loader's scratch-buffer workaround exists precisely because loading is
/// not. `Drop` can express neither that ordering nor a failure.
///
/// `Cx` is whatever releasing needs — for GPU state that is the allocator.
/// Making it a type parameter keeps `atlas-core` free of a dependency on the
/// backend crate while still letting a resource be handed the thing that owns
/// its memory, rather than making every resource carry its own handle.
pub trait ModelResource<Cx: ?Sized>: Send + Sync {
    /// Human name, for the teardown report and for attributing a failure.
    fn label(&self) -> &'static str;

    /// Release everything this owns. Must be idempotent: the host calls it,
    /// and a `Drop` backstop may call it again.
    fn release(&mut self, cx: &Cx) -> anyhow::Result<()>;
}

/// Releases a set of resources in reverse registration order — the inverse of
/// how they were built, which is the only order that is safe when later
/// resources borrow earlier ones.
///
/// One failure does not abandon the rest: every resource is released, and the
/// first error is returned afterwards. A half-torn-down GPU is worse than a
/// reported error.
pub struct Teardown<Cx: ?Sized> {
    resources: Vec<Box<dyn ModelResource<Cx>>>,
}

impl<Cx: ?Sized> Default for Teardown<Cx> {
    fn default() -> Self {
        Self {
            resources: Vec::new(),
        }
    }
}

impl<Cx: ?Sized> Teardown<Cx> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a resource. Registration order is construction order.
    pub fn push(&mut self, resource: Box<dyn ModelResource<Cx>>) {
        self.resources.push(resource);
    }

    pub fn len(&self) -> usize {
        self.resources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Release everything, newest first. Returns the first failure, after
    /// having attempted them all.
    pub fn release_all(&mut self, cx: &Cx) -> anyhow::Result<()> {
        let mut failures = Vec::new();
        while let Some(mut resource) = self.resources.pop() {
            if let Err(e) = resource.release(cx) {
                // Every failure is reported, not just the first: after a
                // partial teardown the operator needs the whole picture to
                // decide whether the GPU is still usable.
                failures.push(format!("{}: {e:#}", resource.label()));
            }
        }
        if failures.is_empty() {
            return Ok(());
        }
        Err(anyhow::anyhow!(
            "{} resource(s) failed to release: {}",
            failures.len(),
            failures.join("; ")
        ))
    }
}

#[cfg(test)]
#[path = "scope_tests.rs"]
mod tests;
