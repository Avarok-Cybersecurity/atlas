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
//! What legitimately remains a static is state derived from the *process* or
//! the *device* rather than the checkpoint — the CUDA context in
//! [`crate::cuda_host`] is the clear case. Every such site must carry a comment
//! arguing why it must be one.
//!
//! # What this module does provide
//!
//! [`Generation`] identifies one loaded model, for diagnostics and for the
//! teardown log. [`ModelResource`] and [`Teardown`] give an ordered, fallible
//! release path, which `Drop` cannot: it is neither ordered across independent
//! values nor able to report a failure, and on GB10 unified memory frees must
//! happen at a quiescent point in a controlled order.

use std::sync::atomic::{AtomicU64, Ordering};

/// Identity of one loaded model, within this process.
///
/// Monotonic and never reused: comparing two `Generation`s answers "were these
/// built for the same model?" and nothing else.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Generation(u64);

impl Generation {
    /// The generation before any model has been loaded. Nothing derived is
    /// ever tagged with it, so a cache initialised at this value always misses.
    pub const NONE: Generation = Generation(0);

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// The epoch counter. Process-global by design — it identifies *which* model is
/// loaded and is itself derived from nothing.
static CURRENT: AtomicU64 = AtomicU64::new(0);

/// The generation currently being served.
pub fn current() -> Generation {
    Generation(CURRENT.load(Ordering::Acquire))
}

/// Begin a new generation. Called exactly once per model load, by the host,
/// **after** the previous model's resources have been released.
///
/// Every [`Scoped`], [`ScopedFlag`] and [`ScopedMap`] in the process is
/// invalidated by this single call — there is no registry of them to walk and
/// therefore no list to forget an entry from.
pub fn advance() -> Generation {
    // `Release` so a thread that observes the new generation also observes
    // everything the loader wrote before publishing it.
    Generation(CURRENT.fetch_add(1, Ordering::Release) + 1)
}

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
