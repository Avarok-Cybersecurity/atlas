// SPDX-License-Identifier: AGPL-3.0-only

//! Model-scoped state — the abstraction that makes in-process model swapping
//! safe.
//!
//! # The problem
//!
//! Atlas caches a great deal of derived state in process-global statics: env
//! toggles, kernel handles, tokenizer-derived token masks, workspaces, and
//! caches keyed by device pointer. Every one of those is derived from the
//! *loaded model*. Serving one model per process makes that correct by
//! construction. Swapping models in-process makes every one of them a way to
//! serve model A's answer to model B's question — **silently**, because a
//! stale `OnceLock` returns a plausible value rather than an error.
//!
//! # The shape of the fix
//!
//! State splits into two populations, and they want different treatment:
//!
//! * **Owned state** — the module set, the weights, the KV/SSM pools, the
//!   tokenizer. These are large, they own device memory, and their teardown
//!   must be *ordered*. They are held in an owning context and **propagated**
//!   (`&T`, `Arc<T>`, `Arc<RwLock<T>>`), never reached through a static. Their
//!   teardown contract is [`ModelResource`], not `Drop`, because `Drop` can be
//!   neither ordered nor fallible.
//!
//! * **Derived leaf state** — an env flag read once, a kernel handle looked up
//!   once, a mask built from the vocabulary. These are small, cheap to rebuild,
//!   and read from deep inside kernel-launch paths that cannot practically take
//!   another parameter. Threading a context through all of them would be a
//!   thousand-line change with no correctness gain over the alternative: keep
//!   the static, but make it **impossible for it to return a value built for a
//!   different model**. That is [`Scoped`], [`ScopedFlag`] and [`ScopedMap`].
//!
//! The [`Generation`] counter is the single authority both populations answer
//! to. It is process-global on purpose: it is an epoch, not model state. It
//! never goes backwards and a value is never reused.
//!
//! # Why this is safe where a hand-audited epoch scheme is not
//!
//! The danger in epoch-guarding by hand is *forgetting a site* — and a
//! forgotten site is silent wrong output. Here the guard is inside the type:
//! a `Scoped<T>` cannot serve a stale value, because every read compares
//! generations before returning. Converting `static X: OnceLock<T>` to
//! `static X: Scoped<T>` is mechanical and greppable, and anything left as a
//! bare `OnceLock` is visible to a search rather than hidden in a closure.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

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

/// A lazily-derived value that is rebuilt when the model changes.
///
/// Drop-in replacement for `static X: OnceLock<T>` in any position where the
/// value is derived from the loaded model. The API is deliberately close to
/// `OnceLock`'s so the conversion is mechanical.
///
/// ```ignore
/// -static MASK: OnceLock<Arc<[bool]>> = OnceLock::new();
/// +static MASK: Scoped<Arc<[bool]>> = Scoped::new();
///  // ...
/// -MASK.get_or_init(|| build_mask(tok)).clone()
/// +MASK.get_or_init(|| build_mask(tok))
/// ```
pub struct Scoped<T> {
    slot: RwLock<Option<(Generation, T)>>,
}

impl<T: Clone> Default for Scoped<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Scoped<T> {
    pub const fn new() -> Self {
        Self {
            slot: RwLock::new(None),
        }
    }

    /// The value for the current generation, building it if this is the first
    /// read since the model changed.
    ///
    /// `init` may run more than once under contention — like `OnceLock`, this
    /// is a cache, not a mutual-exclusion primitive. It must be side-effect
    /// free, which every derivation this replaces already is.
    pub fn get_or_init(&self, init: impl FnOnce() -> T) -> T {
        let generation = current();
        if let Some(value) = self.read_at(generation) {
            return value;
        }
        let built = init();
        self.store(generation, built.clone());
        built
    }

    /// Fallible derivation. A failed build is not cached — the next reader
    /// retries, which is what a transient CUDA failure needs.
    pub fn get_or_try_init<E>(&self, init: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
        let generation = current();
        if let Some(value) = self.read_at(generation) {
            return Ok(value);
        }
        let built = init()?;
        self.store(generation, built.clone());
        Ok(built)
    }

    /// The cached value, if one was built for the current generation.
    pub fn get(&self) -> Option<T> {
        self.read_at(current())
    }

    /// Drop the cached value outright.
    ///
    /// Refusing to *serve* a stale value is enough for a derived value, which
    /// is what this type is for. It is NOT enough when `T` owns a handle to a
    /// model resource — an `Arc` nobody reads is still an `Arc`, and teardown
    /// would block on it. Cells of that shape are the exception; they must be
    /// cleared explicitly during teardown, and
    /// `atlas_core::registry::release`'s reference-count check is what turns a
    /// forgotten `clear` into a named error instead of a silent leak.
    pub fn clear(&self) {
        if let Ok(mut guard) = self.slot.write() {
            *guard = None;
        }
    }

    fn read_at(&self, generation: Generation) -> Option<T> {
        let guard = self.slot.read().ok()?;
        match guard.as_ref() {
            Some((stored, value)) if *stored == generation => Some(value.clone()),
            _ => None,
        }
    }

    fn store(&self, generation: Generation, value: T) {
        if let Ok(mut guard) = self.slot.write() {
            *guard = Some((generation, value));
        }
    }
}

/// An environment toggle, re-read when the model changes.
///
/// The 42 `static X: OnceLock<bool>` env caches in the tree all have the same
/// shape and the same swap hazard: a model loaded with different `ATLAS_*`
/// flags inherits the previous model's decisions. The default is required at
/// the declaration rather than buried in a `get_or_init` closure, so the
/// behaviour when the variable is unset is visible where the flag is declared.
pub struct ScopedFlag {
    var: &'static str,
    /// What the flag means with the variable unset.
    default_on: bool,
    cell: Scoped<bool>,
}

impl ScopedFlag {
    /// `ATLAS_FOO=1` turns it on, `=0` off, unset falls back to `default_on`.
    pub const fn new(var: &'static str, default_on: bool) -> Self {
        Self {
            var,
            default_on,
            cell: Scoped::new(),
        }
    }

    pub fn get(&self) -> bool {
        self.cell.get_or_init(|| match std::env::var(self.var) {
            Ok(v) => matches!(v.trim(), "1" | "true" | "yes" | "on"),
            Err(_) => self.default_on,
        })
    }

    pub fn var(&self) -> &'static str {
        self.var
    }
}

/// A keyed cache that is emptied when the model changes.
///
/// Distinct from [`Scoped`] because of one specific hazard: several caches in
/// the tree are **keyed by raw device pointer**. After a model is released and
/// the next one allocates, the allocator can hand back the same addresses — so
/// a stale entry would be a *hit*, with a value describing different memory.
/// Comparing generations on read is not enough; the map is cleared outright.
pub struct ScopedMap<K, V> {
    /// `None` until first use: `HashMap::new` is not `const`, and this type
    /// has to be constructible in a `static`.
    inner: Mutex<Option<(Generation, HashMap<K, V>)>>,
}

impl<K: Eq + Hash, V: Clone> Default for ScopedMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash, V: Clone> ScopedMap<K, V> {
    pub const fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    /// Look up `key`, inserting `init()`'s value on a miss. Entries from a
    /// previous generation are dropped before the lookup.
    pub fn get_or_insert_with(&self, key: K, init: impl FnOnce() -> V) -> V {
        let generation = current();
        let Ok(mut guard) = self.inner.lock() else {
            // A poisoned cache must not serve entries whose provenance is
            // unknown; rebuilding is always correct, just slower.
            return init();
        };
        let entry = guard.get_or_insert_with(|| (generation, HashMap::new()));
        if entry.0 != generation {
            entry.1.clear();
            entry.0 = generation;
        }
        entry.1.entry(key).or_insert_with(init).clone()
    }

    /// Drop every entry. Call before releasing the device memory the keys
    /// point into, so no entry can outlive its allocation.
    pub fn clear(&self) {
        if let Ok(mut guard) = self.inner.lock()
            && let Some(entry) = guard.as_mut()
        {
            entry.1.clear();
        }
    }

    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .ok()
            .and_then(|g| {
                g.as_ref().map(|(built_for, map)| {
                    if *built_for == current() {
                        map.len()
                    } else {
                        0
                    }
                })
            })
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
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
