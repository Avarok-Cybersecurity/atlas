// SPDX-License-Identifier: AGPL-3.0-only

//! [`OpCache`] — per-backend kernel handles and scratch buffers.
//!
//! Kernel-launching ops memoize two things: the [`KernelHandle`] they resolve
//! from the module registry, and any device scratch they grow on demand. Both
//! were being cached in function-local `static OnceLock` / `static Mutex`, and
//! both are **owned by the model**:
//!
//! * A `KernelHandle` is a raw `CUfunction` from an `AtlasRegistry` module.
//!   The registry unloads its modules on drop, so a handle cached in a static
//!   outlives the module it points into — a launch after a swap is a
//!   use-after-unload, not a stale value.
//! * A scratch `DevicePtr` is an allocation in the model's context. Cached in
//!   a static, the next model writes its activations through a pointer that
//!   was freed with the previous one.
//!
//! Neither fails loudly. Both are the kind of defect that surfaces as
//! corrupted output or an illegal address in an unrelated kernel.
//!
//! An `OpCache` lives on the backend, so its lifetime is exactly the model's:
//! when the backend drops, the handles go with the registry that owns them and
//! the scratch goes with the context that allocated it.

use std::collections::HashMap;
// parking_lot: no poisoning, so a panic in one op cannot turn every later
// lookup — or teardown — into an error path that has to be handled.
use parking_lot::{Mutex, RwLock};

use crate::gpu::{DevicePtr, GpuBackend, KernelHandle};
use anyhow::Result;

/// Memoized kernel handles and scratch allocations for one backend.
#[derive(Default)]
pub struct OpCache {
    /// `(module, function)` → resolved handle. `RwLock` because the steady
    /// state is read-only: every entry is filled on the first launch of its op
    /// and read on every launch after.
    kernels: RwLock<HashMap<(&'static str, &'static str), KernelHandle>>,
    /// Purpose tag → `(pointer, bytes)`. Grow-only within a model's life.
    scratch: Mutex<HashMap<&'static str, (DevicePtr, usize)>>,
    /// Device allocation has failed on this backend at least once.
    alloc_fell_back: std::sync::atomic::AtomicBool,
    /// `(name-hash, M, N, K)` combinations whose route line has been logged.
    /// Backend-scoped like everything else here: the shapes a model dispatches
    /// are its own, and a set shared across a swap suppresses the FIRST route
    /// line for every shape the previous model happened to use — the lines
    /// that say which kernel a model actually took.
    logged_shapes: Mutex<std::collections::HashSet<(u64, u32, u32, u32)>>,
    /// Per-key call counts for the report-the-first-few diagnostics.
    counters: Mutex<HashMap<&'static str, u32>>,
}

impl OpCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve `module::func`, memoized. Equivalent to `gpu.kernel(..)` on a
    /// miss; a map read on a hit.
    pub fn kernel(
        &self,
        gpu: &dyn GpuBackend,
        module: &'static str,
        func: &'static str,
    ) -> Result<KernelHandle> {
        if let Some(k) = self.kernels.read().get(&(module, func)) {
            return Ok(*k);
        }
        let handle = gpu.kernel(module, func)?;
        self.kernels.write().insert((module, func), handle);
        Ok(handle)
    }

    /// A scratch allocation of at least `bytes`, memoized under `tag`.
    ///
    /// Grow-only: a request larger than the current buffer allocates a new one
    /// and abandons the old, which is bounded because the sizes that drive it
    /// (batch × hidden) have a ceiling per model. The abandoned block is
    /// reclaimed when the context goes, which is the point of scoping the
    /// cache to the backend.
    pub fn scratch(
        &self,
        gpu: &dyn GpuBackend,
        tag: &'static str,
        bytes: usize,
    ) -> Result<DevicePtr> {
        let mut g = self.scratch.lock();
        match g.get(tag) {
            Some(&(p, sz)) if sz >= bytes => Ok(p),
            _ => {
                let p = gpu.alloc(bytes)?;
                g.insert(tag, (p, bytes));
                Ok(p)
            }
        }
    }

    /// Has device allocation already failed on this backend?
    ///
    /// Retrying a failing `cuMemAlloc` per tensor wastes minutes of load time
    /// and fragments what is left, so the first failure latches the loader
    /// onto managed memory. Per BACKEND rather than per process: after a
    /// model is unloaded the pressure is gone, and the next model's load
    /// should try device memory again instead of inheriting a UVM sentence
    /// from a model that is no longer resident.
    pub fn alloc_fell_back(&self) -> bool {
        self.alloc_fell_back
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Latch the managed-memory fallback for the rest of this model's load.
    pub fn note_alloc_fallback(&self) {
        self.alloc_fell_back
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// `true` for the first `n` times this backend reaches `key`.
    ///
    /// For the diagnostics that report the first few of something and then
    /// go quiet. Counted per backend, so a second model reports its own.
    pub fn first_n(&self, key: &'static str, n: u32) -> bool {
        let mut g = self.counters.lock();
        let c = g.entry(key).or_insert(0);
        *c += 1;
        *c <= n
    }

    /// `true` the first time this backend reaches `key`, `false` after.
    ///
    /// For the log/dump gates whose call site holds a `GpuBackend` and
    /// nothing else. Backend-scoped, so a second model re-arms them.
    pub fn once(&self, key: &'static str) -> bool {
        self.first_shape(key, 0, 0, 0)
    }

    /// `true` the first time this backend dispatches `(name, m, n, k)`.
    /// Diagnostic de-duplication for the GEMM route/shape log lines.
    pub fn first_shape(&self, name: &str, m: u32, n: u32, k: u32) -> bool {
        let mut h: u64 = 1469598103934665603;
        for b in name.bytes() {
            h = (h ^ b as u64).wrapping_mul(1099511628211);
        }
        self.logged_shapes.lock().insert((h, m, n, k))
    }
}

impl std::fmt::Debug for OpCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kernels = self.kernels.read().len();
        let scratch = self.scratch.lock().len();
        f.debug_struct("OpCache")
            .field("kernels", &kernels)
            .field("scratch", &scratch)
            .finish()
    }
}

/// Release the scratch allocations.
///
/// The kernel handles are not freed here: they are module-scoped and die with
/// the `AtlasRegistry` the backend holds, which `cuda_host::release` unloads
/// once every handle to it is gone. Freeing them here would be a double-unload.
impl atlas_core::scope::ModelResource<dyn crate::gpu::GpuBackend> for OpCache {
    fn label(&self) -> &'static str {
        "op scratch"
    }

    fn release(&mut self, gpu: &dyn crate::gpu::GpuBackend) -> anyhow::Result<()> {
        let mut first_error = None;
        // `drain` makes this idempotent and stops a later launch from finding a
        // pointer into freed memory.
        let taken: Vec<(DevicePtr, usize)> = self
            .scratch
            .lock()
            .drain()
            .map(|(_, entry)| entry)
            .collect();
        for (ptr, _) in taken {
            if let Err(e) = gpu.free(ptr)
                && first_error.is_none()
            {
                first_error = Some(e);
            }
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::mock::MockGpuBackend;

    #[test]
    fn a_kernel_is_resolved_once_and_reused() {
        let gpu = MockGpuBackend::new();
        let c = OpCache::new();
        let a = c.kernel(&gpu, "w4a16", "bf16_to_fp8").expect("resolves");
        let b = c.kernel(&gpu, "w4a16", "bf16_to_fp8").expect("resolves");
        assert_eq!(a.0, b.0);
    }

    #[test]
    fn two_caches_do_not_share_handles_or_scratch() {
        // The property the statics could not have. Each cache belongs to one
        // backend, so nothing a model resolved is reachable from the next.
        let gpu = MockGpuBackend::new();
        let a = OpCache::new();
        let b = OpCache::new();
        let _ = a.scratch(&gpu, "fp8_activation", 1024).expect("allocs");
        assert!(
            format!("{a:?}").contains("scratch: 1"),
            "the first cache holds it"
        );
        assert!(
            format!("{b:?}").contains("scratch: 0"),
            "the second starts empty"
        );
    }

    #[test]
    fn scratch_grows_but_never_shrinks() {
        let gpu = MockGpuBackend::new();
        let c = OpCache::new();
        let small = c.scratch(&gpu, "act", 64).expect("allocs");
        let same = c.scratch(&gpu, "act", 32).expect("reuses");
        assert_eq!(small.0, same.0, "a smaller request reuses the buffer");
        let bigger = c.scratch(&gpu, "act", 4096).expect("reallocs");
        assert_ne!(small.0, bigger.0, "a larger request gets a new buffer");
    }
}
