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
use std::sync::{Mutex, RwLock};

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
        if let Some(k) = self
            .kernels
            .read()
            .expect("op cache kernels poisoned")
            .get(&(module, func))
        {
            return Ok(*k);
        }
        let handle = gpu.kernel(module, func)?;
        self.kernels
            .write()
            .expect("op cache kernels poisoned")
            .insert((module, func), handle);
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
        let mut g = self.scratch.lock().expect("op cache scratch poisoned");
        match g.get(tag) {
            Some(&(p, sz)) if sz >= bytes => Ok(p),
            _ => {
                let p = gpu.alloc(bytes)?;
                g.insert(tag, (p, bytes));
                Ok(p)
            }
        }
    }
}

impl std::fmt::Debug for OpCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kernels = self.kernels.read().map(|k| k.len()).unwrap_or(0);
        let scratch = self.scratch.lock().map(|s| s.len()).unwrap_or(0);
        f.debug_struct("OpCache")
            .field("kernels", &kernels)
            .field("scratch", &scratch)
            .finish()
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
