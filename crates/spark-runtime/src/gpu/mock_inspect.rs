// SPDX-License-Identifier: AGPL-3.0-only

//! `MockGpuBackend` construction and the inspection accessors tests read.
//! A sibling so `mock.rs` stays under the 500-line cap, the same reason
//! `mock_counters.rs` exists; these keep reading the private fields
//! because a `#[path]` child module is still inside the parent module.

use super::*;

impl Default for MockGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MockGpuBackend {
    pub fn new() -> Self {
        Self {
            op_cache: crate::op_cache::OpCache::new(),
            allocs: Mutex::new(HashMap::new()),
            next_ptr: Mutex::new(0x1000_0000),
            max_allocation_bytes: AtomicUsize::new(usize::MAX),
            launches: Mutex::new(Vec::new()),
            kernel_lookups: Mutex::new(Vec::new()),
            max_dynamic_smem: Mutex::new(Vec::new()),
            absent_modules: Mutex::new(std::collections::HashSet::new()),
            denied_kernels: Mutex::new(Vec::new()),
            syncs: AtomicUsize::new(0),
            d2h_blocking: AtomicUsize::new(0),
            d2h_async: AtomicUsize::new(0),
            d2h_async_streams: Mutex::new(Vec::new()),
            sync_d2h_async_counts: Mutex::new(Vec::new()),
            d2d: AtomicUsize::new(0),
            d2d_2d: AtomicUsize::new(0),
            d2d_async_streams: Mutex::new(Vec::new()),
            d2d_2d_async_streams: Mutex::new(Vec::new()),
            host_pinned_allocs: AtomicUsize::new(0),
        }
    }

    pub fn read_alloc(&self, ptr: DevicePtr) -> Option<Vec<u8>> {
        self.allocs.lock().get(&ptr.0).map(|a| a.data.clone())
    }

    /// `bytes` from `src` to `dst` inside the simulated device memory.
    ///
    /// Real byte movement, not a no-op: a D2D that silently succeeds without
    /// moving anything lets a test "pass" while asserting the destination is
    /// still zero — the exact shape of a rollback bug this backend exists to
    /// catch. Source is staged through a temporary so `src` and `dst` may sit
    /// in the same allocation (the borrow checker would otherwise reject it,
    /// and the real `cudaMemcpyAsync` accepts it for non-overlapping ranges).
    pub(super) fn blit(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let mut allocs = self.allocs.lock();
        let staged = {
            let (offset, alloc) = find_alloc(&allocs, src)
                .ok_or_else(|| anyhow::anyhow!("copy_d2d: src {src} not allocated"))?;
            if offset + bytes > alloc.bytes {
                anyhow::bail!("copy_d2d: src {src} + {bytes} overruns its allocation");
            }
            alloc.data[offset..offset + bytes].to_vec()
        };
        let (offset, alloc) = find_alloc_mut(&mut allocs, dst)
            .ok_or_else(|| anyhow::anyhow!("copy_d2d: dst {dst} not allocated"))?;
        if offset + bytes > alloc.bytes {
            anyhow::bail!("copy_d2d: dst {dst} + {bytes} overruns its allocation");
        }
        alloc.data[offset..offset + bytes].copy_from_slice(&staged);
        Ok(())
    }

    /// Every launch recorded so far, in dispatch order. Lets a test assert
    /// WHICH kernel shape ran (grid/block signature), not just how many —
    /// the mock's `kernel()` hands out one shared handle, so geometry is
    /// the only per-launch identity available.
    pub fn launches_snapshot(&self) -> Vec<MockLaunch> {
        self.launches.lock().clone()
    }

    /// Module/function pairs requested through `kernel`, in lookup order.
    /// Declare a module absent from this build, the way a GB10 image lacks a
    /// Hopper-owned twin: `has_module` answers false and a lookup against it
    /// is the caller's mistake.
    pub fn mark_module_absent(&self, module: &str) {
        self.absent_modules.lock().insert(module.to_owned());
    }

    pub fn kernel_lookups_snapshot(&self) -> Vec<(String, String)> {
        self.kernel_lookups.lock().clone()
    }

    /// Next `kernel(module, func)` for this pair fails (records the lookup).
    pub fn deny_kernel(&self, module: &str, func_name: &str) {
        self.denied_kernels
            .lock()
            .push((module.to_owned(), func_name.to_owned()));
    }
}
