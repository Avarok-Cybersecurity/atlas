// SPDX-License-Identifier: AGPL-3.0-only

//! `MockGpuBackend` construction and the test-facing inspection accessors
//! (counters, snapshots, `read_alloc`, the simulated-memory `blit`). Split
//! out of `mock.rs` for the ≤500 LoC cap — a CHILD module so it keeps reading
//! the mock's private fields; the `GpuBackend` impl stays next door.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use parking_lot::Mutex;

use super::{MockGpuBackend, MockLaunch, find_alloc, find_alloc_mut};
use crate::gpu::DevicePtr;

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
            max_dynamic_smem: Mutex::new(Vec::new()),
        }
    }

    pub fn alloc_count(&self) -> usize {
        self.allocs.lock().len()
    }

    /// Reject individual allocations above `bytes`, for exercising
    /// production fallback paths without exhausting host memory.
    pub fn set_max_allocation_bytes(&self, bytes: usize) {
        self.max_allocation_bytes.store(bytes, Ordering::Relaxed);
    }

    pub fn launch_count(&self) -> usize {
        self.launches.lock().len()
    }

    /// `synchronize` calls so far — a proxy for "full stream drains", the cost
    /// a batched gather exists to amortize.
    pub fn sync_count(&self) -> usize {
        self.syncs.load(Ordering::Relaxed)
    }

    /// BLOCKING `copy_d2h` calls (each one drains the stream on the real
    /// backend). A bulk gather must have zero of these.
    pub fn d2h_blocking_count(&self) -> usize {
        self.d2h_blocking.load(Ordering::Relaxed)
    }

    /// `copy_d2h_async` calls (enqueue-only).
    pub fn d2h_async_count(&self) -> usize {
        self.d2h_async.load(Ordering::Relaxed)
    }

    pub fn d2h_async_streams(&self) -> Vec<u64> {
        self.d2h_async_streams.lock().clone()
    }

    pub fn sync_d2h_async_counts(&self) -> Vec<(u64, usize)> {
        self.sync_d2h_async_counts.lock().clone()
    }

    /// `copy_d2d` + `copy_d2d_async` calls so far — one eager launch each on
    /// the real backend.
    pub fn d2d_count(&self) -> usize {
        self.d2d.load(Ordering::Relaxed)
    }

    /// `copy_d2d_2d_async` calls so far — one `cudaMemcpy2DAsync` each,
    /// whatever the row count.
    pub fn d2d_2d_count(&self) -> usize {
        self.d2d_2d.load(Ordering::Relaxed)
    }

    /// Streams supplied to `copy_d2d_async`, in dispatch order.
    pub fn d2d_async_streams(&self) -> Vec<u64> {
        self.d2d_async_streams.lock().clone()
    }

    /// Streams supplied to `copy_d2d_2d_async`, in dispatch order.
    pub fn d2d_2d_async_streams(&self) -> Vec<u64> {
        self.d2d_2d_async_streams.lock().clone()
    }

    /// `alloc_host_pinned` calls — the tripwire for a staging buffer that is
    /// re-allocated per event instead of reused.
    pub fn host_pinned_alloc_count(&self) -> usize {
        self.host_pinned_allocs.load(Ordering::Relaxed)
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
    pub fn kernel_lookups_snapshot(&self) -> Vec<(String, String)> {
        self.kernel_lookups.lock().clone()
    }

    /// `(kernel handle, bytes)` per `set_kernel_max_dynamic_smem` call, in
    /// call order.
    pub fn max_dynamic_smem_calls(&self) -> Vec<(u64, usize)> {
        self.max_dynamic_smem.lock().clone()
    }

    /// Launches that went through the cooperative path.
    pub fn cooperative_launch_count(&self) -> usize {
        self.launches
            .lock()
            .iter()
            .filter(|l| l.cooperative)
            .count()
    }
}
