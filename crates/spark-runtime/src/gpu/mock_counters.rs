// SPDX-License-Identifier: AGPL-3.0-only
//! The mock backend's call counters and stream logs — what a test reads to
//! prove a code path issued (or did not issue) a copy, a sync, a launch.
//! A child of `mock`, so the fields stay private to the mock; split out when
//! `mock.rs` crossed the 500-line cap.

use std::sync::atomic::Ordering;

use super::MockGpuBackend;

impl MockGpuBackend {
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
}
