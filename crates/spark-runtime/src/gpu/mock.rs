// SPDX-License-Identifier: AGPL-3.0-only
//! Mock GPU backend for unit tests (no GPU required).

use super::*;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
pub struct MockAlloc {
    pub bytes: usize,
    pub data: Vec<u8>,
}

/// Records kernel launches and memory operations for test assertions.
pub struct MockGpuBackend {
    op_cache: crate::op_cache::OpCache,
    allocs: Mutex<HashMap<u64, MockAlloc>>,
    next_ptr: Mutex<u64>,
    max_allocation_bytes: AtomicUsize,
    launches: Mutex<Vec<MockLaunch>>,
    kernel_lookups: Mutex<Vec<(String, String)>>,
    /// Copy/sync shape counters. These exist so tests can assert the SHAPE of a
    /// bulk transfer, not just its bytes: the SSM snapshot spill regressed to
    /// 60 blocking `copy_d2h` calls (one full stream drain each, ~400 ms for
    /// 66 MB) and nothing caught it, because the bytes were correct.
    syncs: AtomicUsize,
    d2h_blocking: AtomicUsize,
    d2h_async: AtomicUsize,
    d2h_async_streams: Mutex<Vec<u64>>,
    /// `(stream, completed D2H enqueue count)` at every synchronize call.
    sync_d2h_async_counts: Mutex<Vec<(u64, usize)>>,
    /// `copy_d2d`/`copy_d2d_async` calls — one eager launch each on the real
    /// backend. The SSM verify rollback issued 2 per SSM layer per sequence
    /// (96 on the 27B), so this counter is what proves a batched form
    /// actually batched rather than merely looking different.
    d2d: AtomicUsize,
    /// `copy_d2d_2d_async` calls — ONE `cudaMemcpy2DAsync` each on the real
    /// backend regardless of `height`. Counted apart from `d2d` so a test can
    /// assert the SHAPE of the transfer, not just the bytes.
    d2d_2d: AtomicUsize,
    /// Streams supplied to asynchronous D2D copies, in dispatch order. Byte
    /// movement alone cannot expose an ordering bug caused by enqueuing a copy
    /// on the wrong stream, so stream-sensitive tests inspect this trace.
    d2d_async_streams: Mutex<Vec<u64>>,
    d2d_2d_async_streams: Mutex<Vec<u64>>,
    host_pinned_allocs: AtomicUsize,
}

#[derive(Debug, Clone)]
pub struct MockLaunch {
    pub func: u64,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_mem: u32,
    pub stream: u64,
    pub args: Vec<MockArg>,
}

/// Owned copy of a typed kernel argument at mock dispatch time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MockArg {
    Buffer(DevicePtr),
    Bytes(Vec<u8>),
}

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
    fn blit(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
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
}

/// Find the allocation containing `ptr` (supports offset pointers).
fn find_alloc(allocs: &HashMap<u64, MockAlloc>, ptr: DevicePtr) -> Option<(usize, &MockAlloc)> {
    for (&base, alloc) in allocs.iter() {
        if ptr.0 >= base && ptr.0 < base + alloc.bytes as u64 {
            return Some(((ptr.0 - base) as usize, alloc));
        }
    }
    None
}

/// Mutable version of find_alloc.
fn find_alloc_mut(
    allocs: &mut HashMap<u64, MockAlloc>,
    ptr: DevicePtr,
) -> Option<(usize, &mut MockAlloc)> {
    for (&base, alloc) in allocs.iter_mut() {
        if ptr.0 >= base && ptr.0 < base + alloc.bytes as u64 {
            return Some(((ptr.0 - base) as usize, alloc));
        }
    }
    None
}

impl GpuBackend for MockGpuBackend {
    fn op_cache(&self) -> &crate::op_cache::OpCache {
        &self.op_cache
    }

    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let limit = self.max_allocation_bytes.load(Ordering::Relaxed);
        if bytes > limit {
            anyhow::bail!("alloc: requested {bytes} bytes exceeds mock limit {limit}");
        }
        let mut next = self.next_ptr.lock();
        let ptr = *next;
        *next += bytes as u64;
        // Align to 256 bytes
        *next = (*next + 255) & !255;
        self.allocs.lock().insert(
            ptr,
            MockAlloc {
                bytes,
                data: vec![0u8; bytes],
            },
        );
        Ok(DevicePtr(ptr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.alloc(bytes) // Mock: same as regular alloc
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        if self.allocs.lock().remove(&ptr.0).is_none() {
            anyhow::bail!("free: ptr {ptr} is not an allocation base or is already free");
        }
        Ok(())
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        let mut allocs = self.allocs.lock();
        // Support offset pointers: find the allocation containing dst
        let (offset, alloc) = find_alloc_mut(&mut allocs, dst)
            .ok_or_else(|| anyhow::anyhow!("copy_h2d: ptr {dst} not allocated"))?;
        alloc.data[offset..offset + src.len()].copy_from_slice(src);
        Ok(())
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.d2h_blocking.fetch_add(1, Ordering::Relaxed);
        let allocs = self.allocs.lock();
        // Support offset pointers: find the allocation containing src
        let (offset, alloc) = find_alloc(&allocs, src)
            .ok_or_else(|| anyhow::anyhow!("copy_d2h: ptr {src} not allocated"))?;
        dst.copy_from_slice(&alloc.data[offset..offset + dst.len()]);
        Ok(())
    }

    fn copy_d2h_async(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        // Counted separately from `copy_d2h` and NOT delegating to it, so a
        // test can distinguish the batched shape from the blocking one (the
        // trait's default impl forwards, which would make them indistinguishable).
        self.d2h_async.fetch_add(1, Ordering::Relaxed);
        self.d2h_async_streams.lock().push(stream);
        let allocs = self.allocs.lock();
        let (offset, alloc) = find_alloc(&allocs, src)
            .ok_or_else(|| anyhow::anyhow!("copy_d2h_async: ptr {src} not allocated"))?;
        dst.copy_from_slice(&alloc.data[offset..offset + dst.len()]);
        Ok(())
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.d2d.fetch_add(1, Ordering::Relaxed);
        self.blit(src, dst, bytes)
    }

    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        // NOT delegating to `copy_d2d`: the trait default forwards, which
        // would make the two indistinguishable to `d2d_count` consumers only
        // by accident. Counted here so both forms land in one counter on
        // purpose.
        self.d2d.fetch_add(1, Ordering::Relaxed);
        self.d2d_async_streams.lock().push(stream);
        self.blit(src, dst, bytes)
    }

    fn copy_d2d_2d_async(
        &self,
        src: DevicePtr,
        src_pitch: usize,
        dst: DevicePtr,
        dst_pitch: usize,
        width_bytes: usize,
        height: usize,
        stream: u64,
    ) -> Result<()> {
        // ONE launch on the real backend (`cudaMemcpy2DAsync`), so ONE tick —
        // the row loop below is emulation, not dispatch, and must not inflate
        // `d2d_count` (which exists to prove a batched form batched).
        self.d2d_2d.fetch_add(1, Ordering::Relaxed);
        self.d2d_2d_async_streams.lock().push(stream);
        if width_bytes > src_pitch || width_bytes > dst_pitch {
            anyhow::bail!(
                "copy_d2d_2d_async: width {width_bytes} exceeds pitch \
                 (src {src_pitch}, dst {dst_pitch}) — rows would overlap"
            );
        }
        for r in 0..height {
            self.blit(
                src.offset(r * src_pitch),
                dst.offset(r * dst_pitch),
                width_bytes,
            )?;
        }
        Ok(())
    }

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        _params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.launches.lock().push(MockLaunch {
            func: func.0,
            grid,
            block,
            shared_mem,
            stream,
            args: Vec::new(),
        });
        Ok(())
    }

    fn launch_typed(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        let args = args
            .iter()
            .map(|arg| match arg {
                KernelArg::Buffer(ptr) => MockArg::Buffer(*ptr),
                KernelArg::Bytes(bytes) => MockArg::Bytes(bytes.to_vec()),
            })
            .collect();
        self.launches.lock().push(MockLaunch {
            func: func.0,
            grid,
            block,
            shared_mem,
            stream,
            args,
        });
        Ok(())
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        self.sync_d2h_async_counts
            .lock()
            .push((stream, self.d2h_async.load(Ordering::Relaxed)));
        self.syncs.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        0
    }

    #[track_caller]
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        self.kernel_lookups
            .lock()
            .push((module.to_owned(), func_name.to_owned()));
        Ok(KernelHandle(0xDEAD))
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        let mut allocs = self.allocs.lock();
        let (offset, alloc) = find_alloc_mut(&mut allocs, ptr)
            .ok_or_else(|| anyhow::anyhow!("memset: ptr {ptr} not allocated"))?;
        alloc.data[offset..offset + bytes].fill(value);
        Ok(())
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, _stream: u64) -> Result<()> {
        self.memset(ptr, value, bytes)
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        // Same heap allocation as the trait default (the mock cannot page-lock);
        // overridden solely to COUNT, so a test can prove a staging buffer is
        // allocated once and reused rather than per event.
        self.host_pinned_allocs.fetch_add(1, Ordering::Relaxed);
        let layout = std::alloc::Layout::from_size_align(bytes, 64)
            .map_err(|e| anyhow::anyhow!("invalid layout: {e}"))?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            anyhow::bail!("host alloc failed: {bytes} bytes");
        }
        Ok(ptr)
    }

    fn total_memory(&self) -> Result<usize> {
        Ok(128 * 1024 * 1024 * 1024) // 128 GB
    }

    fn sm_count(&self) -> Result<u32> {
        // Rationale (PCND): tests that exercise occupancy-gated dispatch need
        // SOME machine width; 48 is the GB10 value the model targets, chosen
        // so mock runs take the same branch production does.
        Ok(48)
    }

    fn free_memory(&self) -> Result<usize> {
        Ok(120 * 1024 * 1024 * 1024) // 120 GB
    }
}
