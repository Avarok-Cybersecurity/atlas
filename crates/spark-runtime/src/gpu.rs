// SPDX-License-Identifier: AGPL-3.0-only

//! GPU backend abstraction (SBIO IORouter for GPU operations).
//!
//! All CUDA interactions flow through [`GpuBackend`]. Business logic
//! (model forward pass, KV cache management) never calls cuLaunchKernel
//! or cuMemAlloc directly.

use anyhow::Result;
use std::fmt;

/// Opaque device pointer wrapping a CUDA CUdeviceptr (u64).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePtr(pub u64);

impl DevicePtr {
    pub const NULL: Self = Self(0);

    pub fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Byte offset from this pointer.
    pub fn offset(self, bytes: usize) -> Self {
        Self(self.0 + bytes as u64)
    }
}

/// Handle to a loaded CUDA kernel function.
#[derive(Debug, Clone, Copy)]
pub struct KernelHandle(pub u64);

/// Handle to an instantiated CUDA graph (CUgraphExec).
#[derive(Debug, Clone, Copy)]
pub struct GraphHandle(pub u64);

/// GPU backend trait — SBIO IORouter for all CUDA operations.
///
/// Implementations: `AtlasCudaBackend` (production), `MockGpuBackend` (tests).
pub trait GpuBackend: Send + Sync {
    /// Allocate `bytes` of device memory.
    fn alloc(&self, bytes: usize) -> Result<DevicePtr>;

    /// Allocate managed (unified) memory. On GB10, this allows over-subscribing
    /// physical GPU memory — Linux pages overflow to NVMe swap automatically.
    /// Managed memory is slower than device memory but avoids OOM.
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr>;

    /// Free device memory.
    fn free(&self, ptr: DevicePtr) -> Result<()>;

    /// Copy from host to device.
    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()>;

    /// Copy from device to host.
    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()>;

    /// Synchronous device-to-host copy ordered after work on `stream`.
    ///
    /// Unlike `copy_d2h` (which uses the default stream and only orders
    /// against work already on the default stream), this method enqueues
    /// the copy on `stream`. CUDA serializes the copy after any prior
    /// kernel launches on `stream`, so the bytes read are guaranteed to
    /// reflect post-kernel state.
    ///
    /// Required when reading bytes that were just written by kernels on
    /// a non-default stream — e.g. `high_speed_swap_offload_new_blocks`
    /// reading WHT+quantize output bytes.
    fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        // Default impl for mocks: sync the caller's stream then fall
        // back to copy_d2h. The CUDA backend overrides this for a
        // single-stream copy + sync.
        self.synchronize(stream)?;
        self.copy_d2h(src, dst)
    }

    /// Copy device to device.
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()>;

    /// Launch a kernel on the given CUDA stream.
    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<()>;

    /// Synchronize a CUDA stream (blocks until all work completes).
    fn synchronize(&self, stream: u64) -> Result<()>;

    /// Get the default stream handle.
    fn default_stream(&self) -> u64;

    /// Look up a kernel function by module and function name.
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle>;

    /// Async host-to-device copy (no stream synchronization).
    ///
    /// **Lifetime requirement**: the source buffer must remain valid until the
    /// copy completes (i.e., until the next synchronization point on this
    /// stream). All current callers use stack-local byte arrays or pinned
    /// memory that outlives the stream sync, satisfying this requirement.
    fn copy_h2d_async(&self, src: &[u8], dst: DevicePtr, _stream: u64) -> Result<()> {
        self.copy_h2d(src, dst)
    }

    /// Async device-to-device copy (no stream synchronization).
    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        _stream: u64,
    ) -> Result<()> {
        self.copy_d2d(src, dst, bytes)
    }

    /// Begin capturing CUDA operations on `stream` into a graph.
    ///
    /// All kernel launches and async copies on this stream between
    /// `begin_capture` and `end_capture` are recorded (not executed).
    /// The stream must NOT be the legacy default stream (handle 0).
    fn begin_capture(&self, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// End capture and return an instantiated graph ready for replay.
    fn end_capture(&self, _stream: u64) -> Result<GraphHandle> {
        Ok(GraphHandle(0))
    }

    /// Replay all operations captured in the graph on `stream`.
    fn launch_graph(&self, _graph: GraphHandle, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// Destroy an instantiated graph, freeing resources.
    fn destroy_graph(&self, _graph: GraphHandle) -> Result<()> {
        Ok(())
    }

    /// Set device memory to a byte value (synchronous — waits for completion).
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()>;

    /// Set device memory to a byte value on the given stream (async — does not wait).
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()>;

    /// Total device memory in bytes.
    fn total_memory(&self) -> Result<usize>;

    /// Free device memory in bytes.
    fn free_memory(&self) -> Result<usize>;

    /// Create a new CUDA stream (for overlapping work).
    fn create_stream(&self) -> Result<u64> {
        Ok(0) // Default: return legacy stream
    }

    /// Bind the CUDA context to the current thread.
    ///
    /// Must be called on any thread that uses GPU operations (alloc, launch, etc.)
    /// if it's different from the thread that created the backend.
    fn bind_to_thread(&self) -> Result<()> {
        Ok(()) // No-op for mock backend
    }

    /// Create a CUDA event (for inter-stream synchronization).
    fn create_event(&self) -> Result<u64> {
        Ok(0)
    }

    /// Record an event on a stream (marks a point in the stream's work).
    fn record_event(&self, _event: u64, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// Make a stream wait for an event (GPU-side sync, CPU does not block).
    fn stream_wait_event(&self, _stream: u64, _event: u64) -> Result<()> {
        Ok(())
    }

    /// Destroy an event.
    fn destroy_event(&self, _event: u64) -> Result<()> {
        Ok(())
    }

    /// Allocate page-locked (pinned) host memory for efficient async H2D.
    ///
    /// On DGX Spark (UMA/LPDDR5X), pinned memory enables true async DMA
    /// without internal CUDA staging overhead. Small metadata buffers
    /// should be packed into a single pinned region and copied in one call.
    ///
    /// Returns a raw pointer to `bytes` of page-locked host memory.
    /// Caller must call `free_host_pinned` to release.
    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        // Default: regular heap allocation (mock backend, no pinning)
        let layout = std::alloc::Layout::from_size_align(bytes, 64)
            .map_err(|e| anyhow::anyhow!("invalid layout: {e}"))?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            anyhow::bail!("host alloc failed: {bytes} bytes");
        }
        Ok(ptr)
    }

    /// Free page-locked host memory previously allocated by `alloc_host_pinned`.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn free_host_pinned(&self, ptr: *mut u8, bytes: usize) -> Result<()> {
        if !ptr.is_null() {
            let layout = std::alloc::Layout::from_size_align(bytes, 64)
                .map_err(|e| anyhow::anyhow!("invalid layout: {e}"))?;
            unsafe { std::alloc::dealloc(ptr, layout) };
        }
        Ok(())
    }
}

impl fmt::Display for DevicePtr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DevicePtr(0x{:x})", self.0)
    }
}

#[cfg(any(test, feature = "test-utils"))]
pub mod mock {
    //! Mock GPU backend for unit tests (no GPU required).

    use super::*;
    use parking_lot::Mutex;
    use std::collections::HashMap;

    #[derive(Debug)]
    pub struct MockAlloc {
        pub bytes: usize,
        pub data: Vec<u8>,
    }

    /// Records kernel launches and memory operations for test assertions.
    pub struct MockGpuBackend {
        allocs: Mutex<HashMap<u64, MockAlloc>>,
        next_ptr: Mutex<u64>,
        launches: Mutex<Vec<MockLaunch>>,
    }

    #[derive(Debug, Clone)]
    pub struct MockLaunch {
        pub func: u64,
        pub grid: [u32; 3],
        pub block: [u32; 3],
    }

    impl Default for MockGpuBackend {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MockGpuBackend {
        pub fn new() -> Self {
            Self {
                allocs: Mutex::new(HashMap::new()),
                next_ptr: Mutex::new(0x1000_0000),
                launches: Mutex::new(Vec::new()),
            }
        }

        pub fn alloc_count(&self) -> usize {
            self.allocs.lock().len()
        }

        pub fn launch_count(&self) -> usize {
            self.launches.lock().len()
        }

        pub fn read_alloc(&self, ptr: DevicePtr) -> Option<Vec<u8>> {
            self.allocs.lock().get(&ptr.0).map(|a| a.data.clone())
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
        fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
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
            self.allocs.lock().remove(&ptr.0);
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
            let allocs = self.allocs.lock();
            // Support offset pointers: find the allocation containing src
            let (offset, alloc) = find_alloc(&allocs, src)
                .ok_or_else(|| anyhow::anyhow!("copy_d2h: ptr {src} not allocated"))?;
            dst.copy_from_slice(&alloc.data[offset..offset + dst.len()]);
            Ok(())
        }

        fn copy_d2d(&self, _src: DevicePtr, _dst: DevicePtr, _bytes: usize) -> Result<()> {
            Ok(())
        }

        fn launch(
            &self,
            func: KernelHandle,
            grid: [u32; 3],
            block: [u32; 3],
            _shared_mem: u32,
            _stream: u64,
            _params: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            self.launches.lock().push(MockLaunch {
                func: func.0,
                grid,
                block,
            });
            Ok(())
        }

        fn synchronize(&self, _stream: u64) -> Result<()> {
            Ok(())
        }

        fn default_stream(&self) -> u64 {
            0
        }

        fn kernel(&self, _module: &str, _func_name: &str) -> Result<KernelHandle> {
            Ok(KernelHandle(0xDEAD))
        }

        fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
            let mut allocs = self.allocs.lock();
            let (offset, alloc) = find_alloc_mut(&mut allocs, ptr)
                .ok_or_else(|| anyhow::anyhow!("memset: ptr {ptr} not allocated"))?;
            alloc.data[offset..offset + bytes].fill(value);
            Ok(())
        }

        fn memset_async(
            &self,
            ptr: DevicePtr,
            value: u8,
            bytes: usize,
            _stream: u64,
        ) -> Result<()> {
            self.memset(ptr, value, bytes)
        }

        fn total_memory(&self) -> Result<usize> {
            Ok(128 * 1024 * 1024 * 1024) // 128 GB
        }

        fn free_memory(&self) -> Result<usize> {
            Ok(120 * 1024 * 1024 * 1024) // 120 GB
        }
    }
}

#[cfg(test)]
mod tests {
    use super::mock::MockGpuBackend;
    use super::*;

    #[test]
    fn test_mock_alloc_free() {
        let gpu = MockGpuBackend::new();
        let ptr = gpu.alloc(1024).unwrap();
        assert!(!ptr.is_null());
        assert_eq!(gpu.alloc_count(), 1);
        gpu.free(ptr).unwrap();
        assert_eq!(gpu.alloc_count(), 0);
    }

    #[test]
    fn test_mock_copy_roundtrip() {
        let gpu = MockGpuBackend::new();
        let ptr = gpu.alloc(8).unwrap();
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8];
        gpu.copy_h2d(&src, ptr).unwrap();
        let mut dst = [0u8; 8];
        gpu.copy_d2h(ptr, &mut dst).unwrap();
        assert_eq!(src, dst);
    }

    #[test]
    fn test_device_ptr_offset() {
        let ptr = DevicePtr(0x1000);
        assert_eq!(ptr.offset(256).0, 0x1100);
    }
}
