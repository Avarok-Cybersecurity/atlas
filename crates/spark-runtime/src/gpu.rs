// SPDX-License-Identifier: AGPL-3.0-only

//! GPU backend abstraction (SBIO IORouter for GPU operations).
//!
//! All CUDA interactions flow through [`GpuBackend`]. Business logic
//! (model forward pass, KV cache management) never calls cuLaunchKernel
//! or cuMemAlloc directly.

use anyhow::Result;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Free device memory (bytes) captured once at GPU-context init, BEFORE any
/// model weights or buffers are allocated. Lets the KV-budget sizing measure
/// *this process's own* footprint as `baseline_free - free_now`, which excludes
/// co-tenant memory automatically (vs trusting a hardcoded
/// `ATLAS_KV_EXTERNAL_RESERVE_GB` that goes stale as co-tenants come and go).
/// 0 = unset (e.g. under the mock backend in tests) → callers fall back.
static BASELINE_FREE_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Record the free-memory baseline at GPU-context init. Call once, early,
/// before weight loading. Idempotent-last-write; intended to be set exactly once.
pub fn set_baseline_free_bytes(bytes: usize) {
    BASELINE_FREE_BYTES.store(bytes, Ordering::Relaxed);
}

/// The free-memory baseline captured at context init, or `None` if never set.
pub fn baseline_free_bytes() -> Option<usize> {
    match BASELINE_FREE_BYTES.load(Ordering::Relaxed) {
        0 => None,
        v => Some(v),
    }
}

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

/// Typed kernel argument, used by `launch_typed`.
///
/// CUDA's `cuLaunchKernel` is type-blind — every arg is `void*` and the
/// driver interprets bytes by kernel signature. Metal's
/// `MTLComputeCommandEncoder` is not: buffer arguments require
/// `setBuffer:offset:atIndex:` (the encoder tracks the resource) while
/// scalar/struct args require `setBytes:length:atIndex:`. `KernelArg`
/// preserves that distinction so both backends can dispatch correctly.
#[derive(Debug, Clone, Copy)]
pub enum KernelArg<'a> {
    /// A device buffer at this base GPU address. The metal backend
    /// resolves it to its owning `MTLBuffer` + offset via the alloc
    /// registry; the cuda backend forwards the raw `u64` to the driver.
    Buffer(DevicePtr),
    /// Inline scalar/struct bytes, e.g. a `u32` count or an `f32` eps.
    /// Length is forwarded to Metal's `setBytes:length:`; the cuda
    /// backend zero-pads up to 8 bytes per slot.
    Bytes(&'a [u8]),
}

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

    /// Typed-args kernel launch.
    ///
    /// CUDA's default impl packs args into u64 slots and forwards to
    /// `launch()`. The Metal backend overrides this to map each
    /// `KernelArg::Buffer` to `setBuffer:offset:atIndex:` and each
    /// `KernelArg::Bytes` to `setBytes:length:atIndex:`.
    fn launch_typed(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        // CUDA-compatible default: each arg becomes one u64 slot. The
        // storage stays alive across the launch call so the *mut c_void
        // pointers we hand to `launch()` remain valid.
        let mut storage: Vec<u64> = Vec::with_capacity(args.len());
        for arg in args {
            match arg {
                KernelArg::Buffer(p) => storage.push(p.0),
                KernelArg::Bytes(b) => {
                    let mut slot = [0u8; 8];
                    let n = b.len().min(8);
                    slot[..n].copy_from_slice(&b[..n]);
                    storage.push(u64::from_le_bytes(slot));
                }
            }
        }
        let mut params: Vec<*mut std::ffi::c_void> = storage
            .iter()
            .map(|v| v as *const u64 as *mut std::ffi::c_void)
            .collect();
        self.launch(func, grid, block, shared_mem, stream, &mut params)
    }

    /// Whether `stream` is inside an active CUDA-graph capture. Telemetry
    /// taps MUST check this before any sync/D2H on a potentially-captured
    /// stream — those calls invalidate the capture (CUDA 901) and wedge the
    /// serve. Default `false` (backends without capture, or without a query
    /// API, never capture through this trait's eager paths).
    fn stream_is_capturing(&self, _stream: u64) -> bool {
        false
    }

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

    /// Strided device-to-device 2D (pitched) copy: `height` rows of
    /// `width_bytes`, source rows spaced by `src_pitch`, dest rows by
    /// `dst_pitch`. Default = per-row `copy_d2d_async` loop; the CUDA backend
    /// overrides with ONE `cudaMemcpy2DAsync` (replaces the per-token Z-copy
    /// loop = up to num_tokens×num_ssm_layers launches/forward).
    #[allow(clippy::too_many_arguments)]
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
        for r in 0..height {
            self.copy_d2d_async(
                src.offset(r * src_pitch),
                dst.offset(r * dst_pitch),
                width_bytes,
                stream,
            )?;
        }
        Ok(())
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

    /// Block the calling host thread until all work already
    /// recorded against the event — e.g. an async D2H copy issued on the
    /// graph stream followed by `record_event`, then `event_synchronize`
    /// right before the host dereferences the destination pinned buffer.
    /// Cheaper than `synchronize(stream)` when the stream has work beyond
    /// the event you care about: this only waits for the recorded point,
    /// not for everything subsequently enqueued.
    fn event_synchronize(&self, _event: u64) -> Result<()> {
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
pub mod mock;

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
