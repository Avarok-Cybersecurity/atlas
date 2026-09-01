// SPDX-License-Identifier: AGPL-3.0-only

//! GPU backend abstraction (SBIO IORouter for GPU operations).
//!
//! All CUDA interactions flow through [`GpuBackend`]. Business logic
//! (model forward pass, KV cache management) never calls cuLaunchKernel
//! or cuMemAlloc directly.

use anyhow::Result;
use std::fmt;
use std::sync::atomic::Ordering;
// The free-memory baseline is a field of the single run mailbox,
// `crate::run_metrics::RunMetrics`: it is read by the dashboard and by KV
// sizing from threads with no carrier, and it is cleared at run start so a
// second model measures against its own baseline rather than the first
// model's pre-load free memory.

/// Record the free-memory baseline at GPU-context init. Call once, early,
/// before weight loading. Idempotent-last-write; intended to be set exactly once.
pub fn set_baseline_free_bytes(bytes: usize) {
    crate::run_metrics::metrics()
        .baseline_free_bytes
        .store(bytes, Ordering::Relaxed);
}

/// The free-memory baseline captured at context init, or `None` if never set.
pub fn baseline_free_bytes() -> Option<usize> {
    match crate::run_metrics::metrics()
        .baseline_free_bytes
        .load(Ordering::Relaxed)
    {
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

pub use crate::gpu_args::pack_kernel_args;

/// GPU backend trait — SBIO IORouter for all CUDA operations.
///
/// Implementations: `AtlasCudaBackend` (production), `MockGpuBackend` (tests).
pub trait GpuBackend: Send + Sync {
    /// Allocate `bytes` of device memory.
    ///
    /// `#[track_caller]` so the CUDA backend's ledger records WHICH code
    /// asked for the memory. It must stay on the trait declaration as well as
    /// the impl: nearly every caller goes through `&dyn GpuBackend`, and
    /// without it here the vtable would attribute every allocation in the
    /// process to the one line inside the backend.
    #[track_caller]
    fn alloc(&self, bytes: usize) -> Result<DevicePtr>;

    /// Allocate managed (unified) memory. On GB10, this allows over-subscribing
    /// physical GPU memory — Linux pages overflow to NVMe swap automatically.
    /// Managed memory is slower than device memory but avoids OOM.
    #[track_caller]
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr>;

    /// Free device memory.
    fn free(&self, ptr: DevicePtr) -> Result<()>;

    /// Free every allocation this backend made that nobody released, and
    /// report how many there were.
    ///
    /// The teardown backstop. Enumerating owners does not scale: the loaders
    /// fuse weights into fresh allocations owned by layer structs, which no
    /// pool releases — measured at 15.3 GB per cycle on a 27B, linear over six
    /// cycles. A backend is created per model, so its outstanding set IS that
    /// model's leak.
    ///
    /// Default `0`: a backend that does not track allocations has nothing to
    /// sweep, which is honest for the mock and for Metal.
    fn sweep_unreleased(&self) -> usize {
        0
    }

    /// Live device bytes this backend has allocated and not freed, if it
    /// tracks them. `None` for backends with no ledger (mock/CPU).
    fn live_bytes(&self) -> Option<usize> {
        None
    }

    /// Attribution of live device memory by allocating call site, biggest
    /// first. `None` for backends with no ledger.
    fn alloc_report(&self, _top_n: usize, _min_mb: usize) -> Option<String> {
        None
    }

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
        let (storage, starts) = pack_kernel_args(args);
        let mut params: Vec<*mut std::ffi::c_void> = starts
            .iter()
            .map(|&i| &storage[i] as *const u64 as *mut std::ffi::c_void)
            .collect();
        self.launch(func, grid, block, shared_mem, stream, &mut params)
    }

    /// Cooperative kernel launch (`cuLaunchCooperativeKernel`).
    ///
    /// Every block of the grid is resident on the device for the kernel's
    /// whole lifetime, which is what makes an in-kernel `grid.sync()` legal —
    /// the EXL3 trellis GEMM/GEMV kernels rely on it for their split-k
    /// lock/barrier protocol. The GRID MUST FIT: callers size the grid from
    /// [`GpuBackend::sm_count`] (occupancy-capped, per the upstream formula),
    /// because a grid one block too large fails the launch outright rather
    /// than serializing.
    ///
    /// Unlike `launch`, this path does NOT auto-raise the kernel's dynamic
    /// shared memory cap: a kernel that wants more than the 48 KB default
    /// (EXL3 GEMM asks for 90 KB) must be opted in ONCE at kernel resolution
    /// via [`GpuBackend::set_kernel_max_dynamic_smem`].
    ///
    /// Default: unsupported. A backend without co-resident-grid semantics
    /// must refuse rather than silently launch non-cooperatively — the
    /// kernel's `grid.sync()` would deadlock or race.
    fn launch_cooperative(
        &self,
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        anyhow::bail!("launch_cooperative: not supported by this backend")
    }

    /// Typed-args cooperative launch — `launch_typed`'s cooperative twin,
    /// reached from `KernelLaunch::cooperative()`. The default packs args
    /// into u64 slots exactly like `launch_typed` and forwards to
    /// `launch_cooperative`; the mock overrides it to record the launch.
    fn launch_cooperative_typed(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        let (storage, starts) = pack_kernel_args(args);
        let mut params: Vec<*mut std::ffi::c_void> = starts
            .iter()
            .map(|&i| &storage[i] as *const u64 as *mut std::ffi::c_void)
            .collect();
        self.launch_cooperative(func, grid, block, shared_mem, stream, &mut params)
    }

    /// Raise `kernel`'s dynamic shared memory cap to `bytes`
    /// (`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`).
    ///
    /// Kernels default to a 48 KB dynamic-smem ceiling; anything larger
    /// (EXL3 GEMM: 90 KB) must opt in per FUNCTION before the first launch.
    /// The attribute is sticky on the `CUfunction`, so call this ONCE where
    /// the kernel handle is resolved — a layer constructor — never per
    /// launch. The eager `launch` path auto-raises inside the registry;
    /// `launch_cooperative` deliberately does not, and this method is how
    /// its callers pay that cost up front.
    ///
    /// Default no-op: backends with no such attribute (mock, Metal) have
    /// nothing to raise and must not fail the caller's init for it.
    fn set_kernel_max_dynamic_smem(&self, _kernel: KernelHandle, _bytes: usize) -> Result<()> {
        Ok(())
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
    ///
    /// `#[track_caller]` on the DECLARATION is what makes the caller location
    /// survive the `&dyn GpuBackend` vtable — every lookup in Atlas goes
    /// through dynamic dispatch, so without it the audit can only ever name
    /// the backend's own line. The location is what turns an unresolved-lookup
    /// report from a name list into a work item.
    #[track_caller]
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle>;

    /// This backend's memoized kernel handles and scratch allocations.
    ///
    /// Required rather than defaulted: an op that memoizes a `KernelHandle`
    /// or a `DevicePtr` anywhere else is caching something that belongs to
    /// this backend's model, and a default would let a new backend forget.
    fn op_cache(&self) -> &crate::op_cache::OpCache;

    /// Synchronise the stream after every kernel launch, so an asynchronous
    /// illegal-address fault is reported at the kernel that caused it rather
    /// than at a later sync. Resolved once when the backend is built; read on
    /// the launch path, which is why it is not a per-launch `getenv`.
    fn debug_sync_kernels(&self) -> bool {
        false
    }

    /// This backend's model-scoped kernel modules, for the few callers that
    /// need the registry itself rather than a kernel handle — resolving a
    /// `__device__` symbol, for instance. `None` on backends that have no such
    /// concept, which is why it is an accessor rather than a downcast.
    #[cfg(feature = "cuda")]
    fn kernel_registry(&self) -> Option<std::sync::Arc<atlas_core::registry::AtlasRegistry>> {
        None
    }

    /// Async host-to-device copy: **`src` may be dropped or overwritten the
    /// moment this returns.**
    ///
    /// That is what the ~90 call sites in `spark-model` rely on — nearly all
    /// hand over a stack array or local `Vec` that dies at the end of the
    /// statement — and it used to hold only by accident. See
    /// [`crate::pinned_hosts`] for why, and for how the CUDA backend now MAKES
    /// the promise true (page-locked source ⇒ it buys the ordering that the
    /// pageable path gets from the driver for free) instead of inheriting it.
    ///
    /// Use [`GpuBackend::copy_h2d_async_retained`] when the source outlives the
    /// next synchronisation and the extra ordering is not wanted.
    fn copy_h2d_async(&self, src: &[u8], dst: DevicePtr, _stream: u64) -> Result<()> {
        self.copy_h2d(src, dst)
    }

    /// Async host-to-device copy for a source the CALLER keeps alive.
    ///
    /// `src` must remain valid, and must not be rewritten, until the next
    /// synchronisation point on `stream`. In exchange it never inserts an
    /// implicit sync — what makes a batched scatter out of one pinned staging
    /// blob (N enqueues + one `synchronize`) worth doing; see
    /// [`GpuBackend::copy_d2h_async`] for the measured shape. The name marks,
    /// greppably, every site making a promise the compiler cannot check.
    fn copy_h2d_async_retained(&self, src: &[u8], dst: DevicePtr, stream: u64) -> Result<()> {
        // Default: the transient path. Strictly stronger ordering than promised,
        // so it is always correct — just not always the fastest.
        self.copy_h2d_async(src, dst, stream)
    }

    /// Async device-to-host copy (no stream synchronization).
    ///
    /// The counterpart of [`GpuBackend::copy_h2d_async`], and the ONLY D2H
    /// primitive usable for a batched gather: `copy_d2h` and
    /// `copy_d2h_on_stream` both `cuStreamSynchronize` INSIDE the call, so an
    /// N-chunk gather pays N full stream drains. Measured cost of that shape:
    /// the SSM snapshot spill moved 66,846,720 B as 60 blocking `copy_d2h`
    /// calls in ~400 ms (~165 MB/s), while the mirror-image scatter
    /// (`copy_h2d_async` ×60 + ONE `synchronize`) moved the same bytes through
    /// the same host buffer in ~28 ms.
    ///
    /// **Lifetime requirement** (same as `copy_h2d_async`): the destination
    /// buffer must remain valid, and must not be read or re-used, until the
    /// next synchronization point on this stream.
    fn copy_d2h_async(&self, src: DevicePtr, dst: &mut [u8], _stream: u64) -> Result<()> {
        // Mock/metal fall back to the blocking copy: correct (a stricter
        // ordering than promised), just not batched.
        self.copy_d2h(src, dst)
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

    /// Best-effort: if `stream` is mid graph-capture, end that capture so the
    /// stream returns to normal mode (discarding any partial graph). Call this
    /// on an error path that unwound out of a `begin_capture`/`end_capture`
    /// region (e.g. a fold refuse bailed mid-capture) — otherwise the stream is
    /// left recording and every subsequent op fails with
    /// STREAM_CAPTURE_UNSUPPORTED, bricking the server. No-op if not capturing.
    fn abort_capture_if_active(&self, _stream: u64) {}

    /// Set device memory to a byte value (synchronous — waits for completion).
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()>;

    /// Set device memory to a byte value on the given stream (async — does not wait).
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()>;

    /// Total device memory in bytes.
    fn total_memory(&self) -> Result<usize>;

    /// Free device memory in bytes.
    fn free_memory(&self) -> Result<usize>;

    /// Number of streaming multiprocessors (CUDA SMs / HIP CUs) on the device.
    ///
    /// Queried from the driver, never assumed: dispatch rules that ask "does
    /// this grid still fill the machine?" are wrong on every part whose SM
    /// count differs from the one they were tuned on. Callers must resolve it
    /// ONCE at construction and keep the value, not call it per launch.
    fn sm_count(&self) -> Result<u32>;

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

    /// Device-side alias of a page-locked host pointer from
    /// [`Self::alloc_host_pinned`] (cuMemHostGetDevicePointer). On UMA parts
    /// (GB10) this lets a KERNEL write results directly into host-visible
    /// memory, eliminating the copy-engine op for tiny readbacks entirely.
    /// Default: unsupported.
    fn host_ptr_to_device(&self, _host: *mut u8) -> Result<DevicePtr> {
        anyhow::bail!("host_ptr_to_device: not supported by this backend")
    }

    /// Allocate page-locked (pinned) host memory for efficient async H2D.
    ///
    /// On DGX Spark (UMA/LPDDR5X), pinned memory enables true async DMA
    /// without internal CUDA staging overhead. Small metadata buffers
    /// should be packed into a single pinned region and copied in one call.
    ///
    /// Returns a raw pointer to `bytes` of page-locked host memory.
    /// Caller must call `free_host_pinned` to release.
    ///
    /// **The returned region is ZEROED.** Callers pack these buffers with
    /// alignment padding between fields and then form a `&[u8]` over the whole
    /// packed range for one `copy_h2d`; a slice over a never-written byte is UB
    /// no matter what the device later does with it. Every implementation must
    /// uphold this — `cuMemAllocHost_v2` and `newBufferWithLength` do not zero
    /// on their own and their wrappers memset explicitly.
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
#[path = "gpu_tests.rs"]
mod tests;
