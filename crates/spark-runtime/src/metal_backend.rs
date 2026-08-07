// SPDX-License-Identifier: AGPL-3.0-only

//! Apple Metal GPU backend.
//!
//! Implements [`GpuBackend`] on top of the Metal framework via the
//! `objc2-metal` bindings. Apple Silicon is unified-memory (UMA), so
//! every `MTLBuffer` is allocated with `StorageModeShared` and host
//! `memcpy` against `buffer.contents()` is the canonical H2D/D2H path
//! — no PCIe staging, no pinned-host bounce.
//!
//! # Pointer model
//!
//! `DevicePtr` carries a real GPU virtual address obtained from
//! `MTLBuffer::gpuAddress()` (Metal 3+, native to all Apple Silicon).
//! That makes pointer arithmetic (`DevicePtr::offset`) a plain integer
//! add — no buffer/offset pair to thread through. To recover the
//! owning `MTLBuffer` for `free` / blit-copy / `setBuffer:`, we keep
//! a side table `BTreeMap<base_gpu_address, MTLBuffer>` and look up
//! the largest key ≤ ptr. The buffer's gpuAddress range is
//! `[base, base + length)`, so a binary search is enough.
//!
//! # Streams
//!
//! A stream handle indexes a slab of `MetalStream { queue, in_flight }`.
//! Handle 0 is the default stream and is lazily created on first use.
//! `synchronize(stream)` commits the in-flight `MTLCommandBuffer` and
//! `waitUntilCompleted()`s; the next encoder opens on a fresh buffer.
//!
//! # Kernel handles
//!
//! `KernelHandle` indexes a slab of `MTLComputePipelineState`. The
//! library cache (one `MTLLibrary` per `metallib_modules()` entry) is
//! built once at construction and never mutated; pipeline lookups go
//! through the slab + a `(module, fn_name)` HashMap so repeated
//! `kernel()` calls are O(1) cached.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLEvent, MTLLibrary, MTLResource, MTLResourceOptions, MTLSharedEvent, MTLSize,
};
use parking_lot::Mutex;

use crate::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};

// ── Internal type aliases (Retained<ProtocolObject<dyn _>> is verbose) ────

type ObjDevice = Retained<ProtocolObject<dyn MTLDevice>>;
type ObjBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type ObjQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type ObjCmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type ObjLibrary = Retained<ProtocolObject<dyn MTLLibrary>>;
type ObjPipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type ObjSharedEvent = Retained<ProtocolObject<dyn MTLSharedEvent>>;

// ── Stream + slab types ──────────────────────────────────────────────────

struct MetalStream {
    queue: ObjQueue,
    /// In-flight command buffer accumulating encoded work. Committed +
    /// waited on by `synchronize()`; replaced by a fresh buffer on
    /// next encoder open.
    in_flight: Option<ObjCmdBuf>,
}

/// Tracks one outstanding shared event so `record_event` can write the
/// next signal value and `stream_wait_event` can wait on the same
/// counter.
struct EventSlot {
    event: ObjSharedEvent,
    /// Monotonic value sequence — record_event signals `next`, then
    /// increments. stream_wait_event waits on `next - 1` (the most
    /// recently recorded value). Atomic via the surrounding Mutex.
    next: u64,
}

/// Key for the pipeline cache. Stored as owned strings because the
/// `&str` arguments to `kernel()` come from arbitrary call sites.
type PipelineKey = (String, String);

// ── MetalGpuBackend struct + state ───────────────────────────────────────

pub struct MetalGpuBackend {
    op_cache: crate::op_cache::OpCache,
    device: ObjDevice,
    /// Side table mapping a buffer's base gpuAddress to the owning
    /// `MTLBuffer`. BTreeMap so we can find the buffer containing an
    /// arbitrary `DevicePtr` via `range(..=ptr).next_back()`.
    allocations: Arc<Mutex<BTreeMap<u64, ObjBuffer>>>,
    /// Stream slab. Indexed by `stream_handle - 1`; handle 0 is the
    /// implicit default stream materialized lazily into slot 0.
    streams: Arc<Mutex<Vec<MetalStream>>>,
    /// Loaded metallibs keyed by module name.
    libraries: HashMap<String, ObjLibrary>,
    /// Pipeline-state cache + slab. The HashMap maps `(module, fn)` to
    /// the slab index; the slab owns the `MTLComputePipelineState`.
    /// Both are mutexed so `kernel()` can be called from any thread.
    pipeline_cache: Arc<Mutex<HashMap<PipelineKey, KernelHandle>>>,
    pipeline_slab: Arc<Mutex<Vec<ObjPipeline>>>,
    /// Shared-event slab for cross-stream synchronization.
    events: Arc<Mutex<Vec<EventSlot>>>,
}

unsafe impl Send for MetalGpuBackend {}
unsafe impl Sync for MetalGpuBackend {}

impl MetalGpuBackend {
    /// Initialize the Metal backend with the embedded metallib modules.
    ///
    /// `kernel_modules` is the `metallib_modules()` slice produced by
    /// `atlas-kernels`' build script — `(module_name, metallib_bytes)`.
    /// Each entry is loaded into its own `MTLLibrary` via
    /// `newLibraryWithData_error:`. The default stream (handle 0) is
    /// materialized eagerly so the first launch doesn't pay queue-
    /// creation latency.
    pub fn new(ordinal: usize, kernel_modules: &[(&'static str, &'static [u8])]) -> Result<Self> {
        if ordinal != 0 {
            bail!(
                "Metal: only ordinal 0 is supported (Apple Silicon has one \
                 system default device); requested ordinal {ordinal}"
            );
        }
        let device: ObjDevice = MTLCreateSystemDefaultDevice().ok_or_else(|| {
            anyhow!("MTLCreateSystemDefaultDevice returned null — no Metal-capable GPU")
        })?;

        // Build the library cache up-front. `newLibraryWithData_error`
        // takes a `DispatchData`, which is libdispatch's reference-
        // counted byte container. We wrap the &'static slice via
        // dispatch2::DispatchData (zero-copy) — the metallibs are
        // embedded by include_bytes! and outlive the backend.
        let mut libraries: HashMap<String, ObjLibrary> = HashMap::new();
        for (name, bytes) in kernel_modules {
            let data = dispatch2::DispatchData::from_static_bytes(bytes);
            let lib = device.newLibraryWithData_error(&data).map_err(|e| {
                anyhow!(
                    "newLibraryWithData failed for module '{name}': {}",
                    e.localizedDescription()
                )
            })?;
            libraries.insert((*name).to_string(), lib);
        }

        // Materialize the default stream eagerly (slot 0 = handle 0).
        let default_queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("newCommandQueue returned null on default device"))?;
        let streams = vec![MetalStream {
            queue: default_queue,
            in_flight: None,
        }];

        tracing::info!(
            "MetalGpuBackend initialized on device '{}' with {} metallib modules",
            device.name().to_string(),
            libraries.len()
        );

        Ok(Self {
            op_cache: crate::op_cache::OpCache::new(),
            device,
            allocations: Arc::new(Mutex::new(BTreeMap::new())),
            streams: Arc::new(Mutex::new(streams)),
            libraries,
            pipeline_cache: Arc::new(Mutex::new(HashMap::new())),
            pipeline_slab: Arc::new(Mutex::new(Vec::new())),
            events: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Return the underlying `MTLDevice` (escape hatch for advanced
    /// use cases — graph capture, custom resource creation, etc.).
    pub fn raw_device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    // ── Internal helpers ────────────────────────────────────────────

    /// Look up the `MTLBuffer` owning `ptr` and the byte offset of
    /// `ptr` within it. Returns `None` if no allocation contains it.
    fn find_buffer(
        allocs: &BTreeMap<u64, ObjBuffer>,
        ptr: DevicePtr,
    ) -> Option<(ObjBuffer, usize)> {
        let (base, buf) = allocs.range(..=ptr.0).next_back()?;
        let offset = (ptr.0 - *base) as usize;
        if offset > buf.length() {
            return None;
        }
        Some((buf.clone(), offset))
    }

    /// Resolve a stream handle to its slab index. Handle 0 → slot 0
    /// (default stream); other handles index `handle - 1`.
    fn stream_index(handle: u64, slab: &[MetalStream]) -> Result<usize> {
        let idx = if handle == 0 {
            0
        } else {
            (handle - 1) as usize
        };
        if idx >= slab.len() {
            bail!("Metal: invalid stream handle {handle}");
        }
        Ok(idx)
    }

    /// Borrow (or open) the in-flight command buffer on the given
    /// stream. Returns a clone of the `Retained` so the caller can
    /// encode without holding the streams mutex across encoder calls.
    fn current_cmd_buf(&self, stream_handle: u64) -> Result<ObjCmdBuf> {
        let mut slab = self.streams.lock();
        let idx = Self::stream_index(stream_handle, &slab)?;
        let s = &mut slab[idx];
        if let Some(ref cb) = s.in_flight {
            return Ok(cb.clone());
        }
        let cb = s
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("commandBuffer returned null on stream {stream_handle}"))?;
        s.in_flight = Some(cb.clone());
        Ok(cb)
    }

    /// Commit the in-flight buffer on `stream_handle` (no wait). Used
    /// internally by `synchronize` and `record_event`. Returns the
    /// committed buffer so callers that need to `waitUntilCompleted()`
    /// can.
    fn commit_in_flight(&self, stream_handle: u64) -> Result<Option<ObjCmdBuf>> {
        let mut slab = self.streams.lock();
        let idx = Self::stream_index(stream_handle, &slab)?;
        let s = &mut slab[idx];
        let Some(cb) = s.in_flight.take() else {
            return Ok(None);
        };
        cb.commit();
        Ok(Some(cb))
    }
}

// ── GpuBackend impl ──────────────────────────────────────────────────────

impl GpuBackend for MetalGpuBackend {
    fn op_cache(&self) -> &crate::op_cache::OpCache {
        &self.op_cache
    }

    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        // StorageModeShared is the UMA-friendly mode: `contents()`
        // returns a CPU-mappable pointer that aliases GPU memory.
        let buf: ObjBuffer = self
            .device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("newBufferWithLength failed for {bytes} bytes"))?;
        let addr = buf.gpuAddress();
        if addr == 0 {
            bail!("MTLBuffer::gpuAddress returned 0 — Metal 3 / macOS 13 required");
        }
        self.allocations.lock().insert(addr, buf);
        Ok(DevicePtr(addr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        // Apple Silicon UMA: managed and shared are the same thing.
        // No paged virtual memory swap mechanism (cuMemAllocManaged on
        // GB10) — Metal lets the OS handle pressure via its memory
        // pool. Defer to plain alloc.
        self.alloc(bytes)
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        // Removing the entry drops the last `Retained` reference; the
        // ObjC runtime releases the underlying MTLBuffer.
        self.allocations.lock().remove(&ptr.0);
        Ok(())
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_h2d: ptr {dst} not in any allocation"))?;
        if offset + src.len() > buf.length() {
            bail!(
                "copy_h2d: write overflows buffer ({} + {} > {})",
                offset,
                src.len(),
                buf.length()
            );
        }
        let contents: NonNull<c_void> = buf.contents();
        unsafe {
            let dst_ptr = (contents.as_ptr() as *mut u8).add(offset);
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr, src.len());
        }
        Ok(())
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2h: ptr {src} not in any allocation"))?;
        if offset + dst.len() > buf.length() {
            bail!(
                "copy_d2h: read overflows buffer ({} + {} > {})",
                offset,
                dst.len(),
                buf.length()
            );
        }
        let contents: NonNull<c_void> = buf.contents();
        unsafe {
            let src_ptr = (contents.as_ptr() as *const u8).add(offset);
            std::ptr::copy_nonoverlapping(src_ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        // UMA: synchronize the stream so prior kernels have written
        // their bytes back through the cache, then memcpy.
        self.synchronize(stream)?;
        self.copy_d2h(src, dst)
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (src_buf, src_off) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2d: src ptr {src} not allocated"))?;
        let (dst_buf, dst_off) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_d2d: dst ptr {dst} not allocated"))?;
        drop(allocs);

        let cmd_buf = self.current_cmd_buf(0)?;
        let enc = cmd_buf
            .blitCommandEncoder()
            .ok_or_else(|| anyhow!("blitCommandEncoder returned null"))?;
        unsafe {
            enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src_buf, src_off, &dst_buf, dst_off, bytes,
            );
        }
        enc.endEncoding();
        // Synchronize so the d2d behaves like CUDA's synchronous variant.
        if let Some(cb) = self.commit_in_flight(0)? {
            cb.waitUntilCompleted();
        }
        Ok(())
    }

    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (src_buf, src_off) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2d_async: src {src} not allocated"))?;
        let (dst_buf, dst_off) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_d2d_async: dst {dst} not allocated"))?;
        drop(allocs);
        let cmd_buf = self.current_cmd_buf(stream)?;
        let enc = cmd_buf
            .blitCommandEncoder()
            .ok_or_else(|| anyhow!("blitCommandEncoder returned null"))?;
        unsafe {
            enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src_buf, src_off, &dst_buf, dst_off, bytes,
            );
        }
        enc.endEncoding();
        Ok(())
    }

    fn launch(
        &self,
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _params: &mut [*mut c_void],
    ) -> Result<()> {
        // Metal can't safely interpret untyped `*mut c_void` slots as
        // either buffers or scalars (CUDA gets away with this because
        // the driver cross-references the kernel signature). Callers
        // must use `launch_typed`; the cuda-style untyped path is
        // intentionally unsupported here.
        bail!(
            "Metal backend: launch() requires typed args. Use launch_typed() \
             with KernelArg::Buffer / KernelArg::Bytes — see KernelLaunch builder."
        );
    }

    fn launch_typed(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        _shared_mem: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        // Resolve the pipeline state.
        let pipeline = {
            let slab = self.pipeline_slab.lock();
            slab.get(func.0 as usize)
                .cloned()
                .ok_or_else(|| anyhow!("launch_typed: unknown kernel handle {}", func.0))?
        };

        // Snapshot the alloc registry so we can resolve Buffer args
        // (and so the encoder can `useResource:` every live buffer
        // without holding the alloc lock during encoding).
        let live_buffers: Vec<ObjBuffer> = self.allocations.lock().values().cloned().collect();
        let allocs_snapshot: BTreeMap<u64, ObjBuffer> = self.allocations.lock().clone();

        let cmd_buf = self.current_cmd_buf(stream)?;
        let enc = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("computeCommandEncoder returned null"))?;
        enc.setComputePipelineState(&pipeline);

        // Mark every live allocation as in-use so Metal's automatic
        // hazard tracking keeps them resident. Cheap on Apple Silicon
        // because `useResource:` is a hint, not a copy.
        for buf in &live_buffers {
            let resource: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**buf);
            enc.useResource_usage(
                resource,
                objc2_metal::MTLResourceUsage::Read | objc2_metal::MTLResourceUsage::Write,
            );
        }

        // Bind each typed arg to its index.
        for (idx, arg) in args.iter().enumerate() {
            match arg {
                KernelArg::Buffer(p) => {
                    let (buf, offset) = Self::find_buffer(&allocs_snapshot, *p)
                        .ok_or_else(|| anyhow!("launch_typed: arg #{idx} ptr {p} not allocated"))?;
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&buf), offset, idx);
                    }
                }
                KernelArg::Bytes(b) => {
                    let ptr = NonNull::new(b.as_ptr() as *mut c_void)
                        .ok_or_else(|| anyhow!("launch_typed: arg #{idx} bytes is null"))?;
                    unsafe {
                        enc.setBytes_length_atIndex(ptr, b.len(), idx);
                    }
                }
            }
        }

        let threadgroups = MTLSize {
            width: grid[0] as usize,
            height: grid[1] as usize,
            depth: grid[2] as usize,
        };
        let threads_per_tg = MTLSize {
            width: block[0] as usize,
            height: block[1] as usize,
            depth: block[2] as usize,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        enc.endEncoding();
        Ok(())
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        if let Some(cb) = self.commit_in_flight(stream)? {
            cb.waitUntilCompleted();
        }
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        0
    }

    #[track_caller]
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        let key: PipelineKey = (module.to_string(), func_name.to_string());
        if let Some(handle) = self.pipeline_cache.lock().get(&key) {
            return Ok(*handle);
        }
        let lib = self
            .libraries
            .get(module)
            .ok_or_else(|| anyhow!("Metal: unknown module '{module}'"))?;
        let ns_name = NSString::from_str(func_name);
        let function = lib.newFunctionWithName(&ns_name).ok_or_else(|| {
            anyhow!("Metal: function '{func_name}' not found in module '{module}'")
        })?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                anyhow!(
                    "newComputePipelineStateWithFunction failed for '{func_name}': {}",
                    e.localizedDescription()
                )
            })?;
        let mut slab = self.pipeline_slab.lock();
        let handle = KernelHandle(slab.len() as u64);
        slab.push(pipeline);
        drop(slab);
        self.pipeline_cache.lock().insert(key, handle);
        Ok(handle)
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        // On UMA we can write through `contents()` directly when we
        // own the whole range — much cheaper than a blit fillBuffer.
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, ptr)
            .ok_or_else(|| anyhow!("memset: ptr {ptr} not allocated"))?;
        if offset + bytes > buf.length() {
            bail!(
                "memset: range overflows buffer ({} + {} > {})",
                offset,
                bytes,
                buf.length()
            );
        }
        let contents = buf.contents();
        unsafe {
            let dst = (contents.as_ptr() as *mut u8).add(offset);
            std::ptr::write_bytes(dst, value, bytes);
        }
        Ok(())
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, _stream: u64) -> Result<()> {
        // UMA + StorageModeShared makes the synchronous memset semantically
        // equivalent (no host/device cache split to flush).
        self.memset(ptr, value, bytes)
    }

    fn total_memory(&self) -> Result<usize> {
        // On Apple Silicon UMA, "device memory" = system RAM. Probe
        // hw.memsize via sysctl for the authoritative number; fall
        // back to MTLDevice.recommendedMaxWorkingSetSize otherwise.
        Ok(sysctl_memsize().unwrap_or_else(|| self.device.recommendedMaxWorkingSetSize() as usize))
    }

    fn free_memory(&self) -> Result<usize> {
        // No direct API for "free GPU memory" on UMA. Approximate via
        // `recommendedMaxWorkingSetSize - currentAllocatedSize`,
        // which matches the headroom Metal will let us allocate
        // before performance degrades.
        let max = self.device.recommendedMaxWorkingSetSize() as usize;
        let used = self.device.currentAllocatedSize();
        Ok(max.saturating_sub(used))
    }

    fn sm_count(&self) -> Result<u32> {
        // Metal exposes no SM/core count. Refuse rather than invent one: the
        // only consumers are grid-occupancy dispatch rules, and a fabricated
        // count would mis-tune them silently. No Metal path asks for this
        // today; if one does, wire it to a real device query first.
        anyhow::bail!("MetalGpuBackend does not expose a multiprocessor count")
    }

    fn create_stream(&self) -> Result<u64> {
        let queue = self
            .device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("newCommandQueue returned null"))?;
        let mut slab = self.streams.lock();
        slab.push(MetalStream {
            queue,
            in_flight: None,
        });
        // Handle = slab index + 1 so handle 0 stays reserved for
        // the default stream.
        Ok(slab.len() as u64)
    }

    fn bind_to_thread(&self) -> Result<()> {
        // Metal devices/queues are thread-safe; no binding required.
        Ok(())
    }

    fn create_event(&self) -> Result<u64> {
        let event = self
            .device
            .newSharedEvent()
            .ok_or_else(|| anyhow!("newSharedEvent returned null"))?;
        let mut slab = self.events.lock();
        slab.push(EventSlot { event, next: 1 });
        // Handle = slab index + 1 (0 reserved for "no event").
        Ok(slab.len() as u64)
    }

    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        let value = {
            let mut slab = self.events.lock();
            let idx = (event as usize)
                .checked_sub(1)
                .ok_or_else(|| anyhow!("record_event: invalid event handle {event}"))?;
            let slot = slab
                .get_mut(idx)
                .ok_or_else(|| anyhow!("record_event: event handle {event} out of range"))?;
            let v = slot.next;
            slot.next += 1;
            v
        };
        let cmd_buf = self.current_cmd_buf(stream)?;
        let event_obj = {
            let slab = self.events.lock();
            slab[(event - 1) as usize].event.clone()
        };
        // Encode the signal on the active command buffer. Metal will
        // signal value=`value` once everything queued on this buffer
        // up to this point has completed.
        let proto: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event_obj);
        cmd_buf.encodeSignalEvent_value(proto, value);
        Ok(())
    }

    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        let (event_obj, value) = {
            let slab = self.events.lock();
            let idx = (event as usize)
                .checked_sub(1)
                .ok_or_else(|| anyhow!("stream_wait_event: invalid event handle {event}"))?;
            let slot = slab
                .get(idx)
                .ok_or_else(|| anyhow!("stream_wait_event: event handle {event} out of range"))?;
            // Wait on the most-recently-recorded value (next - 1).
            // If nothing has been recorded yet, slot.next is 1 and
            // value is 0 — Metal treats wait-for-0 as a no-op.
            (slot.event.clone(), slot.next.saturating_sub(1))
        };
        let cmd_buf = self.current_cmd_buf(stream)?;
        let proto: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event_obj);
        cmd_buf.encodeWaitForEvent_value(proto, value);
        Ok(())
    }

    fn destroy_event(&self, event: u64) -> Result<()> {
        if event == 0 {
            return Ok(());
        }
        let mut slab = self.events.lock();
        let idx = (event - 1) as usize;
        if let Some(slot) = slab.get_mut(idx) {
            // Replace with a fresh dummy event so the slab indices
            // stay stable across destroys (matches the cuda backend
            // semantics — handles are not reused).
            slot.next = 0;
        }
        Ok(())
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        // UMA: a Shared MTLBuffer's contents() pointer IS host-pinned
        // memory from the GPU's perspective. We park the buffer in
        // the alloc table keyed by gpuAddress, then return the host
        // pointer. `free_host_pinned` looks the buffer up by host
        // pointer to release it.
        let buf = self
            .device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("alloc_host_pinned: newBufferWithLength failed"))?;
        let host_ptr = buf.contents().as_ptr() as *mut u8;
        // Stash by gpuAddress so plain `free()` on the DevicePtr would
        // also work; the host-pinned variant is purely a CPU view.
        let addr = buf.gpuAddress();
        if addr == 0 {
            bail!("alloc_host_pinned: gpuAddress returned 0");
        }
        // `newBufferWithLength` does not promise zeroed contents. Match the
        // trait contract (and the CUDA backend) so callers may form a `&[u8]`
        // over the whole buffer without every one of them re-establishing
        // initialisation. SAFETY: `host_ptr` is the `contents()` pointer of a
        // live Shared buffer of at least `bytes.max(1)` bytes, uniquely owned
        // here until it is parked in the alloc table.
        unsafe { std::ptr::write_bytes(host_ptr, 0, bytes.max(1)) };
        self.allocations.lock().insert(addr, buf);
        Ok(host_ptr)
    }

    fn free_host_pinned(&self, ptr: *mut u8, _bytes: usize) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        // Find the buffer whose contents() pointer matches.
        let mut allocs = self.allocations.lock();
        let target_addr = allocs.iter().find_map(|(addr, buf)| {
            let host = buf.contents().as_ptr() as *mut u8;
            if host == ptr { Some(*addr) } else { None }
        });
        if let Some(addr) = target_addr {
            allocs.remove(&addr);
        }
        Ok(())
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Probe `hw.memsize` via libc::sysctl on macOS. Returns the total
/// system RAM in bytes — on Apple Silicon UMA this is also the upper
/// bound on Metal-addressable memory.
fn sysctl_memsize() -> Option<usize> {
    use std::ffi::CString;
    let name = CString::new("hw.memsize").ok()?;
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut value as *mut u64 as *mut c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 { Some(value as usize) } else { None }
}

// ── Smoke test ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
