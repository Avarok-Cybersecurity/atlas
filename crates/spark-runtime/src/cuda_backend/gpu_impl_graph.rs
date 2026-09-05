// SPDX-License-Identifier: AGPL-3.0-only

//! CUDA graph capture/replay, stream + event management, memset, memory
//! queries, and pinned-host allocation for [`AtlasCudaBackend`].
//!
//! Split out of `gpu_impl.rs` to keep both files under the repo's 500-LoC cap.
//! Same shape as `spark-model`'s `model/trait_impl/`: these are the inherent
//! bodies (suffixed `_cu` so a delegator can never self-recurse) and the
//! `GpuBackend` impl next door is a one-line delegator to each.
//!
//! The `unsafe` safety contract documented at the top of `gpu_impl.rs` applies
//! verbatim to every driver call here.

use std::ffi::c_void;

use anyhow::{Result, bail};

use super::{
    AtlasCudaBackend, cuCtxGetDevice, cuCtxSetCurrent, cuDeviceGetAttribute, cuEventCreate,
    cuEventDestroy_v2, cuEventRecord, cuEventSynchronize, cuGraphDestroy, cuGraphExecDestroy,
    cuGraphLaunch, cuMemAllocHost_v2, cuMemFreeHost, cuMemGetInfo_v2, cuMemsetD8Async,
    cuStreamBeginCapture, cuStreamCreate, cuStreamEndCapture, cuStreamSynchronize,
    cuStreamWaitEvent,
};
use crate::gpu::{DevicePtr, GraphHandle};

impl AtlasCudaBackend {
    pub(super) fn begin_capture_cu(&self, stream: u64) -> Result<()> {
        // CU_STREAM_CAPTURE_MODE_RELAXED = 2
        // Relaxed mode allows NCCL's internal streams to operate during
        // graph capture (required for EP all-reduce in CUDA graphs).
        let status = unsafe { cuStreamBeginCapture(stream, 2) };
        if status != 0 {
            bail!("cuStreamBeginCapture failed: status {status}");
        }
        Ok(())
    }

    /// Best-effort: end any in-progress capture on this stream to release it
    /// after a mid-capture error. `cuStreamEndCapture` returns an error (which
    /// we intentionally ignore) when the stream is NOT capturing, so no
    /// capture-status probe is needed. This deliberately avoids
    /// cuStreamIsCapturing / cuStreamGetCaptureInfo — neither is exported by
    /// the CUDA-13 driver stub CI links against.
    pub(super) fn abort_capture_if_active_cu(&self, stream: u64) {
        let mut graph: u64 = 0;
        let status = unsafe { cuStreamEndCapture(stream, &mut graph) };
        // Only a stream that WAS capturing yields status==0 + a partial graph
        // to discard; the not-capturing path returns non-zero with graph null.
        if status == 0 && graph != 0 {
            unsafe { cuGraphDestroy(graph) };
        }
    }

    pub(super) fn end_capture_cu(&self, stream: u64) -> Result<GraphHandle> {
        let mut graph: u64 = 0;
        let status = unsafe { cuStreamEndCapture(stream, &mut graph) };
        if status != 0 {
            bail!("cuStreamEndCapture failed: status {status}");
        }
        // Instantiate the graph into an executable. NVIDIA's libcuda exports
        // `cuGraphInstantiateWithFlags`; SCALE (gfx1151) exposes the
        // ABI-identical `cuGraphInstantiate` — see cuda_backend.rs.
        let mut graph_exec: u64 = 0;
        #[cfg(not(atlas_scale))]
        let status = unsafe { super::cuGraphInstantiateWithFlags(&mut graph_exec, graph, 0) };
        #[cfg(atlas_scale)]
        let status = unsafe { super::cuGraphInstantiate(&mut graph_exec, graph, 0) };
        if status != 0 {
            unsafe { cuGraphDestroy(graph) };
            bail!("cuGraphInstantiate failed: status {status}");
        }
        // The graph template is no longer needed after instantiation
        unsafe { cuGraphDestroy(graph) };
        Ok(GraphHandle(graph_exec))
    }

    pub(super) fn launch_graph_cu(&self, graph: GraphHandle, stream: u64) -> Result<()> {
        let status = unsafe { cuGraphLaunch(graph.0, stream) };
        if status != 0 {
            bail!("cuGraphLaunch failed: status {status}");
        }
        Ok(())
    }

    pub(super) fn destroy_graph_cu(&self, graph: GraphHandle) -> Result<()> {
        if graph.0 != 0 {
            let status = unsafe { cuGraphExecDestroy(graph.0) };
            if status != 0 {
                bail!("cuGraphExecDestroy failed: status {status}");
            }
        }
        Ok(())
    }

    pub(super) fn memset_cu(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        let status = unsafe { cuMemsetD8Async(ptr.0, value, bytes, self.default_stream) };
        if status != 0 {
            // The #429 symptom surfaced here: once a kernel launch had
            // poisoned the context, every request died on this memset. Probe
            // so the FIRST such failure is recognised as terminal instead of
            // being reported forever as an unrelated per-request error.
            super::fault_probe::note_failure("cuMemsetD8Async", &format!("status {status}"));
            bail!("cuMemsetD8Async failed: status {status}");
        }
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            super::fault_probe::note_failure(
                "cuStreamSynchronize after memset",
                &format!("status {sync}"),
            );
            bail!("cuStreamSynchronize after memset failed: status {sync}");
        }
        Ok(())
    }

    pub(super) fn memset_async_cu(
        &self,
        ptr: DevicePtr,
        value: u8,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let status = unsafe { cuMemsetD8Async(ptr.0, value, bytes, stream) };
        if status != 0 {
            super::fault_probe::note_failure("cuMemsetD8Async", &format!("status {status}"));
            bail!("cuMemsetD8Async failed: status {status}");
        }
        Ok(())
    }

    pub(super) fn total_memory_cu(&self) -> Result<usize> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
        if status != 0 {
            bail!("cuMemGetInfo_v2 failed: status {status}");
        }
        Ok(total)
    }

    /// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT` on the current context's
    /// device. Fails loudly rather than substituting a guess — a wrong SM
    /// count silently mis-tunes every grid-occupancy dispatch rule.
    pub(super) fn sm_count_cu(&self) -> Result<u32> {
        const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: u32 = 16;
        let mut dev: i32 = 0;
        let status = unsafe { cuCtxGetDevice(&mut dev) };
        if status != 0 {
            bail!("cuCtxGetDevice failed: status {status}");
        }
        let mut count: i32 = 0;
        let status = unsafe {
            cuDeviceGetAttribute(&mut count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)
        };
        if status != 0 {
            bail!("cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed: status {status}");
        }
        if count <= 0 {
            bail!("driver reported {count} multiprocessors on device {dev}");
        }
        Ok(count as u32)
    }

    pub(super) fn free_memory_cu(&self) -> Result<usize> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
        if status != 0 {
            bail!("cuMemGetInfo_v2 failed: status {status}");
        }
        // RULE: host `MemAvailable` substitutes for the driver's device-free
        // figure ONLY on an integrated GPU.
        //
        // On integrated memory (GB10 / DGX Spark, unified LPDDR5X) device and
        // host share one physical pool and `cuMemGetInfo` reports Linux
        // MemFree, which excludes reclaimable buff/cache — MemAvailable is the
        // truer number, so taking the max is right there.
        //
        // On a DISCRETE GPU host RAM is a different pool entirely and the max
        // is nonsense: on a 3x RTX PRO 6000 Blackwell box (95 GiB per card,
        // 1 TB host RAM) MemAvailable read 1,038,438,936 kB, so this returned
        // ~990 GB free for a 95 GB card, `used_so_far` came out 0, and the KV
        // pool was sized as if nothing had been allocated — the load then died
        // in `cuMemAlloc_v2` with 4280.2 MB actually free.
        //
        // `CU_DEVICE_ATTRIBUTE_INTEGRATED` (18) is the discriminator, measured
        // 2026-09-04: 1 on NVIDIA GB10, 0 on RTX PRO 6000 Blackwell (and 0 on
        // every discrete datacenter part).
        // `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS` (99) reads 1 on BOTH,
        // so it does NOT discriminate and must not be used here.
        Ok(effective_free_bytes(
            free,
            super::system_available_memory_bytes(),
            self.device_is_integrated_cu()?,
        ))
    }

    /// `CU_DEVICE_ATTRIBUTE_INTEGRATED` on the current context's device: true
    /// when the GPU shares the host's physical memory (GB10), false on a
    /// discrete card. Cheap enough to query per call — it is a driver-side
    /// table lookup, and `free_memory` is not on any hot path.
    ///
    /// Fails loudly rather than guessing, like `sm_count_cu`: a wrong answer
    /// here mis-sizes the KV pool by hundreds of gigabytes in either
    /// direction.
    pub(super) fn device_is_integrated_cu(&self) -> Result<bool> {
        const CU_DEVICE_ATTRIBUTE_INTEGRATED: u32 = 18;
        let mut dev: i32 = 0;
        let status = unsafe { cuCtxGetDevice(&mut dev) };
        if status != 0 {
            bail!("cuCtxGetDevice failed: status {status}");
        }
        let mut integrated: i32 = 0;
        let status =
            unsafe { cuDeviceGetAttribute(&mut integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev) };
        if status != 0 {
            bail!("cuDeviceGetAttribute(INTEGRATED) failed: status {status}");
        }
        Ok(integrated != 0)
    }

    pub(super) fn create_stream_cu(&self) -> Result<u64> {
        let mut stream: u64 = 0;
        // CU_STREAM_NON_BLOCKING = 1 (does not synchronize with stream 0)
        let status = unsafe { cuStreamCreate(&mut stream, 1) };
        if status != 0 {
            bail!("cuStreamCreate failed: status {status}");
        }
        Ok(stream)
    }

    pub(super) fn bind_to_thread_cu(&self) -> Result<()> {
        let status = unsafe { cuCtxSetCurrent(self.cuda_ctx) };
        if status != 0 {
            bail!("cuCtxSetCurrent failed: status {status}");
        }
        Ok(())
    }

    pub(super) fn create_event_cu(&self) -> Result<u64> {
        let mut event: u64 = 0;
        // CU_EVENT_DISABLE_TIMING = 0x02 (skip timing overhead)
        let status = unsafe { cuEventCreate(&mut event, 0x02) };
        if status != 0 {
            bail!("cuEventCreate failed: status {status}");
        }
        Ok(event)
    }

    pub(super) fn record_event_cu(&self, event: u64, stream: u64) -> Result<()> {
        let status = unsafe { cuEventRecord(event, stream) };
        if status != 0 {
            bail!("cuEventRecord failed: status {status}");
        }
        Ok(())
    }

    pub(super) fn stream_wait_event_cu(&self, stream: u64, event: u64) -> Result<()> {
        let status = unsafe { cuStreamWaitEvent(stream, event, 0) };
        if status != 0 {
            bail!("cuStreamWaitEvent failed: status {status}");
        }
        Ok(())
    }

    pub(super) fn event_synchronize_cu(&self, event: u64) -> Result<()> {
        // Block calling thread until all work recorded against `event`
        // (on whatever stream `record_event` targeted) has completed.
        // Used in Phase E.2: drafter D2H copy is recorded against this
        // event, host blocks here just before reading the pinned buffer.
        let status = unsafe { cuEventSynchronize(event) };
        if status != 0 {
            bail!("cuEventSynchronize failed: status {status}");
        }
        Ok(())
    }

    pub(super) fn destroy_event_cu(&self, event: u64) -> Result<()> {
        if event != 0 {
            let status = unsafe { cuEventDestroy_v2(event) };
            if status != 0 {
                bail!("cuEventDestroy_v2 failed: status {status}");
            }
        }
        Ok(())
    }

    pub(super) fn alloc_host_pinned_cu(&self, bytes: usize) -> Result<*mut u8> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let status = unsafe { cuMemAllocHost_v2(&mut ptr, bytes) };
        if status != 0 {
            bail!("cuMemAllocHost_v2 failed: status {status}, requested {bytes} bytes");
        }
        // `cuMemAllocHost_v2` does NOT zero, unlike the trait's `alloc_zeroed`
        // default and the mock. Callers pack these buffers with alignment
        // padding and then form a `&[u8]` over the whole packed range — a slice
        // over even one never-written byte is UB. Zeroing once here is what
        // makes `GpuBackend::alloc_host_pinned`'s "fully initialised" contract
        // true on every backend, so no caller has to re-establish it. One
        // memset at allocation time; these buffers are allocated at model load
        // and reused for the process lifetime.
        // SAFETY: `cuMemAllocHost_v2` returned success, so `ptr` is a valid,
        // uniquely-owned, writable page-locked region of exactly `bytes`.
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, bytes) };
        // Record it: an H2D from page-locked memory is genuinely async, so
        // `copy_h2d_async` has to know which sources it may not let the caller
        // drop out from under. See `crate::pinned_hosts`.
        crate::pinned_hosts::register(ptr as *const u8, bytes);
        Ok(ptr as *mut u8)
    }

    pub(super) fn free_host_pinned_cu(&self, ptr: *mut u8, _bytes: usize) -> Result<()> {
        if !ptr.is_null() {
            // Before the free, so a reused address is never reported as still
            // page-locked.
            crate::pinned_hosts::unregister(ptr as *const u8);
            let status = unsafe { cuMemFreeHost(ptr as *mut c_void) };
            // The driver tears the primary context down in its own atexit
            // handler, which can run before ours. Pinned host memory allocated
            // against a context that no longer exists was already reclaimed
            // with it — reporting that as a failure is noise at every exit.
            if status != 0 && !atlas_core::registry::is_teardown_noop(status) {
                bail!(
                    "cuMemFreeHost failed: {}",
                    atlas_core::registry::cuda_error_text(status)
                );
            }
        }
        Ok(())
    }
}

/// Free device memory to report, given the driver's figure, host
/// `MemAvailable`, and whether the device is integrated.
///
/// Host memory may stand in for device memory on an INTEGRATED GPU ONLY,
/// where the two are one physical pool. On a discrete GPU they are unrelated,
/// and substituting host RAM reports a free figure many times the card's
/// capacity. Pure so the rule is testable without a GPU.
pub(super) fn effective_free_bytes(
    cu_free: usize,
    mem_available: Option<usize>,
    integrated: bool,
) -> usize {
    match mem_available {
        Some(avail) if integrated => cu_free.max(avail),
        _ => cu_free,
    }
}

#[cfg(test)]
mod tests {
    use super::effective_free_bytes;

    const GIB: usize = 1024 * 1024 * 1024;

    #[test]
    fn discrete_device_ignores_host_mem_available() {
        // Reproduces the RTX PRO 6000 failure: 95 GiB card, 4.2 GiB actually
        // free, host MemAvailable 1,038,438,936 kB (~990 GiB). Host RAM is a
        // different pool; substituting it made free_memory() report ~990 GB
        // and the KV pool was sized as if nothing had been allocated.
        let cu_free = 4 * GIB + 280 * 1024 * 1024;
        let mem_available = Some(1_038_438_936usize * 1024);
        assert_eq!(
            effective_free_bytes(cu_free, mem_available, false),
            cu_free,
            "a discrete GPU must report the driver's device-free figure verbatim"
        );
    }

    #[test]
    fn integrated_device_takes_the_max() {
        // GB10 (DGX Spark): device and host share one LPDDR5X pool, and
        // cuMemGetInfo reports MemFree, which excludes reclaimable buff/cache.
        let cu_free = 20 * GIB;
        let mem_available = Some(90 * GIB);
        assert_eq!(effective_free_bytes(cu_free, mem_available, true), 90 * GIB);
    }

    #[test]
    fn integrated_device_without_meminfo_uses_driver_figure() {
        // /proc/meminfo missing or unparseable (non-Linux, container).
        let cu_free = 20 * GIB;
        assert_eq!(effective_free_bytes(cu_free, None, true), cu_free);
    }

    #[test]
    fn integrated_device_keeps_driver_figure_when_it_is_larger() {
        // MemAvailable can be the smaller of the two; `max` must not shrink
        // the driver's figure.
        let cu_free = 60 * GIB;
        assert_eq!(effective_free_bytes(cu_free, Some(10 * GIB), true), cu_free);
    }
}
