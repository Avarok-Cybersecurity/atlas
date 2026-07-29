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
    AtlasCudaBackend, cuCtxSetCurrent, cuEventCreate, cuEventDestroy_v2, cuEventRecord,
    cuEventSynchronize, cuGraphDestroy, cuGraphExecDestroy, cuGraphLaunch, cuMemAllocHost_v2,
    cuMemFreeHost, cuMemGetInfo_v2, cuMemsetD8Async, cuStreamBeginCapture, cuStreamCreate,
    cuStreamEndCapture, cuStreamSynchronize, cuStreamWaitEvent,
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
            bail!("cuMemsetD8Async failed: status {status}");
        }
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
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

    pub(super) fn free_memory_cu(&self) -> Result<usize> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
        if status != 0 {
            bail!("cuMemGetInfo_v2 failed: status {status}");
        }
        // On unified memory (GB10), cuMemGetInfo reports Linux "free" memory
        // which excludes reclaimable buff/cache. Use MemAvailable instead.
        if let Some(mem_available) = super::system_available_memory_bytes() {
            free = free.max(mem_available);
        }
        Ok(free)
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
        Ok(ptr as *mut u8)
    }

    pub(super) fn free_host_pinned_cu(&self, ptr: *mut u8, _bytes: usize) -> Result<()> {
        if !ptr.is_null() {
            let status = unsafe { cuMemFreeHost(ptr as *mut c_void) };
            if status != 0 {
                bail!("cuMemFreeHost failed: status {status}");
            }
        }
        Ok(())
    }
}
