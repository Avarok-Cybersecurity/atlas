// SPDX-License-Identifier: AGPL-3.0-only

//! `impl GpuBackend for AtlasCudaBackend` — production CUDA backend trait body.
//!
//! ## Safety contract for the `unsafe { cu*(...) }` calls below
//!
//! Every unsafe block in this file wraps a single CUDA Driver API call.
//! The invariants the driver requires are uniform:
//!
//! - **Context bound**: a CUDA primary context for the device is current
//!   on the calling thread. `AtlasCudaBackend::new` binds it once via
//!   `cuCtxSetCurrent`, and we never run on a thread that hasn't been
//!   bound.
//! - **Pointer provenance**: every `DevicePtr` came from a prior
//!   successful `cuMemAlloc_v2` / `cuMemAllocHost_v2` /
//!   `cuMemAllocManaged` and has not yet been freed. `DevicePtr(0)` is
//!   treated as "not allocated" by callers.
//! - **Sizes in bytes**: every `bytes: usize` argument is the exact
//!   byte count of the allocation (callers compute it from typed
//!   sizes); the driver does no bounds-checking.
//! - **Stream / event lifetimes**: handles are owned by `Self` and
//!   freed in `Drop` after `cuStreamSynchronize`, so they outlive every
//!   in-flight launch that captured them.
//! - **`extern "C"` ABI**: matches the cudarc-generated bindings used
//!   in `super::*` imports; see `cudarc` for the full ABI surface.
//!
//! Per-site `// SAFETY:` comments are omitted because the contract is
//! identical for every call. Anything that *deviates* from this
//! contract gets a per-site `// SAFETY:` comment explaining the
//! exception.

use std::ffi::c_void;
use std::sync::OnceLock;

use anyhow::{Result, bail};
use atlas_core::registry::{AtlasRegistry, RawCudaFunc, cuda_error_text};
use cudarc::driver::LaunchConfig;

use super::{
    AtlasCudaBackend, cuCtxSetCurrent, cuEventCreate, cuEventDestroy_v2, cuEventRecord,
    cuEventSynchronize, cuGraphDestroy, cuGraphExecDestroy, cuGraphLaunch, cuMemAlloc_v2,
    cuMemAllocHost_v2, cuMemAllocManaged, cuMemFree_v2, cuMemFreeHost, cuMemGetInfo_v2,
    cuMemcpyDtoDAsync_v2, cuMemcpyDtoHAsync_v2, cuMemcpyHtoDAsync_v2, cuMemsetD8Async,
    cuStreamBeginCapture, cuStreamCreate, cuStreamEndCapture, cuStreamSynchronize,
    cuStreamWaitEvent,
};
use crate::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

impl GpuBackend for AtlasCudaBackend {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let mut dptr: u64 = 0;
        let status = unsafe { cuMemAlloc_v2(&mut dptr, bytes) };
        if status != 0 {
            let mut free: usize = 0;
            let mut total: usize = 0;
            unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
            bail!(
                "cuMemAlloc_v2 failed: status {status}, requested {bytes} bytes \
                 (device reports {:.1} MB free / {:.1} GB total)",
                free as f64 / (1024.0 * 1024.0),
                total as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        Ok(DevicePtr(dptr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        let mut dptr: u64 = 0;
        const CU_MEM_ATTACH_GLOBAL: u32 = 0x1;
        let status = unsafe { cuMemAllocManaged(&mut dptr, bytes, CU_MEM_ATTACH_GLOBAL) };
        if status != 0 {
            bail!(
                "cuMemAllocManaged failed: status {status}, requested {bytes} bytes. \
                 Check system swap space: swapon --show"
            );
        }
        Ok(DevicePtr(dptr))
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        let status = unsafe { cuMemFree_v2(ptr.0) };
        if status != 0 {
            bail!("cuMemFree_v2 failed: status {status}, ptr {ptr}");
        }
        Ok(())
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        let status = unsafe {
            cuMemcpyHtoDAsync_v2(
                dst.0,
                src.as_ptr() as *const c_void,
                src.len(),
                self.default_stream,
            )
        };
        if status != 0 {
            bail!("cuMemcpyHtoDAsync_v2 failed: status {status}");
        }
        // Synchronize to ensure the copy completes before host buffer is freed.
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after H2D failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let status = unsafe {
            cuMemcpyDtoHAsync_v2(
                dst.as_mut_ptr() as *mut c_void,
                src.0,
                dst.len(),
                self.default_stream,
            )
        };
        if status != 0 {
            bail!("cuMemcpyDtoHAsync_v2 failed: status {status}");
        }
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after D2H failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }

    fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        // Enqueue the copy on the caller's stream so CUDA orders it after
        // any prior kernel launches on the same stream. Without this, the
        // copy may run on the default stream concurrently with kernels on
        // `stream` and read torn bytes (HSS Turbo8 race, 2026-04-28).
        let status = unsafe {
            cuMemcpyDtoHAsync_v2(dst.as_mut_ptr() as *mut c_void, src.0, dst.len(), stream)
        };
        if status != 0 {
            bail!("cuMemcpyDtoHAsync_v2 (on_stream) failed: status {status}");
        }
        let sync = unsafe { cuStreamSynchronize(stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after D2H on_stream failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        let status = unsafe { cuMemcpyDtoDAsync_v2(dst.0, src.0, bytes, self.default_stream) };
        if status != 0 {
            bail!("cuMemcpyDtoDAsync_v2 failed: status {status}");
        }
        // Synchronize to ensure copy completes before kernels on other streams read it.
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after D2D failed: {}",
                cuda_error_text(sync)
            );
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
        params: &mut [*mut c_void],
    ) -> Result<()> {
        let raw_func = RawCudaFunc(func.0 as *mut c_void);
        let cfg = LaunchConfig {
            grid_dim: (grid[0], grid[1], grid[2]),
            block_dim: (block[0], block[1], block[2]),
            shared_mem_bytes: shared_mem,
        };
        let registry = AtlasRegistry::get();
        unsafe {
            registry
                .launch_on_stream(raw_func, cfg, stream, params)
                .map_err(|e| anyhow::anyhow!("Kernel launch failed: {e}"))
        }
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        let status = unsafe { cuStreamSynchronize(stream) };
        if status != 0 {
            bail!("cuStreamSynchronize failed: {}", cuda_error_text(status));
        }
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        self.default_stream
    }

    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        // Ephemeral OnceLock — no cross-call caching, but kernel() is only
        // called at model init time. Layers store the returned KernelHandle.
        let cache: OnceLock<RawCudaFunc> = OnceLock::new();
        let registry = AtlasRegistry::get();
        match registry.raw_function_cached(&cache, module, func_name) {
            Ok(raw) => {
                crate::kernel_audit::record(module, func_name, true);
                Ok(KernelHandle(raw.0 as u64))
            }
            Err(e) => {
                // Optional kernels (try_kernel) land here and fall back silently;
                // the audit makes that visible in the startup kernel table.
                crate::kernel_audit::record(module, func_name, false);
                Err(anyhow::anyhow!("Kernel lookup {module}::{func_name}: {e}"))
            }
        }
    }

    fn copy_h2d_async(&self, src: &[u8], dst: DevicePtr, stream: u64) -> Result<()> {
        let status = unsafe {
            cuMemcpyHtoDAsync_v2(dst.0, src.as_ptr() as *const c_void, src.len(), stream)
        };
        if status != 0 {
            bail!("cuMemcpyHtoDAsync_v2 failed: status {status}");
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
        let status = unsafe { cuMemcpyDtoDAsync_v2(dst.0, src.0, bytes, stream) };
        if status != 0 {
            bail!("cuMemcpyDtoDAsync_v2 failed: status {status}");
        }
        Ok(())
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
        // One pitched copy (cudaMemcpyDeviceToDevice = 3) on the caller's stream,
        // replacing a per-row copy_d2d_async loop. cudart is linked (cutlass/
        // flashinfer use the runtime API); a CUstream handle is a valid
        // cudaStream_t.
        unsafe extern "C" {
            fn cudaMemcpy2DAsync(
                dst: *mut c_void,
                dpitch: usize,
                src: *const c_void,
                spitch: usize,
                width: usize,
                height: usize,
                kind: i32,
                stream: u64,
            ) -> i32;
        }
        let status = unsafe {
            cudaMemcpy2DAsync(
                dst.0 as *mut c_void,
                dst_pitch,
                src.0 as *const c_void,
                src_pitch,
                width_bytes,
                height,
                3,
                stream,
            )
        };
        if status != 0 {
            bail!("cudaMemcpy2DAsync failed: status {status}");
        }
        Ok(())
    }

    fn begin_capture(&self, stream: u64) -> Result<()> {
        // CU_STREAM_CAPTURE_MODE_RELAXED = 2
        // Relaxed mode allows NCCL's internal streams to operate during
        // graph capture (required for EP all-reduce in CUDA graphs).
        let status = unsafe { cuStreamBeginCapture(stream, 2) };
        if status != 0 {
            bail!("cuStreamBeginCapture failed: status {status}");
        }
        Ok(())
    }

    fn end_capture(&self, stream: u64) -> Result<GraphHandle> {
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

    fn launch_graph(&self, graph: GraphHandle, stream: u64) -> Result<()> {
        let status = unsafe { cuGraphLaunch(graph.0, stream) };
        if status != 0 {
            bail!("cuGraphLaunch failed: status {status}");
        }
        Ok(())
    }

    fn destroy_graph(&self, graph: GraphHandle) -> Result<()> {
        if graph.0 != 0 {
            let status = unsafe { cuGraphExecDestroy(graph.0) };
            if status != 0 {
                bail!("cuGraphExecDestroy failed: status {status}");
            }
        }
        Ok(())
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
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

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        let status = unsafe { cuMemsetD8Async(ptr.0, value, bytes, stream) };
        if status != 0 {
            bail!("cuMemsetD8Async failed: status {status}");
        }
        Ok(())
    }

    fn total_memory(&self) -> Result<usize> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
        if status != 0 {
            bail!("cuMemGetInfo_v2 failed: status {status}");
        }
        Ok(total)
    }

    fn free_memory(&self) -> Result<usize> {
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

    fn create_stream(&self) -> Result<u64> {
        let mut stream: u64 = 0;
        // CU_STREAM_NON_BLOCKING = 1 (does not synchronize with stream 0)
        let status = unsafe { cuStreamCreate(&mut stream, 1) };
        if status != 0 {
            bail!("cuStreamCreate failed: status {status}");
        }
        Ok(stream)
    }

    fn bind_to_thread(&self) -> Result<()> {
        let status = unsafe { cuCtxSetCurrent(self.cuda_ctx) };
        if status != 0 {
            bail!("cuCtxSetCurrent failed: status {status}");
        }
        Ok(())
    }

    fn create_event(&self) -> Result<u64> {
        let mut event: u64 = 0;
        // CU_EVENT_DISABLE_TIMING = 0x02 (skip timing overhead)
        let status = unsafe { cuEventCreate(&mut event, 0x02) };
        if status != 0 {
            bail!("cuEventCreate failed: status {status}");
        }
        Ok(event)
    }

    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        let status = unsafe { cuEventRecord(event, stream) };
        if status != 0 {
            bail!("cuEventRecord failed: status {status}");
        }
        Ok(())
    }

    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        let status = unsafe { cuStreamWaitEvent(stream, event, 0) };
        if status != 0 {
            bail!("cuStreamWaitEvent failed: status {status}");
        }
        Ok(())
    }

    fn event_synchronize(&self, event: u64) -> Result<()> {
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

    fn destroy_event(&self, event: u64) -> Result<()> {
        if event != 0 {
            let status = unsafe { cuEventDestroy_v2(event) };
            if status != 0 {
                bail!("cuEventDestroy_v2 failed: status {status}");
            }
        }
        Ok(())
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let status = unsafe { cuMemAllocHost_v2(&mut ptr, bytes) };
        if status != 0 {
            bail!("cuMemAllocHost_v2 failed: status {status}, requested {bytes} bytes");
        }
        Ok(ptr as *mut u8)
    }

    fn free_host_pinned(&self, ptr: *mut u8, _bytes: usize) -> Result<()> {
        if !ptr.is_null() {
            let status = unsafe { cuMemFreeHost(ptr as *mut c_void) };
            if status != 0 {
                bail!("cuMemFreeHost failed: status {status}");
            }
        }
        Ok(())
    }
}
