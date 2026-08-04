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
use atlas_core::registry::{RawCudaFunc, cuda_error_text};
use cudarc::driver::LaunchConfig;

use super::{
    AtlasCudaBackend, cuMemAlloc_v2, cuMemAllocManaged, cuMemFree_v2, cuMemGetInfo_v2,
    cuMemcpyDtoDAsync_v2, cuMemcpyDtoHAsync_v2, cuMemcpyHtoDAsync_v2, cuStreamSynchronize,
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
        self.record_alloc(DevicePtr(dptr));
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
        self.record_alloc(DevicePtr(dptr));
        Ok(DevicePtr(dptr))
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        // Off the ledger BEFORE the free: an entry that survives a successful
        // free would be double-freed at teardown.
        self.forget_alloc(ptr);
        let status = unsafe { cuMemFree_v2(ptr.0) };
        // A context that is already being destroyed reports every free as
        // failing, and at process exit that is the normal case, not an error:
        // the driver has reclaimed the allocation by definition. Two other
        // free paths in this crate already consult `is_teardown_noop`; this one
        // did not, so wiring `Model::teardown` into shutdown turned a benign
        // status 4 into `ERROR model teardown reported a failure` on every
        // clean exit — the exact species of false alarm this work set out to
        // remove.
        if status != 0 && !atlas_core::registry::is_teardown_noop(status) {
            bail!("cuMemFree_v2 failed: status {status}, ptr {ptr}");
        }
        Ok(())
    }

    fn sweep_unreleased(&self) -> usize {
        AtlasCudaBackend::sweep_unreleased(self)
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        AtlasCudaBackend::copy_h2d_impl(self, src, dst)
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        AtlasCudaBackend::copy_d2h_impl(self, src, dst)
    }

    fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        AtlasCudaBackend::copy_d2h_on_stream_impl(self, src, dst, stream)
    }

    fn copy_d2h_async(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        // Deliberately NO cuStreamSynchronize — that is the entire point.
        // `copy_d2h`/`copy_d2h_on_stream` drain the stream inside every call,
        // so a multi-chunk gather pays one full drain per chunk (the SSM spill's
        // 60 chunks × 66 MB measured ~400 ms = ~165 MB/s, vs ~28 ms for the
        // async H2D scatter of the same bytes). The caller MUST issue exactly
        // one `synchronize(stream)` before touching `dst`.
        let status = unsafe {
            cuMemcpyDtoHAsync_v2(dst.as_mut_ptr() as *mut c_void, src.0, dst.len(), stream)
        };
        if status != 0 {
            bail!("cuMemcpyDtoHAsync_v2 (async) failed: status {status}");
        }
        Ok(())
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        AtlasCudaBackend::copy_d2d_impl(self, src, dst, bytes)
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
        let registry = self.registry();
        unsafe {
            registry
                .launch_on_stream(raw_func, cfg, stream, params)
                .map_err(|e| anyhow::anyhow!("Kernel launch failed: {e}"))
        }
    }

    fn stream_is_capturing(&self, stream: u64) -> bool {
        // SCALE's libcuda does not export cuStreamIsCapturing; report
        // not-capturing there (gfx1151 telemetry taps then sample eagerly —
        // acceptable for a default-off measurement knob).
        #[cfg(atlas_scale)]
        {
            let _ = stream;
            false
        }
        #[cfg(not(atlas_scale))]
        {
            let mut status: u32 = 0;
            // CU_STREAM_CAPTURE_STATUS_NONE = 0; treat query failure as
            // capturing (conservative: the tap skips its sample).
            let rc = unsafe { super::cuStreamIsCapturing(stream, &mut status) };
            rc != 0 || status != 0
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

    fn op_cache(&self) -> &crate::op_cache::OpCache {
        &self.op_cache
    }

    fn debug_sync_kernels(&self) -> bool {
        AtlasCudaBackend::debug_sync_kernels(self)
    }

    fn kernel_registry(&self) -> Option<std::sync::Arc<atlas_core::registry::AtlasRegistry>> {
        Some(self.registry().clone())
    }

    #[track_caller]
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        // The DISPATCH SITE, not this line: `#[track_caller]` here and on the
        // trait declaration carries the `.kernel(…)` / `try_kernel(…)` caller's
        // `file:line` through, which is the only part of an unresolved-lookup
        // report an operator can act on.
        let site = std::panic::Location::caller();
        // Ephemeral OnceLock — no cross-call caching, but kernel() is only
        // called at model init time. Layers store the returned KernelHandle.
        let cache: OnceLock<RawCudaFunc> = OnceLock::new();
        let registry = self.registry();
        match registry.raw_function_cached(&cache, module, func_name) {
            Ok(raw) => {
                crate::kernel_audit::record(module, func_name, true, site);
                Ok(KernelHandle(raw.0 as u64))
            }
            Err(e) => {
                // Optional kernels (try_kernel) land here and fall back silently;
                // the audit makes that visible in the startup kernel table.
                crate::kernel_audit::record(module, func_name, false, site);
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
        self.begin_capture_cu(stream)
    }
    fn end_capture(&self, stream: u64) -> Result<GraphHandle> {
        self.end_capture_cu(stream)
    }
    fn launch_graph(&self, graph: GraphHandle, stream: u64) -> Result<()> {
        self.launch_graph_cu(graph, stream)
    }
    fn destroy_graph(&self, graph: GraphHandle) -> Result<()> {
        self.destroy_graph_cu(graph)
    }
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.memset_cu(ptr, value, bytes)
    }
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        self.memset_async_cu(ptr, value, bytes, stream)
    }
    fn total_memory(&self) -> Result<usize> {
        self.total_memory_cu()
    }
    fn free_memory(&self) -> Result<usize> {
        self.free_memory_cu()
    }
    fn sm_count(&self) -> Result<u32> {
        self.sm_count_cu()
    }
    fn create_stream(&self) -> Result<u64> {
        self.create_stream_cu()
    }
    fn bind_to_thread(&self) -> Result<()> {
        self.bind_to_thread_cu()
    }
    fn create_event(&self) -> Result<u64> {
        self.create_event_cu()
    }
    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        self.record_event_cu(event, stream)
    }
    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        self.stream_wait_event_cu(stream, event)
    }
    fn event_synchronize(&self, event: u64) -> Result<()> {
        self.event_synchronize_cu(event)
    }
    fn destroy_event(&self, event: u64) -> Result<()> {
        self.destroy_event_cu(event)
    }
    fn host_ptr_to_device(&self, host: *mut u8) -> Result<DevicePtr> {
        let mut dptr: u64 = 0;
        let status =
            unsafe { super::cuMemHostGetDevicePointer_v2(&mut dptr, host as *mut c_void, 0) };
        if status != 0 {
            bail!("cuMemHostGetDevicePointer_v2 failed: status {status}");
        }
        Ok(DevicePtr(dptr))
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        self.alloc_host_pinned_cu(bytes)
    }
    fn free_host_pinned(&self, ptr: *mut u8, _bytes: usize) -> Result<()> {
        self.free_host_pinned_cu(ptr, _bytes)
    }
}
