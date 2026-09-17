// SPDX-License-Identifier: AGPL-3.0-only
//
// Minimal CUDA driver FFI for the Phase-0 probe. Intentionally separate from
// spark-runtime's `cuda_backend.rs` so the probe binary doesn't pull in the
// full Atlas runtime / kernel registry. Only the symbols the probe needs.

use anyhow::{Result, bail};
use std::ffi::c_void;

// Re-export the helpers that used to live here, keeping
// `cuda_min::{CudaModule, CudaEvent, launch_kernel}` paths working after the
// split into `cuda_module.rs`.
pub use crate::cuda_module::{CudaEvent, CudaModule, launch_kernel};

unsafe extern "C" {
    fn cuInit(flags: u32) -> i32;
    fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
    fn cuCtxCreate_v2(pctx: *mut u64, flags: u32, dev: i32) -> i32;
    fn cuCtxDestroy_v2(ctx: u64) -> i32;
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    fn cuMemFree_v2(dptr: u64) -> i32;
    fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> i32;
    fn cuMemFreeHost(p: *mut c_void) -> i32;
    fn cuMemHostGetDevicePointer_v2(pdptr: *mut u64, p: *mut c_void, flags: u32) -> i32;
    fn cuMemcpyHtoDAsync_v2(dst: u64, src: *const c_void, bytes: usize, stream: u64) -> i32;
    fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: u64, bytes: usize, stream: u64) -> i32;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> i32;
    fn cuStreamCreate(phStream: *mut u64, flags: u32) -> i32;
    fn cuStreamDestroy_v2(stream: u64) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
}

/// Query the current context's free/total HBM in bytes. Used by HSS install
/// preflight to fail fast with an actionable error before a multi-GB
/// `cuMemAlloc` blows up cryptically. Phase-7 follow-up to PR #47.
pub fn mem_info() -> Result<(usize, usize)> {
    let mut free = 0usize;
    let mut total = 0usize;
    let s = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
    if s != 0 {
        bail!("cuMemGetInfo_v2 failed: {s}");
    }
    Ok((free, total))
}

pub struct CudaCtx {
    pub ctx: u64,
    pub stream: u64,
}

impl CudaCtx {
    pub fn new(ordinal: i32) -> Result<Self> {
        unsafe {
            let s = cuInit(0);
            if s != 0 {
                bail!("cuInit failed: {s}");
            }
            let mut dev = 0i32;
            let s = cuDeviceGet(&mut dev, ordinal);
            if s != 0 {
                bail!("cuDeviceGet({ordinal}) failed: {s}");
            }
            let mut ctx = 0u64;
            let s = cuCtxCreate_v2(&mut ctx, 0, dev);
            if s != 0 {
                bail!("cuCtxCreate failed: {s}");
            }
            let mut stream = 0u64;
            let s = cuStreamCreate(&mut stream, 0);
            if s != 0 {
                cuCtxDestroy_v2(ctx);
                bail!("cuStreamCreate failed: {s}");
            }
            Ok(Self { ctx, stream })
        }
    }
}

impl Drop for CudaCtx {
    fn drop(&mut self) {
        unsafe {
            let _ = cuStreamDestroy_v2(self.stream);
            let _ = cuCtxDestroy_v2(self.ctx);
        }
    }
}

pub struct DeviceBuffer {
    pub ptr: u64,
    pub bytes: usize,
}

impl DeviceBuffer {
    pub fn new(bytes: usize) -> Result<Self> {
        let mut p = 0u64;
        let s = unsafe { cuMemAlloc_v2(&mut p, bytes) };
        if s != 0 {
            bail!("cuMemAlloc_v2({bytes}) failed: {s}");
        }
        Ok(Self { ptr: p, bytes })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = cuMemFree_v2(self.ptr);
        }
    }
}

pub struct PinnedBuffer {
    pub ptr: *mut c_void,
    pub bytes: usize,
}

// SAFETY: `cuMemAllocHost` returns a process-pinned allocation whose
// virtual address is stable for the buffer's entire lifetime — moving the
// `PinnedBuffer` between threads only transfers a pointer + length + the
// CUcontext handle used by Drop, none of which alias mutable state. The
// inner pointer never escapes through `&self` accessors; concurrent users
// of the underlying memory must coordinate externally (Atlas does this
// via the io_uring submission queue, which is single-threaded per rank).
unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}

impl PinnedBuffer {
    pub fn new(bytes: usize) -> Result<Self> {
        let mut p: *mut c_void = std::ptr::null_mut();
        let s = unsafe { cuMemAllocHost_v2(&mut p, bytes) };
        if s != 0 {
            bail!("cuMemAllocHost_v2({bytes}) failed: {s}");
        }
        Ok(Self { ptr: p, bytes })
    }

    /// Device pointer the GPU uses to address this pinned host allocation.
    ///
    /// On GB10's unified LPDDR (unified addressing) this equals the host
    /// pointer numerically — the property the UMA zero-copy expert arena relies
    /// on. `cuMemAllocHost` memory is portable + page-locked + device-accessible,
    /// so no `Mapped` flag is required.
    pub fn device_ptr(&self) -> Result<u64> {
        let mut dptr = 0u64;
        let s = unsafe { cuMemHostGetDevicePointer_v2(&mut dptr, self.ptr, 0) };
        if s != 0 {
            bail!("cuMemHostGetDevicePointer_v2 failed: {s}");
        }
        Ok(dptr)
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = cuMemFreeHost(self.ptr);
        }
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn copy_h_to_d_async(dst: u64, src: *const c_void, bytes: usize, stream: u64) -> Result<()> {
    let s = unsafe { cuMemcpyHtoDAsync_v2(dst, src, bytes, stream) };
    if s != 0 {
        bail!("cuMemcpyHtoDAsync_v2 failed: {s}");
    }
    Ok(())
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn copy_d_to_h_async(dst: *mut c_void, src: u64, bytes: usize, stream: u64) -> Result<()> {
    let s = unsafe { cuMemcpyDtoHAsync_v2(dst, src, bytes, stream) };
    if s != 0 {
        bail!("cuMemcpyDtoHAsync_v2 failed: {s}");
    }
    Ok(())
}

pub fn stream_sync(stream: u64) -> Result<()> {
    let s = unsafe { cuStreamSynchronize(stream) };
    if s != 0 {
        bail!("cuStreamSynchronize failed: {s}");
    }
    Ok(())
}
