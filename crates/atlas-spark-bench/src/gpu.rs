// SPDX-License-Identifier: AGPL-3.0-only

//! GPU utilities for kernel microbenchmarks.
//!
//! Initializes `AtlasRegistry`, allocates device buffers, and provides
//! CUDA event-based timing for Criterion `iter_custom` benchmarks.

use std::ffi::c_void;
use std::sync::OnceLock;

use anyhow::{Result, bail};
use atlas_core::registry::{AtlasRegistry, RawCudaFunc};
use cudarc::driver::LaunchConfig;

// Raw CUDA driver API for benchmarks.
unsafe extern "C" {
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    fn cuMemFree_v2(dptr: u64) -> i32;
    fn cuMemsetD8Async(dst: u64, value: u8, n: usize, stream: u64) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

static INIT: OnceLock<()> = OnceLock::new();

/// Ensure the AtlasRegistry is initialized (idempotent).
pub fn ensure_registry() -> &'static AtlasRegistry {
    INIT.get_or_init(|| {
        let ptx = atlas_kernels::ptx_modules();
        AtlasRegistry::get_or_init(0, &ptx).expect("AtlasRegistry init failed — is GPU available?");
    });
    AtlasRegistry::get()
}

/// Allocate `bytes` of GPU memory, zero-initialized.
pub fn gpu_alloc_zeroed(stream: u64, bytes: usize) -> Result<u64> {
    let mut dptr: u64 = 0;
    let status = unsafe { cuMemAlloc_v2(&mut dptr, bytes) };
    if status != 0 {
        bail!("cuMemAlloc_v2 failed: status {status}, {bytes} bytes");
    }
    let status = unsafe { cuMemsetD8Async(dptr, 0, bytes, stream) };
    if status != 0 {
        bail!("cuMemsetD8Async failed: status {status}");
    }
    Ok(dptr)
}

/// Free GPU memory.
pub fn gpu_free(dptr: u64) {
    if dptr != 0 {
        unsafe { cuMemFree_v2(dptr) };
    }
}

/// Synchronize the stream.
pub fn gpu_sync(stream: u64) -> Result<()> {
    let status = unsafe { cuStreamSynchronize(stream) };
    if status != 0 {
        bail!("cuStreamSynchronize failed: status {status}");
    }
    Ok(())
}

/// Look up a kernel function handle with caching.
pub fn get_kernel(
    registry: &'static AtlasRegistry,
    cache: &OnceLock<RawCudaFunc>,
    module: &str,
    func: &str,
) -> RawCudaFunc {
    registry
        .raw_function_cached(cache, module, func)
        .unwrap_or_else(|e| panic!("Kernel {module}::{func} not found: {e}"))
}

/// Launch a kernel with raw parameters.
///
/// # Safety
/// `params` must contain valid pointers matching the kernel signature.
pub unsafe fn launch(
    registry: &AtlasRegistry,
    func: RawCudaFunc,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
    stream: u64,
    params: &mut [*mut c_void],
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: shared_mem,
    };
    unsafe { registry.launch_on_stream(func, cfg, stream, params) }
        .map_err(|e| anyhow::anyhow!("Kernel launch failed: {e}"))
}

/// Measure kernel execution time in milliseconds using CUDA events.
/// Runs `warmup` warmup iterations, then `iters` timed iterations,
/// returning the minimum time across 3 rounds.
pub fn bench_kernel_ms(
    stream: u64,
    warmup: usize,
    iters: usize,
    mut kernel_fn: impl FnMut(),
) -> f32 {
    // Create CUDA events
    let mut start: u64 = 0;
    let mut end: u64 = 0;
    unsafe {
        cuEventCreate(&mut start, 0);
        cuEventCreate(&mut end, 0);
    }

    let mut best_ms = f32::MAX;

    for _ in 0..3 {
        // Warmup
        for _ in 0..warmup {
            kernel_fn();
        }
        gpu_sync(stream).unwrap();

        // Timed run
        unsafe { cuEventRecord(start, stream) };
        for _ in 0..iters {
            kernel_fn();
        }
        unsafe { cuEventRecord(end, stream) };
        unsafe { cuEventSynchronize(end) };

        let mut elapsed_ms: f32 = 0.0;
        unsafe { cuEventElapsedTime(&mut elapsed_ms, start, end) };
        let per_iter = elapsed_ms / iters as f32;
        if per_iter < best_ms {
            best_ms = per_iter;
        }
    }

    unsafe {
        cuEventDestroy_v2(start);
        cuEventDestroy_v2(end);
    }

    best_ms
}
