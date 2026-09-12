// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 6b — bring up the GPU backend for whichever backend feature is on.
//!
//! Split out of `preflight.rs` because the CUDA arm gained the device-arch
//! gate: the two `init_gpu_backend` arms are a self-contained unit (construct
//! the backend, latch the free-memory baseline, log the device) and reading
//! them next to each other is how the Metal/CUDA divergence stays visible.

use anyhow::{Context, Result};

use crate::cli;

/// Initialize the GPU backend for the active feature.
///
/// Compile-time dispatch:
/// - `cuda` feature → `AtlasCudaBackend` loading PTX modules from `ptx_set`.
/// - `metal` feature → `MetalGpuBackend` loading metallib modules from
///   `ptx_set` as well. Both arms register the RESOLVED target's modules;
///   `metallib_modules()` is a plain alias of target 0, so registering from
///   it served another model's kernels in a multi-target build.
#[cfg(feature = "cuda")]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<(Box<dyn spark_runtime::gpu::GpuBackend>, usize)> {
    super::super::kernel_gate::gate_device_arch(args.check_kernels, ptx_set, args.gpu_ordinal)?;

    let backend =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(args.gpu_ordinal, &ptx_set.modules)
            .context("Failed to initialize CUDA backend")?;

    let gpu: Box<dyn spark_runtime::gpu::GpuBackend> = Box::new(backend);
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    // Baseline for self-relative KV budgeting: free memory now (post context +
    // PTX modules, pre weights) minus free-at-build = this process's own
    // footprint, co-tenants excluded. See gpu::baseline_free_bytes.
    spark_runtime::gpu::set_baseline_free_bytes(free_mem);
    tracing::info!(
        "GPU {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<(Box<dyn spark_runtime::gpu::GpuBackend>, usize)> {
    // The RESOLVED target's modules, exactly like the CUDA arm above.
    // `metallib_modules()` is an alias of `ptx_modules()`, which build-codegen
    // emits as a plain alias of TARGET 0 in a multi-target build — so this
    // registered another model's kernels and every lookup for the model
    // actually being served failed.
    let gpu: Box<dyn spark_runtime::gpu::GpuBackend> = Box::new(
        spark_runtime::metal_backend::MetalGpuBackend::new(args.gpu_ordinal, &ptx_set.modules)
            .context("Failed to initialize Metal backend")?,
    );
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    spark_runtime::gpu::set_baseline_free_bytes(free_mem);
    tracing::info!(
        "Metal device {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}
