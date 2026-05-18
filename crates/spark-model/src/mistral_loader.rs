// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loader for Mistral Small 4 (MLA + MoE architecture).
//!
//! GQA fallback: MLA LoRA projections are expanded to dense at load time
//! via GPU matmul. Q = wq_b @ wq_a, K/V split from `wkv_b @ wkv_a[:kv_lora]`.
//! Loses MLA's 12.8x KV cache compression but produces coherent output.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;
use crate::weight_map::DenseWeight;

pub struct MistralWeightLoader;

/// GPU matmul: `C[M,N] = A[M,K] × B[K,N]` using dense_gemm_bf16 kernel.
/// Allocate GPU memory, falling back to managed (UVM) if device alloc fails.
/// Uses a static flag to avoid retrying device alloc after the first failure
/// (which wastes time and fragments memory).
fn gpu_alloc_or_managed(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    static USE_MANAGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if USE_MANAGED.load(std::sync::atomic::Ordering::Relaxed) {
        return gpu.alloc_managed(bytes);
    }
    match gpu.alloc(bytes) {
        Ok(p) => Ok(p),
        Err(_) => {
            tracing::warn!(
                "GPU alloc failed ({bytes} bytes) — switching to managed for remaining allocations"
            );
            USE_MANAGED.store(true, std::sync::atomic::Ordering::Relaxed);
            gpu.alloc_managed(bytes)
        }
    }
}

#[allow(dead_code)]
fn gpu_matmul(
    a: DevicePtr,
    b: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let bf16 = 2usize;
    let c = gpu_alloc_or_managed(gpu, m * n * bf16)?;
    let stream = gpu.default_stream();
    let gemm_k = gpu.kernel("gemm", "dense_gemm_bf16")?;
    let b_dense = DenseWeight { weight: b };
    ops::dense_gemm(
        gpu, gemm_k, a, &b_dense, c, m as u32, n as u32, k as u32, stream,
    )?;
    gpu.synchronize(stream)?;
    Ok(c)
}

mod loader_impl;
