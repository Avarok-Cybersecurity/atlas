// SPDX-License-Identifier: AGPL-3.0-only

//! GPU dequant launch wrappers: raw packed GGUF blocks -> BF16 device buffer.
//!
//! Mirrors the verified `dequant_nvfp4_to_bf16` launch shape in
//! spark-model `weight_map/fp8_lut.rs`, but from inside spark-runtime so it
//! uses crate-local `crate::gpu` / `crate::kernel_args` paths.
//!
//! Contract for every wrapper:
//!   * `blocks` is a DevicePtr to the tensor's raw GGUF quant bytes, already
//!     uploaded h2d by the caller (`n_blocks * block_bytes()`).
//!   * returns a freshly-alloc'd BF16 DevicePtr of `n_blocks * qk` elements
//!     (`* 2` bytes); the caller owns it and frees `blocks`.
//!   * module name = the `.cu` file stem `dequant_gguf_bf16`; func name = the
//!     `extern "C" __global__` symbol.
//!
//! The block dim is a fixed 256 with an in-kernel stride loop, so a single
//! launch config serves QK=32 (Q8_0), QK=256 (K-quants), and G in {64,128}
//! (Q2_0) without per-type tuning.
//!
//! NOTE: these kernels are runtime-validated only on a real GB10 backend. Under
//! `MockGpuBackend` (unit tests) `launch` is a no-op, so the CPU reference path
//! (`super::dequant_cpu`) is what the tests exercise for numeric correctness.

use anyhow::{Result, bail};

use crate::gpu::{DevicePtr, GpuBackend};
use crate::kernel_args::KernelLaunch;

const MODULE: &str = "dequant_gguf_bf16";
const BLOCK_DIM: u32 = 256;

/// True if a GPU dequant kernel exists for this ggml type id.
pub(crate) fn supports(id: u32) -> bool {
    matches!(id, 8 | 12 | 14 | 42)
}

/// Dequant a tensor's raw quant blocks (already uploaded to `q_ptr`) to a fresh
/// BF16 device buffer via the GPU kernel for `id`. `q2_group` is the id-42
/// group size (ignored for other types). Errors if `id` has no GPU kernel.
pub(crate) fn to_bf16(
    gpu: &dyn GpuBackend,
    id: u32,
    q_ptr: DevicePtr,
    n_elements: usize,
    q2_group: usize,
) -> Result<DevicePtr> {
    match dispatch_gpu_dequant(gpu, id, q_ptr, n_elements, q2_group)? {
        Some(ptr) => Ok(ptr),
        None => bail!("no GPU dequant kernel for ggml type {id}"),
    }
}

/// Q8_0: QK=32, 34-byte blocks. `n_blocks` = numel / 32.
pub(crate) fn dequant_q8_0(
    gpu: &dyn GpuBackend,
    blocks: DevicePtr,
    n_blocks: usize,
) -> Result<DevicePtr> {
    let out = gpu.alloc(n_blocks * 32 * 2)?; // BF16 = 2 B/elem
    let kernel = gpu.kernel(MODULE, "dequant_q8_0_to_bf16")?;
    let stream = gpu.default_stream();
    KernelLaunch::new(gpu, kernel)
        .grid([n_blocks as u32, 1, 1])
        .block([BLOCK_DIM, 1, 1])
        .arg_ptr(blocks)
        .arg_ptr(out)
        .arg_u32(n_blocks as u32)
        .arg_u32(34) // block_bytes
        .launch(stream)?;
    gpu.synchronize(stream)?;
    Ok(out)
}

/// Q4_K: QK_K=256, 144-byte super-blocks. `n_blocks` = numel / 256.
pub(crate) fn dequant_q4_k(
    gpu: &dyn GpuBackend,
    blocks: DevicePtr,
    n_blocks: usize,
) -> Result<DevicePtr> {
    let out = gpu.alloc(n_blocks * 256 * 2)?;
    let kernel = gpu.kernel(MODULE, "dequant_q4_k_to_bf16")?;
    let stream = gpu.default_stream();
    KernelLaunch::new(gpu, kernel)
        .grid([n_blocks as u32, 1, 1])
        .block([BLOCK_DIM, 1, 1])
        .arg_ptr(blocks)
        .arg_ptr(out)
        .arg_u32(n_blocks as u32)
        .arg_u32(144)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    Ok(out)
}

/// Q6_K: QK_K=256, 210-byte super-blocks. `n_blocks` = numel / 256.
pub(crate) fn dequant_q6_k(
    gpu: &dyn GpuBackend,
    blocks: DevicePtr,
    n_blocks: usize,
) -> Result<DevicePtr> {
    let out = gpu.alloc(n_blocks * 256 * 2)?;
    let kernel = gpu.kernel(MODULE, "dequant_q6_k_to_bf16")?;
    let stream = gpu.default_stream();
    KernelLaunch::new(gpu, kernel)
        .grid([n_blocks as u32, 1, 1])
        .block([BLOCK_DIM, 1, 1])
        .arg_ptr(blocks)
        .arg_ptr(out)
        .arg_u32(n_blocks as u32)
        .arg_u32(210)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    Ok(out)
}

/// Q2_0 group-N (PrismML id 42). `group_size` in {128, 64}; block_bytes =
/// 2 + group_size/4 (34 or 18). `n_blocks` = numel / group_size.
pub(crate) fn dequant_q2_0_gn(
    gpu: &dyn GpuBackend,
    blocks: DevicePtr,
    n_blocks: usize,
    group_size: usize,
) -> Result<DevicePtr> {
    let block_bytes = 2 + group_size / 4;
    let out = gpu.alloc(n_blocks * group_size * 2)?;
    let kernel = gpu.kernel(MODULE, "dequant_q2_0_gn_to_bf16")?;
    let stream = gpu.default_stream();
    KernelLaunch::new(gpu, kernel)
        .grid([n_blocks as u32, 1, 1])
        .block([BLOCK_DIM, 1, 1])
        .arg_ptr(blocks)
        .arg_ptr(out)
        .arg_u32(n_blocks as u32)
        .arg_u32(group_size as u32)
        .arg_u32(block_bytes as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    Ok(out)
}

/// Dispatch keyed by ggml type id, for the P0 GPU-backed types. Returns
/// `Ok(None)` for a type without a GPU kernel so the loader can fall back to
/// the CPU reference dequant. `numel` is the tensor element count; `q2_group`
/// is the id-42 group size (128 or 64), ignored for other types.
fn dispatch_gpu_dequant(
    gpu: &dyn GpuBackend,
    ggml_type: u32,
    blocks: DevicePtr,
    numel: usize,
    q2_group: usize,
) -> Result<Option<DevicePtr>> {
    let out = match ggml_type {
        8 => dequant_q8_0(gpu, blocks, numel / 32)?,   // Q8_0
        12 => dequant_q4_k(gpu, blocks, numel / 256)?, // Q4_K
        14 => dequant_q6_k(gpu, blocks, numel / 256)?, // Q6_K
        42 => dequant_q2_0_gn(gpu, blocks, numel / q2_group, q2_group)?, // Q2_0 id42
        _ => return Ok(None),
    };
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::mock::MockGpuBackend;

    #[test]
    fn supports_matches_dispatch_set() {
        for id in [8u32, 12, 14, 42] {
            assert!(supports(id), "id {id} should be GPU-supported");
        }
        for id in [0u32, 1, 30, 35, 999] {
            assert!(!supports(id), "id {id} must not claim GPU support");
        }
    }

    #[test]
    fn dispatch_returns_none_for_unsupported() {
        // Mock kernels are no-ops; we only assert the dispatch branch logic.
        let gpu = MockGpuBackend::new();
        let q = gpu.alloc(64).unwrap();
        assert!(dispatch_gpu_dequant(&gpu, 0, q, 32, 128).unwrap().is_none());
        assert!(dispatch_gpu_dequant(&gpu, 8, q, 32, 128).unwrap().is_some());
    }
}
