// SPDX-License-Identifier: AGPL-3.0-only

//! cuBLAS / CUTLASS projection routers + their cached weight-prep helpers.
//! Extracted from `dispatch_helpers.rs` during the ≤500-line split. Re-exported
//! at `crate::layers::ops::*` via `ops.rs`.

#![allow(unused_imports)]

use super::*;

/// Route a projection through native-FP8 cuBLASLt block-scaled matmul: quantize
/// the activation to FP8 + per-[token,128-of-K] VEC128 scales (the existing
/// `per_token_group_quant_fp8` kernel), feed the FP8 weight + its per-128×128
/// block scales directly (zero dequant, zero extra weight memory). Both operands
/// 128-block-scaled (cuBLASLt requires it). ~1.8× the bf16 path (152 vs 85 TF).
///
/// `act_fp8_scratch`/`act_scale_scratch` must hold the padded extents (the
/// `buffers.fp8_act`/`fp8_act_scale` arena buffers, sized for max_batch_tokens).
#[allow(clippy::too_many_arguments)]
pub fn cublas_fp8_proj(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    ptg_quant_k: spark_runtime::gpu::KernelHandle,
    act_bf16: spark_runtime::gpu::DevicePtr,
    act_fp8_scratch: spark_runtime::gpu::DevicePtr,
    act_scale_scratch: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    // Quantize the real M tokens → fp8 bytes + VEC128 scales [M, K/128].
    per_token_group_quant_fp8(
        gpu,
        ptg_quant_k,
        act_bf16,
        act_fp8_scratch,
        act_scale_scratch,
        m,
        k,
        stream,
    )?;
    // cuBLASLt requires the scale-tensor M extent to be a multiple of 4; pad to
    // 16 (TC-friendly) and zero the padding scale rows so the phantom output
    // columns (ignored by the caller) are well-defined.
    let m_pad = m.div_ceil(16) * 16;
    if m_pad > m {
        let kg = (k / 128) as usize;
        let pad_off = m as usize * kg * 4;
        let pad_bytes = (m_pad - m) as usize * kg * 4;
        gpu.memset_async(act_scale_scratch.offset(pad_off), 0, pad_bytes, stream)?;
    }
    spark_runtime::cublaslt::fp8_gemm_act_weight_t_blkscaled(
        act_fp8_scratch.0,
        act_scale_scratch.0,
        fp8w.weight.0,
        fp8w.row_scale.0,
        out.0,
        m_pad,
        n,
        k,
        stream,
    )
}

/// Re-quantize a block-scaled FP8 weight `[N,K]` → ROW-WISE FP8 (E4M3 + per-row
/// FP32 scale `[N]`) on-GPU once, cached by the FP8 weight pointer. Path:
/// block-fp8 → BF16 (transient) → row-wise fp8. Backs the GB10-supported
/// `cublas_fp8_rowwise_proj`. Returns `(fp8_weight_ptr, per_row_scale_ptr)`.
fn requant_weight_rowwise_fp8_cached(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<(u64, u64)> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<u64, (u64, u64)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&p) = cache.lock().unwrap().get(&fp8w.weight.0) {
        return Ok(p);
    }
    let (n, k) = (fp8w.n, fp8w.k);
    // 1. block-fp8 → BF16 (transient scratch, freed after re-quant).
    let bf16 = gpu.alloc(n as usize * k as usize * 2)?;
    let block = 128u32;
    let sk = k / block;
    let dq = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    KernelLaunch::new(gpu, dq)
        .grid([div_ceil(k, 64), div_ceil(n, 4), 1])
        .block([64, 4, 1])
        .arg_ptr(fp8w.weight)
        .arg_ptr(fp8w.row_scale)
        .arg_ptr(bf16)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(block)
        .arg_u32(block)
        .arg_u32(sk)
        .arg_u32(1)
        .launch(stream)?;
    // 2. BF16 → row-wise fp8 [N,K] + per-row scale [N].
    let w_fp8 = gpu.alloc(n as usize * k as usize)?;
    let w_scale = gpu.alloc(n as usize * 4)?;
    let qk = gpu.kernel("quant_rowwise_fp8", "quant_rowwise_fp8")?;
    KernelLaunch::new(gpu, qk)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(bf16)
        .arg_ptr(w_fp8)
        .arg_ptr(w_scale)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    gpu.synchronize(stream)?; // re-quant must finish before the transient bf16 is freed
    gpu.free(bf16)?;
    cache
        .lock()
        .unwrap()
        .insert(fp8w.weight.0, (w_fp8.0, w_scale.0));
    Ok((w_fp8.0, w_scale.0))
}

/// Route a projection through ROW-WISE native-FP8 cuBLASLt (the fp8 path GB10
/// supports). Weight is re-quantized once to per-row fp8 (cached); the activation
/// is quantized per-token each call. ~1.8× the bf16 path (152 vs 85 TF), and
/// frees the bf16-dequant memory the bf16 path holds.
/// `act_fp8_scratch` ≥ m*k fp8 bytes; `act_scale_scratch` ≥ m f32 (e.g. the
/// `buffers.fp8_act` / `fp8_act_scale` arena buffers).
#[allow(clippy::too_many_arguments)]
pub fn cublas_fp8_rowwise_proj(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    act_bf16: spark_runtime::gpu::DevicePtr,
    act_fp8_scratch: spark_runtime::gpu::DevicePtr,
    act_scale_scratch: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    use spark_runtime::kernel_args::KernelLaunch;
    let (w_fp8, w_scale) = requant_weight_rowwise_fp8_cached(gpu, fp8w, stream)?;
    // Per-token row-wise quant of the activation → fp8 [M,K] + scale [M].
    let qk = gpu.kernel("quant_rowwise_fp8", "quant_rowwise_fp8")?;
    KernelLaunch::new(gpu, qk)
        .grid([m, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(act_bf16)
        .arg_ptr(act_fp8_scratch)
        .arg_ptr(act_scale_scratch)
        .arg_u32(m)
        .arg_u32(k)
        .launch(stream)?;
    spark_runtime::cublaslt::fp8_gemm_act_weight_t_rowwise(
        act_fp8_scratch.0,
        act_scale_scratch.0,
        w_fp8,
        w_scale,
        out.0,
        m,
        n,
        k,
        stream,
    )
}

/// Dequantize a block-scaled FP8 weight `[N,K]` → BF16 on-GPU once, cached by the
/// FP8 weight pointer (weights are immutable after load). 128×128 blocks + FP32
/// scales (the holo layout). Backs [`cublas_bf16_proj`].
fn dequant_fp8_bf16_cached(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<u64> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<u64, u64>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&p) = cache.lock().unwrap().get(&fp8w.weight.0) {
        return Ok(p);
    }
    let (n, kk) = (fp8w.n, fp8w.k);
    let out = gpu.alloc(n as usize * kk as usize * 2)?; // BF16 [N,K]
    let block = 128u32;
    let sk = kk / block;
    let kernel = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(kk, 64), div_ceil(n, 4), 1])
        .block([64, 4, 1])
        .arg_ptr(fp8w.weight)
        .arg_ptr(fp8w.row_scale)
        .arg_ptr(out)
        .arg_u32(n)
        .arg_u32(kk)
        .arg_u32(block)
        .arg_u32(block)
        .arg_u32(sk)
        .arg_u32(1) // scale_is_fp32
        .launch(stream)?;
    cache.lock().unwrap().insert(fp8w.weight.0, out.0);
    Ok(out.0)
}

fn dequant_fp8_bf16_uncached(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<spark_runtime::gpu::DevicePtr> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    let (n, kk) = (fp8w.n, fp8w.k);
    let out = gpu.alloc(n as usize * kk as usize * 2)?;
    let block = 128u32;
    let sk = kk / block;
    let kernel = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(kk, 64), div_ceil(n, 4), 1])
        .block([64, 4, 1])
        .arg_ptr(fp8w.weight)
        .arg_ptr(fp8w.row_scale)
        .arg_ptr(out)
        .arg_u32(n)
        .arg_u32(kk)
        .arg_u32(block)
        .arg_u32(block)
        .arg_u32(sk)
        .arg_u32(1)
        .launch(stream)?;
    Ok(out)
}

/// Route a projection `out[M,N] = act[M,K] @ weightᵀ` through cuBLASLt BF16.
/// The FP8 weight is dequantized to BF16 once (cached); W16A16 here is strictly
/// more accurate than the blockscaled W8A8 path it replaces.
#[allow(clippy::too_many_arguments)]
pub fn cublas_bf16_proj(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    act: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let w_bf16 = dequant_fp8_bf16_cached(gpu, fp8w, stream)?;
    spark_runtime::cublaslt::bf16_gemm_act_weight_t(act.0, w_bf16, out.0, m, n, k, stream)
}

/// Route a projection `out[M,N] = act[M,K] @ weightᵀ` through CUTLASS BF16.
/// This is the M0 de-risk path for replacing Atlas/cuBLAS GEMMs with
/// CUTLASS-backed kernels on GB10; keep it behind `ATLAS_CUTLASS_GEMM=1`.
#[allow(clippy::too_many_arguments)]
pub fn cutlass_bf16_proj(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    act: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let w_bf16 = dequant_fp8_bf16_cached(gpu, fp8w, stream)?;
    spark_runtime::cutlass::bf16_gemm_act_weight_t(act.0, w_bf16, out.0, m, n, k, stream)
}

/// Route a projection `out[M,N] = act[M,K] @ weightᵀ` through native CUTLASS
/// NVFP4. The activation is packed to CUTLASS NVFP4 inside the runtime wrapper.
/// `weight_t` must be Atlas's transposed NVFP4 layout `[K/2,N]` plus
/// `[K/16,N]` scales, as produced by `QuantizedWeight::transpose_for_gemm`.
#[allow(clippy::too_many_arguments)]
/// Transpose a native NVFP4 checkpoint weight from Atlas `[K/2,N]` into the
/// CUTLASS `[N,K/2]` byte layout the GEMM consumes, caching the result by
/// source weight ptr. Without this the ColumnMajor B operand is read
/// transposed and the GEMM produces garbage (cos≈0 vs reference).
fn cutlass_nvfp4_weight_transposed_cached(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    weight_t: &crate::weight_map::QuantizedWeight,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<u64> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<u64, u64>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&p) = cache.lock().unwrap().get(&weight_t.weight.0) {
        return Ok(p);
    }
    let dst = gpu.alloc((n as usize) * (k as usize) / 2)?;
    spark_runtime::cutlass::transpose_nvfp4_packed_kton(weight_t.weight.0, dst.0, n, k, stream)?;
    gpu.synchronize(stream)?;
    cache.lock().unwrap().insert(weight_t.weight.0, dst.0);
    Ok(dst.0)
}

#[allow(clippy::too_many_arguments)]
pub fn cutlass_nvfp4_proj(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    act: spark_runtime::gpu::DevicePtr,
    weight_t: &crate::weight_map::QuantizedWeight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let packed = cutlass_nvfp4_weight_transposed_cached(gpu, weight_t, n, k, stream)?;
    spark_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t(
        act.0,
        packed,
        weight_t.weight_scale.0,
        weight_t.weight_scale_2,
        out.0,
        m,
        n,
        k,
        stream,
    )
}

fn cutlass_nvfp4_weight_from_fp8_cached(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<(u64, u64)> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<u64, (u64, u64)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&p) = cache.lock().unwrap().get(&fp8w.weight.0) {
        return Ok(p);
    }

    let n = fp8w.n as usize;
    let k = fp8w.k as usize;
    let w_bf16 = dequant_fp8_bf16_uncached(gpu, fp8w, stream)?;
    let packed_t = gpu.alloc(n * k / 2)?;
    let scale_t = gpu.alloc(n * k / 16)?;
    spark_runtime::cutlass::pack_bf16_weight_to_nvfp4_t(
        w_bf16.0, packed_t.0, scale_t.0, fp8w.n, fp8w.k, stream,
    )?;
    gpu.synchronize(stream)?;
    gpu.free(w_bf16)?;
    cache
        .lock()
        .unwrap()
        .insert(fp8w.weight.0, (packed_t.0, scale_t.0));
    Ok((packed_t.0, scale_t.0))
}

/// Native CUTLASS NVFP4 projection for FP8 checkpoint weights. The FP8 weight
/// is dequantized to BF16 using the existing cache, then packed once into
/// Atlas-transposed NVFP4 data/scales and reused for future calls.
#[allow(clippy::too_many_arguments)]
pub fn cutlass_nvfp4_proj_from_fp8(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    act: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let (packed_t, scale_t) = cutlass_nvfp4_weight_from_fp8_cached(gpu, fp8w, stream)?;
    spark_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t(
        act.0, packed_t, scale_t, 1.0, out.0, m, n, k, stream,
    )
}
