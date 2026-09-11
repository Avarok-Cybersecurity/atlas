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
    cublas_fp8_proj_prequant(
        gpu,
        act_fp8_scratch,
        act_scale_scratch,
        fp8w,
        out,
        m,
        n,
        k,
        stream,
    )
}

/// [`cublas_fp8_proj`] for an activation that is ALREADY quantized — the
/// caller ran `per_token_group_quant_fp8` itself.
///
/// WHY the split (#917/#928): the dense FFN's gate and up projections consume
/// the SAME `[M, K]` input, so quantizing inside the GEMM helper would pay the
/// per-token quant twice per layer. The FFN quantizes once and calls this for
/// both, then quantizes the post-SiLU intermediate once for `down`.
///
/// ⚠ PADDED-M EXTENTS. cuBLASLt is handed `ceil16(M)`, so:
///
/// * `out` must hold `ceil16(M) * N` BF16 elements — the phantom rows are
///   WRITTEN (well-defined: their activation scales are zeroed below).
/// * `act_fp8` must hold `ceil16(M) * K` bytes and `act_scale`
///   `ceil16(M) * (K/128)` f32 — the phantom rows are READ.
///
/// The arena sizes that headroom in; see the sizing notes in
/// `spark_runtime::buffers::sizes` (`fp8_act`, `ffn_act_a`, `expert_gate_out`,
/// `moe_output`).
#[allow(clippy::too_many_arguments)]
pub fn cublas_fp8_proj_prequant(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    act_fp8: spark_runtime::gpu::DevicePtr,
    act_scale: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    // cuBLASLt requires the scale-tensor M extent to be a multiple of 4; pad to
    // 16 (TC-friendly) and zero the padding scale rows so the phantom output
    // columns (ignored by the caller) are well-defined.
    let m_pad = cublas_fp8_m_pad(m);
    if m_pad > m {
        let pad_rows = (m_pad - m) as usize;
        // Scales: `[M, K/128]` FP32, row-major — the exact layout
        // `per_token_group_quant_fp8` writes and cuBLASLt reads as the VEC128
        // B-scale, so the pad rows are a contiguous tail.
        let kg = (k / 128) as usize;
        gpu.memset_async(
            act_scale.offset(m as usize * kg * 4),
            0,
            pad_rows * kg * 4,
            stream,
        )?;
        // Activation bytes too: a zero scale kills the phantom rows'
        // CONTRIBUTION, but the FP8 dot product still runs over whatever bytes
        // are there and `NaN * 0.0` is `NaN`. Same reasoning (and same fix) as
        // the row-wise sibling in `dispatch_proj_rowwise.rs`.
        gpu.memset_async(
            act_fp8.offset(m as usize * k as usize),
            0,
            pad_rows * k as usize,
            stream,
        )?;
    }
    spark_runtime::cublaslt::fp8_gemm_act_weight_t_blkscaled(
        act_fp8.0,
        act_scale.0,
        fp8w.weight.0,
        fp8w.row_scale.0,
        out.0,
        m_pad,
        n,
        k,
        stream,
    )
}

/// The M extent [`cublas_fp8_proj_prequant`] actually hands cuBLASLt. SSOT for
/// the callers that must bounds-check their output buffer against it.
pub fn cublas_fp8_m_pad(m: u32) -> u32 {
    m.div_ceil(16) * 16
}

/// Dequantize a block-scaled FP8 weight `[N,K]` → BF16 on-GPU once, cached by the
/// FP8 weight pointer (weights are immutable after load). 128×128 blocks + FP32
/// scales (the holo layout). Backs [`cublas_bf16_proj`].
fn dequant_fp8_bf16_cached(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    derived: &super::DerivedWeights,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<u64> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    let cache_key = fp8w.weight.0;
    if let Some(hit) = derived.get_ptr(super::Derivation::Bf16, cache_key) {
        return Ok(hit);
    }
    let (n, kk) = (fp8w.n, fp8w.k);
    let out = gpu.alloc(n as usize * kk as usize * 2)?; // BF16 [N,K]
    // The kernel reads `scale[(n / block_n) * sk + (k / block_k)]`, so the
    // SAME kernel serves both layouts — the block geometry is what selects
    // between them, not a second kernel:
    //
    //   block-scaled   block_n = block_k = 128, sk = K/128
    //   PER-ROW        block_n = 1, block_k = K, sk = 1
    //                  -> offset = n * 1 + 0 = n, one multiplier per row
    //
    // That per-row case is what a mixed-precision compressed-tensors
    // checkpoint ships, and dequantising it here is lossless: every FP8 E4M3
    // value is exactly representable in BF16, so this is the fold's
    // no-double-quant path even though the GEMM downstream is BF16.
    let per_row = fp8w.scale_format == crate::weight_map::WeightQuantFormat::Fp8PerRow;
    let (block_n, block_k, sk) = if per_row {
        (1u32, kk, 1u32)
    } else {
        (128u32, 128u32, kk / 128)
    };
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
        .arg_u32(block_n)
        .arg_u32(block_k)
        .arg_u32(sk)
        .arg_u32(1) // scale_is_fp32
        .launch(stream)?;
    derived.insert_ptr(super::Derivation::Bf16, cache_key, out.0);
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
    derived: &super::DerivedWeights,
    act: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let w_bf16 = dequant_fp8_bf16_cached(gpu, derived, fp8w, stream)?;
    spark_runtime::cublaslt::bf16_gemm_act_weight_t(act.0, w_bf16, out.0, m, n, k, stream)
}

/// Route a projection `out[M,N] = act[M,K] @ weightᵀ` through cuBLASLt BF16 for
/// a weight that is already native BF16 `[N,K]` (no dequant step). Used by
/// models whose attention/shared-expert weights ship unquantized (e.g. Laguna),
/// which can never satisfy the `as_fp8()` gate of [`cublas_bf16_proj`].
pub fn cublas_bf16_proj_dense(
    act: spark_runtime::gpu::DevicePtr,
    weight_bf16: spark_runtime::gpu::DevicePtr,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    spark_runtime::cublaslt::bf16_gemm_act_weight_t(act.0, weight_bf16.0, out.0, m, n, k, stream)
}

/// Route a projection `out[M,N] = act[M,K] @ weightᵀ` through CUTLASS BF16.
///
/// ★ A REFERENCE PATH FOR BENCHMARKING, NOT A SHIPPING ONE. Opt-in behind
/// `ATLAS_CUTLASS_GEMM=1` and OFF by default; a build without `CUTLASS_HOME`
/// cannot reach it at all. It exists so a shape can be A/B'd against the
/// industry reference on the same box — if CUTLASS wins a shape, the fix is
/// a faster Atlas kernel, not a promotion. See the module docs on
/// `spark_runtime::cutlass` for the full rationale (SSOT).
#[allow(clippy::too_many_arguments)]
pub fn cutlass_bf16_proj(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    derived: &super::DerivedWeights,
    act: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let w_bf16 = dequant_fp8_bf16_cached(gpu, derived, fp8w, stream)?;
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
    derived: &super::DerivedWeights,
    weight_t: &crate::weight_map::QuantizedWeight,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<u64> {
    let cache_key = weight_t.weight.0;
    if let Some(hit) = derived.get_ptr(super::Derivation::CutlassNvfp4Transposed, cache_key) {
        return Ok(hit);
    }
    let dst = gpu.alloc((n as usize) * (k as usize) / 2)?;
    spark_runtime::cutlass::transpose_nvfp4_packed_kton(weight_t.weight.0, dst.0, n, k, stream)?;
    gpu.synchronize(stream)?;
    derived.insert_ptr(super::Derivation::CutlassNvfp4Transposed, cache_key, dst.0);
    Ok(dst.0)
}

#[allow(clippy::too_many_arguments)]
pub fn cutlass_nvfp4_proj(
    // The backend and this model's derived-weight cache travel together
    // everywhere they are used; taking the context instead of the pair keeps
    // the call sites one line each.
    ctx: &crate::layer::ForwardContext<'_>,
    act: spark_runtime::gpu::DevicePtr,
    weight_t: &crate::weight_map::QuantizedWeight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let (gpu, derived) = (ctx.gpu, ctx.derived);
    let packed = cutlass_nvfp4_weight_transposed_cached(gpu, derived, weight_t, n, k, stream)?;
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
    derived: &super::DerivedWeights,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<(u64, u64)> {
    let cache_key = fp8w.weight.0;
    if let Some(hit) = derived.get_pair(super::Derivation::CutlassNvfp4FromFp8, cache_key) {
        return Ok(hit);
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
    derived.insert_pair(
        super::Derivation::CutlassNvfp4FromFp8,
        cache_key,
        (packed_t.0, scale_t.0),
    );
    Ok((packed_t.0, scale_t.0))
}

/// Native CUTLASS NVFP4 projection for FP8 checkpoint weights. The FP8 weight
/// is dequantized to BF16 using the existing cache, then packed once into
/// Atlas-transposed NVFP4 data/scales and reused for future calls.
#[allow(clippy::too_many_arguments)]
pub fn cutlass_nvfp4_proj_from_fp8(
    // The backend and this model's derived-weight cache travel together
    // everywhere they are used; taking the context instead of the pair keeps
    // the call sites one line each.
    ctx: &crate::layer::ForwardContext<'_>,
    act: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let (gpu, derived) = (ctx.gpu, ctx.derived);
    let (packed_t, scale_t) = cutlass_nvfp4_weight_from_fp8_cached(gpu, derived, fp8w, stream)?;
    spark_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t(
        act.0, packed_t, scale_t, 1.0, out.0, m, n, k, stream,
    )
}
