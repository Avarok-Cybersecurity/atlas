// SPDX-License-Identifier: AGPL-3.0-only

//! CUTLASS-backed projection paths: the benchmark-only BF16 reference route
//! and the native NVFP4 routes (checkpoint-native and dequant-from-FP8).
//! Split out of `dispatch_proj.rs` during the ≤500-line split (pure move).

use super::*;

/// [`dequant_fp8_bf16_into`] into a FRESH allocation, memoised by FP8 weight
/// pointer (weights are immutable after load).
///
/// ⚠ OFF-LEDGER. The allocation has no `spark_runtime::buffers::sizes::BufferSizes`
/// entry, so `--gpu-memory-utilization` cannot see it — the #917 defect named
/// on [`dequant_fp8_bf16_into`]. The last caller is [`cutlass_bf16_proj`], a
/// benchmark-only reference path that a shipping recipe cannot reach; the GDN
/// row-wise arms moved to the ledgered slab in
/// `qwen3_ssm/rowwise_bf16.rs`. Do NOT add callers — take a ledgered
/// destination and call [`dequant_fp8_bf16_into`] instead.
fn dequant_fp8_bf16_cached(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    derived: &super::DerivedWeights,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<u64> {
    let cache_key = fp8w.weight.0;
    if let Some(hit) = derived.get_ptr(super::Derivation::Bf16, cache_key) {
        return Ok(hit);
    }
    let out = gpu.alloc(dequant_fp8_bf16_bytes(fp8w))?; // BF16 [N,K]
    dequant_fp8_bf16_into(gpu, fp8w, out, stream)?;
    derived.insert_ptr(super::Derivation::Bf16, cache_key, out.0);
    Ok(out.0)
}

/// [`dequant_fp8_bf16_into`] into a FRESH allocation the caller FREES — the
/// NVFP4 packer's transient, which never outlives the pack.
fn dequant_fp8_bf16_uncached(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<spark_runtime::gpu::DevicePtr> {
    let out = gpu.alloc(dequant_fp8_bf16_bytes(fp8w))?;
    dequant_fp8_bf16_into(gpu, fp8w, out, stream)?;
    Ok(out)
}

/// Route a projection `out[M,N] = act[M,K] @ weightᵀ` through CUTLASS BF16.
///
/// ★ A REFERENCE PATH FOR BENCHMARKING, NOT A SHIPPING ONE. Opt-in behind
/// `AVAROK_CUTLASS_GEMM=1` and OFF by default; a build without `CUTLASS_HOME`
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
