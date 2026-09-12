// SPDX-License-Identifier: AGPL-3.0-only

//! cuBLAS / CUTLASS projection routers + their cached weight-prep helpers.
//! Extracted from `dispatch_helpers.rs` during the ≤500-line split. Re-exported
//! at `crate::layers::ops::*` via `ops.rs`.

#![allow(unused_imports)]

use super::*;

/// `ATLAS_CUBLAS_SCALE_LAYOUT` — which VEC128 activation-scale layout the
/// cuBLASLt block-scaled arm feeds the library.
///
/// * `kmajor` (DEFAULT) — `[K/128, ceil16(M)]`, tokens contiguous. What the
///   cuBLAS manual's "Scaling factors layouts" specifies for the B operand
///   ("N-major for B with shape N x L"); see
///   `spark_runtime::cublaslt::scale_layout` for the full quotes.
/// * `rowmajor` — the quantizer's `[M, K/128]` handed over untransposed, i.e.
///   the pre-fix reading. KEPT ONLY as a measurement control: it is what the
///   2026-09-11 H100 run measured at rel_rms 7.7e-2 / cosine 0.996 vs the
///   in-tree kernel, and an operator comparing the two arms on one box should
///   not have to check out an old commit to reproduce it.
///
/// `OnceLock`-cached for the same reason `ffn_w8a16_only()` is: the selector
/// runs per projection per layer per prefill and `env::var_os` walks the
/// environment block every call.
pub fn cublas_scale_layout_kmajor() -> bool {
    static KMAJOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KMAJOR.get_or_init(|| {
        !matches!(
            std::env::var("ATLAS_CUBLAS_SCALE_LAYOUT").as_deref(),
            Ok("rowmajor")
        )
    })
}

/// Rewrite the quantizer's row-major `[M, K/128]` FP32 activation scales into
/// the `[K/128, M_pad]` cuBLASLt documents for a VEC128 B operand, zero-filling
/// the `M..M_pad` pad rows.
///
/// The index math is pinned on the CPU by
/// `spark_runtime::cublaslt::scale_layout` (SSOT, with the doc quotes); this is
/// only its launcher. The quantizer's own output is left in place — the
/// in-tree `fp8_gemm_t_blockscaled` still reads it directly.
pub fn fp8_act_scale_to_kmajor(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    kernel: spark_runtime::gpu::KernelHandle,
    a_scale: spark_runtime::gpu::DevicePtr,
    a_scale_kmajor: spark_runtime::gpu::DevicePtr,
    m: u32,
    m_pad: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    let l = k / 128;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(m_pad, 256), l, 1])
        .block([256, 1, 1])
        .arg_ptr(a_scale)
        .arg_ptr(a_scale_kmajor)
        .arg_u32(m)
        .arg_u32(m_pad)
        .arg_u32(l)
        .launch(stream)
}

/// Route a projection through native-FP8 cuBLASLt block-scaled matmul: quantize
/// the activation to FP8 + per-[token,128-of-K] scales (the existing
/// `per_token_group_quant_fp8` kernel), adapt those scales to the layout
/// cuBLASLt documents, then feed the FP8 weight + its per-128×128 block scales
/// directly (zero dequant, zero extra weight memory). Both operands
/// 128-block-scaled (cuBLASLt requires it). ~1.8× the bf16 path (152 vs 85 TF).
///
/// `act_fp8_scratch`/`act_scale_scratch`/`act_scale_kmajor_scratch` must hold
/// the padded extents (the `buffers.fp8_act`/`fp8_act_scale` arena buffers,
/// sized for max_batch_tokens).
#[allow(clippy::too_many_arguments)]
pub fn cublas_fp8_proj(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    ptg_quant_k: spark_runtime::gpu::KernelHandle,
    scale_kmajor_k: spark_runtime::gpu::KernelHandle,
    act_bf16: spark_runtime::gpu::DevicePtr,
    act_fp8_scratch: spark_runtime::gpu::DevicePtr,
    act_scale_scratch: spark_runtime::gpu::DevicePtr,
    act_scale_kmajor_scratch: spark_runtime::gpu::DevicePtr,
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
        scale_kmajor_k,
        act_fp8_scratch,
        act_scale_scratch,
        act_scale_kmajor_scratch,
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
/// ⚠ SCALE LAYOUT. cuBLASLt reads the VEC128 B-scale tensor with the TOKEN
/// index contiguous (`[K/128, ceil16(M)]`), not the `[M, K/128]` the quantizer
/// writes — cuBLAS "Scaling factors layouts", and the reason this helper needs
/// `act_scale_kmajor` at all. Handing the quantizer's buffer over directly is
/// what the 2026-09-11 H100 run measured at rel_rms 7.7e-2 / ~33 000 BF16 ULP
/// against the in-tree kernel on identical FP8 bytes; `ATLAS_CUBLAS_SCALE_LAYOUT
/// =rowmajor` reproduces that reading deliberately.
///
/// ⚠ PADDED-M EXTENTS. cuBLASLt is handed `ceil16(M)`, so:
///
/// * `out` must hold `ceil16(M) * N` BF16 elements — the phantom rows are
///   WRITTEN (well-defined: their activation scales are zeroed below).
/// * `act_fp8` must hold `ceil16(M) * K` bytes, `act_scale`
///   `M * (K/128)` f32 and `act_scale_kmajor` `ceil16(M) * (K/128)` f32 — the
///   phantom rows are READ.
///
/// The arena sizes that headroom in; see the sizing notes in
/// `spark_runtime::buffers::sizes` (`fp8_act`, `ffn_act_a`, `ffn_act_scale`,
/// `ffn_act_scale_kmajor`, `expert_gate_out`, `moe_output`).
#[allow(clippy::too_many_arguments)]
pub fn cublas_fp8_proj_prequant(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    scale_kmajor_k: spark_runtime::gpu::KernelHandle,
    act_fp8: spark_runtime::gpu::DevicePtr,
    act_scale: spark_runtime::gpu::DevicePtr,
    act_scale_kmajor: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    // cuBLASLt requires the scale-tensor M extent to be a multiple of 4; pad to
    // 16 (TC-friendly) and zero the padding so the phantom output columns
    // (ignored by the caller) are well-defined.
    let m_pad = cublas_fp8_m_pad(m);
    let kg = (k / 128) as usize;
    if m_pad > m {
        // A zero scale kills the phantom rows' CONTRIBUTION, but the FP8 dot
        // product still runs over whatever bytes are there and `NaN * 0.0` is
        // `NaN`. Same reasoning (and same fix) as the row-wise sibling in
        // `dispatch_proj_rowwise.rs`.
        gpu.memset_async(
            act_fp8.offset(m as usize * k as usize),
            0,
            (m_pad - m) as usize * k as usize,
            stream,
        )?;
    }
    let b_scale = if cublas_scale_layout_kmajor() {
        if scale_kmajor_k.0 == 0 || act_scale_kmajor.0 == 0 {
            anyhow::bail!(
                "cuBLASLt block-scaled FP8 needs the fp8_act_scale_to_kmajor adapter \
                 (kernel={:#x}, scratch={:#x}) — see cublas_scale_layout_kmajor()",
                scale_kmajor_k.0,
                act_scale_kmajor.0
            );
        }
        // Writes every [K/128, m_pad] slot, pad rows included, so no separate
        // memset of the scale pad is needed.
        fp8_act_scale_to_kmajor(
            gpu,
            scale_kmajor_k,
            act_scale,
            act_scale_kmajor,
            m,
            m_pad,
            k,
            stream,
        )?;
        act_scale_kmajor
    } else {
        // Measurement control only (`ATLAS_CUBLAS_SCALE_LAYOUT=rowmajor`): the
        // pad rows are a contiguous tail in THIS layout, so zero them here.
        if m_pad > m {
            gpu.memset_async(
                act_scale.offset(m as usize * kg * 4),
                0,
                (m_pad - m) as usize * kg * 4,
                stream,
            )?;
        }
        act_scale
    };
    spark_runtime::cublaslt::fp8_gemm_act_weight_t_blkscaled(
        act_fp8.0,
        b_scale.0,
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

/// Dequantize a block-scaled OR per-row FP8 weight `[N,K]` → BF16 into a
/// CALLER-OWNED buffer of `n*k*2` bytes. Allocates nothing.
///
/// The kernel reads `scale[(n / block_n) * sk + (k / block_k)]`, so the SAME
/// kernel serves both layouts — the block geometry is what selects between
/// them, not a second kernel:
///
///   block-scaled   block_n = block_k = 128, sk = K/128
///   PER-ROW        block_n = 1, block_k = K, sk = 1
///                  -> offset = n * 1 + 0 = n, one multiplier per row
///
/// That per-row case is what a mixed-precision compressed-tensors checkpoint
/// ships, and dequantising it is lossless: every FP8 E4M3 value is exactly
/// representable in BF16, so this is the fold's no-double-quant path even
/// though the GEMM downstream is BF16.
///
/// SSOT for every FP8→BF16 weight expansion in this file. It takes a
/// destination rather than producing one because the #917 H100 receipt
/// (2026-09-11, `Qwen/Qwen3.8-27B-FP8`) was a `gpu.alloc` hidden in here:
/// `167772160` B per GDN layer with no `BufferSizes` entry, invisible to
/// `--gpu-memory-utilization`, which killed a 28-token prefill at layer 36
/// with `cuMemAlloc_v2 failed: status 2`. Who owns the bytes is now the
/// caller's decision, and the row-wise GDN arms answer it with the ledgered
/// `buffers.take_ssm_rowwise_w_bf16` slab.
pub fn dequant_fp8_bf16_into(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    fp8w: &crate::weight_map::Fp8Weight,
    dst: spark_runtime::gpu::DevicePtr,
    stream: u64,
) -> anyhow::Result<()> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    let (n, kk) = (fp8w.n, fp8w.k);
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
        .arg_ptr(dst)
        .arg_u32(n)
        .arg_u32(kk)
        .arg_u32(block_n)
        .arg_u32(block_k)
        .arg_u32(sk)
        .arg_u32(1) // scale_is_fp32
        .launch(stream)
}

/// BF16 bytes [`dequant_fp8_bf16_into`] writes for `fp8w`.
pub fn dequant_fp8_bf16_bytes(fp8w: &crate::weight_map::Fp8Weight) -> usize {
    fp8w.n as usize * fp8w.k as usize * 2
}

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

/// Route a projection `out[M,N] = act[M,K] @ weightᵀ` through cuBLASLt BF16 for
/// a weight that is already BF16 `[N,K]`. Two kinds of caller: models whose
/// attention/shared-expert weights ship unquantized (e.g. Laguna), and the
/// row-wise GDN prefill arms, which hand it the ledgered BF16 dequant
/// `qwen3_ssm/rowwise_bf16.rs` writes once per layer.
///
/// There is deliberately NO `cublas_bf16_proj` beside it any more — the
/// dequant-and-cache variant that used to own the FP8→BF16 expansion is the
/// #917 off-ledger allocation (see [`dequant_fp8_bf16_into`]). Splitting
/// "who owns the BF16 bytes" from "multiply them" is what keeps the ledger
/// honest: this function cannot allocate.
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
