// SPDX-License-Identifier: AGPL-3.0-only

//! W8A8 block-scaled cuBLASLt arm for the SSM/GDN **`out_proj` PREFILL**
//! projection — the `ATLAS_CUBLAS_GEMM=ssm` route for the second half of the
//! GDN block, alongside `prefill_w8a8.rs`'s `in_proj_qkvz` arm.
//!
//! WHY THIS EXISTS (#917 / #928). nsys on 1xH100, 2026-09-11 round 9,
//! `Qwen/Qwen3.8-27B-FP8`, best recipe `ATLAS_CUBLAS_GEMM=ffn,ssm,attn`
//! (`nsys-r9-prefill`, kernel-sum CSV). A 1193-token prefill is **368.263 ms**
//! of GPU-busy union, and `w8a16_gemm_pipelined` is **100.582 ms = 27.31%** of
//! it across **112 launches**. Resolved by launch geometry — the kernel's grid
//! is `(ceil(N/32), ceil(M/128))`, so GrdX names N and GrdY names the chunk —
//! those 112 are NOT the dense FFN (which the same trace shows on `nvjet…`
//! cuBLASLt lines for gate/up/down at every M):
//!
//! | launches | grid | shape | site | µs |
//! |---|---|---|---|---|
//! | 48 | 160x10 | N=5120 K=6144 M=1168 | **SSM `out_proj`, chunk 0** | 54 119.7 |
//! | 16 | 384x10 | N=12288 K=5120 M=1168 | attn `q_proj`, chunk 0 (`cache_skip_qkv.rs`) | 32 466.8 |
//! | 48 | 160x1 | N=5120 K=6144 M=25 | **SSM `out_proj`, chunk 1** | 13 995.4 |
//!
//! i.e. **96 of the 112 launches and 68.1 ms of the 100.6 ms are this one
//! projection**, once per GDN layer per prefill chunk, on all 48 GDN layers.
//! At 4593 tokens it is 163.5 ms of the kernel's 285.5 ms.
//!
//! WHY IT WAS EXCLUDED. `prefill_out_proj_dispatch` had no cuBLASLt arm at all,
//! and its only W8A8 arm was gated on a bare `ATLAS_FP8_W8A8=1` env read —
//! **not** on `ctx.dispatch.fp8_blockscaled_prefill` (the gate the QKVZ,
//! attention and dense-FFN prefills use) and **not** on any `ATLAS_CUBLAS_GEMM`
//! family. So the serve recipe that armed `ssm` moved `in_proj_qkvz` to
//! cuBLASLt and left `out_proj` — the same layer's other GEMM — on the W8A16
//! tensor-core kernel, where BF16 activations against E4M3 weights turn the
//! FP8 bytes into pure memory savings and none of the FLOPs.
//!
//! WHAT THIS ADDS. The same quantize-once + k-major-scale pattern as
//! `prefill_w8a8.rs` / `dense_ffn_w8a8_prefill.rs` / `paged_oproj.rs`: the
//! `per_token_group_quant_fp8` activation already on this path, re-laid-out by
//! `fp8_act_scale_to_kmajor`, handed to `ops::cublas_fp8_proj_prequant`. No
//! dequant, no cache, no device memory — every buffer is arena scratch that
//! already exists. **The W8A16 kernels remain the fallback** and are what an
//! un-armed serve still runs, byte for byte.
//!
//! ACCURACY. W8A8 quantizes the activation to E4M3 per 128-wide K group, so it
//! is lossier than W8A16 by construction — the same deliberate trade the other
//! three prefill families already made. `native_fp8_prefill_proj_w8a8_microtest`
//! gates it at cosine >= 0.999 / rel_rms <= 3e-2 against the W8A16 reference at
//! the real shapes. `ATLAS_SSM_OUT_W8A16_ONLY` (presence) restores W8A16.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

/// `ATLAS_SSM_OUT_W8A16_ONLY` kill switch: PRESENCE (any value, including
/// empty) keeps the SSM `out_proj` prefill on the W8A16 kernels. Presence
/// rather than `=1` for the same reason as `ATLAS_FFN_W8A16_ONLY` — this is an
/// escape hatch reached for mid-incident, and `=0` meaning "on" is a trap.
///
/// `OnceLock`-cached: the selector runs once per GDN layer per prefill chunk
/// (48x per chunk on a 27B) and `var_os` walks the whole environment block.
pub(super) fn ssm_out_w8a16_only() -> bool {
    static ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONLY.get_or_init(|| std::env::var_os("ATLAS_SSM_OUT_W8A16_ONLY").is_some())
}

/// Everything the `out_proj` cuBLASLt arm needs, as a pure function so the CPU
/// tests can pin each clause without a GPU.
///
/// Clauses, each load-bearing:
///
/// * `!w8a16_only` — the kill switch above.
/// * `cublas_ssm` — `ATLAS_CUBLAS_GEMM` must name the `ssm` family. Scoped,
///   deliberately: arming the dense FFN must not arm this.
/// * `fp8_blockscaled_prefill` — the `ATLAS_FP8_SINGLE_SCALE` kill switch that
///   already governs every other W8A8 prefill arm.
/// * `m > 4` — the same floor `w8a8_prefill_selected` uses; at M<=4 the GEMV
///   tier streams each weight once and beats any MMA tile.
/// * `Fp8BlockScaled` — cuBLASLt is told the weight scales are a `[N/128,
///   K/128]` BLK128x128 grid; a per-row `row_scale` has a different shape and
///   would be read as garbage.
/// * `n % 128 == 0`, `k % 128 == 0` — that grid, and the activation
///   quantizer's one-scale-per-128-of-K contract.
/// * `blk128x128_stride_ok(k)` (`k % 512 == 0`) — cuBLAS requires the weight
///   scale column stride `K/128` to be a multiple of 4.
/// * output room for `ceil16(M) * N` BF16 — the padded rows are WRITTEN.
///   `moe_output` is already sized `ceil16(m) * hidden` (`sizes.rs`), so this
///   is a check that a future re-sizing cannot silently invalidate.
/// * activation scratch room for `ceil16(M) * K` FP8 and its `ceil16(M) *
///   K/128` FP32 scales — `cublas_fp8_proj_prequant` ZEROES the pad rows of
///   both before the matmul, so it writes past M in each.
/// * the quantizer kernel, and the VEC128 scale-layout adapter, kernel AND
///   scratch AND its capacity. cuBLASLt reads the activation scales
///   token-contiguous; handing over the quantizer's `[M, K/128]` order is fast
///   and WRONG (H100 2026-09-11: rel_rms 7.7e-2 on identical FP8 bytes).
///   Falling back to W8A16 is the only safe answer when it is missing.
#[allow(clippy::too_many_arguments)]
pub(super) fn out_proj_cublas_selected(
    cublas_ssm: bool,
    fp8_blockscaled_prefill: bool,
    w8a16_only: bool,
    scale_format: WeightQuantFormat,
    m: u32,
    n: u32,
    k: u32,
    out_capacity_bytes: usize,
    act_capacity_bytes: usize,
    act_scale_capacity_bytes: usize,
    quant_k: KernelHandle,
    scale_kmajor_k: KernelHandle,
    scale_kmajor_buf: DevicePtr,
    scale_kmajor_capacity_bytes: usize,
) -> bool {
    let m_pad = ops::cublas_fp8_m_pad(m) as usize;
    let kg = k as usize / 128;
    let kmajor_ready = scale_kmajor_k.0 != 0
        && scale_kmajor_buf.0 != 0
        && scale_kmajor_capacity_bytes >= m_pad * kg * 4;
    !w8a16_only
        && cublas_ssm
        && fp8_blockscaled_prefill
        && m > 4
        && scale_format == WeightQuantFormat::Fp8BlockScaled
        && n.is_multiple_of(128)
        && k.is_multiple_of(128)
        && spark_runtime::cublaslt::scale_layout::blk128x128_stride_ok(k as usize)
        && m_pad * (n as usize) * 2 <= out_capacity_bytes
        && m_pad * (k as usize) <= act_capacity_bytes
        && m_pad * kg * 4 <= act_scale_capacity_bytes
        && quant_k.0 != 0
        && (kmajor_ready || !ops::cublas_scale_layout_kmajor())
}

impl Qwen3SsmLayer {
    /// Whether ONE SSM `out_proj` prefill takes the W8A8 cuBLASLt arm.
    ///
    /// `n` is the hidden size (the projection's output width) and `k` the GDN
    /// `value_dim` it contracts over — the 5120 x 6144 of the round-9 receipt.
    pub(super) fn prefill_out_proj_w8a8_selected(
        &self,
        ctx: &ForwardContext,
        m: u32,
        n: u32,
        k: u32,
        fp8w: &Fp8Weight,
    ) -> bool {
        out_proj_cublas_selected(
            ctx.dispatch.cublas.ssm,
            ctx.dispatch.fp8_blockscaled_prefill,
            ssm_out_w8a16_only(),
            fp8w.scale_format,
            m,
            n,
            k,
            ctx.buffers.moe_output_bytes(),
            ctx.buffers.fp8_act_bytes(),
            ctx.buffers.fp8_act_scale_bytes(),
            self.per_token_group_quant_fp8_k,
            self.fp8_act_scale_kmajor_k,
            ctx.buffers.fp8_act_scale_kmajor(),
            ctx.buffers.fp8_act_scale_kmajor_bytes(),
        )
    }

    /// `out[m, n] = quant(normed_out)[m, k] @ weight[n, k]ᵀ` through cuBLASLt,
    /// with the per-token 1x128 activation scales and the checkpoint's 128x128
    /// weight scales folded in an FP32 epilogue.
    ///
    /// Quantizes ONCE into the arena's `fp8_act` / `fp8_act_scale` scratch —
    /// the same buffers the QKVZ arm used earlier in this layer, safely reused
    /// because the whole chain is same-stream ordered — then hands both to
    /// `ops::cublas_fp8_proj_prequant`, which does the k-major scale transpose
    /// and the matmul. Allocates nothing.
    ///
    /// Callers MUST have checked [`Qwen3SsmLayer::prefill_out_proj_w8a8_selected`]
    /// first; this method does not re-check and does not fall back.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_out_proj_w8a8_cublas(
        &self,
        ctx: &ForwardContext,
        normed_out: DevicePtr,
        fp8w: &Fp8Weight,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let a_fp8 = ctx.buffers.fp8_act();
        let a_scale = ctx.buffers.fp8_act_scale();
        // Padded extents, because `cublas_fp8_proj_prequant` zeroes and then
        // READS rows `m..ceil16(m)` of both. The selector already gated on
        // these; the debug asserts are the loud version for a dev build.
        let m_pad = ops::cublas_fp8_m_pad(m) as usize;
        debug_assert!(m_pad * k as usize <= ctx.buffers.fp8_act_bytes());
        debug_assert!(m_pad * (k as usize / 128) * 4 <= ctx.buffers.fp8_act_scale_bytes());
        ops::per_token_group_quant_fp8(
            ctx.gpu,
            self.per_token_group_quant_fp8_k,
            normed_out,
            a_fp8,
            a_scale,
            m,
            k,
            stream,
        )?;
        ops::cublas_fp8_proj_prequant(
            ctx.gpu,
            self.fp8_act_scale_kmajor_k,
            a_fp8,
            a_scale,
            ctx.buffers.fp8_act_scale_kmajor(),
            fp8w,
            out,
            m,
            n,
            k,
            stream,
        )
    }

    /// Log-once route line for the SSM `out_proj` prefill. Emitted from BOTH
    /// arms so the absence of the W8A8 line is never ambiguous between "not
    /// armed" and "log lost" — the failure mode that let round 9's 100.6 ms be
    /// attributed to the dense FFN, which the same trace shows on `nvjet…`.
    pub(super) fn log_out_proj_prefill_route(&self, ctx: &ForwardContext, cublas: bool) {
        if ctx.stats.once("log:ssm_out_proj_prefill") {
            if cublas {
                tracing::info!(
                    "[atlas] SSM out_proj prefill: W8A8 block-scaled via cuBLASLt \
                     (per-token 1x128 act scales x 128x128 weight scales, FP32 epilogue). \
                     ATLAS_CUBLAS_GEMM=ssm selected it; ATLAS_SSM_OUT_W8A16_ONLY restores W8A16. \
                     This arm allocates nothing."
                );
            } else {
                tracing::info!(
                    "[atlas] SSM out_proj prefill: W8A16 (BF16 act x FP8 weight). \
                     W8A8 cuBLASLt not selected — add `ssm` to ATLAS_CUBLAS_GEMM; see #917/#928."
                );
            }
        }
    }
}

#[cfg(test)]
#[path = "prefill_out_w8a8_tests.rs"]
mod tests;
