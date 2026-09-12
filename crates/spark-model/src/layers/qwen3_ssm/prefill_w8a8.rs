// SPDX-License-Identifier: AGPL-3.0-only

//! W8A8 block-scaled GEMM selection for the SSM/GDN fused `in_proj_qkvz`
//! prefill projection: the cuBLASLt arm (`ATLAS_CUBLAS_GEMM=ssm`) and the
//! in-tree `fp8_gemm_t_blockscaled` kernel it falls back to.
//!
//! WHY this file exists — H100, 2026-09-11, `Qwen/Qwen3.8-27B-FP8` native FP8,
//! tip `5f78270dc`. `ATLAS_CUBLAS_GEMM=1` used to route this projection to
//! `ops::cublas_bf16_proj`, which DEQUANTIZES the FP8 weight to BF16 and caches
//! the copy: `[10240,5120] + [6144,5120]` fused, x 2 B = `167772160` bytes per
//! layer, ~10.3 GiB across 48 SSM layers, allocated lazily on the first prefill
//! and owned by nothing the memory planner can see. The preflight reserve had
//! 6.2 GiB free, so ONE 28-token prefill consumed 6120 MiB and died at layer 36
//! with `cuMemAlloc_v2 failed: status 2, requested 167772160 bytes`.
//!
//! The replacement keeps the projection in FP8 end to end and spends NO device
//! memory at all: the same `per_token_group_quant_fp8` activation the in-tree
//! kernel already quantizes into the arena's `fp8_act` / `fp8_act_scale`
//! scratch, re-laid-out into `fp8_act_scale_kmajor` by the shared
//! `fp8_act_scale_to_kmajor` adapter, then handed to
//! `ops::cublas_fp8_proj_prequant` — the same helper, the same k-major VEC128
//! scale layout, and therefore the same fix, as the dense-FFN arm in
//! `dense_ffn_w8a8_prefill.rs`.
//!
//! WHAT IS LOST: the old arm ran W16A16 (BF16 activation x dequantized BF16
//! weight), which is strictly more accurate than W8A8. That accuracy was never
//! free — it was paid for in 10.3 GiB of off-ledger weight copies — and the
//! precision-preserving path for a checkpoint that needs it is
//! `ATLAS_FP8_ROWWISE=1`, which shadows this arm entirely (see
//! `trait_prefill_proj.rs`). Turning `ssm` off in `ATLAS_CUBLAS_GEMM` restores
//! the in-tree W8A8 kernel, and `ATLAS_FP8_SINGLE_SCALE=1` the W8A16 one.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

/// Everything the cuBLASLt arm needs beyond "the W8A8 arm was selected", as a
/// pure function so the CPU tests can pin each clause without a GPU.
///
/// Clauses, each load-bearing:
///
/// * `cublas_ssm` — `ATLAS_CUBLAS_GEMM` must name the `ssm` family. The whole
///   point of the scoped lever: arming the dense FFN must not arm this.
/// * `Fp8BlockScaled` — cuBLASLt is told the weight scales are a BLK128x128
///   grid; a per-row or single-scale `row_scale` has a different shape and
///   would be read as garbage.
/// * `n % 128 == 0` — that grid is `[N/128, K/128]`.
/// * `blk128x128_stride_ok(k)` (i.e. `k % 512 == 0`) — cuBLAS requires the
///   weight-scale column stride `K/128` to be a multiple of 4.
/// * output room for `ceil16(M) * N` BF16 — the padded rows are WRITTEN.
/// * the VEC128 scale-layout adapter, kernel AND scratch AND its capacity.
///   cuBLASLt reads the activation scales token-contiguous; handing over the
///   quantizer's `[M, K/128]` order is fast and WRONG (H100 2026-09-11:
///   rel_rms 7.7e-2 / ~33 000 BF16 ULP against the in-tree kernel on identical
///   FP8 bytes). Falling back is the only safe answer when it is missing.
#[allow(clippy::too_many_arguments)]
pub(super) fn qkvz_cublas_selected(
    cublas_ssm: bool,
    scale_format: WeightQuantFormat,
    m_pad: u32,
    n: u32,
    k: u32,
    out_capacity_bytes: usize,
    scale_kmajor_k: KernelHandle,
    scale_kmajor_buf: DevicePtr,
    scale_kmajor_capacity_bytes: usize,
) -> bool {
    let kmajor_ready = scale_kmajor_k.0 != 0
        && scale_kmajor_buf.0 != 0
        && scale_kmajor_capacity_bytes >= (m_pad as usize) * (k as usize / 128) * 4;
    cublas_ssm
        && scale_format == WeightQuantFormat::Fp8BlockScaled
        && n.is_multiple_of(128)
        && k.is_multiple_of(128)
        && spark_runtime::cublaslt::scale_layout::blk128x128_stride_ok(k as usize)
        && (m_pad as usize) * (n as usize) * 2 <= out_capacity_bytes
        && (kmajor_ready || !ops::cublas_scale_layout_kmajor())
}

impl Qwen3SsmLayer {
    /// `out[m, n] = a_fp8[m, k] @ weight[n, k]ᵀ` with both block-scale sets
    /// folded in an FP32 epilogue — cuBLASLt when the `ssm` family is armed and
    /// every clause of [`qkvz_cublas_selected`] holds, else the in-tree kernel.
    ///
    /// `out_capacity_bytes` is the allocated size of `out`'s ARENA buffer
    /// (`ssm_deinterleaved` on a sequential model, `ssm_qkvz` otherwise).
    /// Neither arm allocates: the activation scratch, its scales and the
    /// k-major transpose are all arena buffers sized in
    /// `spark_runtime::buffers::sizes`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qkvz_w8a8_gemm(
        &self,
        ctx: &ForwardContext,
        a_fp8: DevicePtr,
        a_scale: DevicePtr,
        fp8w: &Fp8Weight,
        out: DevicePtr,
        out_capacity_bytes: usize,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let m_pad = ops::cublas_fp8_m_pad(m);
        let cublas = qkvz_cublas_selected(
            ctx.dispatch.cublas.ssm,
            fp8w.scale_format,
            m_pad,
            n,
            k,
            out_capacity_bytes,
            self.fp8_act_scale_kmajor_k,
            ctx.buffers.fp8_act_scale_kmajor(),
            ctx.buffers.fp8_act_scale_kmajor_bytes(),
        );
        if ctx.stats.once("log:ssm_qkvz_w8a8_prefill") {
            let how = if cublas { "cuBLASLt" } else { "kernel" };
            tracing::info!(
                "[atlas] SSM QKVZ prefill: W8A8 block-scaled via {how} \
                 (per-token 1x128 act scales x 128x128 weight scales, FP32 epilogue). \
                 ATLAS_CUBLAS_GEMM=ssm selects cuBLASLt; neither arm allocates."
            );
        }
        if cublas {
            return ops::cublas_fp8_proj_prequant(
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
            );
        }
        ops::fp8_gemm_t_blockscaled(
            ctx.gpu,
            self.fp8_gemm_t_blockscaled_k,
            a_fp8,
            a_scale,
            fp8w.weight,
            fp8w.row_scale,
            out,
            m,
            n,
            k,
            stream,
        )
    }
}

#[cfg(test)]
#[path = "prefill_w8a8_tests.rs"]
mod tests;
