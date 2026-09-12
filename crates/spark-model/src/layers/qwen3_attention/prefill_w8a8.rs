// SPDX-License-Identifier: AGPL-3.0-only

//! The cuBLASLt half of the attention **prefill** W8A8 block-scaled arm — the
//! `ATLAS_CUBLAS_GEMM=attn` route that replaced an off-ledger BF16 dequant.
//!
//! WHY THIS EXISTS (#917 round 3 / #927). `ctx.dispatch.cublas.attn` used to
//! route the FP8 attention projections to `ops::cublas_bf16_proj`, which
//! DEQUANTIZES the FP8 weight to BF16 and caches the copy — 2 bytes per weight
//! element, allocated lazily on the first prefill and owned by nothing the
//! memory planner can see. That is the same leak the SSM QKVZ arm hit on H100
//! on 2026-09-11, where 48 layers of it came to ~10.3 GiB and one 28-token
//! prefill died at layer 36 with `cuMemAlloc_v2 ... status 2`.
//!
//! It matters again now because the serve recipe for the 5..16-row decode
//! projections is `ATLAS_CUBLAS_GEMM=ffn,ssm,attn` — arming `attn` for decode
//! must not re-arm a prefill arm that allocates a BF16 twin of every
//! attention weight. So the BF16 arm is GONE, and this is what stands in its
//! place: the same W8A8 block-scaled arithmetic the prefill path already
//! computes, handed to cuBLASLt instead of the in-tree kernel. No dequant, no
//! cache, no device memory at all — the activation scratch and the k-major
//! scale transpose are arena buffers that already exist.
//!
//! Structurally identical to `qwen3_ssm::prefill_w8a8` and
//! `layers::dense_ffn_w8a8_prefill`; the clauses are the same clauses, and the
//! fallback is always the in-tree `fp8_gemm_t_blockscaled`.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

impl Qwen3AttentionLayer {
    /// `out[m, n] = a_fp8[m, k] @ weight[n, k]ᵀ` with both block-scale sets
    /// folded in an FP32 epilogue — cuBLASLt when the `attn` family is armed
    /// and every clause holds, else the in-tree kernel.
    ///
    /// `out_capacity_bytes` is the allocated size of `out`'s ARENA buffer.
    /// cuBLASLt is handed `ceil16(M)` and WRITES those phantom rows (their
    /// activation scales are zeroed, so the values are defined, but the stores
    /// happen), and prefill `M` is a token count that need not be a multiple of
    /// 16 — so this bound is doing real work, not ceremony.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attn_prefill_w8a8_gemm(
        &self,
        ctx: &ForwardContext,
        a_fp8: DevicePtr,
        a_scale: DevicePtr,
        w: &Fp8Weight,
        out: DevicePtr,
        out_capacity_bytes: usize,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let m_pad = ops::cublas_fp8_m_pad(m);
        let kmajor_ready = self.fp8_act_scale_kmajor_k.0 != 0
            && ctx.buffers.fp8_act_scale_kmajor().0 != 0
            && ctx.buffers.fp8_act_scale_kmajor_bytes()
                >= (m_pad as usize) * (k as usize / 128) * 4;
        let cublas = ctx.dispatch.cublas.attn
            && w.scale_format == WeightQuantFormat::Fp8BlockScaled
            && n.is_multiple_of(128)
            && k.is_multiple_of(128)
            && spark_runtime::cublaslt::scale_layout::blk128x128_stride_ok(k as usize)
            && (m_pad as usize) * (n as usize) * 2 <= out_capacity_bytes
            && (kmajor_ready || !ops::cublas_scale_layout_kmajor());
        if ctx.stats.once("log:attn_w8a8_prefill") {
            let how = if cublas { "cuBLASLt" } else { "kernel" };
            tracing::info!(
                "[atlas] attention prefill: W8A8 block-scaled via {how} \
                 (per-token 1x128 act scales x 128x128 weight scales, FP32 epilogue). \
                 ATLAS_CUBLAS_GEMM=attn selects cuBLASLt; neither arm allocates."
            );
        }
        if cublas {
            return ops::cublas_fp8_proj_prequant(
                ctx.gpu,
                self.fp8_act_scale_kmajor_k,
                a_fp8,
                a_scale,
                ctx.buffers.fp8_act_scale_kmajor(),
                w,
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
            w.weight,
            w.row_scale,
            out,
            m,
            n,
            k,
            stream,
        )
    }
}
