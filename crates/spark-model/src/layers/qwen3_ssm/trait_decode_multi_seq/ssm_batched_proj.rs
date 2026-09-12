// SPDX-License-Identifier: AGPL-3.0-only

//! The two BIG projections of the batched-decode SSM mixer — `in_proj_qkvz`
//! and `out_proj` — and the tier resolution they share.
//!
//! Split out of `ssm_batched.rs` at the 500-LoC cap when the W8A8 cuBLASLt arm
//! (#927) landed. The orchestrator keeps the phase ORDER (norm, qkvz,
//! recurrent, out_proj, residual); this file owns only "which GEMM serves a
//! projection at n rows", which is now a four-way choice per projection:
//! W8A8 cuBLASLt (5..16 rows, `decode_w8a8_proj.rs`), the block-scaled
//! W8A16 GEMV tiers, the pipelined W8A16 GEMM, or the NVFP4/BF16 siblings.

use super::super::decode_w8a8_proj::SsmDecodeProj;
use super::super::*;
use super::ssm_batched::ssm_tc_proj_min_n;

/// Which GEMV/GEMM instantiation this step's row count selects, resolved ONCE
/// and used by both projections. `qkvz` and `out_proj` have the same row count
/// and the same weight format, so resolving twice would be two chances to
/// disagree.
pub(super) struct BatchedProjTier {
    /// `w8a16_gemm_pipelined` (cp.async) is present — bit-identical to the base
    /// `w8a16_gemm` and ~4.6x faster.
    pub(super) w8a16_pipe: bool,
    /// The contiguous block-scaled GEMV wrapper paired with its handle.
    pub(super) gemv_batch: ops::ContiguousBatchGemv,
    pub(super) gemv_batch_k: KernelHandle,
    /// Whether that GEMV tier is eligible at this row count.
    pub(super) use_batch4: bool,
    /// The NVFP4 sibling's handle for this row count.
    pub(super) fp4_gemv_batch_k: KernelHandle,
}

impl Qwen3SsmLayer {
    /// Resolve the GEMV/GEMM tier for `n` decode rows.
    pub(super) fn batched_proj_tier(&self, n: usize) -> BatchedProjTier {
        // FP8 (w8a16) when the decode overlay is installed, else BF16 dense.
        // Prefer the pipelined (cp.async) w8a16 kernel — bit-identical, ~4.6×
        // faster than the base w8a16_gemm, which nsys showed as 44.6% of the
        // C>1 decode step. `.0 == 0` → fall back to the base kernel.
        let w8a16_pipe = self.w8a16_gemm_pipelined_k.0 != 0;
        // Weight-streaming block-scaled GEMV for batched decode: avoids the
        // pipelined kernel's M->128 MMA pad. batch4 (M<=4) common path, batch16
        // (M<=16) for C=8/16; bit-identical per row to w8a16_gemv, disabled by
        // ATLAS_SSM_GEMV_BATCH4=0. Wrapper pairs with handle: batch4 caps at 4.
        let (gemv_batch, gemv_batch_k): (ops::ContiguousBatchGemv, KernelHandle) = if n <= 4 {
            (ops::w8a16_gemv_batch4, self.w8a16_gemv_batch4_k)
        } else {
            (ops::w8a16_gemv_batch16, self.w8a16_gemv_batch16_k)
        };
        let use_batch4 = gemv_batch_k.0 != 0
            && n <= 16
            && crate::layers::ops::ModelLevers::get().ssm_gemv_batch4;
        // FP4 sibling: the narrow w4a16_gemv batch{4..8} family (M<=8), else
        // batch16 (M<=16). Single NVFP4 weight pass for the QKVZ + out_proj
        // GEMVs (amortizes the weight read). The narrow tiers size acc/smem —
        // and, because the row loop is unrolled, the CODE — to the real row
        // count instead of batch16's 16; 0-handle → batch16 as before.
        let narrow = self.w4a16_batchm.kernel(n as u32);
        let fp4_gemv_batch_k = if narrow.0 != 0 {
            narrow
        } else {
            self.w4a16_gemv_batch16_k
        };
        BatchedProjTier {
            w8a16_pipe,
            gemv_batch,
            gemv_batch_k,
            use_batch4,
            fp4_gemv_batch_k,
        }
    }

    /// Batched QKVZ projection: ONE `[N,h]` -> `[N,qkvz]` GEMM (weights read
    /// once for all `n` rows).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ms_batched_qkvz(
        &self,
        ctx: &ForwardContext,
        tier: &BatchedProjTier,
        n: usize,
        normed_base: DevicePtr,
        deinterleaved: DevicePtr,
        qkvz_size: usize,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        let BatchedProjTier {
            w8a16_pipe,
            gemv_batch,
            gemv_batch_k,
            use_batch4,
            fp4_gemv_batch_k,
        } = *tier;
        if let Some(ref fp8) = self.qkvz_fp8w {
            // W8A8 cuBLASLt at 5..16 rows (#927): round 7 put THIS projection
            // at 25.9% of the n=16 step, 357 GB/s. Declining falls through to
            // the unchanged GEMV tiers below. SSOT: `decode_w8a8_proj.rs`.
            if self.try_ssm_decode_w8a8(
                ctx,
                SsmDecodeProj::Qkvz,
                normed_base,
                fp8,
                deinterleaved,
                ctx.buffers.ssm_deinterleaved_bytes(),
                n,
                qkvz_size as u32,
                h as u32,
                stream,
            )? {
                // Routed through cuBLASLt W8A8 above.
            } else if use_batch4 {
                gemv_batch(
                    ctx.gpu,
                    gemv_batch_k,
                    normed_base,
                    fp8.weight,
                    fp8.row_scale,
                    deinterleaved,
                    n as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else if w8a16_pipe {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    normed_base,
                    fp8.weight,
                    fp8.row_scale,
                    deinterleaved,
                    n as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    normed_base,
                    fp8.weight,
                    fp8.row_scale,
                    deinterleaved,
                    n as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
            match (ssm_tc_proj_min_n(), self.qkvz_nvfp4_t.as_ref()) {
                (Some(min_n), Some(nvfp4_t)) if n >= min_n => {
                    // Tile GEMM on the transposed twin — the same call the SSM
                    // prefill path makes on this same weight. `ms_proj_gemm`
                    // picks the 128-row M-tile at wide batches so the weight
                    // is streamed once instead of ceil(n/64) times.
                    self.ms_proj_gemm(
                        ctx.gpu,
                        normed_base,
                        nvfp4_t,
                        deinterleaved,
                        n as u32,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
                // FP4 batched QKVZ: ONE NVFP4 weight pass for all n seqs
                // (sequential layout writes the deinterleaved buffer directly).
                _ => {
                    // w4a16_gemv_batch16 is a MAX_M=16 template: at M>16 it
                    // silently computes rows 0..15 and never writes rows 16..
                    // — garbage, not a crash. The eligibility gate makes this
                    // arm unreachable at n>16 today; fail fast if that drifts.
                    anyhow::ensure!(
                        n <= 16,
                        "SSM batchm QKVZ GEMV caps at M=16 (n={n}); tile-GEMM twins required"
                    );
                    ops::w4a16_gemv_batchm(
                        ctx.gpu,
                        fp4_gemv_batch_k,
                        normed_base,
                        nvfp4,
                        deinterleaved,
                        n as u32,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?
                }
            }
        } else {
            // BF16-kept GDN build: scalar `dense_gemm` costs ~1.03 ms/layer
            // at n=2 (measured) — cuBLASLt tensor-cores it (381 us).
            ops::cublas_bf16_proj_dense(
                normed_base,
                self.ssm.in_proj_qkvz.weight,
                deinterleaved,
                n as u32,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        }
        Ok(())
    }

    /// Batched out_proj: ONE `[N,value_dim]` -> `[N,h]` GEMM (weights read
    /// once for all `n` rows).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ms_batched_out_proj(
        &self,
        ctx: &ForwardContext,
        tier: &BatchedProjTier,
        n: usize,
        normed_out_base: DevicePtr,
        ssm_out_base: DevicePtr,
        h: usize,
        value_dim: usize,
        stream: u64,
    ) -> Result<()> {
        let BatchedProjTier {
            w8a16_pipe,
            gemv_batch,
            gemv_batch_k,
            use_batch4,
            fp4_gemv_batch_k,
        } = *tier;
        if let Some(ref fp8) = self.out_proj_fp8w {
            // W8A8 cuBLASLt at 5..16 rows (#927): `out_proj` is ~5.11 ms of
            // the n=16 step inside the GrdX=1280 `w8a16_gemv_batch16` group.
            if self.try_ssm_decode_w8a8(
                ctx,
                SsmDecodeProj::OutProj,
                normed_out_base,
                fp8,
                ssm_out_base,
                ctx.buffers.moe_output_bytes(),
                n,
                h as u32,
                value_dim as u32,
                stream,
            )? {
                // Routed through cuBLASLt W8A8 above.
            } else if use_batch4 {
                gemv_batch(
                    ctx.gpu,
                    gemv_batch_k,
                    normed_out_base,
                    fp8.weight,
                    fp8.row_scale,
                    ssm_out_base,
                    n as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else if w8a16_pipe {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    normed_out_base,
                    fp8.weight,
                    fp8.row_scale,
                    ssm_out_base,
                    n as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    normed_out_base,
                    fp8.weight,
                    fp8.row_scale,
                    ssm_out_base,
                    n as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        } else if let Some(ref out_proj_dense) = self.out_proj_dense {
            // Same cuBLASLt swap as the QKVZ arm (513 -> 194 us at n=2).
            ops::cublas_bf16_proj_dense(
                normed_out_base,
                out_proj_dense.weight,
                ssm_out_base,
                n as u32,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if self.qkvz_nvfp4.is_some() {
            match (ssm_tc_proj_min_n(), self.out_proj_nvfp4_t.as_ref()) {
                (Some(min_n), Some(nvfp4_t)) if n >= min_n => {
                    // Tile GEMM on the transposed twin — mirrors the SSM
                    // prefill out_proj call on this same weight. `ms_proj_gemm`
                    // picks the 128-row M-tile at wide batches so the weight
                    // is streamed once instead of ceil(n/64) times.
                    self.ms_proj_gemm(
                        ctx.gpu,
                        normed_out_base,
                        nvfp4_t,
                        ssm_out_base,
                        n as u32,
                        h as u32,
                        value_dim as u32,
                        stream,
                    )?;
                }
                // FP4 batched out_proj: ONE NVFP4 weight pass for all n seqs.
                // (qkvz_nvfp4.is_some() ⇒ the NVFP4 SSM build, where
                // ssm.out_proj is the NVFP4 weight the per-seq path uses.)
                _ => {
                    // Same MAX_M=16 template as the QKVZ arm — silent row
                    // truncation above 16. Unreachable at n>16 today; fail
                    // fast if the eligibility gate drifts.
                    anyhow::ensure!(
                        n <= 16,
                        "SSM batchm out_proj GEMV caps at M=16 (n={n}); tile-GEMM twins required"
                    );
                    ops::w4a16_gemv_batchm(
                        ctx.gpu,
                        fp4_gemv_batch_k,
                        normed_out_base,
                        &self.ssm.out_proj,
                        ssm_out_base,
                        n as u32,
                        h as u32,
                        value_dim as u32,
                        stream,
                    )?
                }
            }
        }
        Ok(())
    }
}
