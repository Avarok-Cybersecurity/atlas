// SPDX-License-Identifier: AGPL-3.0-only

//! QKVZ projection GEMM dispatch for `Qwen3SsmLayer::prefill_inner`.
//!
//! Hoisted from `trait_prefill.rs` to keep that file under the 500 LoC
//! cap. [`Qwen3SsmLayer::prefill_qkvz_proj`] mirrors the original step
//! 2+3 block 1:1 — same FP8 / NVFP4 / BF16 dispatch, same deinterleave,
//! same kernel launches and buffer wiring.

use super::*;

impl Qwen3SsmLayer {
    /// QKVZ projection GEMM (+ deinterleave when QKVZ is interleaved).
    ///
    /// Writes the sequential `[Q|K|V|Z]` projection into the
    /// `ssm_deinterleaved` buffer. `force_bf16` (= `ATLAS_GDN_BF16_WEIGHTS`)
    /// bypasses both the FP8 and NVFP4 weight-quant paths.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_qkvz_proj(
        &self,
        normed: DevicePtr,
        deinterleaved: DevicePtr,
        k: u32,
        qkvz_size: usize,
        h: usize,
        nk: usize,
        kd: usize,
        vpg: usize,
        vd: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let proj_dst = if self.sequential_qkvz {
            deinterleaved
        } else {
            ctx.buffers.ssm_qkvz()
        };
        // Env override: ATLAS_GDN_BF16_WEIGHTS=1 forces the BF16 dense
        // GEMM path for QKVZ — bypassing both FP8 and NVFP4 weight-quant
        // paths. Tests whether weight-quantization noise on qkvz (esp.
        // the W_z slice that feeds gnorm's silu gate) is the dominant
        // source of long-context layer-1+ drift.
        let force_bf16 = matches!(
            std::env::var("ATLAS_GDN_BF16_WEIGHTS").ok().as_deref(),
            Some("1")
        );
        if force_bf16 {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &self.ssm.in_proj_qkvz,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "ssm prefill: QKVZ BF16 dense GEMM failed (M={k}, N={qkvz_size}): {e}"
                )
            })?;
        } else if let Some(fp8) = self.qkvz_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                normed,
                fp8,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("ssm prefill: QKVZ FP8 GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else if let Some(ref nvfp4_t) = self.qkvz_nvfp4_t {
            if k > 128 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "ssm prefill: QKVZ m128 GEMM failed (M={k}, N={qkvz_size}): {e}"
                    )
                })?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("ssm prefill: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
                })?;
            }
        } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                nvfp4,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("ssm prefill: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &self.ssm.in_proj_qkvz,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        }
        if !self.sequential_qkvz {
            ops::deinterleave_qkvz(
                ctx.gpu,
                self.deinterleave_k,
                proj_dst,
                deinterleaved,
                k,
                nk as u32,
                kd as u32,
                vpg as u32,
                vd as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
