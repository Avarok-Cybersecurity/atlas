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
        let force_w8a8 = matches!(std::env::var("ATLAS_FP8_W8A8").ok().as_deref(), Some("1"));
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
        } else if force_w8a8
            && let Some(ref fp8w) = self.qkvz_fp8w
            && self.per_token_group_quant_fp8_k.0 != 0
            && self.fp8_gemm_t_blockscaled_k.0 != 0
        {
            tracing::info!(
                "ssm prefill: QKVZ via W8A8+FP32-epilogue (vLLM-equivalent, M={k} K={h} N={qkvz_size})"
            );
            let m = k as usize;
            let k_dim = h;
            let a_fp8_bytes = m * k_dim;
            let a_scale_bytes = m * (k_dim / 128) * 4;
            let a_fp8_buf = ctx.gpu.alloc(a_fp8_bytes)?;
            let a_scale_buf = ctx.gpu.alloc(a_scale_bytes)?;
            // BISECT TEST 1: quant kernel only — call it then fall through
            // to w8a16_gemm (which we know works) for the actual GEMM.
            // If this still crashes, the bug is in per_token_group_quant_fp8.
            // If it works, the bug is in fp8_gemm_t_blockscaled.
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                normed,
                a_fp8_buf,
                a_scale_buf,
                k,
                k_dim as u32,
                stream,
            )?;
            // BISECT TEST 2: now call the GEMM with no-fold variant. If
            // this crashes, the bug is in the MMA loop itself.
            tracing::info!("ssm prefill: calling fp8_gemm_t_blockscaled (no-fold bisect)");
            ops::fp8_gemm_t_blockscaled(
                ctx.gpu,
                self.fp8_gemm_t_blockscaled_k,
                a_fp8_buf,
                a_scale_buf,
                fp8w.weight,
                fp8w.row_scale,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
            ctx.gpu.synchronize(stream)?;
            ctx.gpu.free(a_fp8_buf)?;
            ctx.gpu.free(a_scale_buf)?;
        } else if let Some(ref fp8w) = self.qkvz_fp8w
            && self.w8a16_gemm_pipelined_k.0 != 0
        {
            // Block-scaled W8A16 prefill: matches vLLM's per-128-block FP32
            // scale precision (vs the single-scale fp8_gemm_n128 below
            // which bakes ALL per-block scales into one global scale,
            // dropping per-block dynamic range). This is the SSM-side of
            // the W8A8+FP32-epilogue fix shipped for the attention layer.
            //
            // Block-scaled W8A16 QKVZ routed through the bit-identical
            // (cosine=1.0) ~4.6× faster tensor-core w8a16_gemm_pipelined kernel
            // where available (NVIDIA). gfx1151/HIP has no cp.async, so that
            // kernel is absent there â fall through to the cp.async-free
            // non-pipelined w8a16_gemm branch below.
            ops::w8a16_gemm_pipelined(
                ctx.gpu,
                self.w8a16_gemm_pipelined_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "ssm prefill: QKVZ w8a16_gemm_pipelined failed (M={k}, N={qkvz_size}): {e}"
                )
            })?;
        } else if let Some(ref fp8w) = self.qkvz_fp8w
            && self.w8a16_gemm_k.0 != 0
        {
            // cp.async-free fallback (gfx1151/HIP): non-pipelined block-scaled
            // W8A16 GEMM. Same per-128-block FP32-scale math as the pipelined
            // kernel, without the sm_80+ cp.async multistage prefetch.
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "ssm prefill: QKVZ w8a16_gemm (block-scaled) failed (M={k}, N={qkvz_size}): {e}"
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
