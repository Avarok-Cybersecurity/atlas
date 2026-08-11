// SPDX-License-Identifier: AGPL-3.0-only

//! `NemotronMamba2Layer` prefill projection dispatch: the in_proj and out_proj
//! GEMM arm selection (native BF16 → native FP8 → W4A4 → pre-dequant FP8 →
//! transposed NVFP4 → plain W4A16). Split from `prefill.rs` (500-LoC cap);
//! the surrounding scan/conv/norm pipeline stays there.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::NemotronMamba2Layer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl NemotronMamba2Layer {
    /// in_proj GEMM: `[N, h] × [h, in_proj_size] → [N, in_proj_size]`.
    ///
    /// `fp8_a` / `w4a4` / `pd_fp8_ok` are computed once in `prefill_ssm` (they
    /// gate both projections identically); see the comments there.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_in_proj(
        &self,
        normed: DevicePtr,
        proj: DevicePtr,
        n: u32,
        h: usize,
        fp8_a: bool,
        w4a4: bool,
        pd_fp8_ok: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Native FP8 first: when the checkpoint's own block-scaled FP8 weights are
        // installed, `self.ssm.in_proj` is NULL (no NVFP4 copy was ever built), so
        // every arm below would fault. This also keeps prefill on the checkpoint's
        // per-128×128-block scales instead of the single global NVFP4 scale the
        // legacy load derived across all 73.9M elements.
        //
        // `native_fp8_prefill` (not merely the presence of the weights) selects
        // this arm: in the `decode` bisect mode the FP8 weights are installed
        // for `w8a16_gemv` while the NVFP4 copies below are still built, and
        // prefill must keep using those.
        // NATIVE BF16 first. When installed, the checkpoint's projections were
        // never quantized, so `ssm.in_proj` is a NULL `QuantizedWeight` and every
        // arm below would dereference it — this must take precedence.
        if let Some(ref w) = self.in_proj_bf16 {
            ops::dense_gemm_bf16_pipelined(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                normed,
                w,
                proj,
                n,
                self.in_proj_size as u32,
                h as u32,
                stream,
            )?;
        } else if let Some(fp8w) = self
            .in_proj_fp8
            .as_ref()
            .filter(|_| self.native_fp8_prefill)
        {
            if self.w8a16_gemm_pipelined_k.0 != 0 {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    normed,
                    fp8w.weight,
                    fp8w.row_scale,
                    proj,
                    n,
                    self.in_proj_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "ssm prefill: in_proj w8a16_gemm_pipelined failed (M={n}, N={}): {e}",
                        self.in_proj_size
                    )
                })?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    normed,
                    fp8w.weight,
                    fp8w.row_scale,
                    proj,
                    n,
                    self.in_proj_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if w4a4 {
            let a4 = ctx.buffers.fp8_act();
            let a4_sf = a4.offset((n as usize) * h / 2);
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                normed,
                a4,
                a4_sf,
                n,
                h as u32,
                stream,
            )?;
            ops::w4a4_gemm_mfast(
                ctx.gpu,
                self.w4a4_gemm_k,
                a4,
                a4_sf,
                &self.ssm.in_proj,
                proj,
                n,
                self.in_proj_size as u32,
                h as u32,
                stream,
            )?;
        } else if let Some(w_fp8) = self.in_proj_pd_fp8.filter(|_| pd_fp8_ok) {
            // Weights already FP8: no per-K-step dequant, no M-block redundancy.
            if fp8_a {
                let a8 = ctx.buffers.fp8_act();
                ops::bf16_to_fp8(
                    ctx.gpu,
                    self.bf16_to_fp8_k,
                    normed,
                    a8,
                    n * h as u32,
                    stream,
                )?;
                ops::fp8_fp8_gemm_m128_mfast(
                    ctx.gpu,
                    self.fp8_fp8_gemm_t_k,
                    a8,
                    w_fp8,
                    proj,
                    n,
                    self.in_proj_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::fp8_gemm_m128_mfast(
                    ctx.gpu,
                    self.fp8_gemm_t_k,
                    normed,
                    w_fp8,
                    proj,
                    n,
                    self.in_proj_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if let Some(ref wt) = self.in_proj_t {
            // Fast path: transposed weights + FP8 MMA (N128, K32, cp.async pipeline)
            if n > 128 && self.w4a16_gemm_t_m128_k.0 != 0 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    normed,
                    wt,
                    proj,
                    n,
                    self.in_proj_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    wt,
                    proj,
                    n,
                    self.in_proj_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                &self.ssm.in_proj,
                proj,
                n,
                self.in_proj_size as u32,
                h as u32,
                stream,
            )?;
        }
        Ok(())
    }

    /// out_proj GEMM: `[N, d_inner] × [d_inner, h] → [N, h]`.
    /// Mirrors the in_proj dispatch above arm for arm.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_out_proj(
        &self,
        gated_out: DevicePtr,
        out: DevicePtr,
        n: u32,
        h: usize,
        fp8_a: bool,
        w4a4: bool,
        pd_fp8_ok: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(ref w) = self.out_proj_bf16 {
            ops::dense_gemm_bf16_pipelined(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                gated_out,
                w,
                out,
                n,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        } else if let Some(fp8w) = self
            .out_proj_fp8
            .as_ref()
            .filter(|_| self.native_fp8_prefill)
        {
            if self.w8a16_gemm_pipelined_k.0 != 0 {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    gated_out,
                    fp8w.weight,
                    fp8w.row_scale,
                    out,
                    n,
                    h as u32,
                    self.d_inner as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "ssm prefill: out_proj w8a16_gemm_pipelined failed (M={n}, N={h}): {e}"
                    )
                })?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    gated_out,
                    fp8w.weight,
                    fp8w.row_scale,
                    out,
                    n,
                    h as u32,
                    self.d_inner as u32,
                    stream,
                )?;
            }
        } else if w4a4 {
            let a4 = ctx.buffers.fp8_act();
            let a4_sf = a4.offset((n as usize) * self.d_inner / 2);
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                gated_out,
                a4,
                a4_sf,
                n,
                self.d_inner as u32,
                stream,
            )?;
            ops::w4a4_gemm_mfast(
                ctx.gpu,
                self.w4a4_gemm_k,
                a4,
                a4_sf,
                &self.ssm.out_proj,
                out,
                n,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        } else if let Some(w_fp8) = self.out_proj_pd_fp8.filter(|_| pd_fp8_ok) {
            if fp8_a {
                let a8 = ctx.buffers.fp8_act();
                ops::bf16_to_fp8(
                    ctx.gpu,
                    self.bf16_to_fp8_k,
                    gated_out,
                    a8,
                    n * self.d_inner as u32,
                    stream,
                )?;
                ops::fp8_fp8_gemm_m128_mfast(
                    ctx.gpu,
                    self.fp8_fp8_gemm_t_k,
                    a8,
                    w_fp8,
                    out,
                    n,
                    h as u32,
                    self.d_inner as u32,
                    stream,
                )?;
            } else {
                ops::fp8_gemm_m128_mfast(
                    ctx.gpu,
                    self.fp8_gemm_t_k,
                    gated_out,
                    w_fp8,
                    out,
                    n,
                    h as u32,
                    self.d_inner as u32,
                    stream,
                )?;
            }
        } else if let Some(ref wt) = self.out_proj_t {
            if n > 128 && self.w4a16_gemm_t_m128_k.0 != 0 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    gated_out,
                    wt,
                    out,
                    n,
                    h as u32,
                    self.d_inner as u32,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    gated_out,
                    wt,
                    out,
                    n,
                    h as u32,
                    self.d_inner as u32,
                    stream,
                )?;
            }
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                gated_out,
                &self.ssm.out_proj,
                out,
                n,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
