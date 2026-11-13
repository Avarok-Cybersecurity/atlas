// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the decode forward of `Lfm3ConvLayer` (one token for each
//! of N sequences: the single-sequence decode and `decode_multi_seq`), the
//! projection dispatch shared with prefill, and the sublayer tail (post norms,
//! residuals, FFN) both paths end with.
//!
//! Buffers (all from the arena, rows `N`, BF16): `norm_output` holds the
//! normed rows, `ssm_qkvz` the in_proj output BCx `[N, 3 * hidden]` (sized
//! from `ssm_qkvz_size()`, which the LFM3 config sets to exactly `3 * hidden`),
//! `ssm_deinterleaved` the gated conv output y `[N, hidden]`, and `qkv_output`
//! the out_proj output (both carry cuBLASLt's `ceil16(M)` row headroom).
//!
//! Projection widths and kernels:
//!
//! | rows | FP8 decode copies installed | BF16 only |
//! |---|---|---|
//! | 1 | `fp8_gemv_ada` / `w8a16_gemv` | `dense_gemv_bf16` |
//! | 2..=16 | `w8a16_gemv_batch16` | `dense_gemv_bf16_batchm` (row bits = M 1) |
//! | > 16 | BF16 cuBLASLt | BF16 cuBLASLt |
//!
//! No step syncs the host, so the decode is capturable into a CUDA graph.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::Lfm3ConvLayer;
use super::launch::{self, Strides};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8Weight};

/// Which conv projection a dispatch is for (selects the FP8 copy).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Proj {
    In,
    Out,
}

impl Lfm3ConvLayer {
    fn proj_weights(&self, which: Proj) -> (&DenseWeight, Option<&Fp8Weight>, u32) {
        let h = self.cfg.hidden as u32;
        match which {
            Proj::In => (&self.in_proj, self.fp8.as_ref().map(|f| &f.in_proj), 3 * h),
            Proj::Out => (&self.out_proj, self.fp8.as_ref().map(|f| &f.out_proj), h),
        }
    }

    /// `out[rows, n] = x[rows, hidden] @ W^T` for one conv projection.
    pub(crate) fn project(
        &self,
        ctx: &ForwardContext,
        which: Proj,
        x: DevicePtr,
        rows: usize,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (w, fp8, n) = self.proj_weights(which);
        let k = self.cfg.hidden as u32;
        let m = rows as u32;
        if let Some(f) = fp8 {
            if m == 1 && self.k.fp8_gemv.0 != 0 {
                return ops::w8a16_gemv(
                    ctx.gpu,
                    self.k.fp8_gemv,
                    x,
                    f.weight,
                    f.row_scale,
                    out,
                    n,
                    k,
                    stream,
                );
            }
            if (2..=16).contains(&m) && self.k.fp8_gemv_batch16.0 != 0 {
                return ops::w8a16_gemv_batch16(
                    ctx.gpu,
                    self.k.fp8_gemv_batch16,
                    x,
                    f.weight,
                    f.row_scale,
                    out,
                    m,
                    n,
                    k,
                    stream,
                );
            }
        }
        if m == 1 {
            return ops::dense_gemv(ctx.gpu, self.k.gemv, x, w, out, n, k, stream);
        }
        if m <= ops::DENSE_GEMV_BATCHM_MAX_M && self.k.gemv_batchm.0 != 0 {
            return ops::dense_gemv_batchm(
                ctx.gpu,
                self.k.gemv_batchm,
                x,
                w,
                out,
                m,
                n,
                k,
                n,
                stream,
            );
        }
        ops::cublas_bf16_proj_dense(x, w.weight, out, m, n, k, stream)
    }

    /// Operator norm and in_proj over `rows` rows of `hidden`; returns BCx.
    pub(crate) fn head(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.k.rms_norm_residual,
            hidden,
            &self.operator_norm,
            normed,
            residual,
            rows as u32,
            self.cfg.hidden as u32,
            self.cfg.eps,
            stream,
        )?;
        let bcx = ctx.buffers.ssm_qkvz();
        self.project(ctx, Proj::In, normed, rows, bcx, stream)?;
        Ok(bcx)
    }

    /// out_proj of y, then the sublayer tail:
    /// `h += post_operator_norm(o)`, `h += post_ffn_norm(ffn(ffn_norm(h)))`.
    pub(crate) fn tail(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        y: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (h, n, eps) = (self.cfg.hidden as u32, rows as u32, self.cfg.eps);
        let o = ctx.buffers.qkv_output();
        self.project(ctx, Proj::Out, y, rows, o, stream)?;
        ops::rms_norm(
            ctx.gpu,
            self.k.rms_norm,
            o,
            &self.post_operator_norm,
            o,
            n,
            h,
            eps,
            stream,
        )?;
        let normed2 = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.k.residual_add_rms_norm,
            hidden,
            o,
            &self.ffn_norm,
            normed2,
            residual,
            n,
            h,
            eps,
            stream,
        )?;
        let ffn_out = if rows == 1 {
            self.ffn.forward(normed2, ctx, stream)?
        } else {
            self.ffn.forward_prefill(normed2, rows, ctx, stream)?;
            ctx.buffers.moe_output()
        };
        ops::rms_norm(
            ctx.gpu,
            self.k.rms_norm,
            ffn_out,
            &self.post_ffn_norm,
            ffn_out,
            n,
            h,
            eps,
            stream,
        )?;
        ops::residual_add(ctx.gpu, self.k.residual_add, hidden, ffn_out, n * h, stream)
    }

    /// One decode token for each of `states.len()` sequences: row `r` of
    /// `hidden` / `residual` belongs to the window `states[r]`.
    pub(crate) fn decode_rows(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        states: &[DevicePtr],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rows = states.len();
        if rows == 0 {
            return Ok(());
        }
        let bcx = self.head(hidden, residual, rows, ctx, stream)?;
        let y = ctx.buffers.ssm_deinterleaved();
        launch::decode(
            ctx.gpu,
            &self.k.conv,
            bcx,
            self.conv.weight,
            y,
            Strides::dense(self.cfg.hidden as u32),
            states,
            stream,
        )?;
        self.tail(hidden, residual, y, rows, ctx, stream)
    }
}
