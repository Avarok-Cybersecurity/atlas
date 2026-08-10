// SPDX-License-Identifier: AGPL-3.0-only

//! K + V projection branch of `attention_forward` (decode path). Picks
//! one of: MLA-skip (K/V already produced), FP8 native dual GEMV, fused
//! NVFP4 dual `w4a16_gemv_dual`, or per-projection NVFP4/dense fallback.
//! Extracted from `attention_forward.rs` to keep that file under 500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_forward_kv(
        &self,
        normed: DevicePtr,
        k_out: DevicePtr,
        v_out: DevicePtr,
        nkv: u32,
        hd: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.mla.is_some() {
            // MLA branch already wrote K and V into k_out/v_out
            return Ok(());
        }

        if let (Some(k_q2), Some(v_q2)) = (
            self.k_weight.as_ref().and_then(|w| w.as_packed_q2()),
            self.v_weight.as_ref().and_then(|w| w.as_packed_q2()),
        ) {
            // Keep-packed Q2_0 (Tier-1c): per-projection 2-bit GEMV.
            ops::q2_0_gemv_vec(ctx.gpu, self.q2_0_gemv_k, normed, k_q2, k_out, stream)?;
            ops::q2_0_gemv_vec(ctx.gpu, self.q2_0_gemv_k, normed, v_q2, v_out, stream)?;
            return Ok(());
        }

        if let (Some(k_fp8), Some(v_fp8)) = (
            self.k_weight.as_ref().and_then(|w| w.as_fp8()),
            self.v_weight.as_ref().and_then(|w| w.as_fp8()),
        ) {
            // FP8 native: individual w8a16_gemv for K and V
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed,
                k_fp8.weight,
                k_fp8.row_scale,
                k_out,
                nkv * hd,
                h,
                stream,
            )?;
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed,
                v_fp8.weight,
                v_fp8.row_scale,
                v_out,
                nkv * hd,
                h,
                stream,
            )?;
            return Ok(());
        }

        // Fuse K+V projections into a single dual GEMV when both are NVFP4
        match (
            self.k_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.v_weight.as_ref().and_then(|w| w.as_nvfp4()),
        ) {
            (Some(k_fp4), Some(v_fp4)) => {
                ops::w4a16_gemv_dual(
                    ctx.gpu,
                    self.w4a16_gemv_dual_k,
                    normed,
                    k_fp4,
                    k_out,
                    v_fp4,
                    v_out,
                    nkv * hd,
                    h,
                    stream,
                )?;
            }
            _ => {
                if let Some(nvfp4) = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        normed,
                        nvfp4,
                        k_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed,
                        &self.attn.k_proj,
                        k_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                }
                if let Some(nvfp4) = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        normed,
                        nvfp4,
                        v_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed,
                        &self.attn.v_proj,
                        v_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }
}
