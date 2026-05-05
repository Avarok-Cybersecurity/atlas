// SPDX-License-Identifier: AGPL-3.0-only

//! Section 10 of `prefill_attention_paged`: O-projection GEMM
//! `[N, nq*hd] → [N, h]`. 6-way quantization dispatch (FP8 transposed,
//! FP8, FP8 col-scale, NVFP4 transposed, BF16 dense, NVFP4 default).
//! Extracted from `paged.rs` to keep that file under 500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_attention_paged_oproj(
        &self,
        attn_out: DevicePtr,
        n: u32,
        h: u32,
        nq: u32,
        hd: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let o_out = ctx.buffers.norm_output();
        if let Some(ref fp8t) = self.o_fp8w_t {
            ops::w8a16_gemm_t(
                ctx.gpu,
                self.w8a16_gemm_t_k,
                attn_out,
                fp8t.weight_t,
                fp8t.scale_t,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if self.o_weight.as_ref().and_then(|w| w.as_fp8()).is_some()
            && self.w8a16_gemm_k.0 != 0
        {
            let fp8w = self.o_weight.as_ref().and_then(|w| w.as_fp8()).unwrap();
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                attn_out,
                fp8w.weight,
                fp8w.row_scale,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if let Some(fp8) = self.o_fp8 {
            if n > 128 {
                ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    attn_out,
                    fp8,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    attn_out,
                    fp8,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4_t) = self.o_nvfp4_t {
            if n > 128 {
                self.w4a16_gemm_m128_dispatch(
                    ctx.gpu,
                    attn_out,
                    nvfp4_t,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    attn_out,
                    nvfp4_t,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
        } else if let Some(o_bf16) = self.o_dense_bf16.as_ref() {
            // BF16 dense fallback (Gemma-4 dense per Nvidia ModelOpt's
            // ignore list — all self_attn projections must stay BF16).
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                attn_out,
                o_bf16,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                attn_out,
                &self.attn.o_proj,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        }
        Ok(o_out)
    }
}
