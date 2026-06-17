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
        let force_w8a8 = matches!(std::env::var("ATLAS_FP8_W8A8").ok().as_deref(), Some("1"));
        if force_w8a8
            && let Some(fp8w) = self.o_weight.as_ref().and_then(|w| w.as_fp8())
            && self.per_token_group_quant_fp8_k.0 != 0
            && self.fp8_gemm_t_blockscaled_k.0 != 0
        {
            // o_proj GEMM: C[M, N] = A[M, K] @ B[N, K]
            //   A = attn_out  : [num_tokens, nq*hd]  (input,  K = nq*hd)
            //   B = o_weight  : [h, nq*hd]           (stored row-major, N = h)
            //   C = o_out     : [num_tokens, h]      (output, N = h)
            let m = n as usize;
            let k_dim = (nq * hd) as usize; // inner contract dim — input width of o_proj
            let n_out = h as usize; // output width of o_proj
            let a_fp8_bytes = m * k_dim;
            let a_scale_bytes = m * (k_dim / 128) * 4;
            let a_fp8_buf = ctx.gpu.alloc(a_fp8_bytes)?;
            let a_scale_buf = ctx.gpu.alloc(a_scale_bytes)?;
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                attn_out,
                a_fp8_buf,
                a_scale_buf,
                n,
                nq * hd,
                stream,
            )?;
            ops::fp8_gemm_t_blockscaled(
                ctx.gpu,
                self.fp8_gemm_t_blockscaled_k,
                a_fp8_buf,
                a_scale_buf,
                fp8w.weight,
                fp8w.row_scale,
                o_out,
                n,
                n_out as u32,
                k_dim as u32,
                stream,
            )?;
            ctx.gpu.synchronize(stream)?;
            ctx.gpu.free(a_fp8_buf)?;
            ctx.gpu.free(a_scale_buf)?;
        } else if let Some(ref fp8t) = self.o_fp8w_t
            && self.w8a16_gemm_t_pipelined_k.0 != 0
        {
            // o_proj via the byte-identical ~4.2x faster pipelined transposed
            // tensor-core kernel where available (NVIDIA). gfx1151/HIP has no
            // cp.async -> pipelined absent -> non-pipelined w8a16_gemm_t below.
            ops::w8a16_gemm_t_pipelined(
                ctx.gpu,
                self.w8a16_gemm_t_pipelined_k,
                attn_out,
                fp8t.weight_t,
                fp8t.scale_t,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if let Some(ref fp8t) = self.o_fp8w_t
            && self.w8a16_gemm_t_k.0 != 0
        {
            // cp.async-free fallback (gfx1151/HIP): non-pipelined transposed W8A16.
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
        // ATLAS_OP_DUMP hook: post-O-projection — this is the FULL attention
        // block output (Q*K^T*V * O_proj). Compares 1:1 against the HF
        // module hooked on `full_attention.o_proj.forward` for the last
        // token. Use `n` as token-count so we slice the last token.
        let bf16 = 2usize;
        let num_tokens = n as usize;
        if num_tokens > 0 {
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                o_out,
                (num_tokens - 1) * h as usize * bf16,
                h as usize,
                self.attn_layer_idx,
                "o_proj",
                stream,
            )?;
        }
        Ok(o_out)
    }
}
