// SPDX-License-Identifier: AGPL-3.0-only

//! Standard Q/K/V projection branch of `prefill_attention_paged`.
//! 6-way quantization dispatch (transposed-FP8, FP8, FP8 col-scale,
//! NVFP4 transposed, NVFP4, BF16) shared across Q, K, V; extracted to
//! keep `paged.rs` under the 500-LoC budget.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

/// Identifies which projection (Q/K/V) — selects the correct weight bank
/// from `Qwen3AttentionLayer`.
pub(super) enum Proj {
    Q,
    K,
    V,
}

impl Qwen3AttentionLayer {
    /// Run the Q, K, and V GEMMs (in that order) for non-MLA prefill.
    /// Output destinations follow the existing buffer layout in `paged.rs`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_attention_paged_qkv(
        &self,
        normed: DevicePtr,
        n: u32,
        h: u32,
        nkv: u32,
        hd: u32,
        q_proj_dim: usize,
        kv_dim: usize,
        num_tokens: usize,
        bf16: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let qg_out = ctx.buffers.qkv_output();
        self.prefill_one_proj(
            Proj::Q,
            normed,
            qg_out,
            n,
            q_proj_dim as u32,
            h,
            ctx,
            stream,
        )?;
        // ATLAS_OP_DUMP hook: q_proj output (last token, BF16 → f32).
        // For gated Qwen3.6, q_proj_dim = 2*q_dim (Q+Gate interleaved).
        // We dump the FULL Q+Gate buffer; the HF reference will only
        // contain the deinterleaved Q so partial cosine on first half
        // is the comparable metric.
        super::super::op_dump::dump_bf16(
            ctx.gpu,
            qg_out,
            (num_tokens - 1) * q_proj_dim * bf16,
            q_proj_dim,
            self.attn_layer_idx,
            "q_proj_full",
            stream,
        )?;

        let k_contiguous = ctx.buffers.ssm_qkvz();
        self.prefill_one_proj(Proj::K, normed, k_contiguous, n, nkv * hd, h, ctx, stream)?;
        super::super::op_dump::dump_bf16(
            ctx.gpu,
            k_contiguous,
            (num_tokens - 1) * kv_dim * bf16,
            kv_dim,
            self.attn_layer_idx,
            "k_proj",
            stream,
        )?;

        let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);
        self.prefill_one_proj(Proj::V, normed, v_contiguous, n, nkv * hd, h, ctx, stream)?;
        super::super::op_dump::dump_bf16(
            ctx.gpu,
            v_contiguous,
            (num_tokens - 1) * kv_dim * bf16,
            kv_dim,
            self.attn_layer_idx,
            "v_proj",
            stream,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_one_proj(
        &self,
        proj: Proj,
        normed: DevicePtr,
        out: DevicePtr,
        n: u32,
        out_dim: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (fp8w_t, weight_opt, fp8, nvfp4_t, dense) = match proj {
            Proj::Q => (
                self.q_fp8w_t.as_ref(),
                self.q_weight.as_ref(),
                self.q_fp8,
                self.q_nvfp4_t.as_ref(),
                &self.attn.q_proj,
            ),
            Proj::K => (
                self.k_fp8w_t.as_ref(),
                self.k_weight.as_ref(),
                self.k_fp8,
                self.k_nvfp4_t.as_ref(),
                &self.attn.k_proj,
            ),
            Proj::V => (
                self.v_fp8w_t.as_ref(),
                self.v_weight.as_ref(),
                self.v_fp8,
                self.v_nvfp4_t.as_ref(),
                &self.attn.v_proj,
            ),
        };

        let force_w8a8 = matches!(std::env::var("ATLAS_FP8_W8A8").ok().as_deref(), Some("1"));
        // W8A8 + FP32 epilogue: requires NON-transposed FP8 weights with
        // block scales (matches the kernel signature). The attn layer stores
        // those via set_fp8_weights — accessible via weight_opt.as_fp8().
        if force_w8a8
            && let Some(fp8w) = weight_opt.and_then(|w| w.as_fp8())
            && self.per_token_group_quant_fp8_k.0 != 0
            && self.fp8_gemm_t_blockscaled_k.0 != 0
        {
            let m = n as usize;
            let k_dim = h as usize;
            let a_fp8_bytes = m * k_dim;
            let a_scale_bytes = m * (k_dim / 128) * 4;
            let a_fp8_buf = ctx.gpu.alloc(a_fp8_bytes)?;
            let a_scale_buf = ctx.gpu.alloc(a_scale_bytes)?;
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                normed,
                a_fp8_buf,
                a_scale_buf,
                n,
                h,
                stream,
            )?;
            ops::fp8_gemm_t_blockscaled(
                ctx.gpu,
                self.fp8_gemm_t_blockscaled_k,
                a_fp8_buf,
                a_scale_buf,
                fp8w.weight,
                fp8w.row_scale,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
            ctx.gpu.synchronize(stream)?;
            ctx.gpu.free(a_fp8_buf)?;
            ctx.gpu.free(a_scale_buf)?;
        } else if let Some(fp8t) = fp8w_t {
            ops::w8a16_gemm_t(
                ctx.gpu,
                self.w8a16_gemm_t_k,
                normed,
                fp8t.weight_t,
                fp8t.scale_t,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else if weight_opt.and_then(|w| w.as_fp8()).is_some() && self.w8a16_gemm_k.0 != 0 {
            let fp8w = weight_opt.and_then(|w| w.as_fp8()).unwrap();
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else if let Some(fp8p) = fp8 {
            if n > 128 {
                ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    normed,
                    fp8p,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            } else {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    normed,
                    fp8p,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            }
        } else if let Some(nvfp4_t) = nvfp4_t {
            if n > 128 {
                self.w4a16_gemm_m128_dispatch(
                    ctx.gpu, normed, nvfp4_t, out, n, out_dim, h, stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    nvfp4_t,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            }
        } else if let Some(nvfp4) = weight_opt.and_then(|w| w.as_nvfp4()) {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                nvfp4,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else {
            // BF16 dense fallback. For native-FP8 models with
            // ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1, `dense` (= attn.{q,k,v}_proj)
            // holds the FP8→BF16 dequanted weight; otherwise it is the
            // model's native dense weight.
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                dense,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        }
        Ok(())
    }
}
