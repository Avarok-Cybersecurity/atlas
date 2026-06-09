// SPDX-License-Identifier: AGPL-3.0-only

//! Standard Q/K/V projection branch of `prefill_attention_with_cache_skip`.
//! Differs from `paged.rs` by preferring `w8a16_gemm` (W8A16 with E4M3
//! LUT + block scales) over the transposed FP8 path, and using the
//! pre-converted `normed_fp8` activations for the FP8×FP8 GEMM. Extracted
//! to keep `cache_skip.rs` under 500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

pub(super) enum SkipProj {
    Q,
    K,
    V,
}

impl Qwen3AttentionLayer {
    /// Run all three projections (Q, then K, then V) for the cache-skip
    /// non-MLA path. Output buffer addresses match the inline body.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_attention_cache_skip_qkv(
        &self,
        normed: DevicePtr,
        normed_fp8: DevicePtr,
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
        self.cache_skip_one_proj(
            SkipProj::Q,
            normed,
            normed_fp8,
            qg_out,
            n,
            q_proj_dim as u32,
            h,
            ctx,
            stream,
        )?;
        let k_contiguous = ctx.buffers.ssm_qkvz();
        self.cache_skip_one_proj(
            SkipProj::K,
            normed,
            normed_fp8,
            k_contiguous,
            n,
            nkv * hd,
            h,
            ctx,
            stream,
        )?;
        let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);
        self.cache_skip_one_proj(
            SkipProj::V,
            normed,
            normed_fp8,
            v_contiguous,
            n,
            nkv * hd,
            h,
            ctx,
            stream,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn cache_skip_one_proj(
        &self,
        proj: SkipProj,
        normed: DevicePtr,
        normed_fp8: DevicePtr,
        out: DevicePtr,
        n: u32,
        out_dim: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Per-projection weight bundle. Q has no `q_fp8w_t` shortcut in this
        // path (matches the inline body pre-refactor).
        let (fp8w_t, weight_opt, fp8, nvfp4_t, dense, label) = match proj {
            SkipProj::Q => (
                None,
                self.q_weight.as_ref(),
                self.q_fp8,
                self.q_nvfp4_t.as_ref(),
                &self.attn.q_proj,
                "q_proj",
            ),
            SkipProj::K => (
                self.k_fp8w_t.as_ref(),
                self.k_weight.as_ref(),
                self.k_fp8,
                self.k_nvfp4_t.as_ref(),
                &self.attn.k_proj,
                "k_proj",
            ),
            SkipProj::V => (
                self.v_fp8w_t.as_ref(),
                self.v_weight.as_ref(),
                self.v_fp8,
                self.v_nvfp4_t.as_ref(),
                &self.attn.v_proj,
                "v_proj",
            ),
        };

        if let Some(fp8t) = fp8w_t {
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
        } else if weight_opt.and_then(|w| w.as_fp8()).is_some() {
            anyhow::bail!("w8a16_gemm kernel not loaded — cannot prefill with FP8 weights");
        } else if let Some(fp8p) = fp8 {
            if n > 128 {
                ops::fp8_fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_fp8_gemm_t_m128_k,
                    normed_fp8,
                    fp8p,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            } else {
                ops::fp8_fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_fp8_gemm_k,
                    normed_fp8,
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
            )
            .map_err(|e| {
                anyhow::anyhow!("{label} w4a16_gemm failed: m={n} n={out_dim} k={h}: {e}")
            })?;
        } else {
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
