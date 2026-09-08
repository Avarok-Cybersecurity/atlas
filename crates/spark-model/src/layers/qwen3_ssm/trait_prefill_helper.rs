// SPDX-License-Identifier: AGPL-3.0-only

//! Output-projection GEMM dispatch for `Qwen3SsmLayer::prefill_inner`.
//!
//! Hoisted from `trait_prefill.rs` to keep that file under the 500 LoC cap.
//! The single helper `prefill_out_proj_dispatch` mirrors the original
//! Section 10 block 1:1: routes through dense / FP8 (with `n128_m128` fast
//! path for k>128) / NVFP4-transposed / NVFP4 paths based on which weight
//! variant is loaded.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3SsmLayer {
    /// GDN HeadParallel tensor-parallel all-reduce of the row-parallel
    /// `out_proj` output.
    ///
    /// Each TP rank ran the GDN scan over its LOCAL value-head slice and
    /// projected with the row-parallel `out_proj` (columns = local value_dim),
    /// so its `[num_tokens, h]` BF16 buffer holds a PARTIAL sum over the full
    /// hidden dim. Summing across ranks reconstructs the complete SSM output,
    /// exactly mirroring attention's post-`o_proj` reduce
    /// (`qwen3_attention/trait_impl/decode_inner.rs`). Must run BEFORE the
    /// residual add / post-norm that consumes `out_proj_buf`.
    ///
    /// No-op when `tp_world_size == 1` or no communicator is present (single
    /// GPU, or a path that already holds the complete output).
    ///
    /// ALSO applies the `out_proj` LoRA delta, and does so AFTER the reduce.
    /// The two belong together: before the reduce `out_proj_buf` holds only
    /// this rank's partial row-parallel product, and a delta added there would
    /// be summed once per rank and scaled by the TP width.
    ///
    /// Folding the delta into this helper rather than into each caller is
    /// deliberate. Every SSM dispatch path — single-token decode, prefill,
    /// prefill phase-3, batched decode, multi-seq batched — finishes its
    /// output projection by calling exactly this function, so applying here
    /// makes it structurally impossible to wire the adapter into some paths
    /// and forget others. An adapter that applies in prefill but not decode is
    /// a model that contradicts itself mid-sequence.
    pub(super) fn ssm_tp_all_reduce(
        &self,
        out_proj_buf: DevicePtr,
        normed_out: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            // BF16 [num_tokens, hidden_size] — same byte count attention
            // reduces after o_proj.
            let bytes = num_tokens * ctx.config.hidden_size * 2;
            comm.all_reduce_async(out_proj_buf.0, bytes, stream)?;
        }
        self.apply_lora_out_proj(ctx, normed_out, out_proj_buf, num_tokens as u32, stream)
    }

    pub(super) fn prefill_out_proj_dispatch(
        &self,
        ctx: &ForwardContext,
        normed_out_buf: DevicePtr,
        out_proj_buf: DevicePtr,
        k: u32,
        h: usize,
        value_dim: usize,
        stream: u64,
    ) -> Result<()> {
        // Native EXL3 (ATLAS_EXL3_NATIVE_DENSE=1): the packed trellis is the
        // ONLY live out_proj weight on this layer (every FP8/NVFP4/BF16 slot
        // below is None — enforced at install), so it sits first without
        // shadowing any precision lever. Row-batched cooperative GEMM into
        // the contiguous [k, h] block.
        if let Some(ref g) = self.exl3_gdn {
            return self.exl3_out_proj(g, ctx, normed_out_buf, out_proj_buf, k as usize, stream);
        }
        let force_w8a8 = matches!(std::env::var("ATLAS_FP8_W8A8").ok().as_deref(), Some("1"));
        // PER-ROW FP8 from the checkpoint (`ATLAS_FP8_ROWWISE=1`), dequantised
        // once to BF16 — see the matching arm in `trait_prefill_proj.rs` for
        // why BF16 and not the row-wise FP8 GEMM. First because it is the only
        // arm that never re-quantises.
        if let Some(ref fp8w) = self.out_proj_fp8w_rowwise {
            return ops::cublas_bf16_proj(
                ctx.gpu,
                ctx.derived,
                normed_out_buf,
                fp8w,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            );
        }
        if ctx.dispatch.cutlass_nvfp4_ssm_out
            && let Some(ref nvfp4_t) = self.out_proj_nvfp4_t
        {
            ops::log_cutlass_nvfp4_route(ctx.gpu, "ssm_out_nvfp4", k, h as u32, value_dim as u32);
            ops::cutlass_nvfp4_proj(
                ctx,
                normed_out_buf,
                nvfp4_t,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else if ctx.dispatch.cutlass_nvfp4_ssm_out
            && let Some(ref fp8w) = self.out_proj_fp8w
        {
            ops::log_cutlass_nvfp4_route(ctx.gpu, "ssm_out_fp8pack", k, h as u32, value_dim as u32);
            ops::cutlass_nvfp4_proj_from_fp8(
                ctx,
                normed_out_buf,
                fp8w,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else if let Some(ref dense_out) = self.out_proj_dense {
            // SSM out_proj is kept BF16 dense for accuracy (decode uses FP8
            // block-scaled, prefill stays BF16). Always routed through the
            // tensor-core dense_gemm_bf16_pipelined kernel (~40× vs the old
            // scalar dense_gemm, identical BF16 math, cosine=1.0).
            ops::dense_gemm_bf16_pipelined(
                ctx.gpu,
                self.dense_gemm_pipelined_k,
                normed_out_buf,
                dense_out,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else if force_w8a8
            && let Some(ref fp8w) = self.out_proj_fp8w
            && self.per_token_group_quant_fp8_k.0 != 0
            && self.fp8_gemm_t_blockscaled_k.0 != 0
        {
            tracing::debug!(
                "ssm prefill: out_proj via W8A8+FP32-epilogue (M={k} K={h} N={value_dim})"
            );
            let m = k as usize;
            let k_dim = h;
            // Persistent arena scratch (no per-projection alloc/sync/free).
            let a_fp8_buf = ctx.buffers.fp8_act();
            let a_scale_buf = ctx.buffers.fp8_act_scale();
            debug_assert!(m * k_dim <= ctx.buffers.fp8_act_bytes());
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                normed_out_buf,
                a_fp8_buf,
                a_scale_buf,
                k,
                k_dim as u32,
                stream,
            )?;
            ops::fp8_gemm_t_blockscaled(
                ctx.gpu,
                self.fp8_gemm_t_blockscaled_k,
                a_fp8_buf,
                a_scale_buf,
                fp8w.weight,
                fp8w.row_scale,
                out_proj_buf,
                k,
                value_dim as u32,
                h as u32,
                stream,
            )?;
            Ok(())
        } else if let Some(ref fp8w) = self.out_proj_fp8w
            && self.w8a16_gemm_pipelined_k.0 != 0
        {
            ops::w8a16_gemm_pipelined(
                ctx.gpu,
                self.w8a16_gemm_pipelined_k,
                normed_out_buf,
                fp8w.weight,
                fp8w.row_scale,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else if let Some(ref fp8w) = self.out_proj_fp8w
            && self.w8a16_gemm_k.0 != 0
        {
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                normed_out_buf,
                fp8w.weight,
                fp8w.row_scale,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else if let Some(fp8) = self.out_proj_fp8 {
            if k > 128 {
                ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    normed_out_buf,
                    fp8,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )
            } else {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    normed_out_buf,
                    fp8,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )
            }
        } else if let Some(ref nvfp4_t) = self.out_proj_nvfp4_t {
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t_k,
                normed_out_buf,
                nvfp4_t,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed_out_buf,
                &self.ssm.out_proj,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        }
        .map_err(|e| anyhow::anyhow!("ssm prefill: out_proj GEMM failed: {e}"))
    }
}
