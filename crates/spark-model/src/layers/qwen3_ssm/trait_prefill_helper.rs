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
        // W4 pair (`w4_proj.rs`, port lane L22a): the only live out_proj
        // weight once installed, so it precedes every arm below.
        if self.w4_proj.is_some() {
            return self.w4_out_prefill(ctx, normed_out_buf, out_proj_buf, k as usize, stream);
        }
        // The in-tree W8A8 arm below: `AVAROK_FP8_W8A8=1` as before, or by
        // default on a layer the load-time BF16 -> FP8 GDN route built
        // (`out_proj_w8a8_prefill`), which `AVAROK_FP8_SINGLE_SCALE` still
        // turns off. Both need the 128-multiple grid the weight scales are on.
        let force_w8a8 = (matches!(std::env::var("AVAROK_FP8_W8A8").ok().as_deref(), Some("1"))
            || (self.out_proj_w8a8_prefill && ctx.dispatch.fp8_blockscaled_prefill))
            && h.is_multiple_of(128)
            && value_dim.is_multiple_of(128);
        // PER-ROW FP8 from the checkpoint (`AVAROK_FP8_ROWWISE=1`), dequantised
        // once to BF16 into the LEDGERED arena slab — see the matching arm in
        // `trait_prefill_proj.rs` for why BF16 and not the row-wise FP8 GEMM,
        // and `rowwise_bf16.rs` for why the bytes are the arena's. First
        // because it is the only arm that never re-quantises.
        if let Some(ref fp8w) = self.out_proj_fp8w_rowwise {
            // LEDGERED BF16 weight (`rowwise_bf16.rs`, #917): dequantised once
            // per layer into `BufferSizes::ssm_rowwise_w_bf16`, so this arm
            // allocates nothing. The predecessor cached a `gpu.alloc` by
            // weight pointer, outside the ledger the fitter reads.
            let w_bf16 = self.rowwise_out_proj_bf16(ctx, fp8w, stream)?;
            return ops::cublas_bf16_proj_dense(
                normed_out_buf,
                w_bf16,
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
        // W8A8 block-scaled cuBLASLt — 96 of the 112 `w8a16_gemm_pipelined`
        // launches in the round-9 H100 prefill trace (68.1 ms of its 100.6 ms
        // at 1193 tokens) were THIS projection, once per GDN layer per chunk,
        // and nothing in `AVAROK_CUBLAS_GEMM` could reach it. See
        // `prefill_out_w8a8.rs` for the full receipt and the clause list.
        // Ahead of the `AVAROK_FP8_W8A8` arm below because both compute the same
        // W8A8 arithmetic and this one is the faster implementation of it; the
        // env lever alone (without `ssm` in `AVAROK_CUBLAS_GEMM`) still picks
        // the in-tree kernel.
        } else if let Some(ref fp8w) = self.out_proj_fp8w
            && self.prefill_out_proj_w8a8_selected(ctx, k, h as u32, value_dim as u32, fp8w)
        {
            self.log_out_proj_prefill_route(ctx, true);
            self.prefill_out_proj_w8a8_cublas(
                ctx,
                normed_out_buf,
                fp8w,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else if force_w8a8
            && let Some(ref fp8w) = self.out_proj_fp8w
            && self.per_token_group_quant_fp8_k.available()
            && self.fp8_gemm_t_blockscaled_k.0 != 0
        {
            if ctx.stats.once("log:ssm_out_proj_w8a8_kernel_prefill") {
                tracing::info!(
                    "[avarok] SSM out_proj prefill: W8A8 block-scaled via kernel \
                     (per-token 1x128 act scales x 128x128 weight scales, FP32 epilogue; \
                     N={h} K={value_dim}). AVAROK_FP8_SINGLE_SCALE=1 restores W8A16."
                );
            }
            // `out_proj` maps `[M, value_dim]` to `[M, h]`: the activation is
            // quantized over K = value_dim and the GEMM is (M, N = h,
            // K = value_dim). This arm used to pass K = h and N = value_dim,
            // which is right only when the two are equal (they are not on any
            // Qwen3.5 / 3.6 dense GDN: 2560 vs 4096 on the 4B).
            let m = k as usize;
            let k_dim = value_dim;
            // Persistent arena scratch (no per-projection alloc/sync/free),
            // sized for `max_proj_k`, which includes value_dim.
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
                h as u32,
                value_dim as u32,
                stream,
            )?;
            Ok(())
        } else if let Some(ref fp8w) = self.out_proj_fp8w
            && self.w8a16_gemm_pipelined_k.0 != 0
        {
            // The 100.6 ms / 27.31% kernel of the round-9 H100 prefill trace,
            // 96 of whose 112 launches were this line. The counterpart log says
            // so out loud, so "no W8A8 line" is never read as "log lost".
            self.log_out_proj_prefill_route(ctx, false);
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
