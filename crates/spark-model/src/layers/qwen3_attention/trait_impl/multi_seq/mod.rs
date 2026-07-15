// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-sequence batched-decode body for [`super::super::Qwen3AttentionLayer`].
//!
//! Split into phase modules under the `_inner` delegation pattern:
//! - `ctx`  — `MultiSeqCtx` shared scalars + buffer pointers
//! - `qkv`  — phase 2: per-token Q/K/V projections (batch3/batch2/seq)
//! - `attn` — phases 3-6: RoPE → cache write → paged decode → O proj
//! - `ffn`  — phase 7: residual + post-norm + MoE/dense FFN
//!
//! The trait impl in `super::trait_impl` calls
//! [`Qwen3AttentionLayer::decode_multi_seq_inner`] which simply builds
//! the ctx, runs phase 1 inline (RMS norm), and dispatches the rest.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;

mod attn;
mod ctx;
mod ffn;
mod mla;
mod mla_gemv;
mod qkv;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let _ = states; // Attention layers use EmptyLayerState — no per-seq state.
        let bs = kv_cache.block_size() as u32;
        let c = ctx::MultiSeqCtx::new(self, ctx, hidden, residual, num_seqs, bs, stream);

        // DeepSeek-V4: Manifold-Constrained Hyper-Connections (mHC).
        if self.hc.is_some() {
            return self.decode_multi_seq_inner_hc(c, kv_cache, ctx, stream);
        }

        // ── Phase 1: RMS norm + residual for N tokens ──
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            c.hidden,
            &self.input_norm,
            c.normed,
            c.residual,
            c.n as u32,
            c.h as u32,
            c.eps,
            c.stream,
        )?;

        let meta = ctx
            .attn_metadata
            .expect("attention layer requires metadata");

        // ── Phases 2-6: attention ──
        // MLA models (Mistral-Small-4) take the dedicated absorbed-MLA
        // batched path (issue #84). The standard `ms_phase_qkv` reads
        // `attn.q_proj`, a NULL stub for MLA loaders — see `mla.rs`.
        let o_out = if let Some(ref _mla) = self.mla {
            self.ms_mla_decode(&c, kv_cache, meta)?
        } else {
            // ── Phase 2: QKV projections (batch3 / batch2 / sequential) ──
            self.ms_phase_qkv(&c)?;

            // ── Phase 3: RoPE per-sequence ──
            self.ms_phase_rope(&c, meta)?;

            // ── Phase 4: KV cache write ──
            self.ms_phase_cache_write(&c, kv_cache, meta)?;

            // ── Phase 5: paged decode attention (batched) ──
            let attn_out = self.ms_phase_paged_decode(&c, kv_cache, meta)?;

            // ── Phase 6: gate multiply + O projection ──
            self.ms_phase_o_proj(&c, attn_out)?
        };

        // TP all-reduce on o_out after o_proj (Megatron row-parallel
        // pattern). Mirrors decode_inner.rs and prefill_inner.rs. Without
        // this, multi-token decode (K=2 / K=3 / K=γ verify) under
        // tp_world_size>1 reads a partial attention output from each
        // rank, corrupting the FFN/MoE input and producing degenerate
        // logits — observed as `/`/`,` repetition spirals on
        // Qwen3.6 FP8 + TP=2 + MTP for HTML/code prompts.
        if c.fwd.config.tp_world_size > 1
            && let Some(comm) = c.fwd.comm
        {
            let bytes = c.n * c.h * c.bf16;
            comm.all_reduce_async(o_out.0, bytes, c.stream)?;
        }

        // ── Phase 7: residual + post-norm + MoE ──
        self.ms_phase_ffn(&c, o_out)?;

        Ok(())
    }

    /// HC-enabled batched multi-sequence decode.  Only the sequential
    /// per-token FFN branch is implemented (DeepSeek-V4 MLA always
    /// takes this path).
    fn decode_multi_seq_inner_hc(
        &self,
        c: ctx::MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = c.n;
        let hc = self.hc.as_ref().unwrap();
        let hc_mult = hc.hc_mult as u32;
        let is_first_layer = self.attn_layer_idx == 0;
        let is_last_layer = self.attn_layer_idx + 1 == ctx.config.num_hidden_layers;
        let hc_streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        let diag_this =
            std::env::var("ATLAS_DIAG_V4_ALL_LAYERS").is_ok_and(|v| v == "1" || v == "true");

        if is_first_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                c.hidden,
                hc_streams,
                n as u32,
                h as u32,
                hc_mult,
                stream,
            )?;
        }

        // ── Phase 1: collapse + norm for N tokens ──
        ops::hc_pre(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
            hc.attn.hc_fn,
            hc.attn.hc_scale,
            hc.attn.hc_base,
            c.hidden,
            post,
            comb,
            n as u32,
            h as u32,
            hc_mult,
            hc.sinkhorn_iters as u32,
            eps,
            hc.hc_eps,
            stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                c.hidden,
                n * h,
                stream,
                &format!("V4-msdecode L{} hc_pre-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                n * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} post-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                n * (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} comb-attn", self.attn_layer_idx),
            );
        }
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            c.hidden,
            &self.input_norm,
            c.normed,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;

        let meta = ctx
            .attn_metadata
            .expect("attention layer requires metadata");

        // ── Phases 2-6: attention ──
        let o_out = if let Some(ref _mla) = self.mla {
            self.ms_mla_decode(&c, kv_cache, meta)?
        } else {
            self.ms_phase_qkv(&c)?;
            self.ms_phase_rope(&c, meta)?;
            self.ms_phase_cache_write(&c, kv_cache, meta)?;
            let attn_out = self.ms_phase_paged_decode(&c, kv_cache, meta)?;
            self.ms_phase_o_proj(&c, attn_out)?
        };

        if c.fwd.config.tp_world_size > 1
            && let Some(comm) = c.fwd.comm
        {
            let bytes = c.n * c.h * c.bf16;
            comm.all_reduce_async(o_out.0, bytes, c.stream)?;
        }

        // Expand attention output back into multi-stream state.
        ops::hc_post(
            ctx.gpu,
            self.hc_post_k,
            o_out,
            hc_streams,
            post,
            comb,
            hc_streams,
            n as u32,
            h as u32,
            hc_mult,
            stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-msdecode L{} hc_post-attn", self.attn_layer_idx),
            );
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                n * (hc_mult as usize) * h,
                stream,
                &format!(
                    "V4-msdecode L{} hc_post-attn ALL_STREAMS",
                    self.attn_layer_idx
                ),
            );
        }

        // Standalone attention (no FFN)
        if self.ffn.is_none() {
            if is_last_layer && let Some(ref head) = hc.head {
                ops::hc_head(
                    ctx.gpu,
                    self.hc_head_k,
                    hc_streams,
                    head.hc_fn,
                    head.hc_scale,
                    head.hc_base,
                    c.hidden,
                    n as u32,
                    h as u32,
                    hc_mult,
                    eps,
                    hc.hc_eps,
                    stream,
                )?;
                if diag_this {
                    super::diag_norm(
                        ctx.gpu,
                        c.hidden,
                        n * h,
                        stream,
                        &format!("V4-msdecode L{} hc_head", self.attn_layer_idx),
                    );
                }
            } else if is_last_layer {
                tracing::warn!(
                    "V4-msdecode L{}: hc_head SKIPPED (no head weights)",
                    self.attn_layer_idx
                );
            }
            return Ok(());
        }

        // ── Phase 7: FFN + hc_post (per-token sequential only) ──
        ops::hc_pre(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
            hc.ffn.hc_fn,
            hc.ffn.hc_scale,
            hc.ffn.hc_base,
            c.hidden,
            post,
            comb,
            n as u32,
            h as u32,
            hc_mult,
            hc.sinkhorn_iters as u32,
            eps,
            hc.hc_eps,
            stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                c.hidden,
                n * h,
                stream,
                &format!("V4-msdecode L{} hc_pre-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                n * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} post-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                n * (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} comb-ffn", self.attn_layer_idx),
            );
        }
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            c.hidden,
            &self.post_attn_norm,
            c.normed,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;

        // Per-token sequential FFN (MLA models always take this path).
        for i in 0..n {
            let normed2_i = c.normed.offset(i * c.h * c.bf16);
            let moe_out = self.ffn.forward(normed2_i, ctx, stream)?;
            // hc_streams is the FP32 mHC highway (4 bytes/elem), not BF16.
            let hc_streams_i = hc_streams.offset(i * hc.hc_mult * c.h * 4);
            let post_i = post.offset(i * hc.hc_mult * 4);
            let comb_i = comb.offset(i * hc.hc_mult * hc.hc_mult * 4);
            ops::hc_post(
                ctx.gpu,
                self.hc_post_k,
                moe_out,
                hc_streams_i,
                post_i,
                comb_i,
                hc_streams_i,
                1,
                h as u32,
                hc_mult,
                stream,
            )?;
        }
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-msdecode L{} hc_post-ffn", self.attn_layer_idx),
            );
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                n * (hc_mult as usize) * h,
                stream,
                &format!(
                    "V4-msdecode L{} hc_post-ffn ALL_STREAMS",
                    self.attn_layer_idx
                ),
            );
        }

        if is_last_layer && let Some(ref head) = hc.head {
            ops::hc_head(
                ctx.gpu,
                self.hc_head_k,
                hc_streams,
                head.hc_fn,
                head.hc_scale,
                head.hc_base,
                c.hidden,
                n as u32,
                h as u32,
                hc_mult,
                eps,
                hc.hc_eps,
                stream,
            )?;
            if diag_this {
                super::diag_norm(
                    ctx.gpu,
                    c.hidden,
                    n * h,
                    stream,
                    &format!("V4-msdecode L{} hc_head", self.attn_layer_idx),
                );
            }
        } else if is_last_layer {
            tracing::warn!(
                "V4-msdecode L{}: hc_head SKIPPED (no head weights)",
                self.attn_layer_idx
            );
        }

        Ok(())
    }
}
