// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! K-row mHC verify body for [`super::super::Qwen3AttentionLayer`].
//!
//! Under the mHC highway the MTP verify used to run every attention layer as
//! K sequential one-row `decode_inner_hc` bodies (verify_hc.rs). Each body
//! carried six hyper-connection sites at T=1 (two `hc_pre`, two `hc_post`,
//! plus the norms), so a K=3 verify paid 72 `hc_pre` and 72 `hc_post` sites
//! per step on the 12 attention layers while the 36 GDN layers ran theirs
//! once at T=3.
//!
//! Only the attention core itself (QKV, RoPE, KV append, paged attention,
//! o_proj) needs the rows in order: row `t` must see rows `< t` in the KV
//! cache. Everything around it is row-independent. This body therefore runs
//! `hc_pre` / `rms_norm` / `hc_post` / the FFN at T=K, exactly the dispatch
//! the GDN layers already take (`hc_pre_site(n)`, the small-M FFN arms), and
//! keeps the attention core per row, unchanged, through `attention_forward`
//! with the same per-row metadata the one-row bodies received.
//!
//! Buffer contract (all pre-existing, all sized for the K-row verify arena):
//! `hidden` is `[K, H]` BF16 and doubles as the per-row attention-output
//! staging area once `rms_norm` has consumed it; `norm_output()` is `[K, H]`
//! and is ALSO where o_proj writes its single row (row 0), so each row's
//! output is copied into `hidden + t*H` before the next row runs;
//! `moe_output()` receives the K-row FFN result.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use super::multi_seq::ctx::MultiSeqCtx;
use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::ops;

/// K-row attention body under the mHC verify. ON by default;
/// `AVAROK_QWEN4EXP_MTP_HC_ATTN_ROWS=0` restores the per-row decode bodies
/// (the A/B and rollback switch).
pub fn verify_attn_rows_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AVAROK_QWEN4EXP_MTP_HC_ATTN_ROWS").as_deref() != Ok("0"))
}

/// Inside the K-row body, also run the attention projections at T=K through
/// the multi-sequence phases (QKV, RoPE, cache write, o_proj batched; paged
/// decode per row). ON by default; `AVAROK_QWEN4EXP_MTP_HC_ATTN_ROWS_QKV=0`
/// keeps the K-row body with per-row projections. Needs the K-row body.
pub fn verify_attn_rows_qkv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AVAROK_QWEN4EXP_MTP_HC_ATTN_ROWS_QKV").as_deref() != Ok("0"))
}

impl Qwen3AttentionLayer {
    /// True when this layer can take the K-row body: an mHC layer with an
    /// FFN (the standalone-attention shape keeps the per-row path).
    pub fn verify_rows_hc_ok(&self) -> bool {
        self.hc.is_some() && !self.ffn.is_none()
    }

    /// One K-row pass over this attention layer under the mHC highway.
    ///
    /// * `hidden`: `[K, H]` BF16 rows at `hc_row_offset = 0` of the highway.
    /// * `row_metas[t]` / `row_seq_lens[t]` / `tokens[t]`: exactly what the
    ///   one-row decode body for row `t` received.
    /// * `ctx`: the K-row verify context (`hc_row_offset == 0`).
    #[allow(clippy::too_many_arguments)]
    pub fn decode_verify_rows_hc(
        &self,
        hidden: DevicePtr,
        k: usize,
        state: &mut (dyn LayerState + 'static),
        kv_cache: &mut PagedKvCache,
        row_metas: &[AttnMetadataDev],
        row_seq_lens: &[usize],
        tokens: &[u32],
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_verify_rows_hc_inner(
            hidden,
            k,
            state,
            kv_cache,
            row_metas,
            row_seq_lens,
            tokens,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
            true,
        )
    }

    /// The attention sublayer ONLY — identical to [`Self::decode_verify_rows_hc`]
    /// up to and including the attention `hc_post`, then returns. Pair with
    /// [`Self::decode_verify_ffn_rows_hc`] once over the whole batch. See the
    /// module note: the FFN sublayer is row-wise and sequence-independent, so
    /// running it per sequence paid the highway bracket n times.
    pub fn decode_verify_rows_hc_attn_only(
        &self,
        hidden: DevicePtr,
        k: usize,
        state: &mut (dyn LayerState + 'static),
        kv_cache: &mut PagedKvCache,
        row_metas: &[AttnMetadataDev],
        row_seq_lens: &[usize],
        tokens: &[u32],
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_verify_rows_hc_inner(
            hidden,
            k,
            state,
            kv_cache,
            row_metas,
            row_seq_lens,
            tokens,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
            false,
        )
    }

    /// The FFN sublayer of the highway attention block, ONCE over all
    /// `ks.iter().sum()` rows at highway base 0: hc_pre at T=R, block-input
    /// norm, the FFN per sequence (each fused arm writes `moe_output()[0, k)`,
    /// staged into `norm_output` at its batch offset), hc_post at T=R, and the
    /// head site on the last model layer.
    ///
    /// `hidden` is the BASE hidden buffer (all R rows), not a per-sequence
    /// offset — this call covers every sequence. `ctx.hc_row_offset` must be 0
    /// for the same reason.
    pub fn decode_verify_ffn_rows_hc(
        &self,
        hidden: DevicePtr,
        ks: &[usize],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            ctx.hc_row_offset == 0,
            "decode_verify_ffn_rows_hc covers every sequence and runs at highway base 0, \
             got hc_row_offset={}",
            ctx.hc_row_offset
        );
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let bf16 = 2usize;
        let rows: usize = ks.iter().sum();
        anyhow::ensure!(rows >= 1, "decode_verify_ffn_rows_hc: empty batch");
        let n = rows as u32;
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decode_verify_ffn_rows_hc on a layer without mHC"))?;
        let hc_streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        // Staging for the per-sequence FFN outputs — free here because the
        // block-input norm below is applied IN PLACE on `hidden` rather than
        // into `norm_output` as the per-sequence body does.
        let stage = ctx.buffers.norm_output();

        let timing = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var("AVAROK_HC_VERIFY_STAGE_TIMING").as_deref() == Ok("1"))
        };
        let t0 = std::time::Instant::now();

        // ── hc_pre (ffn) at T=R, collapse into `hidden` ──
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
            &hc.ffn,
            hc,
            hidden,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n,
            h as u32,
            eps,
            stream,
        )?;
        if ops::HcVariant::of(hc).applies_block_input_norm() {
            // In place: same input==output use the post-FFN norm below relies on.
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                hidden,
                &self.post_attn_norm,
                hidden,
                n,
                h as u32,
                eps,
                stream,
            )?;
        }

        // ── FFN per sequence (the fused K-row arms), staged at batch offsets ──
        // The MoE cannot amortise across rows — top-10-of-512 routing means
        // different rows light different experts and it is already at ~86% of
        // peak DRAM — so per sequence is the right width. What this call
        // saves is the bracket around it, not the FFN itself.
        let mut off = 0usize;
        for &k in ks {
            self.verify_rows_ffn(hidden.offset(off * h * bf16), k, ctx, stream)?;
            ctx.gpu.copy_d2d_async(
                ctx.buffers.moe_output(),
                stage.offset(off * h * bf16),
                k * h * bf16,
                stream,
            )?;
            off += k;
        }
        let ffn_out = stage;
        if let Some(ref post_norm) = self.post_ffn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                ffn_out,
                post_norm,
                ffn_out,
                n,
                h as u32,
                eps,
                stream,
            )?;
        }
        if let Some(scalar) = self.layer_scalar {
            self.apply_layer_scalar(ctx.gpu, ffn_out, rows * h, scalar, stream)?;
        }
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            ffn_out,
            hc_streams,
            post,
            comb,
            hc_streams,
            n,
            h as u32,
            stream,
        )?;

        if hc.is_last_model_layer
            && let Some(ref head) = hc.head
        {
            ops::hc_head_site(
                ctx.gpu,
                self.hc_head_k,
                hc_streams,
                head,
                hc,
                hidden,
                ctx.buffers.hc_lowrank_scratch(),
                n,
                h as u32,
                eps,
                stream,
            )?;
        } else if hc.is_last_model_layer {
            tracing::warn!(
                "V4-verify-ffn-rows L{}: hc_head SKIPPED (no head weights)",
                self.attn_layer_idx
            );
        }

        if timing {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static CALLS: AtomicUsize = AtomicUsize::new(0);
            let call = CALLS.fetch_add(1, Ordering::Relaxed);
            if call.is_multiple_of(1024) && call > 0 {
                let _ = ctx.gpu.synchronize(stream);
                tracing::info!(
                    call,
                    rows,
                    n_seqs = ks.len(),
                    ffn_sublayer_us = t0.elapsed().as_micros() as u64,
                    "attention hc verify FFN sublayer, BATCHED across sequences (ONE layer)"
                );
            }
        }
        Ok(())
    }
}

#[path = "verify_rows_hc_attn.rs"]
mod verify_rows_hc_attn;

#[path = "verify_rows_hc_inner.rs"]
mod verify_rows_hc_inner;
