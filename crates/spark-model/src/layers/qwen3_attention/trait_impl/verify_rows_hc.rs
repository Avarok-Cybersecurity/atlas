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
/// `ATLAS_QWEN4EXP_MTP_HC_ATTN_ROWS=0` restores the per-row decode bodies
/// (the A/B and rollback switch).
pub fn verify_attn_rows_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_HC_ATTN_ROWS").as_deref() != Ok("0"))
}

/// Inside the K-row body, also run the attention projections at T=K through
/// the multi-sequence phases (QKV, RoPE, cache write, o_proj batched; paged
/// decode per row). ON by default; `ATLAS_QWEN4EXP_MTP_HC_ATTN_ROWS_QKV=0`
/// keeps the K-row body with per-row projections. Needs the K-row body.
pub fn verify_attn_rows_qkv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_HC_ATTN_ROWS_QKV").as_deref() != Ok("0"))
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
            *ON.get_or_init(|| {
                std::env::var("ATLAS_HC_VERIFY_STAGE_TIMING").as_deref() == Ok("1")
            })
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
            if call % 1024 == 0 && call > 0 {
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

    fn decode_verify_rows_hc_inner(
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
        run_ffn: bool,
    ) -> Result<()> {
        anyhow::ensure!(
            k >= 1 && row_metas.len() == k && row_seq_lens.len() == k && tokens.len() == k,
            "decode_verify_rows_hc: k={k} but {} metas / {} seq_lens / {} tokens",
            row_metas.len(),
            row_seq_lens.len(),
            tokens.len()
        );
        // `hc_row_offset` is the highway base this pass's K rows live at — 0
        // for the single-sequence verify, `off[i]` when the cross-sequence
        // verify drives one sequence at a time.
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = k as u32;
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decode_verify_rows_hc on a layer without mHC"))?;
        let hc_mult = hc.hc_mult as u32;
        // Highway base row. The K-row verify runs at 0; the CROSS-SEQUENCE
        // verify calls this body once per sequence, with sequence i's rows
        // parked at `off[i]`, so the streams have to start there — the same
        // `hc_row_offset * hc_mult * H * 4` arithmetic the GDN hc bodies use.
        // `hidden` is already offset by the caller; `norm_output` / `hc_post`
        // / `hc_comb` / `moe_output` are per-CALL scratch consumed before the
        // next sequence runs, so they stay at their base.
        let hc_streams = ctx
            .buffers
            .hc_streams()
            .offset(ctx.hc_row_offset * hc.hc_mult * h * 4);
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        let normed = ctx.buffers.norm_output();

        if hc.is_first_model_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                hc_streams,
                n,
                h as u32,
                hc_mult,
                stream,
            )?;
        }

        // ── Phase timing (ATLAS_HC_VERIFY_STAGE_TIMING=1) ──
        // See the module note: attention layers cost MORE per layer than SSM
        // layers and the residual after the FFN is ~40x the projection floor.
        let phase_timing = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                std::env::var("ATLAS_HC_VERIFY_STAGE_TIMING").as_deref() == Ok("1")
            })
        };
        let mut at = std::time::Instant::now();
        let (mut a1, mut a2, mut a3) = (0u128, 0u128, 0u128);
        let aphase = |t: &mut std::time::Instant, acc: &mut u128| {
            if phase_timing {
                let _ = ctx.gpu.synchronize(stream);
                *acc += t.elapsed().as_micros();
                *t = std::time::Instant::now();
            }
        };

        // ── Attention sublayer: hc_pre at T=K ──
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
            &hc.attn,
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
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                hidden,
                &self.input_norm,
                normed,
                n,
                h as u32,
                eps,
                stream,
            )?;
        } else {
            ctx.gpu.copy_d2d_async(hidden, normed, k * h * 2, stream)?;
        }

        aphase(&mut at, &mut a1);
        // ── Attention core ──
        // Batched arm (projections at T=K, paged decode per row) when the
        // shape allows; otherwise per row, unchanged: `attention_forward`
        // writes o_proj into `norm_output()` row 0, which is row 0 of
        // `normed`. Row 0's input has already been consumed by then; rows > 0
        // read their own `normed` row. Each output is moved into
        // `hidden + t*H` (free once `rms_norm` ran) before the next row.
        let batched_out = self.attention_rows_batched(
            hidden,
            k,
            state,
            kv_cache,
            row_metas,
            row_seq_lens,
            ctx,
            stream,
        )?;
        let attn_block_out = if let Some(o) = batched_out { o } else { hidden };
        for t in 0..k {
            if batched_out.is_some() {
                break;
            }
            let row_ctx = ForwardContext {
                buffers: ctx.buffers,
                hc_row_offset: t,
                gpu: ctx.gpu,
                config: ctx.config,
                dispatch: ctx.dispatch,
                derived: ctx.derived,
                levers: ctx.levers,
                stats: ctx.stats,
                attn_metadata: Some(row_metas[t]),
                profile: ctx.profile,
                comm: ctx.comm,
                graph_capture: ctx.graph_capture,
                gdn_exact_replay: ctx.gdn_exact_replay,
                token_ids: ctx.token_ids,
                host_token_ids: Some(&tokens[t..t + 1]),
                routed_lora_layers: ctx.routed_lora_layers,
                midchunk_capture: None,
                moe_lora_route: ctx.moe_lora_route,
            };
            let attn_out = self.attention_forward(
                state,
                normed.offset(t * h * 2),
                row_seq_lens[t],
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                kv_cache,
                &row_ctx,
                stream,
            )?;
            if ctx.config.tp_world_size > 1
                && let Some(comm) = ctx.comm
            {
                comm.all_reduce_async(attn_out.0, h * 2, stream)?;
            }
            ctx.gpu
                .copy_d2d_async(attn_out, hidden.offset(t * h * 2), h * 2, stream)?;
        }
        if let Some(ref post_norm) = self.post_attn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                attn_block_out,
                post_norm,
                attn_block_out,
                n,
                h as u32,
                eps,
                stream,
            )?;
        }
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            attn_block_out,
            hc_streams,
            post,
            comb,
            hc_streams,
            n,
            h as u32,
            stream,
        )?;

        aphase(&mut at, &mut a2);
        if !run_ffn {
            // The caller runs the FFN sublayer once over the whole batch
            // (`decode_verify_ffn_rows_hc`); this sequence's rows are done.
            return Ok(());
        }
        // ── FFN sublayer: hc_pre at T=K, K-row FFN, hc_post at T=K ──
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
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                hidden,
                &self.post_attn_norm,
                normed,
                n,
                h as u32,
                eps,
                stream,
            )?;
        } else {
            ctx.gpu.copy_d2d_async(hidden, normed, k * h * 2, stream)?;
        }
        self.verify_rows_ffn(normed, k, ctx, stream)?;
        let ffn_out = ctx.buffers.moe_output();
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
            self.apply_layer_scalar(ctx.gpu, ffn_out, k * h, scalar, stream)?;
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
                "V4-verify-rows L{}: hc_head SKIPPED (no head weights)",
                self.attn_layer_idx
            );
        }
        aphase(&mut at, &mut a3);
        if phase_timing {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static CALLS: AtomicUsize = AtomicUsize::new(0);
            let call = CALLS.fetch_add(1, Ordering::Relaxed);
            if call % 1024 == 0 && call > 0 {
                tracing::info!(
                    call,
                    rows = k,
                    a1_hc_pre_us = a1 as u64,
                    a2_core_us = a2 as u64,
                    a3_ffn_us = a3 as u64,
                    "attention hc verify phase split (ONE layer, synced per phase)"
                );
            }
        }
        Ok(())
    }

    /// Attention projections at T=K through the multi-sequence phases.
    ///
    /// The K rows are one sequence, so they are NOT K independent sequences:
    /// the KV rows are all written first (every row's K/V is known after the
    /// batched projection), then the paged decode runs per row against
    /// `row_metas[t]`, whose device `seq_len` is `base + t + 1`, so row `t`
    /// attends over rows `<= t` and never over rows `> t`. Rows run in
    /// DESCENDING order: the one-row decode writes `attn_output()` row 0, and
    /// row 0 is the last one computed, so its output lands in place while the
    /// higher rows were copied out to their own row before it ran.
    /// QSA ingest, when the layer has it, then advances the single sequence
    /// state row by row, ascending, exactly as the per-row bodies did.
    /// `None` = shape outside the phases (MLA, TP, QSA selection active, the
    /// flag unset); the caller falls back to the per-row core.
    #[allow(clippy::too_many_arguments)]
    fn attention_rows_batched(
        &self,
        hidden: DevicePtr,
        k: usize,
        state: &mut (dyn LayerState + 'static),
        kv_cache: &mut PagedKvCache,
        row_metas: &[AttnMetadataDev],
        row_seq_lens: &[usize],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        if !verify_attn_rows_qkv_enabled()
            || k < 2
            || self.mla.is_some()
            || ctx.config.tp_world_size > 1
            || self.ms_qsa_selection_active(row_seq_lens, k)
        {
            return Ok(None);
        }
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            tracing::info!(
                "mHC verify: attention projections BATCHED at T=K through the multi-seq phases \
                 (default on; ATLAS_QWEN4EXP_MTP_HC_ATTN_ROWS_QKV=0 disables), first pass k={k}"
            );
        });
        let h = ctx.config.hidden_size;
        let bs = kv_cache.block_size() as u32;
        let mut c = MultiSeqCtx::new(self, ctx, hidden, hidden, k, bs, stream);
        c.seq_slot = ctx.attn_metadata.map_or(DevicePtr(0), |m| m.seq_slot);
        // Row-walking phases index `positions` / `slot` by row; the K-row
        // pack holds them contiguously, and `row_metas[0]` carries the bases.
        let meta_k = AttnMetadataDev {
            num_seqs: k as u32,
            ..row_metas[0]
        };
        // ── Phase timing (ATLAS_HC_VERIFY_STAGE_TIMING=1), see module note ──
        let core_timing = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                std::env::var("ATLAS_HC_VERIFY_STAGE_TIMING").as_deref() == Ok("1")
            })
        };
        let mut ct = std::time::Instant::now();
        let (mut c1, mut c2, mut c3, mut c4) = (0u128, 0u128, 0u128, 0u128);
        let cphase = |t: &mut std::time::Instant, acc: &mut u128| {
            if core_timing {
                let _ = ctx.gpu.synchronize(stream);
                *acc += t.elapsed().as_micros();
                *t = std::time::Instant::now();
            }
        };

        self.ms_phase_qkv(&c)?;
        self.ms_phase_rope(&c, meta_k)?;
        self.ms_phase_cache_write(&c, kv_cache, meta_k)?;
        cphase(&mut ct, &mut c1);

        let attn_out = ctx.buffers.attn_output();
        let q_row = c.q_dim as usize * c.bf16;
        let row_view = |t: usize| MultiSeqCtx {
            fwd: c.fwd,
            hidden: c.hidden.offset(t * h * c.bf16),
            residual: c.residual,
            n: 1,
            stream: c.stream,
            h: c.h,
            nq: c.nq,
            nkv: c.nkv,
            hd: c.hd,
            eps: c.eps,
            bs: c.bs,
            bf16: c.bf16,
            q_dim: c.q_dim,
            q_proj_dim: c.q_proj_dim,
            q_proj_bytes: c.q_proj_bytes,
            per_seq_qkv: c.per_seq_qkv,
            normed: c.normed.offset(t * h * c.bf16),
            qkv_buf: c.qkv_buf.offset(t * c.per_seq_qkv),
            seq_slot: c.seq_slot,
        };
        for t in (0..k).rev() {
            let out = self.ms_phase_paged_decode(&row_view(t), kv_cache, row_metas[t])?;
            if t > 0 {
                ctx.gpu
                    .copy_d2d_async(out, attn_out.offset(t * q_row), q_row, stream)?;
            } else {
                anyhow::ensure!(
                    out == attn_out,
                    "paged decode row 0 must land in attn_output() row 0"
                );
            }
        }
        cphase(&mut ct, &mut c2);
        if self.qsa.is_some() {
            for t in 0..k {
                let mut states: [&mut (dyn LayerState + 'static); 1] = [&mut *state];
                self.ms_qsa_ingest_only(
                    &row_view(t),
                    &mut states,
                    &row_seq_lens[t..t + 1],
                    kv_cache,
                    row_metas[t],
                )?;
            }
        }
        cphase(&mut ct, &mut c3);
        let o_out = self.ms_phase_o_proj(&c, attn_out)?;
        cphase(&mut ct, &mut c4);
        if core_timing {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static CALLS: AtomicUsize = AtomicUsize::new(0);
            let call = CALLS.fetch_add(1, Ordering::Relaxed);
            if call % 1024 == 0 && call > 0 {
                tracing::info!(
                    call,
                    rows = k,
                    c1_proj_us = c1 as u64,
                    c2_paged_us = c2 as u64,
                    c3_qsa_us = c3 as u64,
                    c4_oproj_us = c4 as u64,
                    "attention core phase split (ONE sequence, synced per phase)"
                );
            }
        }
        Ok(Some(o_out))
    }

    /// The K-row FFN, same arms as the attention prefill body's small-M
    /// dispatch (prefill_inner.rs) so the two verify bodies cannot drift:
    /// 1 -> `forward`, 2 -> `forward_k2`, 3 -> `forward_k3`, else prefill.
    /// Every arm writes `moe_output()`.
    fn verify_rows_ffn(
        &self,
        rows: DevicePtr,
        k: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let small_m = {
            static SMALL_M: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *SMALL_M.get_or_init(|| {
                std::env::var("ATLAS_QWEN4EXP_HC_SMALL_M_FFN").as_deref() != Ok("0")
            })
        };
        match k {
            1 if small_m => {
                let out = self.ffn.forward(rows, ctx, stream)?;
                anyhow::ensure!(
                    out == ctx.buffers.moe_output(),
                    "verify rows FFN: single-token MoE returned a buffer other than moe_output()"
                );
                Ok(())
            }
            2 if small_m => self.ffn.forward_k2(rows, ctx, stream),
            3 if small_m => self.ffn.forward_k3(rows, ctx, stream),
            _ => self.ffn.forward_prefill(rows, k, ctx, stream),
        }
    }
}
