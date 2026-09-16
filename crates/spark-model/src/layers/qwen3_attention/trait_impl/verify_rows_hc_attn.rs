// SPDX-License-Identifier: AGPL-3.0-only

//! The per-stage attention helpers the mHC verify body calls, split out of
//! `verify_rows_hc.rs` to keep it under the 500-line cap.

use super::*;

impl Qwen3AttentionLayer {
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
    /// Cross-sequence attention sublayer, part 1 of 3: hc_expand (first
    /// model layer), hc_pre(attn), the input norm and the QKV projection, ONCE
    /// over all `ks.iter().sum()` rows at highway base 0. See the module note
    /// on the order contract. `all_row_seq_lens` is every sequence's rows in
    /// batch order (for the QSA-selection decline); `seq_slot` is the
    /// per-request adapter slot buffer (`DevicePtr(0)` without LoRA).
    ///
    /// Returns `None` when the batched projections decline — the caller must
    /// then run this layer per sequence, exactly as before.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn verify_attn_pre_hc<'a>(
        &self,
        hidden: DevicePtr,
        ks: &[usize],
        all_row_seq_lens: &[usize],
        seq_slot: DevicePtr,
        bs: u32,
        ctx: &'a ForwardContext<'a>,
        stream: u64,
    ) -> Result<Option<MultiSeqCtx<'a>>> {
        let r: usize = ks.iter().sum();
        // ★ THE `tp_world_size > 1` TERM IS GONE. It was never about the
        // projections being wrong under TP — it was that this arm ends at
        // `ms_phase_o_proj` and never reduced, while the per-row twin reduces
        // each row inline right after `attention_forward`. `verify_attn_post_hc`
        // now issues that reduction over all R rows at once, so the arm is
        // TP-correct and the term has nothing left to guard.
        //
        // `ms_qsa_selection_active` STAYS: the select+attend path is not wired
        // into this arm yet, so at ISL >= 2051 (inert_bound) it still declines.
        if !verify_attn_rows_qkv_enabled()
            || r < 2
            || self.mla.is_some()
            || self.ms_qsa_selection_active(all_row_seq_lens, r)
        {
            return Ok(None);
        }
        anyhow::ensure!(
            ctx.hc_row_offset == 0 && all_row_seq_lens.len() == r,
            "verify_attn_pre_hc: runs at highway base 0 over {r} rows (got offset {}, {} seq_lens)",
            ctx.hc_row_offset,
            all_row_seq_lens.len()
        );
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = r as u32;
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("verify_attn_pre_hc on a layer without mHC"))?;
        let hc_streams = ctx.buffers.hc_streams();
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
                hc.hc_mult as u32,
                stream,
            )?;
        }
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
            ctx.gpu.copy_d2d_async(hidden, normed, r * h * 2, stream)?;
        }
        let mut c = MultiSeqCtx::new(self, ctx, hidden, hidden, r, bs, stream);
        c.seq_slot = seq_slot;
        self.ms_phase_qkv(&c)?;
        Ok(Some(c))
    }

    /// Part 2 of 3: ONE sequence's rows `off..off+k` of the R-row context —
    /// rope, cache write, per-row paged decode, QSA ingest — against ITS
    /// state, KV and metadata. Sequences MUST be called in descending `off`
    /// order (module note).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn verify_attn_seq_hc(
        &self,
        c: &MultiSeqCtx<'_>,
        off: usize,
        k: usize,
        state: &mut (dyn LayerState + 'static),
        kv_cache: &mut PagedKvCache,
        row_metas: &[AttnMetadataDev],
        row_seq_lens: &[usize],
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            k >= 1 && row_metas.len() == k && row_seq_lens.len() == k && off + k <= c.n,
            "verify_attn_seq_hc: k={k} at off={off} of {} rows, {} metas / {} seq_lens",
            c.n,
            row_metas.len(),
            row_seq_lens.len()
        );
        let h = c.h;
        let view = |row0: usize, rows: usize| MultiSeqCtx {
            fwd: c.fwd,
            hidden: c.hidden.offset(row0 * h * c.bf16),
            residual: c.residual,
            n: rows,
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
            normed: c.normed.offset(row0 * h * c.bf16),
            qkv_buf: c.qkv_buf.offset(row0 * c.per_seq_qkv),
            seq_slot: c.seq_slot,
        };
        // This sequence's k rows are a contiguous pack in the verify metadata,
        // and `row_metas[0]` carries its bases.
        let view_k = view(off, k);
        let meta_k = AttnMetadataDev {
            num_seqs: k as u32,
            ..row_metas[0]
        };
        self.ms_phase_rope(&view_k, meta_k)?;
        self.ms_phase_cache_write(&view_k, kv_cache, meta_k)?;

        let attn_out = c.fwd.buffers.attn_output();
        let q_row = c.q_dim as usize * c.bf16;
        for t in (0..k).rev() {
            let g = off + t;
            let out = self.ms_phase_paged_decode(&view(g, 1), kv_cache, row_metas[t])?;
            if g > 0 {
                c.fwd
                    .gpu
                    .copy_d2d_async(out, attn_out.offset(g * q_row), q_row, stream)?;
            } else {
                anyhow::ensure!(
                    out == attn_out,
                    "paged decode row 0 must land in attn_output() row 0"
                );
            }
        }
        if self.qsa.is_some() {
            for t in 0..k {
                let mut states: [&mut (dyn LayerState + 'static); 1] = [&mut *state];
                self.ms_qsa_ingest_only(
                    &view(off + t, 1),
                    &mut states,
                    &row_seq_lens[t..t + 1],
                    kv_cache,
                    row_metas[t],
                )?;
            }
        }
        Ok(())
    }

    /// Part 3 of 3: o_proj, the post-attention norm and hc_post(attn), ONCE
    /// over all R rows of the context.
    pub(crate) fn verify_attn_post_hc(
        &self,
        c: &MultiSeqCtx<'_>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = c.n as u32;
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("verify_attn_post_hc on a layer without mHC"))?;
        let attn_out = ctx.buffers.attn_output();
        let o_out = self.ms_phase_o_proj(c, attn_out)?;
        // ── TP reduction for the batched arm ──
        //
        // o_proj is the ROW-split half of tensor parallelism, so every rank
        // holds a partial sum here and they must be summed before ANYTHING
        // reads them. That "anything" is immediate: the post-attention norm
        // directly below would otherwise normalise a partial, and `hc_post_site`
        // would fold it into the highway — silently wrong hidden states rather
        // than a crash.
        //
        // One collective over all R rows: `ms_phase_o_proj` documents that
        // "o_out rows are h BF16 elements apart", i.e. R contiguous rows, so
        // `c.n * h * 2` bytes is exactly the batch. The per-row twin issues R
        // of these; this issues 1.
        //
        // Symmetric by construction: both ranks run this same unconditional
        // code with the same `c.n`, so neither can issue a collective the other
        // does not.
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(o_out.0, c.n * h * 2, stream)?;
        }
        if let Some(ref post_norm) = self.post_attn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                o_out,
                post_norm,
                o_out,
                n,
                h as u32,
                eps,
                stream,
            )?;
        }
        let hc_streams = ctx.buffers.hc_streams();
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            o_out,
            hc_streams,
            ctx.buffers.hc_post(),
            ctx.buffers.hc_comb(),
            hc_streams,
            n,
            h as u32,
            stream,
        )
    }

    pub(super) fn attention_rows_batched(
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
        // See `verify_attn_pre_hc`: the TP term guarded a MISSING REDUCTION,
        // not a wrong computation, and this body now reduces its own o_proj
        // over all k rows before returning.
        if !verify_attn_rows_qkv_enabled()
            || k < 2
            || self.mla.is_some()
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
            *ON.get_or_init(|| std::env::var("ATLAS_HC_VERIFY_STAGE_TIMING").as_deref() == Ok("1"))
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
        // TP reduction — see `verify_attn_post_hc` for why it must land here,
        // before the caller's post-attention norm reads these rows.
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(o_out.0, k * h * 2, stream)?;
        }
        cphase(&mut ct, &mut c4);
        if core_timing {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static CALLS: AtomicUsize = AtomicUsize::new(0);
            let call = CALLS.fetch_add(1, Ordering::Relaxed);
            if call.is_multiple_of(1024) && call > 0 {
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
    pub(super) fn verify_rows_ffn(
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
