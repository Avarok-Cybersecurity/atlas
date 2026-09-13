// SPDX-License-Identifier: AGPL-3.0-only

//! The K-row mHC verify body, split out of `verify_rows_hc.rs` to keep it
//! under the 500-line cap.

use super::*;

impl Qwen3AttentionLayer {
    pub(super) fn decode_verify_rows_hc_inner(
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
            *ON.get_or_init(|| std::env::var("ATLAS_HC_VERIFY_STAGE_TIMING").as_deref() == Ok("1"))
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
                decode_step: false,
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
            if call.is_multiple_of(1024) && call > 0 {
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
}
