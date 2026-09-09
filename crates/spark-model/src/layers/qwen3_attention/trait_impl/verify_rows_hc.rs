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
use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::ops;

/// `ATLAS_QWEN4EXP_MTP_HC_ATTN_ROWS=1`: K-row attention body under the mHC
/// verify. Opt-in while it is A/B'd against the per-row decode bodies.
pub fn verify_attn_rows_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_HC_ATTN_ROWS").as_deref() == Ok("1"))
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
        state: &mut dyn LayerState,
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
        anyhow::ensure!(
            k >= 1 && row_metas.len() == k && row_seq_lens.len() == k && tokens.len() == k,
            "decode_verify_rows_hc: k={k} but {} metas / {} seq_lens / {} tokens",
            row_metas.len(),
            row_seq_lens.len(),
            tokens.len()
        );
        anyhow::ensure!(
            ctx.hc_row_offset == 0,
            "decode_verify_rows_hc: the K-row body addresses highway rows 0..K (hc_row_offset={})",
            ctx.hc_row_offset
        );
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = k as u32;
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decode_verify_rows_hc on a layer without mHC"))?;
        let hc_mult = hc.hc_mult as u32;
        let hc_streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        let normed = ctx.buffers.norm_output();

        if hc.is_first_model_layer {
            ops::hc_expand(ctx.gpu, self.hc_expand_k, hidden, hc_streams, n, h as u32, hc_mult, stream)?;
        }

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
            ops::rms_norm(ctx.gpu, self.rms_norm_w_k, hidden, &self.input_norm, normed, n, h as u32, eps, stream)?;
        } else {
            ctx.gpu.copy_d2d_async(hidden, normed, k * h * 2, stream)?;
        }

        // ── Attention core, per row, unchanged ──
        // `attention_forward` writes o_proj into `norm_output()` row 0, which
        // is row 0 of `normed`. Row 0's input has already been consumed by
        // then; rows > 0 read their own `normed` row. Each output is moved
        // into `hidden + t*H` (free once `rms_norm` ran) before the next row.
        for t in 0..k {
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
            ctx.gpu.copy_d2d_async(attn_out, hidden.offset(t * h * 2), h * 2, stream)?;
        }
        if let Some(ref post_norm) = self.post_attn_out_norm {
            ops::rms_norm(ctx.gpu, self.rms_norm_w_k, hidden, post_norm, hidden, n, h as u32, eps, stream)?;
        }
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            hidden,
            hc_streams,
            post,
            comb,
            hc_streams,
            n,
            h as u32,
            stream,
        )?;

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
            ops::rms_norm(ctx.gpu, self.rms_norm_w_k, hidden, &self.post_attn_norm, normed, n, h as u32, eps, stream)?;
        } else {
            ctx.gpu.copy_d2d_async(hidden, normed, k * h * 2, stream)?;
        }
        self.verify_rows_ffn(normed, k, ctx, stream)?;
        let ffn_out = ctx.buffers.moe_output();
        if let Some(ref post_norm) = self.post_ffn_out_norm {
            ops::rms_norm(ctx.gpu, self.rms_norm_w_k, ffn_out, post_norm, ffn_out, n, h as u32, eps, stream)?;
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

        if hc.is_last_model_layer && let Some(ref head) = hc.head {
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
        Ok(())
    }

    /// The K-row FFN, same arms as the attention prefill body's small-M
    /// dispatch (prefill_inner.rs) so the two verify bodies cannot drift:
    /// 1 -> `forward`, 2 -> `forward_k2`, 3 -> `forward_k3`, else prefill.
    /// Every arm writes `moe_output()`.
    fn verify_rows_ffn(&self, rows: DevicePtr, k: usize, ctx: &ForwardContext, stream: u64) -> Result<()> {
        let small_m = {
            static SMALL_M: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *SMALL_M.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_HC_SMALL_M_FFN").as_deref() != Ok("0"))
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
