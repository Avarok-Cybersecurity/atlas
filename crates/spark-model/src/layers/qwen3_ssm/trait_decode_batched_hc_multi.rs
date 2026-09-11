// SPDX-License-Identifier: AGPL-3.0-only

//! R-row (N sequences x ks[i] tokens) batched GDN verify under an mHC
//! highway — the cross-sequence analogue of `trait_decode_batched_hc.rs`.
//!
//! # Why this can exist at all
//!
//! The three files around it each own one axis:
//!
//! * `trait_decode_multi_seq/hc.rs` — N sequences x 1 token. Rows are
//!   independent; the recurrence runs per row.
//! * `trait_decode_batched_hc.rs` — 1 sequence x K tokens. Row `t+1`'s state
//!   depends on row `t`'s, and every row boundary must be materialised for
//!   `commit_accepted_prefix` to rewind onto.
//! * THIS file — N sequences x ks[i] tokens, i.e. both at once. Rows
//!   `off_i..off_i+ks[i]` are sequence i's, ordered within the sequence and
//!   independent across sequences.
//!
//! The highway stages do not care which sequence a row belongs to.
//! `hc_expand`, both `hc_pre` sites, both `hc_post` sites and the MoE are all
//! ROW-PARALLEL over `[R, ...]` layouts — the same kernels the K-row body
//! launches, with R substituted for K. That is what makes this a small file
//! rather than a second verify engine: the only two stages that carry
//! per-sequence state are PLE and the GDN recurrence, and the recurrence
//! already has its cross-sequence form in
//! `GdnStates::Multi` (`trait_decode_batched_conv_gdn_multi.rs`, whose
//! two-launch fast path engages at every ladder width k in 2..=4).
//!
//! # The two per-sequence carries
//!
//! 1. **GDN `h_state` / `conv_state`** — `GdnStates::Multi` walks each
//!    sequence's own rows against its own state and writes that sequence's
//!    `h_state_intermediates[t]` / `conv_state_intermediates[t]`, which is
//!    the contract `commit_accepted_prefix` rewinds from. Unchanged from the
//!    non-highway batched verify.
//!
//! 2. **PLE's rolling conv/history window** — per sequence, ROW BY ROW, for
//!    exactly the reason the K-row body gives: a multi-row `forward` advances
//!    conv + history in one shot, so the only snapshot point would be
//!    PRE-verify, and a partial accept would leave the carry advanced over
//!    rejected rows with no way to rebuild it (conv is a rolling FP32 state;
//!    history's oldest ids have already rolled off). Each sequence gets its
//!    own `begin_verify_rows` / `push_verify_row` bracket over ITS row range,
//!    because the snapshot stack is per-`PleSeqState`.
//!
//!    🪤 The snapshot boundaries are `hc_verify_snapshot_rows(ks[i])` — the
//!    range within the SEQUENCE, not within the batch. `commit_accepted_prefix`
//!    rewinds each sequence against its own `num_accepted`, so a boundary
//!    numbered in batch rows would restore another sequence's carry.
//!
//! # QSA marks are not here
//!
//! They advance on the 12 full-attention layers, which the caller drives
//! through `decode_multi_seq` with per-row block tables and seq lens.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::Qwen3SsmLayer;
use super::trait_decode_batched_hc::hc_verify_snapshot_rows;
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;

impl Qwen3SsmLayer {
    /// R-row batched GDN verify under the highway, R = `ks.iter().sum()`.
    ///
    /// `states[i]` / `ks[i]` describe sequence i, which owns batch rows
    /// `off_i..off_i + ks[i]` where `off_i = ks[..i].iter().sum()`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_verify_multi_inner_hc<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        n_seqs: usize,
        ks: &[usize],
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        wy_tables: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hc = self.hc.as_ref().ok_or_else(|| {
            anyhow::anyhow!("decode_verify_multi_inner_hc without mHC weights")
        })?;
        anyhow::ensure!(
            states.len() == n_seqs && ks.len() == n_seqs,
            "decode_verify_multi_inner_hc: states/ks/n mismatch"
        );
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let rows: usize = ks.iter().sum();
        let n = rows as u32;

        // Same refusal the other three hc bodies carry: `hc_norm` inside
        // `hc_pre` replaces the fused gate-f32 norm, so ATLAS_FP32_ROUTING
        // would have the router read the PREVIOUS layer's activations.
        anyhow::ensure!(
            !self.ffn.fp32_routing_active(),
            "qwen3_ssm mHC multi-seq verify: ATLAS_FP32_ROUTING needs the fused \
             gate-f32 norm, which the highway path replaces. Unset it."
        );

        let streams = ctx
            .buffers
            .hc_streams()
            .offset(ctx.hc_row_offset * hc.hc_mult * h * 4);
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        let hc_row = hc.hc_mult * h * 4;

        // The row-exact leg is a SINGLE-SEQUENCE bit-parity tool: it replays
        // `hc_pre` one row at a time so a K-row verify reproduces K one-row
        // decodes exactly. Across sequences there is no serial decode to be
        // bit-identical TO — the batch is the unit — and honouring it here
        // would silently cost R launches per site for a guarantee nobody can
        // observe. Refuse instead of pretending, and name the knob.
        anyhow::ensure!(
            !crate::layers::qwen3_ssm::verify_row_exact_leg(
                ctx.gdn_exact_replay,
                crate::layers::qwen3_ssm::RowExactLeg::HcPre,
            ),
            "row-exact mHC verify is single-sequence only; the cross-sequence \
             batched verify has no serial decode to match bit-for-bit. Unset \
             the row-exact leg or disable the batched multi-seq verify."
        );

        if hc.is_first_model_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                streams,
                n,
                h as u32,
                hc.hc_mult as u32,
                stream,
            )?;
        }

        // ── Carry 2: PLE, per sequence, row by row ──
        if let Some(ple) = self.ple.as_ref() {
            let host = ctx.host_token_ids.ok_or_else(|| {
                anyhow::anyhow!("hc multi-seq verify: PLE needs host_token_ids threaded")
            })?;
            anyhow::ensure!(
                host.len() >= rows,
                "hc multi-seq verify: {} host ids for {rows} rows",
                host.len()
            );
            let mut off = 0usize;
            for (i, state) in states.iter_mut().enumerate().take(n_seqs) {
                let k = ks[i];
                let ssm = state
                    .as_any_mut()
                    .downcast_mut::<crate::layer::SsmLayerState>()
                    .ok_or_else(|| {
                        anyhow::anyhow!("PLE host layer state is not SsmLayerState for seq {i}")
                    })?;
                let st = ssm.ple.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("PLE multi-seq verify before prefill: no seq state {i}")
                })?;
                ple.begin_verify_rows(st);
                for t in 0..k {
                    let row = off + t;
                    ple.forward_row(
                        st,
                        streams.offset(row * hc_row),
                        &host[row..row + 1],
                        ctx,
                        stream,
                    )?;
                    // Boundaries within THIS sequence — see the header trap.
                    if hc_verify_snapshot_rows(k).contains(&t) {
                        ple.push_verify_row(st, ctx.gpu, stream)?;
                    }
                }
                off += k;
            }
        }

        // ── GDN sublayer. `hidden` is scratch; the highway is the residual. ──
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
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
        // ── Carry 1: the recurrence + its per-row intermediates ──
        let out_proj_buf = self.decode_batched_block(
            hidden,
            rows,
            super::trait_decode_batched::GdnStates::Multi {
                states,
                ks,
                wy_tables,
            },
            ctx,
            stream,
        )?;
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            out_proj_buf,
            streams,
            post,
            comb,
            streams,
            n,
            h as u32,
            stream,
        )?;

        // ── MoE sublayer ──
        // `decode_batched_block` returned `moe_output()`, which the FFN is
        // about to overwrite — safe only because `hc_post` above already
        // consumed it into the highway. Keep that order.
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
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
        self.hc_small_m_ffn(hidden, rows, ctx, stream)?;
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            ctx.buffers.moe_output(),
            streams,
            post,
            comb,
            streams,
            n,
            h as u32,
            stream,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::trait_decode_batched_hc::hc_verify_snapshot_rows;

    /// The trap the header names: snapshot boundaries are numbered within the
    /// SEQUENCE, so sequence 1 of a [2, 3] batch snapshots at its own rows
    /// 0..2, never at batch rows 2..4.
    #[test]
    fn snapshot_boundaries_are_per_sequence_not_per_batch() {
        let ks = [2usize, 3];
        let per_seq: Vec<Vec<usize>> = ks
            .iter()
            .map(|&k| hc_verify_snapshot_rows(k).collect())
            .collect();
        assert_eq!(per_seq[0], vec![0]);
        assert_eq!(per_seq[1], vec![0, 1]);
    }

    /// A width-1 sequence has no interior boundary: `commit_accepted_prefix`
    /// short-circuits on a full accept, so there is nothing to restore onto.
    #[test]
    fn width_one_sequence_snapshots_nothing() {
        assert_eq!(hc_verify_snapshot_rows(1).count(), 0);
    }
}
