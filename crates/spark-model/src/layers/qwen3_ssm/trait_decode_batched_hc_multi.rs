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

        let stage_timing = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                std::env::var("ATLAS_HC_VERIFY_STAGE_TIMING").as_deref() == Ok("1")
            })
        };
        let mark = |t: &mut std::time::Instant, acc: &mut u128| {
            if stage_timing {
                let _ = ctx.gpu.synchronize(stream);
                *acc += t.elapsed().as_micros();
                *t = std::time::Instant::now();
            }
        };
        let mut tk = std::time::Instant::now();
        let (mut us_ple, mut us_pre_a, mut us_gdn, mut us_post_a,
             mut us_pre_f, mut us_ffn, mut us_post_f) = (0u128, 0u128, 0u128, 0u128, 0u128, 0u128, 0u128);

        // Same refusal the other three hc bodies carry: `hc_norm` inside
        // `hc_pre` replaces the fused gate-f32 norm, so ATLAS_FP32_ROUTING
        // would have the router read the PREVIOUS layer's activations.
        anyhow::ensure!(
            !self.ffn.fp32_routing_active(ctx.levers),
            "qwen3_ssm mHC multi-seq verify: ATLAS_FP32_ROUTING needs the fused \
             gate-f32 norm, which the highway path replaces. Unset it."
        );

        // Provable engagement, once per process. Without this the arm is
        // selected by a silent `&&` chain upstream
        // (`can_batch_verify_dispatch`), and a decline is indistinguishable
        // from a pass: the per-sequence verify produces the SAME answers, so
        // known-answer probes go 4/4 either way and the only visible
        // difference is throughput — which is exactly how a fast path that
        // fails closed gets recorded as a working one.
        {
            static ON: std::sync::Once = std::sync::Once::new();
            ON.call_once(|| {
                tracing::info!(
                    "mHC cross-sequence batched verify ACTIVE: {n_seqs} seqs, \
                     ks={ks:?} ({rows} rows) in one highway pass \
                     (ATLAS_HC_BATCH_VERIFY=1; unset restores the per-seq verify)"
                );
            });
        }

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

        mark(&mut tk, &mut us_ple);

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
        mark(&mut tk, &mut us_pre_a);

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
        mark(&mut tk, &mut us_gdn);
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

        mark(&mut tk, &mut us_post_a);

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
        mark(&mut tk, &mut us_pre_f);
        // ── MoE: ONE token-major call at a PADDED width ──
        //
        // History, all measured on this path:
        //  * one call at R rows, unpadded: `hc_ffn_dispatch` has fused arms
        //    only at 1/2/3 rows and falls to `Prefill` — the grouped GEMM that
        //    streams ALL 512 experts regardless of row count. 3793 us, 90% of
        //    a 4182 us layer.
        //  * per SEQUENCE (ks[i]=3 -> fused K3): 14.99 -> 27.44 tok/s. But the
        //    weight-heavy op then runs n times, which is what the control
        //    already does — no amortisation, just less waste.
        //  * one call at R=11 via `forward_token_major_decode`: 9492 us, WORSE
        //    than 6541 for the per-sequence loop. 11 is a width that arm never
        //    sees: `padded_batch_n` yields 2, 4, 8, 12, 16, 24 ... and the
        //    decode path only ever hands it those.
        //
        //  * one call at a PADDED width, which is what this arm does. It was
        //    the default on the strength of R=12 (1163 us against the
        //    per-sequence loop's 1216) and that was a MISTAKE: end-to-end it
        //    costs 15.4% at C=2 (48.90 -> 56.42 tok/s with it off) and buys
        //    nothing at C=4 (56.65 -> 58.35, ranges overlapping). The reason
        //    is `padded_batch_n`, which has no 6 — a C=2 verify is R=6 and
        //    pads to 8, where token-major measures 1151 us against 608 for
        //    two fused K3 calls. NOW OPT-IN (`ATLAS_HC_VERIFY_MOE_PADDED=1`).
        //
        // Pad rows compute garbage from whatever `hidden` holds above row R;
        // only rows [0, R) are consumed below, exactly as the decode path
        // relies on for its own padded batches, and VERIFY_ROW_CAP (96) keeps
        // them in bounds.
        // -- One-shot MoE cost-vs-rows sweep (ATLAS_MOE_ROW_SWEEP=1) --
        // See the module note: at C=4 the control batches its sequences into
        // ONE 4-row MoE call, so MTP pays cost(R rows) to earn tok_step
        // tokens per sequence. This prints the curve that decides it.
        {
            static SWEPT: std::sync::Once = std::sync::Once::new();
            let on = {
                static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *ON.get_or_init(|| std::env::var("ATLAS_MOE_ROW_SWEEP").as_deref() == Ok("1"))
            };
            if on && !self.ffn.is_none() {
                let mut err: Option<anyhow::Error> = None;
                SWEPT.call_once(|| {
                    let run = || -> anyhow::Result<()> {
                        for &w in &[1usize, 2, 3, 4, 6, 8, 12, 16, 24] {
                            // Two arms per width where each is legal: the fused
                            // ladder the verify uses today, and the token-major
                            // decode kernel the padded arm uses.
                            for arm in ["ladder", "token_major"] {
                                if arm == "token_major" && w < 4 {
                                    continue;
                                }
                                let call = || -> anyhow::Result<()> {
                                    if arm == "ladder" {
                                        self.hc_small_m_ffn(hidden, w, ctx, stream)
                                    } else {
                                        self.ffn.forward_token_major_decode(hidden, w, ctx, stream)
                                    }
                                };
                                // Warm: first touch of a width pays plan setup.
                                let mut ok = true;
                                for _ in 0..3 {
                                    if let Err(e) = call() {
                                        tracing::info!(rows = w, arm, error = %e, "MoE row sweep: arm REFUSED");
                                        ok = false;
                                        break;
                                    }
                                }
                                if !ok {
                                    continue;
                                }
                                ctx.gpu.synchronize(stream)?;
                                let t = std::time::Instant::now();
                                const ITERS: usize = 20;
                                for _ in 0..ITERS {
                                    call()?;
                                }
                                ctx.gpu.synchronize(stream)?;
                                let us = t.elapsed().as_micros() as f64 / ITERS as f64;
                                tracing::info!(
                                    rows = w,
                                    arm,
                                    us_per_call = us,
                                    us_per_row = us / w as f64,
                                    "MoE row sweep"
                                );
                            }
                        }
                        Ok(())
                    };
                    if let Err(e) = run() {
                        err = Some(e);
                    }
                });
                if let Some(e) = err {
                    return Err(e);
                }
            }
        }

        // OPT-IN (`=1`). Default OFF: measured a 15.4% LOSS at C=2 and nothing
        // at C=4 — see the module note. `padded_batch_n` has no 6, so a C=2
        // verify (R=6) pads to 8, the arm's worst width.
        let moe_padded = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                std::env::var("ATLAS_HC_VERIFY_MOE_PADDED").as_deref() == Ok("1")
            })
        };
        let moe_rows = if moe_padded && !self.ffn.is_none() && rows > 1 {
            let padded = crate::traits::padded_batch_n(rows);
            {
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| {
                    tracing::info!(rows, padded, "hc verify MoE: ONE token-major call, padded");
                });
            }
            self.ffn
                .forward_token_major_decode(hidden, padded, ctx, stream)?;
            ctx.buffers.moe_output()
        } else {
            // Per-sequence fallback: ks[i] rows each, which lands on the fused
            // K3 arm on the usual ladder. Every arm writes `moe_output()` at
            // rows [0, k), so each result is staged into `norm_output()` at its
            // batch offset; `norm_output` is free here because this body sends
            // `hc_pre`'s collapse to `hidden`.
            let stage = ctx.buffers.norm_output();
            let mut off = 0usize;
            for i in 0..n_seqs {
                let k = ks[i];
                self.hc_small_m_ffn(hidden.offset(off * h * 2), k, ctx, stream)?;
                ctx.gpu.copy_d2d_async(
                    ctx.buffers.moe_output(),
                    stage.offset(off * h * 2),
                    k * h * 2,
                    stream,
                )?;
                off += k;
            }
            stage
        };
        mark(&mut tk, &mut us_ffn);
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            moe_rows,
            streams,
            post,
            comb,
            streams,
            n,
            h as u32,
            stream,
        )?;
        mark(&mut tk, &mut us_post_f);
        if stage_timing {
            // PERIODIC, not one-shot: a `Once` here only ever sampled the
            // first call of the process, which is cold (kernel load + plan
            // setup) and misreports the FFN by ~20x. Every 1024th call is
            // ~21 steps at 48 layers — past warmup, and still resampling.
            use std::sync::atomic::{AtomicUsize, Ordering};
            static CALLS: AtomicUsize = AtomicUsize::new(0);
            let call = CALLS.fetch_add(1, Ordering::Relaxed);
            if call.is_multiple_of(1024) && call > 0 {
                tracing::info!(
                    call,
                    rows,
                    ple_us = us_ple as u64,
                    hc_pre_attn_us = us_pre_a as u64,
                    gdn_block_us = us_gdn as u64,
                    hc_post_attn_us = us_post_a as u64,
                    hc_pre_ffn_us = us_pre_f as u64,
                    moe_ffn_us = us_ffn as u64,
                    hc_post_ffn_us = us_post_f as u64,
                    "hc multi-seq GDN body stage split (ONE layer, synced per stage)"
                );
            }
        }
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
