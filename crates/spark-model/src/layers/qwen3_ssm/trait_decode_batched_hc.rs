// SPDX-License-Identifier: AGPL-3.0-only

//! K-row (ONE sequence, K tokens) batched GDN decode under an mHC highway —
//! the speculative-verify body.
//!
//! # This is not the N-sequence path
//!
//! `trait_decode_multi_seq/hc.rs` is N sequences x 1 token: every row owns
//! its own recurrent state and the rows are independent, so it runs the
//! recurrence per row in a loop. THIS file is 1 sequence x K tokens, where
//! row `t+1`'s state depends on row `t`'s and the state at every row boundary
//! has to be materialised for `commit_accepted_prefix` to rewind onto. The
//! two share only the highway bracketing.
//!
//! # Why it exists
//!
//! `model/trait_impl/verify_hc.rs` had to run every layer through `prefill()`,
//! because `decode_batched` refuses under the highway
//! (`refuse_batched_under_hc`). Two costs followed:
//!
//! * the GDN body ran the CHUNK SCAN (`prefill_block`) for K rows, measured at
//!   862 us per layer per verify row — 36 layers x 862 us ~= 31 ms/row; and
//! * the chunk scan writes NO `h_state_intermediates`, which is the contract
//!   `commit_accepted_prefix` rewinds from, so a partial accept copied
//!   never-written pool memory into 36 layers of live recurrent state. The fix
//!   in e53b78427 publishes that state explicitly from K SINGLE-ROW passes —
//!   correct, but K passes.
//!
//! The batched conv+GDN kernels (`trait_decode_batched_conv_gdn*.rs`) already
//! do both jobs: they advance the recurrence over K rows in decode-shaped
//! kernels AND write `h_state_intermediates[t]` / `conv_state_intermediates[t]`
//! for `t in 0..K-1` as a side effect — exactly `commit_rewind_index`'s range.
//! They were unreachable only because they sat inside `decode_batched_inner`'s
//! RESIDUAL bracket.
//!
//! # The shape
//!
//! `decode_batched_inner` is now `residual bracket + decode_batched_block`,
//! and `decode_batched_block` (steps 2-9) touches neither `hidden` nor
//! `residual`. So this file is `prefill_inner_hc`'s bracket with
//! `prefill_block` swapped for `decode_batched_block`:
//!
//! ```text
//! hc_expand(hidden -> streams)              # MODEL layer 0 only
//! PLE forward over K rows                   # rolling window, one call
//! hc_pre(streams, attn_site) -> hidden
//!   decode_batched_block(hidden, K) -> moe_output   # writes intermediates
//! hc_post(moe_output, streams) -> streams
//! hc_pre(streams, ffn_site)  -> hidden
//!   ffn(hidden)              -> moe_output
//! hc_post(moe_output, streams) -> streams
//! ```
//!
//! # The three per-row carries
//!
//! 1. **SSM `h_state` / `conv_state`** — written by the conv+GDN kernels into
//!    the pool intermediates, natively, at the row granularity
//!    `commit_accepted_prefix` reads. Nothing to publish by hand.
//! 2. **PLE's rolling conv/history window** — ONE `PleLayer::forward` call
//!    over K rows, byte-for-byte the call the K-row mini-prefill already made
//!    (`trait_prefill_hc.rs`), with `fresh = false`: a verify never starts a
//!    sequence. PLE lives on exactly one model layer.
//! 3. **QSA `ingested` / `pooled` marks** — NOT this layer's. They advance on
//!    the 12 full-attention layers, which keep running `prefill()` for the
//!    same K rows, and `verify_hc_rows` still aligns them to `seq_len` before
//!    the pass.

use super::*;

impl Qwen3SsmLayer {
    /// K-row batched GDN decode under the highway. `state` is the ONE
    /// sequence's layer state; the recurrence walks all `num_tokens` rows.
    pub(super) fn decode_batched_inner_hc(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decode_batched_inner_hc without mHC weights"))?;
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_tokens as u32;

        // Same refusal the other two hc bodies carry: `hc_norm` inside
        // `hc_pre` replaces the fused gate-f32 norm, so ATLAS_FP32_ROUTING
        // would have the router read the PREVIOUS layer's activations.
        anyhow::ensure!(
            !self.ffn.fp32_routing_active(),
            "qwen3_ssm mHC batched verify: ATLAS_FP32_ROUTING needs the fused \
             gate-f32 norm, which the highway path replaces. Unset it."
        );

        // Mixed steps park the chunk's highway rows above the decode rows;
        // verify runs at offset 0, but honour it rather than assume it.
        let streams = ctx
            .buffers
            .hc_streams()
            .offset(ctx.hc_row_offset * hc.hc_mult * h * 4);
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();

        // ── The row-exact contract for THIS pass ──
        //
        // `hc_pre` is the FIRST place a K-row verify stops reproducing K
        // decode rows, and it is not in the GDN body at all. Its low-rank
        // collapse runs three cuBLASLt GEMMs at M = num_tokens
        // (`hyper_connection_lowrank_gemm.rs::hc_pre_gemm`), and cuBLASLt
        // picks its kernel per M — so row 0 of an M=2 collapse is not the
        // bits the M=1 decode collapse produced. MEASURED on
        // qwen3.8-flash-next (native EXL3, gamma=1, first verify pass, SSM
        // layer 0, sum|x| over row 0): the highway going IN is bit-identical
        // (37.252170 both arms), `hc_pre`'s output is not — 1479.630222
        // batched vs 1479.306446 for the one-row decode body — and every
        // stage after it inherits that. `hc_post` needs no such treatment:
        // it is one elementwise kernel at grid=[T], row-parallel by
        // construction.
        //
        // So under the row-exact contract the two `hc_pre` sites run once
        // per row at n = 1, at that row's bases: streams by `hc_row`, the
        // block input by `h` BF16, and the injection vector by `hc_mult`
        // FP32 — the same `[T, ...]` layouts the K-row call writes, so the
        // K-row `hc_post` that follows reads them unchanged.
        let row_exact = crate::layers::qwen3_ssm::verify_row_exact_leg(
            ctx.gdn_exact_replay,
            crate::layers::qwen3_ssm::RowExactLeg::HcPre,
        );
        let hc_row = hc.hc_mult * h * 4;
        anyhow::ensure!(
            !row_exact
                || matches!(
                    ops::HcVariant::of(hc),
                    ops::HcVariant::LowRank
                ),
            "row-exact mHC verify: the per-row hc_pre passes ONE `comb` for every \
             row, which only the low-rank variant (which never writes it) admits. \
             A Sinkhorn site here would have its rows clobber each other — refuse \
             rather than corrupt. Set ATLAS_NO_VERIFY_ROW_HC to run the K-row \
             collapse instead."
        );
        // One closure so the attn and ffn sites cannot drift apart.
        let hc_pre_rows = |site: &crate::layers::qwen3_attention::HcSiteWeights| -> Result<()> {
            if !row_exact {
                return ops::hc_pre_site(
                    ctx.gpu,
                    self.hc_pre_k,
                    streams,
                    site,
                    hc,
                    hidden,
                    post,
                    comb,
                    ctx.buffers.hc_lowrank_scratch(),
                    n,
                    h as u32,
                    eps,
                    stream,
                );
            }
            // Only the projections need M=1. The other three stages are
            // row-independent, so share their launches across the batch.
            if !ops::hc_decode_split_forced()
                && !ctx.buffers.hc_lowrank_scratch().is_null()
                && let Some(w) = &site.lowrank
            {
                return ops::hc_pre_gemm(
                    ctx.gpu,
                    streams,
                    w,
                    hidden,
                    post,
                    ctx.buffers.hc_lowrank_scratch(),
                    n,
                    h as u32,
                    hc.hc_mult as u32,
                    eps,
                    true,
                    true,
                    true,
                    stream,
                );
            }
            for t in 0..num_tokens {
                ops::hc_pre_site(
                    ctx.gpu,
                    self.hc_pre_k,
                    streams.offset(t * hc_row),
                    site,
                    hc,
                    hidden.offset(t * h * 2),
                    post.offset(t * hc.hc_mult * 4),
                    comb,
                    ctx.buffers.hc_lowrank_scratch(),
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
            Ok(())
        };

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

        // ── Carry 2: PLE's rolling window, ONE ROW AT A TIME ──
        //
        // This is the per-TOKEN analogue of the per-SEQ mini-loop in
        // `trait_decode_multi_seq/hc.rs`, and it is deliberately NOT the
        // K-row `PleLayer::forward` the mini-prefill uses.
        //
        // WHY. A K-row `forward` advances conv + history for all K rows in one
        // shot, so the only snapshot point available is PRE-verify. A partial
        // accept then leaves the carry ADVANCED over the rejected rows, and
        // unlike QSA's contiguous marks NOTHING about PLE can be rebuilt by
        // truncation: `conv` is a rolling FP32 state and `history` is a fixed
        // window whose oldest ids have already rolled off. That is the
        // measured degeneration class. Running row by row costs K launches on
        // the ONE layer that carries PLE — a rounding error against 36 layers
        // of GDN — and buys a real checkpoint at every row boundary a commit
        // can land on.
        //
        // `fresh` is implicitly false: a verify never starts a sequence, so
        // both the conv state and the token history carry in from the
        // committed prefix.
        if let Some(ple) = self.ple.as_ref() {
            let host = ctx.host_token_ids.ok_or_else(|| {
                anyhow::anyhow!("hc batched verify: PLE needs host_token_ids threaded")
            })?;
            anyhow::ensure!(
                host.len() >= num_tokens,
                "hc batched verify: {} host ids for {num_tokens} rows",
                host.len()
            );
            let ssm = state
                .as_any_mut()
                .downcast_mut::<crate::layer::SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("PLE host layer state is not SsmLayerState"))?;
            let st = ssm
                .ple
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("PLE batched verify before prefill: no seq state"))?;
            ple.begin_verify_rows(st);
            for t in 0..num_tokens {
                ple.forward_row(st, streams.offset(t * hc_row), &host[t..t + 1], ctx, stream)?;
                // Row boundaries a partial accept can land on are exactly
                // `0..num_tokens-1` — the same range as `hc_publish_rows`,
                // because `commit_accepted_prefix` short-circuits on a full
                // accept and reads `num_accepted - 1` otherwise.
                if hc_verify_snapshot_rows(num_tokens).contains(&t) {
                    ple.push_verify_row(st, ctx.gpu, stream)?;
                }
            }
        }

        super::debug::hc_stage_probe(
            ctx,
            "bat_ple",
            streams,
            hc.hc_mult * h,
            true,
            stream,
        );

        // ── GDN sublayer. `hidden` is scratch; the highway is the residual. ──
        hc_pre_rows(&hc.attn)?;
        super::debug::hc_stage_probe(ctx, "bat_pre_attn", hidden, h, false, stream);
        // ── Carry 1: the recurrence + its per-row intermediates ──
        let out_proj_buf = self.decode_batched_block(
            hidden,
            num_tokens,
            super::trait_decode_batched::GdnStates::Single(state),
            ctx,
            stream,
        )?;
        super::debug::hc_stage_probe(ctx, "bat_ssm_out", out_proj_buf, h, false, stream);
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
        hc_pre_rows(&hc.ffn)?;
        // Same small-M substitution the mini-prefill body makes, and for the
        // same reason: `forward_prefill` routes the MoE through the grouped
        // GEMM, which streams all 512 experts' weights regardless of row
        // count (2700 us -> 191 us per row, measured). All four arms write
        // `moe_output()`, so this is a kernel-shape choice, not a math change.
        super::debug::hc_stage_probe(ctx, "bat_pre_ffn", hidden, h, false, stream);
        self.hc_small_m_ffn(hidden, num_tokens, ctx, stream)?;
        super::debug::hc_stage_probe(
            ctx,
            "bat_moe_out",
            ctx.buffers.moe_output(),
            h,
            false,
            stream,
        );
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
        super::debug::hc_stage_probe(ctx, "bat_end", streams, hc.hc_mult * h, true, stream);
        Ok(())
    }

    /// Row-count-shaped MoE dispatch for the mHC verify bodies, all arms
    /// writing `ctx.buffers.moe_output()`.
    ///
    /// `ATLAS_QWEN4EXP_HC_SMALL_M_FFN=0` restores the grouped-GEMM path for an
    /// A/B. Shared by `prefill_inner_hc` and `decode_batched_inner_hc` so the
    /// two verify bodies cannot drift apart on the FFN.
    pub(super) fn hc_small_m_ffn(
        &self,
        rows: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let small_m = {
            static SMALL_M: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *SMALL_M
                .get_or_init(|| std::env::var("ATLAS_QWEN4EXP_HC_SMALL_M_FFN").as_deref() != Ok("0"))
        };
        // ROW-EXACT (`ForwardContext::gdn_exact_replay`, kill switch
        // ATLAS_NO_VERIFY_ROW_EXACT): every arm below dispatches on ROW COUNT
        // — `forward_k2` fuses both rows' expert GEMVs into 5 launches where
        // `forward` runs 5 per row — and row-count-shaped MoE arms round
        // differently from the single-row one (#459). A verify whose row 0
        // re-processes an already-committed token has to reproduce that
        // token's serial `decode()`, and decode's MoE is `ffn.forward` at ONE
        // row. So under the row-exact contract the FFN runs one row at a time.
        //
        // DESCENDING, and that is load-bearing: `forward` always writes ROW 0
        // of `moe_output()`, so each row's result has to be moved aside before
        // the next call overwrites it. The FFN is stateless across rows, so
        // visiting them K-1..0 lets every row but 0 be copied out immediately
        // and leaves row 0's result in the place it already belongs — no
        // staging buffer, no lost row.
        if crate::layers::qwen3_ssm::verify_row_exact_leg(
            ctx.gdn_exact_replay,
            crate::layers::qwen3_ssm::RowExactLeg::Ffn,
        ) {
            let h = ctx.config.hidden_size;
            let moe_out = ctx.buffers.moe_output();
            for t in (0..num_tokens).rev() {
                let out = self.ffn.forward(rows.offset(t * h * 2), ctx, stream)?;
                anyhow::ensure!(
                    out == moe_out,
                    "row-exact FFN: single-token MoE returned a buffer other than \
                     moe_output(), which the hc post-site reads unconditionally"
                );
                if t > 0 {
                    ctx.gpu
                        .copy_d2d_async(moe_out, moe_out.offset(t * h * 2), h * 2, stream)?;
                }
            }
            return Ok(());
        }
        match num_tokens {
            1 if small_m => {
                let out = self.ffn.forward(rows, ctx, stream)?;
                anyhow::ensure!(
                    out == ctx.buffers.moe_output(),
                    "small-M FFN: single-token MoE returned a buffer other than \
                     moe_output(), which the hc post-site reads unconditionally"
                );
            }
            2 if small_m => self.ffn.forward_k2(rows, ctx, stream)?,
            3 if small_m => self.ffn.forward_k3(rows, ctx, stream)?,
            _ => self.ffn.forward_prefill(rows, num_tokens, ctx, stream)?,
        }
        Ok(())
    }
}

/// `ATLAS_QWEN4EXP_MTP_HC_BATCHED=1` arms the K-row batched GDN verify under
/// the highway (this file). DEFAULT OFF: the e53b78427 per-row path
/// (`verify_hc_rows` once per token + `publish_verify_row_state`) stays the
/// reference until this one is proven equal-or-better on correctness AND
/// speed, and stays in the tree as the A/B arm either way.
pub(crate) fn hc_batched_verify_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_HC_BATCHED").as_deref() == Ok("1"))
}

/// The verify row boundaries the PLE carry must be snapshotted at, for a
/// verify of width `k`.
///
/// SSOT-paired with `verify_hc::hc_publish_rows` (the SSM carry's range) and
/// with `async_chkpt::commit_rewind_index` (what a commit actually reads).
/// All three describe the same fact: a partial accept commits `1..k` rows and
/// lands on index `num_accepted - 1`, so boundaries `0..k-1` are reachable and
/// the last row's is not — a full accept keeps the live carry.
/// `hc_ple_snapshot_range_matches_the_ssm_one` pins the agreement.
pub(crate) const fn hc_verify_snapshot_rows(k: usize) -> std::ops::Range<usize> {
    0..k.saturating_sub(1)
}
