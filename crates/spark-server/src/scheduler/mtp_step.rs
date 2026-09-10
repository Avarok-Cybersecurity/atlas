// SPDX-License-Identifier: AGPL-3.0-only

//! MTP speculative draft proposal step.

use super::*;

/// MTP-aware step: bootstrap sequences without drafts, then verify via CUDA graph.
/// Supports K=2 (num_drafts=1) and K=3 (num_drafts=2).
///
/// `verify_ctx` carries the tokenizer special-token IDs the verify
/// pipeline needs (`<think>` / `</think>` / `<tool_call>` /
/// `</tool_call>`). Threaded down to every verify call site so the
/// 8-stage [`crate::scheduler::logit_processors`] pipeline can run on
/// each verify-position's logits — the fix for MTP-emitted tokens
/// bypassing all pre-sample masks. See `verify_pipeline_helper`.
pub fn step_mtp(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    // ATLAS_MTP_TIMING outer bracket: `step_mtp` minus the per-chunk verify
    // guard's TOTAL is the driver's own host prep/tail (classification,
    // bootstrap, D-Cut plan, chunk sort) — one component of the out-of-step
    // GAP. One Instant::now() when disarmed, same cost note as StepTimer.
    let t_step_outer = std::time::Instant::now();
    let mut bootstrap_idxs: Vec<usize> = Vec::new();
    let mut verify_idxs: Vec<usize> = Vec::new();
    for (i, a) in active.iter().enumerate() {
        if !a.pending_drafts.is_empty() {
            verify_idxs.push(i);
        } else {
            bootstrap_idxs.push(i);
        }
    }

    // K-vs-batch ladder (task #35): the per-step draft count is a function
    // of the CURRENT concurrency. Default ladder `4:3,8:3,16:1,32:1` holds
    // 3 drafts through n=8 and drops to 1 through the cap (32), so R =
    // Σ(drafts+1) tops out at the 64-row buffer bound at n=32 (32x2); n=8
    // (8x4) and n=16 (16x2) sit at 32 rows. The depth step-down that used
    // to sit at n>4 was an artifact of the chunk cap below, not of GDN
    // depth cost (see that comment); the one at n>8 is real (16:2 -> 94.1).
    // SSOT + overrides (`ATLAS_MTP_K_LADDER`, `ATLAS_NO_MTP_K_LADDER`):
    // `spark_model::speculative::ladder`. DFlash keeps its own γ economics.
    // Wave 28: at the n=16 rung the draft count is ACCEPT-RATE-AWARE — the
    // static rung cannot win both regimes (prose wants k=1, tool-shaped
    // wants k=2, same binary, same boot). `adaptive_rung::drafts_for`
    // returns the static ladder at every other width.
    let ladder_nd = if dflash_verify_raw_argmax {
        num_drafts
    } else {
        crate::scheduler::adaptive_rung::drafts_for(active.len(), num_drafts)
    };
    // Tiered verify-pool capacity clamp (2026-08-16): the step's draft
    // count must respect the MINIMUM slot capacity across the active
    // sequences — a sequence in a K=2-sized slot must never receive K=4
    // drafts. Capacities are the model's ACTUAL pool geometry
    // (`mtp_slot_draft_capacity`); full-width pools (kill switch, DFlash-γ,
    // pure-attention) report usize::MAX and leave the ladder untouched.
    // See `spec_capacity` for the invariant and its two trigger shapes.
    let ladder_nd = crate::scheduler::spec_capacity::clamp_drafts_to_slot_capacity(
        ladder_nd,
        active
            .iter()
            .map(|a| model.mtp_slot_draft_capacity(a.seq.slot_idx)),
    );

    // ── Phase A: Bootstrap decode for sequences without a draft ──
    if !bootstrap_idxs.is_empty() {
        // The previous verify commit's live-state restore runs async on the
        // secondary stream; order it before the bootstrap decode reads
        // h_state/conv_state (and before start_checkpoint_async snapshots
        // the live state). GPU-side event wait, zero CPU cost.
        if let Err(e) = model.sync_secondary() {
            tracing::error!("bootstrap sync_secondary: {e:#}");
        }
    }
    // Batched form: ONE `decode_batch` for every draftless sequence plus a
    // batched cross-sequence propose, replacing n M=1 weight sweeps of the
    // target and n of the drafter. Falls back to the per-sequence loop below
    // whenever the envelope does not hold (`mtp_bootstrap_step`); kill switch
    // ATLAS_NO_MTP_BATCH_BOOTSTRAP.
    if can_batch_bootstrap(model, sched, bootstrap_idxs.len(), dflash_verify_raw_argmax) {
        step_mtp_bootstrap_batched(model, active, sched, &bootstrap_idxs, ladder_nd, verify_ctx);
        bootstrap_idxs.clear();
    }
    for &idx in &bootstrap_idxs {
        let a = &mut active[idx];

        // DFlash path: skip the standalone M=1 decode. The fused pass already
        // computes every position's logit in one weight sweep, so the "next
        // decoded token" is the bonus token at result[num_accepted] — the logit
        // at the position immediately after the accepted prefix (§8 vLLM
        // bonus-token pattern). Propose initial drafts using the DFlash hidden
        // already captured at row 0 by the previous step's fused pass (or
        // prefill), then route through step_verify_k3/k2 which handles the
        // fused forward, accept/reject, bonus-token emit, and re-propose for
        // the next step. This replaces the two-sweep sequence (M=1 decode here
        // + M=1+k fused in Phase B) with a single M=1+k fused sweep.
        if dflash_verify_raw_argmax
            && !sched.levers.dflash_seam_serial
            && crate::scheduler::adaptive_spec::spec_allowed(a, sched)
        {
            let eff = if a.grammar_state.is_some() {
                1
            } else {
                num_drafts
            };
            let _gmask = mtp_grammar_mask_for(a);
            match model.run_mtp_propose_multi(
                a.last_token,
                a.seq.seq_len,
                eff,
                &mut a.seq,
                0,
                _gmask.as_deref(),
            ) {
                Ok(init) if !init.is_empty() => {
                    if eff >= 3 && init.len() >= 3 {
                        step_verify_k4(
                            model,
                            a,
                            sched,
                            &init,
                            num_drafts,
                            verify_ctx,
                            dflash_verify_raw_argmax,
                        );
                    } else if eff >= 2 && init.len() >= 2 {
                        step_verify_k3(
                            model,
                            a,
                            sched,
                            &init,
                            num_drafts,
                            verify_ctx,
                            dflash_verify_raw_argmax,
                        );
                    } else {
                        step_verify_k2(
                            model,
                            a,
                            sched,
                            &init,
                            num_drafts,
                            verify_ctx,
                            dflash_verify_raw_argmax,
                        );
                    }
                    continue;
                }
                Ok(_) => {
                    tracing::warn!(
                        "DFlash bootstrap propose returned empty; falling back to standalone decode"
                    );
                }
                Err(e) => {
                    tracing::error!("DFlash bootstrap propose: {e:#}");
                }
            }
            // Rare fallback: propose failed or returned empty (e.g. drafter not
            // yet primed). Fall through to the standalone decode below so the
            // sequence emits its next token rather than stalling.
        }

        // Non-DFlash path (or DFlash-propose fallback): EP broadcast + standalone decode.
        // EP: broadcast token to worker before decode (worker runs decode in lockstep).
        if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, a.last_token) {
            tracing::error!("EP broadcast bootstrap token: {e:#}");
            a.finished = true;
            continue;
        }
        let logits = match model.decode(a.last_token, &mut a.seq, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("bootstrap decode error: {e:#}");
                a.finished = true;
                continue;
            }
        };
        // Build the seq's configured penalties (rep/presence/frequency/LZ/DRY)
        // so the MTP bootstrap token sees the SAME penalties+history the
        // non-MTP path applies — the root-cause fix for repetition_penalty /
        // dry_multiplier never reaching MTP-emitted tokens. Cloned before the
        // mutable `grammar_state` borrow to satisfy the borrow checker.
        let penalties = crate::scheduler::sample_step::penalty_params_for(
            a,
            crate::scheduler::sample_step::PositionKind::Verify,
            0.0,
            None,
            Vec::new(),
        );
        // #192: same per-tool-call-segment scoping as the main pipeline
        // (`penalty_history_scope`) so MTP bootstrap tokens see the identical
        // penalty landscape.
        let history = crate::scheduler::sample_step::penalty_history_scope(
            &a.output_tokens,
            a.tool_call_end_token,
        )
        .to_vec();
        // P1-4 (2026-07-09): the bootstrap token is one of only two
        // stochastic sample points under MTP, and its stochastic branch
        // previously sampled with a hardcoded `min_p: 0.0` deep inside
        // `sample_token_with_grammar` — bypassing the MODEL.toml
        // `min_p_floor` (0.05 on this family) that exists precisely to stop
        // FP8/NVFP4 argmax-flip tail tokens. The sampler now reads
        // `penalties.min_p`, which `penalty_params_for` copies from
        // `a.min_p` (request value + floor, resolved in `sampling_setup`) —
        // SSOT, no new channel. Kill-switch: ATLAS_NO_MTP_MINP=1.
        let tok = match sample_token_with_grammar(
            model,
            logits,
            a.temperature,
            a.top_k,
            a.top_p,
            &[],
            a.grammar_state.as_mut(),
            &penalties,
            &history,
            &sched.levers.sampling(),
        ) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("bootstrap sample error: {e:#}");
                a.finished = true;
                continue;
            }
        };

        // Extract logprobs from bootstrap decode logits (single position).
        let lp = if let Some(k) = a.top_logprobs {
            extract_single_logprobs(model, logits, tok, k)
        } else {
            None
        };

        emit_token(a, tok, lp, sched);
        if a.finished {
            continue;
        }
        a.last_token = tok;
        // Adaptive speculation: count serial tokens toward the re-probe window.
        crate::scheduler::adaptive_spec::tick_serial(a, sched);

        // Ctx-holes fix (ATLAS_DFLASH_SERIAL_APPEND=1), COMPLEMENT-GATED:
        // the serial ctx-append fires iff propose() will NOT run this
        // iteration, so append and propose decode-append can never both
        // cover one token — double-append impossible by construction
        // (that was the cuMemcpyDtoDAsync status-1 crash).
        // `spec_allowed` is evaluated exactly once (it mutates re-probe
        // state); its verdict is reused for the propose gate below.
        // Exception — re-probe RESUME: the token decoded on the un-suspend
        // iteration would otherwise fall in a hole (the stale
        // `skip_next_decode_append` set by the last suspended token makes
        // the propose below skip its decode-append). Append it here; the
        // skip flag this sets is consumed by that propose — one append,
        // no duplicate, seam covered.
        let was_suspended = crate::scheduler::adaptive_spec::is_suspended(a, sched);
        let will_propose = crate::scheduler::adaptive_spec::spec_allowed(a, sched);
        let reprobe_resume = was_suspended && will_propose;
        if sched.levers.dflash_unified_ctx {
            // Unified ctx commit: same complement-gate as the old serial
            // append — fire iff propose() will NOT run (or re-probe resume),
            // so commit and propose decode-append never both cover a token.
            if !will_propose || reprobe_resume {
                let base_pos = a.seq.seq_len.saturating_sub(1);
                if let Err(e) = model.commit_ctx(&mut a.seq, 1, base_pos, 0) {
                    tracing::error!("commit_ctx (mtp serial): {e:#}");
                }
            }
        } else if sched.levers.dflash_serial_append
            && (!will_propose || reprobe_resume)
            && let Err(e) = model.dflash_serial_ctx_append(&mut a.seq)
        {
            tracing::error!("dflash_serial_ctx_append: {e:#}");
        }

        if let Err(e) = model.save_hidden_for_mtp(0, 0) {
            tracing::error!("save_hidden_for_mtp: {e:#}");
            continue;
        }
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        // BUG#4 (2026-06-02): when a grammar is active, generate only ONE draft.
        // run_mtp_propose_multi (mtp_multi.rs) masks only draft[0] with the
        // position-0 bitmask and leaves draft[1..] UNMASKED, so multi-draft +
        // grammar desyncs — a draft[1] token can violate its true per-position
        // mask, get verified+accepted, then be refused by the matcher later
        // (→ truncation). A single draft uses its own up-to-date mask and is
        // sound; drafts.len()==1 routes verify to the K=2 path. Mask is a no-op
        // when grammar is inactive, so NVFP4/non-tool paths keep full K.
        // 2026-07-09: hoisted to the `effective_drafts_under_grammar` SSOT,
        // now also applied at the five verify-path re-propose sites that
        // previously bypassed this clamp (the "mask held fixed" warn spam).
        // Composed with the K-vs-batch ladder: the bootstrap propose is
        // sized for the current concurrency so the next verify is uniform
        // at the ladder width (no surplus drafts to truncate).
        let effective_num_drafts =
            crate::scheduler::spec_step::effective_drafts_under_grammar(a, ladder_nd);
        // Adaptive speculation: a suspended seq skips proposing entirely and
        // stays on this serial bootstrap path until the re-probe fires.
        // (`will_propose` is the single spec_allowed evaluation above.)
        if will_propose {
            match model.run_mtp_propose_multi(
                tok,
                a.seq.seq_len,
                effective_num_drafts,
                &mut a.seq,
                0,
                _mtp_grammar_mask.as_deref(),
            ) {
                Ok(drafts) if !drafts.is_empty() => {
                    tracing::debug!("MTP bootstrap: tok={tok} → drafts={drafts:?}");
                    a.pending_drafts = drafts;
                }
                Ok(_) => tracing::warn!("MTP propose returned empty"),
                Err(e) => {
                    tracing::error!("run_mtp_propose_multi: {e:#}");
                }
            }
        }

        if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
            tracing::error!("bootstrap start_checkpoint_async: {e:#}");
        }
    }

    // ── Phase B: Verify with pipelined checkpoint ──
    //
    // Batched multi-seq K-row verify (batched-MTP E11 + the ladder). Only
    // reachable when `ATLAS_MTP_MAX_SEQS > 1` (default 32 with the ladder)
    // puts >= 2 verify-ready sequences in one step (`ATLAS_MTP_MAX_SEQS=1`
    // ⇒ this partition is a no-op and every seq takes the per-seq loop
    // below, byte-identical to the pre-batched HEAD). Batchable =
    // grammarless, non-DFlash, >= ladder_nd pending drafts (surplus from a
    // ladder step-down is truncated — the same draft-tail truncation the
    // grammar-boundary path already does; `after_verify`'s
    // `last_num_drafted` trim contract stays consistent). The model
    // additionally self-gates (non-EP, non-HSS, no LoRA) via
    // `can_batch_verify(&ks)`. Kill switch `ATLAS_NO_MTP_BATCH_VERIFY`
    // (PRESENCE check) forces the serialized loop for A/B.
    let mut serial_idxs: Vec<usize> = Vec::new();
    let mut batchable_idxs: Vec<usize> = Vec::new();
    // ── DFlash: cross-sequence batched K=γ verify ──
    // The block drafter has no ragged ladder, so the batch is the set of
    // grammarless sequences carrying the SAME γ drafts; anything else falls
    // to the per-sequence step. One R=n*(γ+1)-row forward replaces n full
    // weight sweeps (the per-step verify wall was flat ~115ms per SEQUENCE
    // from C=1..4 before this). Kill switch: ATLAS_DFLASH_BATCH_VERIFY=0.
    // One-shot attribution for "why is the batched path not running": each
    // of these four is individually capable of silently keeping every
    // sequence on the per-sequence verify, which reads as "no concurrency
    // amortisation" rather than as a disabled feature.
    {
        static WHY: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        WHY.get_or_init(|| {
            tracing::debug!(
                raw_argmax = dflash_verify_raw_argmax,
                n_verify = verify_idxs.len(),
                lever = sched.levers.dflash_batch_verify,
                killed = batch_verify_disabled(),
                "DFlash batched verify gate (first tick with drafts)"
            );
        });
    }
    if dflash_verify_raw_argmax
        && verify_idxs.len() >= 2
        && sched.levers.dflash_batch_verify
        && !batch_verify_disabled()
    {
        let mut gamma = 0usize;
        for &idx in &verify_idxs {
            let a = &active[idx];
            let g = a.pending_drafts.len();
            if a.grammar_state.is_some() || g < 1 {
                serial_idxs.push(idx);
            } else if gamma == 0 || g == gamma {
                gamma = g;
                batchable_idxs.push(idx);
            } else {
                serial_idxs.push(idx);
            }
        }
        // Widest group the model accepts. `can_batch_verify` bounds R = n*k
        // by the verify row budget (96 rows), so at γ=7 the cap is 12
        // sequences — and this gate used to be ALL-OR-NOTHING: a C=16 tick
        // asked for 16×8 = 128 rows, was refused, and every sequence fell to
        // the per-sequence serial loop, which is exactly the flat
        // no-amortization wall the batched path exists to remove (measured
        // 2026-08-21: the C=16 sweep cell ROLLED OVER below C=8, 63.5 vs
        // 65.9 aggregate). Chunking to the widest accepted group batches
        // 12 + 4 instead of serializing 16 — the same shape the batched
        // PROPOSE already uses (`pending.chunks(group_cap)`).
        let k_rows = gamma + 1;
        let mut m_max = batchable_idxs.len();
        while m_max >= 2 && !model.can_batch_verify(&vec![k_rows; m_max]) {
            m_max -= 1;
        }
        if batchable_idxs.len() >= 2 && m_max >= 2 {
            // SLOT ORDER, not arrival order. The batched conv+WY GDN path
            // requires the batch to sit on CONSECUTIVE ssm-pool slots in
            // batch order (checked on the pointers in
            // trait_decode_batched_conv_gdn_multi.rs). The pool freelist is
            // LIFO, so the first fill hands out slots 0..n in arrival order
            // and engages, but every later fill re-issues released slots in
            // reverse finish order and the same batch DECLINED to the
            // per-sequence loop (~47% of steps engaged in a two-round C=16
            // prose bench, 2026-09-02). Drafts travel inside each
            // `ActiveSeq`, so batch order is free to the caller. Sequences
            // without a slot sort last; the model re-checks the layout and
            // declines safely if a gap remains.
            let mut by_slot = batchable_idxs.clone();
            by_slot.sort_by_key(|&i| (active[i].seq.ssm_slot_idx().unwrap_or(usize::MAX), i));
            for chunk in by_slot.chunks(m_max) {
                if chunk.len() >= 2 {
                    // Borrow in ascending active-index order (one forward
                    // walk), then restore the chunk's slot order.
                    let mut asc: Vec<usize> = chunk.to_vec();
                    asc.sort_unstable();
                    let mut tagged: Vec<(usize, &mut ActiveSeq)> = Vec::with_capacity(asc.len());
                    let mut it = active.iter_mut();
                    let mut consumed = 0usize;
                    for &i in &asc {
                        let a = it.nth(i - consumed).expect("verify index within active");
                        consumed = i + 1;
                        let pos = chunk.iter().position(|&c| c == i).expect("chunk member");
                        tagged.push((pos, a));
                    }
                    tagged.sort_by_key(|t| t.0);
                    let mut refs: Vec<&mut ActiveSeq> = tagged.into_iter().map(|t| t.1).collect();
                    step_verify_dflash_batched(
                        model, &mut refs, sched, gamma, num_drafts, verify_ctx,
                    );
                } else {
                    // A trailing single cannot batch — it takes the serial
                    // loop below, same as before this gate existed.
                    serial_idxs.extend_from_slice(chunk);
                }
            }
        } else {
            // Loud on the FIRST decline only: a silently-refused gate looks
            // exactly like "the batched path is off", which cost a whole
            // validation cycle when k=γ-1+1 failed an over-strict width check.
            if !batchable_idxs.is_empty() && sched.stats.once("log:dflash_batch_decline") {
                tracing::info!(
                    "DFlash batched verify DECLINED: n={} k={} (model.can_batch_verify said no) \
                     — running the per-sequence loop",
                    batchable_idxs.len(),
                    gamma + 1,
                );
            }
            serial_idxs.extend_from_slice(&batchable_idxs);
        }
        batchable_idxs.clear();
        for &idx in &serial_idxs {
            let a = &mut active[idx];
            let mut drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
            a.pending_draft_conf.clear();
            if drafts.is_empty() {
                continue;
            }
            if let Some(ref mut gs) = a.grammar_state {
                let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
                drafts.truncate(kept);
                if drafts.is_empty() {
                    continue;
                }
            }
            step_verify_dflash(
                model,
                a,
                sched,
                &drafts,
                num_drafts,
                verify_ctx,
                dflash_verify_raw_argmax,
            );
        }
        // Same StepOuter record the shared tail makes — this arm returns
        // early, and dropping it would blind the phase telemetry exactly
        // where the batched path is meant to show its win.
        sched
            .timing
            .record(crate::scheduler::mtp_timing::Phase::StepOuter, t_step_outer);
        return;
    }
    if verify_idxs.len() >= 2
        && spark_model::speculative::mtp_multi_seq_mode()
        && !dflash_verify_raw_argmax
        && !batch_verify_disabled()
        && ladder_nd >= 1
    {
        for &idx in &verify_idxs {
            let a = &mut active[idx];
            if a.grammar_state.is_none() && a.pending_drafts.len() >= ladder_nd {
                if a.pending_drafts.len() > ladder_nd {
                    a.pending_drafts.truncate(ladder_nd);
                }
                batchable_idxs.push(idx);
            } else {
                serial_idxs.push(idx);
            }
        }
    } else {
        serial_idxs.extend_from_slice(&verify_idxs);
    }
    let rows = ladder_nd + 1;

    // ── D-Cut: per-sequence verify depth from drafter confidence ──
    // Default ON at ratio 0.75 (+2.6% at C=8; kill switch `ATLAS_NO_MTP_DCUT`,
    // PRESENCE). Ranks every prunable draft position ACROSS the batch by its
    // prefix-product survival score and keeps the top `ATLAS_MTP_DCUT_RATIO`
    // fraction (`mtp_dcut`). The retained set is a per-sequence PREFIX by
    // construction, so the only downstream effect is a RAGGED row count. OFF
    // (or `ladder_nd < 2`, or the batch wider than `dcut_width_cap()` = 8 —
    // the D-Cut-at-depth policy: pruning the 16:2 rung's n=16 measured -9%)
    // ⇒ `ks` is the uniform ladder shape and everything below reduces to the
    // pre-D-Cut path exactly.
    let ks = mtp_dcut::plan(active, &mut batchable_idxs, ladder_nd, rows);
    // The width the ASSIGNMENT was gated on (`plan` reorders `batchable_idxs`,
    // never resizes it): the per-chunk re-ordering below must ask the gate with
    // THIS width, never the chunk's, or a chunked batch could take the opposite
    // arm from the one its depths were assigned under.
    let batch_n = batchable_idxs.len();

    // Chunking: the 96-row buffer bound `can_batch_verify` enforces, with the
    // per-chunk sequence cap DERIVED from it (`VERIFY_ROW_BUDGET / widest
    // rows` — chunk_ranges): rows=4 → 24 seqs, rows=3 → 32 (R=96 at n=32,
    // the 32:2 depth-at-width shape), rows=2 → 48 (n is separately bounded
    // at 32). Default-ladder shapes chunk exactly as before (every default
    // rung's row total already fit the old 64-row budget in one chunk).
    //
    // History — the SAME stale-cap artifact, twice: rows=4 was once capped at
    // 4 seqs, which split 8 batchable sequences into TWO serialized 4-wide
    // verify forwards (2x the weight reads per step) and is what the
    // "8:3 collapses" measurements (57.9, 62.6 on 2026-07-28) actually
    // recorded — NOT depth-3 at width 8. Then rows=3/4 stayed hardcoded at
    // 8 seqs after the budget widened 32→64, which serialized every depth
    // shape above n=8 the same way (fixer r2 2026-07-30: a 16:2 env-ladder
    // leg read `n=8 k_drafts=2` in its accept telemetry — two chunks). The
    // cap is now anchored to the row budget so the artifact class is closed.
    for (lo, hi) in mtp_dcut::chunk_ranges(&ks) {
        let chunk = &batchable_idxs[lo..hi];
        let chunk_ks = &ks[lo..hi];
        if chunk.len() >= 2 && model.can_batch_verify(chunk_ks) {
            // Collect disjoint &mut refs — the iterator walk requires ASCENDING
            // indices, so sort a copy of the chunk before walking and restore
            // the batch order (with each sequence's k) immediately after.
            let mut asc: Vec<(usize, usize)> = chunk
                .iter()
                .copied()
                .zip(chunk_ks.iter().copied())
                .collect();
            asc.sort_unstable();
            let mut refs: Vec<(&mut ActiveSeq, usize)> = Vec::with_capacity(chunk.len());
            let mut it = active.iter_mut();
            let mut consumed = 0usize;
            for &(i, k) in &asc {
                let a = it.nth(i - consumed).expect("chunk index within active");
                consumed = i + 1;
                refs.push((a, k));
            }
            // Batch order, from the ONE ordering rule shared with the graph
            // key (`verify_key`), asked with the SAME width gate `plan` used so
            // order and assignment can never disagree. Canonical (n >=
            // `CANONICAL_KEY_MIN_WIDTH` = 8): ssm slots ascending = also
            // deepest-first under the canonical assignment, so the key is a
            // function of the depth MULTISET not its arrangement (266 keys → 3
            // at n=8). When the selected pool slots have no gaps, each depth
            // run owns a consecutive slot block for the batched-GDN fast path;
            // fragmented runs are checked and declined by the model. Below the
            // gate, deepest-first then slot — the
            // pre-canonical order byte for byte. Idempotent on `plan`'s ordered
            // batch under both arms; it still runs because `plan` returns the
            // batch UNORDERED whenever D-Cut declines, and that uniform-`k`
            // case is the pre-D-Cut sort by slot. PERMUTATION ONLY — depths
            // stay attached to the sequence
            // `plan` truncated for (`verify_k4_batch_step` pins
            // `drafts + 1 == ks[i]`). Verdicts are index-mapped inside the
            // step, so batch order is free to the caller.
            let chunk_slots: Vec<usize> = refs
                .iter()
                .map(|(a, _)| a.seq.ssm_slot_idx().unwrap_or(usize::MAX))
                .collect();
            let chunk_depths: Vec<usize> = refs.iter().map(|&(_, k)| k).collect();
            let order = spark_model::speculative::verify_key::verify_batch_permutation(
                &chunk_slots,
                &chunk_depths,
                spark_model::speculative::verify_key::canonical_assignment(batch_n),
            );
            let sorted_ks: Vec<usize> = order.iter().map(|&p| chunk_depths[p]).collect();
            let mut slotted: Vec<Option<&mut ActiveSeq>> =
                refs.into_iter().map(|(a, _)| Some(a)).collect();
            let mut batch: Vec<&mut ActiveSeq> = order
                .iter()
                .map(|&p| {
                    slotted[p]
                        .take()
                        .expect("verify_batch_permutation is a permutation")
                })
                .collect();
            step_verify_k4_batched(model, &mut batch, sched, &sorted_ks, ladder_nd, verify_ctx);
        } else {
            // Model can't batch this width (or a lone leftover): fall back
            // to the existing per-seq dispatch for these sequences.
            serial_idxs.extend_from_slice(chunk);
        }
    }
    for &idx in &serial_idxs {
        let a = &mut active[idx];
        let mut drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
        // Confidences describe the taken drafts; clearing here is the single
        // place the two vectors are kept in lock-step for the serial path.
        a.pending_draft_conf.clear();
        if drafts.is_empty() {
            continue;
        }

        // Spec-decode boundary awareness (arXiv:2512.15834): when a
        // grammar is active, validate the draft sequence against the
        // matcher and truncate at the first token that crosses a
        // grammar transition. Without this, a draft span that crosses
        // `</function>` (or any other structural boundary) gets
        // accepted by the verifier and emitted, but the post-emit
        // `accept_token` silently fails — desync'ing the grammar
        // from the output stream. Truncating here downgrades K=4 →
        // K=3 → K=2 cleanly.
        if let Some(ref mut gs) = a.grammar_state {
            let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
            if kept < drafts.len() {
                drafts.truncate(kept);
            }
            if drafts.is_empty() {
                continue;
            }
        }

        // DFlash γ-block drafters return ≥4 drafts per step (γ=16 typical).
        // The K=2/3/4 graphed paths are MTP-shaped and don't generalize past
        // K=4 cleanly, so γ-block verify routes through `step_verify_dflash`.
        // MTP keeps using the existing graphed paths; this dispatch is purely
        // additive.
        if drafts.len() >= 4 {
            step_verify_dflash(
                model,
                a,
                sched,
                &drafts,
                num_drafts,
                verify_ctx,
                dflash_verify_raw_argmax,
            );
        } else if num_drafts >= 3 && drafts.len() >= 3 {
            step_verify_k4(
                model,
                a,
                sched,
                &drafts,
                num_drafts,
                verify_ctx,
                dflash_verify_raw_argmax,
            );
        } else if num_drafts >= 2 && drafts.len() >= 2 {
            step_verify_k3(
                model,
                a,
                sched,
                &drafts,
                num_drafts,
                verify_ctx,
                dflash_verify_raw_argmax,
            );
        } else {
            step_verify_k2(
                model,
                a,
                sched,
                &drafts,
                num_drafts,
                verify_ctx,
                dflash_verify_raw_argmax,
            );
        }
    }
    sched
        .timing
        .record(crate::scheduler::mtp_timing::Phase::StepOuter, t_step_outer);
}
