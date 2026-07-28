// SPDX-License-Identifier: AGPL-3.0-only

//! Batched multi-sequence K-row verify step (batched-MTP E10 + the
//! K-vs-batch ladder, task #35).
//!
//! Runs the n weight-reading verify forwards of a chunk of verify-ready
//! sequences as ONE R = n*(k+1)-row forward (`decode_verify_batched`,
//! seq-major rows `r = i*rows + j`), then applies each sequence's verdict
//! with the EXISTING single-seq machinery (`k4_apply_verdict`, K-generic).
//! `k_drafts` (drafts per sequence this step) comes from the ladder —
//! `speculative::ladder::mtp_ladder_drafts(active.len())` — 3 at n<=4
//! (today's proven K=4 regime), 2 at n<=8, 1 at n<=16, keeping R <= 32.
//!
//! Phase ordering is load-bearing (shared-buffer clobber hazard,
//! `mtp_multi.rs:165`): every drafter propose writes the shared
//! `hidden_states` buffer and its lm_head writes the shared `logits`
//! buffer, so ALL row reads (pipeline picks + logprobs, Phase 1) and the
//! accepted-row hidden stash (Phase 2) MUST complete for every sequence
//! before ANY sequence's verdict/propose runs (Phase 3).
//!
//! Reachability: only via `step_mtp` Phase B when `ATLAS_MTP_MAX_SEQS > 1`
//! (default 16 with the ladder) puts >= 2 verify-ready grammarless
//! sequences holding `k_drafts` drafts in one step AND the model says
//! `can_batch_verify(n, k_drafts+1)`. `ATLAS_MTP_MAX_SEQS=1` keeps this
//! path dead and the single-seq path byte-unchanged.

use super::*;

/// Kill switch `ATLAS_NO_MTP_BATCH_VERIFY` — PRESENCE check per the
/// house convention (`=0` is NOT off): any set value forces the
/// serialized per-seq verify loop at n > 1 for A/B against the batched
/// forward.
pub(super) fn batch_verify_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_NO_MTP_BATCH_VERIFY").is_ok())
}

/// Batched K-row verify for `batch.len() >= 2` sequences, each holding
/// exactly `k_drafts` pending drafts. Caller (Phase B in `mtp_step.rs`)
/// guarantees: grammarless, non-DFlash, uniform `pending_drafts.len() ==
/// k_drafts` (ladder-truncated), chunk size <= the per-rows cap, batch
/// sorted by ssm slot (canonical graph key), and
/// `model.can_batch_verify(batch.len(), k_drafts+1)` true.
///
/// `k_drafts` is also the re-propose width (Phase 4): the next round's
/// drafts are sized for the CURRENT concurrency, so a ladder step-change
/// truncates at most one round's surplus.
pub(super) fn step_verify_k4_batched(
    model: &dyn Model,
    batch: &mut [&mut ActiveSeq],
    k_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    let n = batch.len();
    let rows = k_drafts + 1;
    debug_assert!((2..=16).contains(&n) && (2..=4).contains(&rows) && n * rows <= 32);

    // ATLAS_MTP_TIMING step summary (same Drop-guard pattern as the
    // single-seq step; one timer for the whole batched step).
    let _step_timer = crate::scheduler::mtp_timing::StepTimer::new(batch[0].seq.seq_len);

    // One secondary-stream sync for the whole batch: orders the previous
    // verify commits' async live-state restores before this forward reads
    // h_state/conv_state (same semantics as the per-seq entry sync).
    if let Err(e) = model.sync_secondary() {
        tracing::error!("batched-verify sync_secondary: {e:#}");
        for a in batch.iter_mut() {
            a.finished = true;
        }
        return;
    }

    // Per-seq verify rows [last_verified, d0, .., d_{k-1}], flat seq-major.
    // Drafts are taken here (not by the driver) so a batch-level bail
    // leaves no half-taken state behind.
    let mut drafts_per_seq: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut tokens: Vec<u32> = Vec::with_capacity(n * rows);
    for a in batch.iter_mut() {
        let d = std::mem::take(&mut a.pending_drafts);
        debug_assert!(
            d.len() == k_drafts,
            "batchable classification requires exactly {k_drafts} drafts"
        );
        tokens.push(a.last_token);
        tokens.extend_from_slice(&d);
        drafts_per_seq.push(d);
    }

    // ── ONE batched verify forward: R = n*rows rows, weights read once ──
    // Contract (`decode_verify_batched`): on Ok every seq has tokens+=rows
    // and seq_len+=rows (verdict rewind below is caller arithmetic, same as
    // the per-seq path); on Err NO host seq state advanced.
    let t_verify = Instant::now();
    let results: Vec<u32> = {
        let mut seq_refs: Vec<&mut SequenceState> = batch.iter_mut().map(|a| &mut a.seq).collect();
        match model.decode_verify_batched(&tokens, rows, &mut seq_refs, 0) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("decode_verify_batched (n={n} rows={rows}): {e:#}");
                for a in batch.iter_mut() {
                    a.finished = true;
                }
                return;
            }
        }
    };
    let verify_us = t_verify.elapsed().as_micros();

    // ── Phase 1: consume every sequence's logits rows BEFORE any propose ──
    // Rows for seq i live at row_base = i*rows in the shared logits buffer.
    let mut verdicts: Vec<(Vec<u32>, usize, Vec<crate::api::TokenLogprobs>)> =
        Vec::with_capacity(n);
    for (i, a) in batch.iter_mut().enumerate() {
        let r = &results[i * rows..(i + 1) * rows];
        // Full pre-sample pipeline per verify position, reading this
        // sequence's rows (row_base = i*rows) — same 8-stage semantics as
        // the single-seq MTP path.
        let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model,
            r,
            a,
            verify_ctx,
            i * rows,
        );
        let v: Vec<u32> = (0..rows)
            .map(|j| processed.get(j).copied().unwrap_or(r[j]))
            .collect();
        let drafts = &drafts_per_seq[i];
        let mut num_accepted = 0usize;
        while num_accepted < k_drafts && drafts[num_accepted] == v[num_accepted] {
            num_accepted += 1;
        }
        // Unconditional per-position draft match (same counters as the
        // single-seq step) — scored before the accept chain short-circuits.
        // The positional telemetry is K=4-shaped (p1/p2/p3); ladder steps
        // with fewer drafts record outcomes only (`k4_record_outcome`).
        if k_drafts == 3 {
            k4_record_positional(
                drafts[0] == v[0],
                drafts[1] == v[1],
                drafts[2] == v[2],
                a.seq.seq_len,
            );
        }
        let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
            extract_verify_logprobs(model, &v, top_logprobs, i * rows)
        } else {
            Vec::new()
        };
        a.last_token_time = Instant::now();
        verdicts.push((v, num_accepted, verify_lps));
    }

    // ── Phase 2: stash the accepted-position hiddens before any propose ──
    // Absolute forward row i*rows + num_accepted_i → stash slot i; the
    // verdict proposes below overwrite the live rows, so
    // `save_hidden_for_mtp` must read from the stash instead.
    let stash_rows: Vec<usize> = verdicts
        .iter()
        .enumerate()
        .map(|(i, &(_, num_accepted, _))| i * rows + num_accepted)
        .collect();
    if let Err(e) = model.stash_verify_hidden_rows(&stash_rows, 0) {
        // Degraded, not fatal: verdicts still apply; each seq's
        // save-from-stash will fail and skip only its re-propose.
        tracing::error!("stash_verify_hidden_rows: {e:#}");
    }

    // ── Phase 3: per-seq verdict via the EXISTING single-seq machinery ──
    // (greedy accept + rewind arithmetic + emit + trim; propose is DEFERRED
    // so Phase 4 can batch it across sequences — the per-seq drafter forward
    // reads ~850 MB of BF16 drafter weights.)
    for (i, (a, (v, num_accepted, verify_lps))) in
        batch.iter_mut().zip(verdicts.into_iter()).enumerate()
    {
        k4_apply_verdict(
            model,
            a,
            &drafts_per_seq[i],
            &v,
            verify_lps,
            k_drafts,
            num_accepted,
            K4Hidden::DeferPropose,
            verify_us,
        );
    }

    // ── Phase 4: batched cross-sequence propose, chunked by 4 ──
    // Sequences still alive after their verdict need fresh drafts. The
    // batched propose reads each sequence's accepted-position hidden
    // straight from its stash slot. `can_propose_batch` caps a propose
    // batch at 4 sequences (drafter meta staging envelope), so wider
    // batches run in groups of <= 4; singles and unsupported groups fall
    // back per-seq (re-saving the stash slot into the single-slot MTP
    // input buffer immediately before each propose).
    let t_propose = Instant::now();
    let pending: Vec<usize> = (0..n)
        .filter(|&i| !batch[i].finished && batch[i].pending_drafts.is_empty())
        .collect();
    if pending.is_empty() {
        return;
    }
    let mut need_fallback: Vec<usize> = Vec::new();
    let mut groups_batched = 0usize;
    if pending.len() >= 2 && !batch_propose_disabled() {
        for group in pending.chunks(4) {
            if group.len() < 2 {
                need_fallback.extend_from_slice(group);
                continue;
            }
            let tokens: Vec<u32> = group.iter().map(|&i| batch[i].last_token).collect();
            let positions: Vec<usize> = group.iter().map(|&i| batch[i].seq.seq_len).collect();
            let stash_idx: Vec<usize> = group.to_vec();
            let result = {
                let mut seq_refs: Vec<&mut SequenceState> = Vec::with_capacity(group.len());
                let mut it = batch.iter_mut();
                let mut prev = 0usize;
                for (j, &i) in group.iter().enumerate() {
                    let step = if j == 0 { i } else { i - prev - 1 };
                    let a = it.nth(step).expect("group index in batch");
                    seq_refs.push(&mut a.seq);
                    prev = i;
                }
                model.run_mtp_propose_batched(
                    &tokens,
                    &positions,
                    &stash_idx,
                    k_drafts,
                    &mut seq_refs,
                    0,
                )
            };
            match result {
                Ok(Some(all)) => {
                    for (j, &i) in group.iter().enumerate() {
                        if !all[j].is_empty() {
                            batch[i].pending_drafts = all[j].clone();
                        }
                    }
                    groups_batched += 1;
                }
                Ok(None) => need_fallback.extend_from_slice(group),
                Err(e) => {
                    // Failed mid-chain: `last_num_drafted` tracks exactly the
                    // drafter rows written, so `after_verify` stays consistent
                    // — but a SECOND (fallback) propose on top would append
                    // more rows than the next trim accounts for. Skip
                    // proposing this group this step; the affected sequences
                    // decode serially next step.
                    tracing::error!("run_mtp_propose_batched: {e:#}");
                }
            }
        }
    } else {
        need_fallback = pending.clone();
    }
    for &i in &need_fallback {
        let a = &mut batch[i];
        if let Err(e) = model.save_hidden_for_mtp_from_stash(i, 0) {
            tracing::error!("save_hidden_for_mtp_from_stash({i}): {e:#}");
            continue;
        }
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            a.last_token,
            a.seq.seq_len,
            k_drafts,
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {}
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
            }
        }
    }
    tracing::debug!(
        "K{rows} batched propose: n={} groups_batched={groups_batched} fallback={} propose={}μs",
        pending.len(),
        need_fallback.len(),
        t_propose.elapsed().as_micros()
    );
}

/// Kill switch `ATLAS_NO_MTP_BATCH_PROPOSE` — PRESENCE check (`=0` is NOT
/// off): forces the per-seq propose fallback inside the batched verify step,
/// for A/B attribution of the propose-batching sub-lever.
fn batch_propose_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_NO_MTP_BATCH_PROPOSE").is_ok())
}
