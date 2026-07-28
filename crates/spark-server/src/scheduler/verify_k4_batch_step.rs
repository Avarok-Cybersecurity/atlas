// SPDX-License-Identifier: AGPL-3.0-only

//! Batched multi-sequence K=4 verify step (batched-MTP E10).
//!
//! Runs the n weight-reading verify forwards of a chunk of verify-ready
//! sequences as ONE eager R = n*4-row forward (`decode_verify_batched_k4`,
//! seq-major rows `r = i*4 + j`), then applies each sequence's verdict with
//! the EXISTING single-seq machinery (`k4_apply_verdict`).
//!
//! Phase ordering is load-bearing (shared-buffer clobber hazard,
//! `mtp_multi.rs:165`): every drafter propose writes the shared
//! `hidden_states` buffer and its lm_head writes the shared `logits`
//! buffer, so ALL row reads (pipeline picks + logprobs, Phase 1) and the
//! accepted-row hidden stash (Phase 2) MUST complete for every sequence
//! before ANY sequence's verdict/propose runs (Phase 3).
//!
//! Reachability: only via `step_mtp` Phase B when `ATLAS_MTP_MAX_SEQS > 1`
//! puts >= 2 verify-ready grammarless K=4 sequences in one step AND the
//! model says `can_batch_verify_k4(n)`. Default cap = 1 keeps this path
//! dead and the single-seq path byte-unchanged.

use super::*;

/// Kill switch `ATLAS_NO_MTP_BATCH_VERIFY` — PRESENCE check per the
/// house convention (`=0` is NOT off): any set value forces the
/// serialized per-seq verify loop at n > 1 for A/B against the batched
/// forward.
pub(super) fn batch_verify_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_NO_MTP_BATCH_VERIFY").is_ok())
}

/// Batched K=4 verify for `batch.len() >= 2` sequences, each holding exactly
/// 3 pending drafts. Caller (Phase B in `mtp_step.rs`) guarantees: grammarless,
/// non-DFlash, `num_drafts >= 3`, chunk size <= 4, and
/// `model.can_batch_verify_k4(batch.len())` true.
pub(super) fn step_verify_k4_batched(
    model: &dyn Model,
    batch: &mut [&mut ActiveSeq],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    let n = batch.len();
    debug_assert!((2..=4).contains(&n));

    // ATLAS_MTP_TIMING step summary (same Drop-guard pattern as the
    // single-seq step; one timer for the whole batched step).
    let _step_timer = crate::scheduler::mtp_timing::StepTimer::new(batch[0].seq.seq_len);

    // One secondary-stream sync for the whole batch: orders the previous
    // verify commits' async live-state restores before this forward reads
    // h_state/conv_state (same semantics as the per-seq entry sync).
    if let Err(e) = model.sync_secondary() {
        tracing::error!("batched-k4 sync_secondary: {e:#}");
        for a in batch.iter_mut() {
            a.finished = true;
        }
        return;
    }

    // Per-seq verify rows [last_verified, d0, d1, d2]. Drafts are taken
    // here (not by the driver) so a batch-level bail leaves no half-taken
    // state behind.
    let mut drafts_per_seq: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut tokens: Vec<[u32; 4]> = Vec::with_capacity(n);
    for a in batch.iter_mut() {
        let d = std::mem::take(&mut a.pending_drafts);
        debug_assert!(d.len() >= 3, "batchable classification requires 3 drafts");
        tokens.push([a.last_token, d[0], d[1], d[2]]);
        drafts_per_seq.push(d);
    }

    // ── ONE batched verify forward: R = n*4 rows, weights read once ──
    // Contract (`decode_verify_batched_k4`): on Ok every seq has tokens+=4
    // and seq_len+=4 (verdict rewind below is caller arithmetic, same as
    // the per-seq path); on Err NO host seq state advanced.
    let t_verify = Instant::now();
    let results: Vec<[u32; 4]> = {
        let mut seq_refs: Vec<&mut SequenceState> =
            batch.iter_mut().map(|a| &mut a.seq).collect();
        match model.decode_verify_batched_k4(&tokens, &mut seq_refs, 0) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("decode_verify_batched_k4 (n={n}): {e:#}");
                for a in batch.iter_mut() {
                    a.finished = true;
                }
                return;
            }
        }
    };
    let verify_us = t_verify.elapsed().as_micros();

    // ── Phase 1: consume every sequence's logits rows BEFORE any propose ──
    // Rows for seq i live at row_base = i*4 in the shared logits buffer.
    let mut verdicts: Vec<([u32; 4], usize, Vec<crate::api::TokenLogprobs>)> =
        Vec::with_capacity(n);
    for (i, a) in batch.iter_mut().enumerate() {
        let r = results[i];
        // Full pre-sample pipeline per verify position, reading this
        // sequence's rows (row_base = i*4) — same 8-stage semantics as the
        // single-seq MTP path.
        let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model,
            &r,
            a,
            verify_ctx,
            i * 4,
        );
        let v = [
            processed.first().copied().unwrap_or(r[0]),
            processed.get(1).copied().unwrap_or(r[1]),
            processed.get(2).copied().unwrap_or(r[2]),
            processed.get(3).copied().unwrap_or(r[3]),
        ];
        let drafts = &drafts_per_seq[i];
        let num_accepted = if drafts[0] != v[0] {
            0
        } else if drafts[1] != v[1] {
            1
        } else if drafts[2] != v[2] {
            2
        } else {
            3
        };
        // Unconditional per-position draft match (same counters as the
        // single-seq step) — scored before the accept chain short-circuits.
        k4_record_positional(
            drafts[0] == v[0],
            drafts[1] == v[1],
            drafts[2] == v[2],
            a.seq.seq_len,
        );
        let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
            extract_verify_logprobs(model, &v, top_logprobs, i * 4)
        } else {
            Vec::new()
        };
        a.last_token_time = Instant::now();
        verdicts.push((v, num_accepted, verify_lps));
    }

    // ── Phase 2: stash the accepted-position hiddens before any propose ──
    // Absolute forward row i*4 + num_accepted_i → stash slot i; the verdict
    // proposes below overwrite the live rows, so `save_hidden_for_mtp` must
    // read from the stash (K4Hidden::Stash) instead.
    let rows: Vec<usize> = verdicts
        .iter()
        .enumerate()
        .map(|(i, &(_, num_accepted, _))| i * 4 + num_accepted)
        .collect();
    if let Err(e) = model.stash_verify_hidden_rows(&rows, 0) {
        // Degraded, not fatal: verdicts still apply; each seq's
        // save-from-stash will fail and skip only its re-propose.
        tracing::error!("stash_verify_hidden_rows: {e:#}");
    }

    // ── Phase 3: per-seq verdict via the EXISTING single-seq machinery ──
    // (greedy accept + rewind arithmetic + emit + trim; propose is DEFERRED
    // so Phase 4 can batch it across sequences — the per-seq drafter forward
    // reads ~850 MB of BF16 drafter weights, so n x num_drafts serial
    // forwards were ~62 ms of the ~180 ms C=4 step.)
    for (i, (a, (v, num_accepted, verify_lps))) in
        batch.iter_mut().zip(verdicts.into_iter()).enumerate()
    {
        k4_apply_verdict(
            model,
            a,
            &drafts_per_seq[i],
            v,
            verify_lps,
            num_drafts,
            num_accepted,
            K4Hidden::DeferPropose,
            verify_us,
        );
    }

    // ── Phase 4: ONE batched cross-sequence propose ──
    // Sequences still alive after their verdict need fresh drafts. The
    // batched propose reads each sequence's accepted-position hidden straight
    // from its stash slot; per-seq fallback re-saves the slot into the
    // single-slot MTP input buffer immediately before each propose (the
    // verdict loop no longer saves, and one slot cannot hold n hiddens).
    let t_propose = Instant::now();
    let pending: Vec<usize> = (0..n)
        .filter(|&i| !batch[i].finished && batch[i].pending_drafts.is_empty())
        .collect();
    if pending.is_empty() {
        return;
    }
    let mut batched_done = false;
    if pending.len() >= 2 && !batch_propose_disabled() {
        let tokens: Vec<u32> = pending.iter().map(|&i| batch[i].last_token).collect();
        let positions: Vec<usize> = pending.iter().map(|&i| batch[i].seq.seq_len).collect();
        let stash_idx: Vec<usize> = pending.clone();
        let result = {
            let mut seq_refs: Vec<&mut SequenceState> = Vec::with_capacity(pending.len());
            let mut it = batch.iter_mut();
            let mut prev = 0usize;
            for (j, &i) in pending.iter().enumerate() {
                let step = if j == 0 { i } else { i - prev - 1 };
                let a = it.nth(step).expect("pending index in batch");
                seq_refs.push(&mut a.seq);
                prev = i;
            }
            model.run_mtp_propose_batched(
                &tokens,
                &positions,
                &stash_idx,
                num_drafts,
                &mut seq_refs,
                0,
            )
        };
        match result {
            Ok(Some(all)) => {
                for (j, &i) in pending.iter().enumerate() {
                    if !all[j].is_empty() {
                        batch[i].pending_drafts = all[j].clone();
                    }
                }
                batched_done = true;
            }
            Ok(None) => {}
            Err(e) => {
                // Failed mid-chain: `last_num_drafted` tracks exactly the
                // drafter rows written, so `after_verify` stays consistent —
                // but a SECOND (fallback) propose on top would append more
                // rows than the next trim accounts for. Skip proposing this
                // step; the affected sequences decode serially next step.
                tracing::error!("run_mtp_propose_batched: {e:#}");
                batched_done = true;
            }
        }
    }
    if !batched_done {
        for &i in &pending {
            let a = &mut batch[i];
            if let Err(e) = model.save_hidden_for_mtp_from_stash(i, 0) {
                tracing::error!("save_hidden_for_mtp_from_stash({i}): {e:#}");
                continue;
            }
            let _mtp_grammar_mask = mtp_grammar_mask_for(a);
            match model.run_mtp_propose_multi(
                a.last_token,
                a.seq.seq_len,
                num_drafts,
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
    }
    tracing::debug!(
        "K4 batched propose: n={} batched={batched_done} propose={}μs",
        pending.len(),
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
