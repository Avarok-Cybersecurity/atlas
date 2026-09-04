// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-sequence batched DFlash K=γ verify.
//!
//! The single-sequence [`super::verify_dflash_step::step_verify_dflash`] runs
//! ONE target forward per sequence, so a C=4 round pays four full weight
//! sweeps. Measured 2026-08-19 on qwen3.8-27B+DFlash2: the per-step verify
//! wall is FLAT at ~115 ms from C=1 to C=4 (propose ~40 ms likewise) — i.e.
//! DFlash had zero concurrency amortisation, which is exactly why the
//! throughput gate arbitrates verify away at C>=3 despite it holding ~72%
//! acceptance there.
//!
//! This step packs `n` sequences' `[last_token, d0..d_{γ-1}]` rows into ONE
//! `R = n*(γ+1)`-row forward. Every weight-bearing op (QKVZ / o_proj / FFN /
//! lm_head) reads the weights ONCE for the whole batch; only the GDN
//! recurrent body stays per-sequence (K=γ+1 has no fused WY kernel, so
//! `decode_verify_multi` takes its byte-identical per-sequence fallback —
//! the same arm the single-sequence K=γ path already used). Kernel evidence
//! for the win: `gemm_t` costs 3679 us at M=9 and 3837 us at M=36 — 4x the
//! rows for +4%.
//!
//! PHASE ORDER IS LOAD-BEARING (same hazard as `verify_k4_batch_step`): the
//! forward leaves per-row logits and per-sequence capture bands live in
//! SHARED buffers. Every read of them (accept walk, `commit_ctx`, hidden
//! stash) must complete for ALL sequences before the first `propose`, which
//! overwrites `hidden_states`.

use super::*;

/// Batched DFlash verify for `batch.len()` sequences at uniform K = γ+1.
///
/// `drafts_per_seq` is γ — every sequence must carry exactly that many
/// pending drafts (the caller gates on it; the block drafter has no ragged
/// ladder, unlike the MTP D-Cut path).
pub fn step_verify_dflash_batched(
    model: &dyn Model,
    batch: &mut [&mut ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts_per_seq: usize,
    num_drafts: usize,
    _verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    let n = batch.len();
    let k = drafts_per_seq + 1;
    debug_assert!(n >= 2 && drafts_per_seq >= 1);

    // ONE secondary-stream sync for the whole batch: the previous step's
    // commit/restore must land before this forward reads SSM state.
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary (dflash batched): {e:#}");
        for a in batch.iter_mut() {
            a.finished = true;
        }
        return;
    }

    // Flat seq-major token rows + the per-sequence drafts they encode.
    let mut tokens: Vec<u32> = Vec::with_capacity(n * k);
    let mut drafts_all: Vec<Vec<u32>> = Vec::with_capacity(n);
    for a in batch.iter_mut() {
        let drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
        a.pending_draft_conf.clear();
        debug_assert_eq!(drafts.len(), drafts_per_seq);
        tokens.push(a.last_token);
        tokens.extend_from_slice(&drafts);
        drafts_all.push(drafts);
    }
    let ks = vec![k; n];

    let t_verify = std::time::Instant::now();
    let results = {
        let mut seq_refs: Vec<&mut SequenceState> = batch.iter_mut().map(|a| &mut a.seq).collect();
        // Write-on-accept is OUR request: this step folds every verdict
        // below before any commit, which is what makes the K=4 twin sound.
        let opts = spark_model::traits::VerifyBatchedOpts {
            write_on_accept: true,
        };
        match model.decode_verify_batched(&tokens, &ks, &mut seq_refs, 0, opts) {
            Ok(v) => v,
            Err(e) => {
                // No sequence state was advanced on Err — restore the drafts
                // so the caller's next tick re-verifies them serially rather
                // than losing a step's work.
                tracing::error!("decode_verify_batched (dflash): {e:#}");
                for (a, d) in batch.iter_mut().zip(drafts_all.into_iter()) {
                    a.pending_drafts = d;
                }
                return;
            }
        }
    };
    let verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0;
    if results.len() < n * k {
        tracing::error!(
            "decode_verify_batched (dflash): short result {} < {}",
            results.len(),
            n * k
        );
        for a in batch.iter_mut() {
            a.finished = true;
        }
        return;
    }

    // ── Phase A: per-sequence verdict, ctx commit, emit, SSM commit ──
    // All shared-buffer reads live here, before ANY propose.
    let mut accepted_per_seq: Vec<usize> = Vec::with_capacity(n);
    let mut stash_rows: Vec<usize> = Vec::with_capacity(n);
    let now = Instant::now();
    for (i, a) in batch.iter_mut().enumerate() {
        let off = i * k;
        let verified = &results[off..off + k];
        let drafts = &drafts_all[i];
        a.last_token_time = now;

        // DFlash judges on RAW argmax so the verifier and the drafter share
        // one basis (mirrors the single-sequence path's default arm).
        let mut num_accepted = 0usize;
        for j in 0..drafts.len() {
            if j + 1 >= verified.len() || drafts[j] != verified[j] {
                break;
            }
            num_accepted += 1;
        }
        accepted_per_seq.push(num_accepted);
        crate::scheduler::adaptive_spec::record_verify(a, num_accepted, sched);

        // Rewind the forward's unconditional +k to the accepted prefix plus
        // the bonus slot (identical arithmetic to the single-seq path).
        let pre_verify_len = a.seq.seq_len.saturating_sub(k);
        let target_seq_len = pre_verify_len + num_accepted + 1;
        let to_drop = a.seq.seq_len.saturating_sub(target_seq_len);
        if to_drop > 0 {
            a.seq.seq_len = target_seq_len;
            let pop_n = to_drop.min(a.seq.tokens.len());
            for _ in 0..pop_n {
                a.seq.tokens.pop();
            }
        }

        // Commit this sequence's ctx rows from ITS OWN capture band. The
        // band base is the same `i * kgamma` the model captured into; the
        // model exposes it so the two can never disagree.
        tracing::debug!(
            "CTX_VERIFY slot={} pre_verify_len={} na={} k={} band={}",
            a.seq.slot_idx,
            pre_verify_len,
            num_accepted,
            k,
            i * model.dflash_capture_band(),
        );
        if sched.levers.dflash_unified_ctx
            && let Err(e) = model.commit_ctx(
                &mut a.seq,
                num_accepted + 1,
                pre_verify_len,
                i * model.dflash_capture_band(),
            )
        {
            tracing::error!("commit_ctx (dflash batched): {e:#}");
        }

        for j in 0..num_accepted {
            emit_token(a, drafts[j], None, sched);
            if a.finished {
                break;
            }
        }
        if !a.finished && num_accepted < verified.len() {
            let bonus = verified[num_accepted];
            emit_token(a, bonus, None, sched);
            a.last_token = bonus;
        }

        crate::metrics::SPEC_DECODE_VERIFY
            .with_label_values(&[
                "dflash",
                if num_accepted == drafts.len() {
                    "accept_all"
                } else {
                    "accept_partial"
                },
            ])
            .inc();

        // SSM commit is deferred to after this loop: the write-on-accept
        // fold needs every sequence's verdict in one batched launch first.
        // Row of THIS sequence's bonus generator in the shared hidden buffer.
        stash_rows.push(off + num_accepted);
    }

    // ── Write-on-accept fold (one launch per GDN layer over the batch), then
    // the per-sequence commit (conv restore; h too when nothing folded). ──
    {
        let slots: Vec<usize> = batch.iter().map(|a| a.seq.slot_idx).collect();
        let rows: Vec<u32> = accepted_per_seq.iter().map(|&na| (na + 1) as u32).collect();
        if let Err(e) = model.gdn_fold_accepted(&slots, &rows, k) {
            // The h states of the whole batch are no longer trustworthy
            // (the twin wrote nothing and the fold did not land): finish
            // every sequence, the same verdict a commit error gets.
            tracing::error!("gdn_fold_accepted (dflash batched): {e:#}");
            for a in batch.iter_mut() {
                a.finished = true;
            }
            return;
        }
        for (i, a) in batch.iter_mut().enumerate() {
            let num_accepted = accepted_per_seq[i];
            if let Err(e) = model.commit_accepted_prefix(&mut a.seq, num_accepted + 1, k) {
                tracing::error!("commit_accepted_prefix (dflash batched): {e:#}");
                a.finished = true;
            }
        }
    }

    // Park every sequence's bonus hidden in the 32-slot stash while the rows
    // are still live — a single `save_hidden_for_mtp` would keep only the
    // last sequence's row once proposes start overwriting the buffer.
    if let Err(e) = model.stash_verify_hidden_rows(&stash_rows, 0) {
        tracing::warn!("stash_verify_hidden_rows (dflash batched): {e:#}");
    }

    // ── Phase B: per-sequence trim, then ONE batched re-propose ──
    // Safe to clobber the shared buffers from here on.
    let t_propose = std::time::Instant::now();
    // Trim first for everyone: the batched propose reads each sequence's
    // proposer state, so every state must already reflect what its verify
    // accepted.
    // `spec_allowed` takes &mut, so eligibility is decided in this pass while
    // the mutable borrow is already held.
    let mut eligible = vec![false; batch.len()];
    for (i, a) in batch.iter_mut().enumerate() {
        if a.finished {
            continue;
        }
        let num_accepted = accepted_per_seq[i];
        // NO save_hidden_for_mtp_from_stash here. It stages into a SHARED
        // destination slot, so calling it for every sequence before any
        // propose leaves only the LAST sequence's hidden staged and every
        // other sequence appends the wrong row to its drafter context. The
        // batched arm gets each sequence's hidden from its own stash row
        // (`verify_hidden_stash.offset(i * h * 2)`, built in the dispatch);
        // the per-sequence arm below stages per sequence, immediately before
        // that sequence's own propose, which is the only ordering that works.
        if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
            tracing::error!("trim_proposer_state (dflash batched): {e:#}");
        }
        eligible[i] =
            crate::scheduler::adaptive_spec::spec_allowed(a, sched) && a.grammar_state.is_none();
    }
    // Eligible = not finished, speculation allowed, no grammar (the batched
    // propose is grammarless by contract — a per-position mask cannot be
    // applied to a shared forward).
    let prop_idx: Vec<usize> = (0..batch.len()).filter(|&i| eligible[i]).collect();
    let group_cap = model.mtp_propose_batch_max().max(1);
    // Chunk by the width the proposer DECLARED, not by however many
    // sequences happen to be eligible: handing it more than it said it can
    // carry is how a declared cap becomes a silent lie.
    let mut batched_done = !prop_idx.is_empty();
    for group in prop_idx.chunks(group_cap.max(1)) {
        if group_cap < 2 || group.len() < 2 {
            batched_done = false;
            break;
        }
        let prop_idx: Vec<usize> = group.to_vec();
        let tokens: Vec<u32> = prop_idx.iter().map(|&i| batch[i].last_token).collect();
        let positions: Vec<usize> = prop_idx.iter().map(|&i| batch[i].seq.seq_len).collect();
        let stash_idx: Vec<usize> = prop_idx.clone();
        let result = {
            let mut seq_refs: Vec<&mut SequenceState> = Vec::with_capacity(prop_idx.len());
            for (i, a) in batch.iter_mut().enumerate() {
                if prop_idx.contains(&i) {
                    seq_refs.push(&mut a.seq);
                }
            }
            model.run_mtp_propose_batched(
                &tokens,
                &positions,
                &stash_idx,
                num_drafts,
                &mut seq_refs,
                0,
                None,
            )
        };
        match result {
            Ok(Some(all)) if all.len() == prop_idx.len() => {
                for (g, &row) in prop_idx.iter().enumerate() {
                    if !all[g].is_empty() {
                        batch[row].pending_drafts = all[g].clone();
                    }
                }
            }
            Ok(_) => {
                batched_done = false;
                break;
            }
            Err(e) => {
                tracing::warn!("DFlash batched propose: {e:#} — per-sequence path");
                batched_done = false;
                break;
            }
        }
    }
    if batched_done {
        // Batched path covered every eligible sequence; nothing left to do.
    } else {
        for (i, a) in batch.iter_mut().enumerate() {
            if a.finished || !crate::scheduler::adaptive_spec::spec_allowed(a, sched) {
                continue;
            }
            if let Err(e) = model.save_hidden_for_mtp_from_stash(i, 0) {
                tracing::warn!("save_hidden_for_mtp_from_stash (dflash batched): {e:#}");
            }
            let gmask = mtp_grammar_mask_for(a);
            match model.run_mtp_propose_multi(
                a.last_token,
                a.seq.seq_len,
                num_drafts,
                &mut a.seq,
                0,
                gmask.as_deref(),
            ) {
                Ok(d) if !d.is_empty() => a.pending_drafts = d,
                Ok(_) => {}
                Err(e) => tracing::error!("run_mtp_propose_multi (dflash batched): {e:#}"),
            }
        }
    }

    let total_accepted: usize = accepted_per_seq.iter().sum();
    tracing::info!(
        "DFLASH BATCHED verify: n={n} γ={drafts_per_seq} accepted={total_accepted}/{} ({:.0}%) \
         verify={verify_ms:.1}ms propose={:.1}ms",
        n * drafts_per_seq,
        100.0 * (total_accepted as f64) / ((n * drafts_per_seq) as f64),
        t_propose.elapsed().as_secs_f64() * 1000.0,
    );
}
