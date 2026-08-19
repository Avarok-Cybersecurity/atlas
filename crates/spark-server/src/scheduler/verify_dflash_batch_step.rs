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
        match model.decode_verify_batched(&tokens, &ks, &mut seq_refs, 0) {
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

        // Draft TOKENS, not steps — see verify_dflash_step.rs for why the
        // old accept_all/accept_partial pair read as 100% in the TUI.
        let rejected = drafts.len().saturating_sub(num_accepted);
        crate::metrics::SPEC_DECODE_VERIFY
            .with_label_values(&["dflash", "accept"])
            .inc_by(num_accepted as u64);
        if rejected > 0 {
            crate::metrics::SPEC_DECODE_VERIFY
                .with_label_values(&["dflash", "reject"])
                .inc_by(rejected as u64);
        }

        // STree-style in-place SSM commit: h_state is canonical, a partial
        // accept restores intermediate[total_accepted-1].
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, num_accepted + 1, k) {
            tracing::error!("commit_accepted_prefix (dflash batched): {e:#}");
            a.finished = true;
        }
        // Row of THIS sequence's bonus generator in the shared hidden buffer.
        stash_rows.push(off + num_accepted);
    }

    // Park every sequence's bonus hidden in the 32-slot stash while the rows
    // are still live — a single `save_hidden_for_mtp` would keep only the
    // last sequence's row once proposes start overwriting the buffer.
    if let Err(e) = model.stash_verify_hidden_rows(&stash_rows, 0) {
        tracing::warn!("stash_verify_hidden_rows (dflash batched): {e:#}");
    }

    // ── Phase B: per-sequence trim, then re-propose ──
    // Safe to clobber the shared buffers from here on.
    //
    // Propose is now the dominant per-step term (measured 36ms/sequence:
    // 23ms drafter layers + 13ms tail, both weight-bearing) and it is the
    // last piece that still scales linearly with n. `run_mtp_propose_batched`
    // amortises it across the batch when the proposer implements
    // `propose_batch`; it returns Ok(None) otherwise, which drops us onto the
    // per-sequence loop below with no behaviour change.
    let t_propose = std::time::Instant::now();
    let mut proposing: Vec<usize> = Vec::with_capacity(n);
    for (i, a) in batch.iter_mut().enumerate() {
        if a.finished {
            continue;
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, accepted_per_seq[i], 0) {
            tracing::error!("trim_proposer_state (dflash batched): {e:#}");
        }
        // `spec_allowed` MUTATES re-probe state, so it is evaluated exactly
        // once per sequence here and its verdict reused by both arms.
        if crate::scheduler::adaptive_spec::spec_allowed(a, sched) {
            proposing.push(i);
        }
    }

    // The batched drafter forward carries no grammar bitmask, so a
    // grammar-constrained sequence must take the per-sequence arm (which
    // passes its mask) rather than be dropped from both. The caller already
    // keeps grammar sequences out of this batch — this is the belt-and-braces
    // that keeps that a caller CHOICE and not a silent correctness contract.
    let grammarless = proposing.iter().all(|&i| batch[i].grammar_state.is_none());

    let mut batched_ok = false;
    if proposing.len() >= 2 && grammarless {
        let toks: Vec<u32> = proposing.iter().map(|&i| batch[i].last_token).collect();
        let positions: Vec<usize> = proposing.iter().map(|&i| batch[i].seq.seq_len).collect();
        // Stash slot == batch index (Phase A stashed row i into slot i).
        let stash_idx: Vec<usize> = proposing.clone();
        let mut seq_refs: Vec<&mut SequenceState> = Vec::with_capacity(proposing.len());
        for (i, a) in batch.iter_mut().enumerate() {
            if proposing.contains(&i) {
                seq_refs.push(&mut a.seq);
            }
        }
        match model.run_mtp_propose_batched(
            &toks,
            &positions,
            &stash_idx,
            num_drafts,
            &mut seq_refs,
            0,
            None,
        ) {
            Ok(Some(drafts)) if drafts.len() == proposing.len() => {
                for (slot, d) in proposing.iter().zip(drafts.into_iter()) {
                    if !d.is_empty() {
                        batch[*slot].pending_drafts = d;
                    }
                }
                batched_ok = true;
            }
            Ok(_) => {}
            Err(e) => tracing::error!("run_mtp_propose_batched (dflash): {e:#}"),
        }
    }

    if !batched_ok {
        for &i in &proposing {
            let a = &mut batch[i];
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
