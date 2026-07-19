// SPDX-License-Identifier: AGPL-3.0-only

//! K=2 verify step.

use super::*;

// Periodic ACCEPT/REJECT summary counters (P4, 2026-05-24).
// Replaces the earlier `is_multiple_of(50)` per-step gate which hid
// accept events behind seq_len happenstance. We now log a single
// summary line every K2_SUMMARY_PERIOD verify steps and reset.
// Atomics are fine: scheduler runs on a single thread today, but
// future multi-scheduler builds remain race-free.
const K2_SUMMARY_PERIOD: u64 = 100;
static K2_ACCEPTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static K2_REJECTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[inline]
fn k2_record_outcome(accepted: bool, seq_len: usize) {
    let counter = if accepted { &K2_ACCEPTS } else { &K2_REJECTS };
    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let total = K2_ACCEPTS.load(std::sync::atomic::Ordering::Relaxed)
        + K2_REJECTS.load(std::sync::atomic::Ordering::Relaxed);
    if total >= K2_SUMMARY_PERIOD {
        let accepts = K2_ACCEPTS.swap(0, std::sync::atomic::Ordering::Relaxed);
        let rejects = K2_REJECTS.swap(0, std::sync::atomic::Ordering::Relaxed);
        let total = (accepts + rejects).max(1);
        let pct = 100.0 * (accepts as f64) / (total as f64);
        // A6 (2026-05-26): MTP K=2 acceptance rate as a free drift
        // gauge. The HF z-lab/Qwen3.6-27B-DFlash#2 report and our own
        // research3_spec_verify finding show that FP8 Qwen3.6 with
        // sustained <30% MTP accept indicates the target model's
        // logits have entered the "confidently wrong" attractor — the
        // exact failure mode driving the opencode multi-turn drift.
        // For now we just WARN to surface the signal in production
        // logs; a future Tier-B refinement adds per-sequence state +
        // `finish_reason="drift_detected"` to actually terminate the
        // response when the gauge trips.
        const DRIFT_THRESHOLD_PCT: f64 = 30.0;
        if pct < DRIFT_THRESHOLD_PCT && total >= K2_SUMMARY_PERIOD {
            tracing::warn!(
                "K2 drift gauge: accept rate {pct:.1}% < {DRIFT_THRESHOLD_PCT}% over last {total} steps (seq_len={seq_len}). Model logits likely in 'confidently wrong' attractor."
            );
        } else {
            tracing::info!(
                "K2 summary: {accepts} accept / {rejects} reject in last {total} steps ({pct:.1}% accept) seq_len={seq_len}"
            );
        }
    }
}

/// K=2 verify: [last_token, draft] → [v0, v1]. Two outcomes: ACCEPT or REJECT.
///
/// `verify_ctx` carries tokenizer special-token IDs the pre-sample
/// pipeline needs. Each verify position's logits are now copied D2H
/// and run through the same 8-stage processor pipeline used by the
/// non-MTP path — the fix for MTP-emitted tokens bypassing mid-word
/// defer, post-close mask, forced think-end, grammar bitmask, etc.
/// See `verify_pipeline_helper` for the root-cause writeup.
pub fn step_verify_k2(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    use crate::scheduler::mtp_timing::{self, Phase};
    let t_step = Instant::now();
    let t_sync = Instant::now();
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        a.finished = true;
        return;
    }
    mtp_timing::record(Phase::SyncSecondary, t_sync);
    let sync_us = t_sync.elapsed().as_micros();

    // EP: broadcast verify K=2 command + tokens so worker runs decode_verify_graphed in lockstep.
    let t_ep = Instant::now();
    let tokens_k2 = [a.last_token, drafts[0]];
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, 0xFFFFFFF2) {
        tracing::error!("EP broadcast verify_k2 cmd: {e:#}");
        a.finished = true;
        return;
    }
    for &t in &tokens_k2 {
        if let Err(e) = model.ep_broadcast_cmd(t) {
            tracing::error!("EP broadcast verify_k2 token: {e:#}");
            a.finished = true;
            return;
        }
    }

    mtp_timing::record(Phase::EpBroadcast, t_ep);
    let ep_us = t_ep.elapsed().as_micros();

    let t_verify = Instant::now();
    // Fused single-sweep path: DFlash only AND single-rank only. Under EP
    // (multi-rank) the worker ranks dispatch `decode_verify_graphed` on the
    // broadcast cmd above, so the master MUST run the same method to stay in
    // NCCL lockstep — the fused forward is not EP-coherent. The MTP path
    // (non-raw-argmax) also stays on the legacy graphed verify unchanged.
    let result_vec: Vec<u32> = if dflash_verify_raw_argmax && !model.is_ep() {
        // Fused path: single M=2 forward, DFlash hidden captured at row 0.
        match model.decode_and_verify_fused(&tokens_k2, &mut a.seq, 0) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("decode_and_verify_fused (k2): {e:#}");
                a.finished = true;
                return;
            }
        }
    } else {
        match model.decode_verify_graphed(&tokens_k2, &mut a.seq, 0) {
            Ok(r) => r.to_vec(),
            Err(e) => {
                tracing::error!("decode_verify_graphed: {e:#}");
                a.finished = true;
                return;
            }
        }
    };
    mtp_timing::record(Phase::VerifyForward, t_verify);
    let verify_us = t_verify.elapsed().as_micros();
    a.last_token_time = Instant::now();
    let (v0_argmax, v1_argmax) = (result_vec[0], result_vec[1]);

    let (v0, v1) = if dflash_verify_raw_argmax
        && !crate::scheduler::verify_pipeline_helper::dflash_masked_verify_enabled()
    {
        // DFlash drafter proposes on raw argmax; verify on the SAME (GOLD) basis.
        (v0_argmax, v1_argmax)
    } else {
        // MTP path: full pre-sample pipeline (rep_pen + DRY) unchanged.
        // Phase C-2 (2026-05-24): apply the logits-processor pipeline to each
        // verify position so MTP-emitted tokens see the same masks as non-MTP.
        let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model,
            &[v0_argmax, v1_argmax],
            a,
            verify_ctx,
        );
        (
            processed.first().copied().unwrap_or(v0_argmax),
            processed.get(1).copied().unwrap_or(v1_argmax),
        )
    };
    let accepted = drafts[0] == v0;

    // Extract logprobs from verify logits buffer (K=2 positions) when requested.
    let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
        extract_verify_logprobs(model, &[v0, v1], top_logprobs)
    } else {
        Vec::new()
    };

    // EP: always broadcast accept/reject to worker (prevents deadlock on EOS).
    if let Err(e) = model.ep_broadcast_cmd(accepted as u32) {
        tracing::error!("EP broadcast verify_k2 result: {e:#}");
        a.finished = true;
        return;
    }

    // EASD scaffolding (A.2): track baseline accept rate to decide
    // whether activating entropy-aware-spec-decode (per-step D2H of
    // verify logits + entropy threshold per arXiv:2512.23765) is
    // worth its overhead. Baseline acceptance is the prerequisite
    // signal — high accept rate means EASD has little to gain.
    crate::metrics::SPEC_DECODE_VERIFY
        .with_label_values(&["2", if accepted { "accept" } else { "reject" }])
        .inc();

    if accepted {
        // ── ACCEPTED ──
        emit_token(a, drafts[0], verify_lps.first().cloned());
        if !a.finished {
            emit_token(a, v1, verify_lps.get(1).cloned());
        }
        if a.finished {
            return;
        }
        a.last_token = v1;

        // Item #2 (STree-style in-place K=2 verify commit). Full accept
        // (num_accepted=k=2): the verify kernel already wrote the canonical
        // h_state, so the commit is a no-op.
        let t_commit = Instant::now();
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 2, 2) {
            tracing::error!("commit_accepted_prefix (accept): {e:#}");
            return;
        }
        mtp_timing::record(Phase::Commit, t_commit);

        // EAGLE-fix (ATLAS_DFLASH_EAGLE_FIX=1, K=2 accept only): append row 0 @ N
        // then row 1 @ N+1 BEFORE propose so forward_block conditions on row 1
        // (the hidden that generated bonus). This also sets the proposer's
        // skip-flag so propose does NOT re-append row 0.
        let eagle_fix = std::env::var("ATLAS_DFLASH_EAGLE_FIX").ok().as_deref() == Some("1");
        if eagle_fix && let Err(e) = model.dflash_eagle_accept_append(&mut a.seq) {
            tracing::error!("dflash_eagle_accept_append: {e:#}");
        }
        let t_save = Instant::now();
        if let Err(e) = model.save_hidden_for_mtp(1, 0) {
            tracing::error!("save_hidden_for_mtp(1): {e:#}");
            return;
        }
        mtp_timing::record(Phase::SaveHidden, t_save);
        let t_trim = Instant::now();
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 1, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        mtp_timing::record(Phase::TrimProposer, t_trim);
        let t_mask = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        mtp_timing::record(Phase::ProposeMask, t_mask);
        let t_propose = Instant::now();
        match model.run_mtp_propose_multi(
            v1,
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
        mtp_timing::record(Phase::Propose, t_propose);
        let propose_us = t_propose.elapsed().as_micros();
        // Per-step ACCEPT trace at debug — fires every step during
        // spec-decode. Power-user diagnostics:
        // `RUST_LOG=spark::scheduler::verify_k2_step=debug`. The
        // summary line emitted by `k2_record_outcome` is the
        // production signal.
        tracing::debug!(
            "K2 ACCEPT: ep={ep_us}μs sync={sync_us}μs verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k2_record_outcome(true, a.seq.seq_len);
        // #155 iter3: block-aligned Marconi checkpoint (live SSM state is
        // canonical post-commit). Fires only at interval boundaries.
        let t_marconi = Instant::now();
        model.decode_marconi_checkpoint(&mut a.seq);
        mtp_timing::record(Phase::MarconiCkpt, t_marconi);
        mtp_timing::step_done(t_step, a.seq.seq_len);
    } else {
        // ── REJECTED ──
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();

        let t_trim = Instant::now();
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        mtp_timing::record(Phase::TrimProposer, t_trim);
        // Item #2 (STree-style in-place K=2 verify commit). K=2 reject means
        // num_accepted=1 (last_token is always accepted): the in-place
        // commit rewinds live h_state to intermediate[0] (state after the
        // accepted token). Verify draft state is discarded.
        let t_commit = Instant::now();
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 1, 2) {
            tracing::error!("commit_accepted_prefix (reject): {e:#}");
            a.finished = true;
            return;
        }
        mtp_timing::record(Phase::Commit, t_commit);

        emit_token(a, v0, verify_lps.first().cloned());
        if a.finished {
            return;
        }
        a.last_token = v0;

        let t_save = Instant::now();
        if let Err(e) = model.save_hidden_for_mtp(0, 0) {
            tracing::error!("save_hidden_for_mtp(0): {e:#}");
            return;
        }
        mtp_timing::record(Phase::SaveHidden, t_save);
        let t_mask = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        mtp_timing::record(Phase::ProposeMask, t_mask);
        let t_propose = Instant::now();
        match model.run_mtp_propose_multi(
            v0,
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
        mtp_timing::record(Phase::Propose, t_propose);
        let propose_us = t_propose.elapsed().as_micros();
        let new_draft = a.pending_drafts.first().copied().unwrap_or(0);
        // REJECT log demoted from `info!` to `debug!` to match the
        // ACCEPT path. Rejection is informative when investigating
        // draft quality but spams Docker logs at production load.
        // Periodic accept-rate summary lives in `k2_record_outcome`.
        tracing::debug!(
            "K2 REJECT: ep={ep_us}μs sync={sync_us}μs verify={verify_us}μs propose={propose_us}μs seq_len={} last_tok={} prev_draft={} v0_verified={} new_draft={}",
            a.seq.seq_len,
            a.last_token,
            drafts[0],
            v0,
            new_draft,
        );
        k2_record_outcome(false, a.seq.seq_len);
        // #155 iter3: block-aligned Marconi checkpoint (live SSM state is
        // canonical post-commit). Fires only at interval boundaries.
        let t_marconi = Instant::now();
        model.decode_marconi_checkpoint(&mut a.seq);
        mtp_timing::record(Phase::MarconiCkpt, t_marconi);
        mtp_timing::step_done(t_step, a.seq.seq_len);
    }
}
