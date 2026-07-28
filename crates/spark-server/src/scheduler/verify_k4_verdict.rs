// SPDX-License-Identifier: AGPL-3.0-only

//! K=4 verify verdict application: accept/rewind/emit/re-propose.
//!
//! Extracted VERBATIM from `verify_k4_step.rs` (behavior-identical refactor,
//! batched-MTP E9) so the single-seq `step_verify_k4` and the batched
//! `step_verify_k4_batched` share ONE copy of the four accept branches —
//! rewind arithmetic, `trim_proposer_state`, `commit_accepted_prefix`,
//! emit order, and `k4_record_outcome` are the existing machinery unchanged.
//! The only parameterization is WHERE the accepted-row hidden for the next
//! propose comes from ([`K4Hidden`]): the live verify row (single-seq path,
//! `save_hidden_for_mtp`) or the pre-propose stash slot (batched path, whose
//! phase-3 proposes have already clobbered the live rows).

use super::*;

/// Source of the accepted-position hidden fed to `run_mtp_propose_multi`.
#[derive(Clone, Copy)]
pub(super) enum K4Hidden {
    /// Read the live verify forward's row `num_accepted` (single-seq path).
    VerifyRow,
    /// Read stash slot `i` written by `stash_verify_hidden_rows` BEFORE any
    /// propose ran (batched path).
    Stash(usize),
}

#[inline]
fn save_hidden(model: &dyn Model, hidden: K4Hidden, na: usize) -> anyhow::Result<()> {
    match hidden {
        K4Hidden::VerifyRow => model.save_hidden_for_mtp(na, 0),
        K4Hidden::Stash(i) => model.save_hidden_for_mtp_from_stash(i, 0),
    }
}

/// Apply a K=4 verify verdict to one sequence: emit the accepted prefix +
/// correction/bonus token, rewind `seq_len`/`tokens` for rejected drafts,
/// roll back proposer + SSM state, save the accepted-position hidden, and
/// re-propose. Body is the verbatim four-branch tail of the pre-refactor
/// `step_verify_k4`.
#[allow(clippy::too_many_arguments)]
pub(super) fn k4_apply_verdict(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    v: [u32; 4],
    verify_lps: Vec<crate::api::TokenLogprobs>,
    num_drafts: usize,
    num_accepted: usize,
    hidden: K4Hidden,
    verify_us: u128,
) {
    let [v0, v1, v2, v3] = v;
    if num_accepted == 3 {
        emit_token(a, drafts[0], verify_lps.first().cloned());
        if !a.finished {
            emit_token(a, drafts[1], verify_lps.get(1).cloned());
        }
        if !a.finished {
            emit_token(a, drafts[2], verify_lps.get(2).cloned());
        }
        if !a.finished {
            emit_token(a, v3, verify_lps.get(3).cloned());
        }
        if a.finished {
            return;
        }
        a.last_token = v3;

        // Item #2 (STree-style in-place K=4 verify commit). Full accept
        // (num_accepted=k=4): the verify kernel already wrote the canonical
        // h_state, so the commit is a no-op.
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 4, 4) {
            // SSM state is no longer trustworthy — terminate, do not continue.
            tracing::error!("commit_accepted_prefix (K=4 accept-4): {e:#}");
            a.finished = true;
            return;
        }
        if let Err(e) = save_hidden(model, hidden, 3) {
            tracing::error!("save_hidden_for_mtp(3): {e:#}");
            return;
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 3, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            v3,
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
        let propose_us = t_propose.elapsed().as_micros();
        tracing::debug!(
            "K4 ACCEPT-3: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k4_record_outcome(3, a.seq.seq_len);
    } else if num_accepted == 2 {
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 2, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // Item #2 (STree-style in-place K=4 verify commit). Partial accept
        // (num_accepted=3 < k=4): rewind live h_state to intermediate[2]
        // (state after the third accepted token).
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 3, 4) {
            tracing::error!("commit_accepted_prefix (K=4 accept-3): {e:#}");
            a.finished = true;
            return;
        }
        emit_token(a, drafts[0], verify_lps.first().cloned());
        if !a.finished {
            emit_token(a, drafts[1], verify_lps.get(1).cloned());
        }
        if !a.finished {
            emit_token(a, v2, verify_lps.get(2).cloned());
        }
        if a.finished {
            return;
        }
        a.last_token = v2;
        if let Err(e) = save_hidden(model, hidden, 2) {
            tracing::error!("save_hidden_for_mtp(2): {e:#}");
            return;
        }
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            v2,
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
        let propose_us = t_propose.elapsed().as_micros();
        tracing::debug!(
            "K4 ACCEPT-2: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k4_record_outcome(2, a.seq.seq_len);
    } else if num_accepted == 1 {
        a.seq.seq_len -= 2;
        a.seq.tokens.pop();
        a.seq.tokens.pop();
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 1, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // Item #2 (STree-style in-place K=4 verify commit). Partial accept
        // (num_accepted=2 < k=4): rewind live h_state to intermediate[1].
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 2, 4) {
            tracing::error!("commit_accepted_prefix (K=4 accept-2): {e:#}");
            a.finished = true;
            return;
        }
        emit_token(a, drafts[0], verify_lps.first().cloned());
        if !a.finished {
            emit_token(a, v1, verify_lps.get(1).cloned());
        }
        if a.finished {
            return;
        }
        a.last_token = v1;
        if let Err(e) = save_hidden(model, hidden, 1) {
            tracing::error!("save_hidden_for_mtp(1): {e:#}");
            return;
        }
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
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
        let propose_us = t_propose.elapsed().as_micros();
        tracing::debug!(
            "K4 ACCEPT-1: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k4_record_outcome(1, a.seq.seq_len);
    } else {
        a.seq.seq_len -= 3;
        a.seq.tokens.pop();
        a.seq.tokens.pop();
        a.seq.tokens.pop();
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // Item #2 (STree-style in-place K=4 verify commit). Partial accept
        // (num_accepted=1 < k=4): rewind live h_state to intermediate[0]
        // (state after the always-accepted bonus token).
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 1, 4) {
            tracing::error!("commit_accepted_prefix (K=4 accept-1): {e:#}");
            a.finished = true;
            return;
        }
        emit_token(a, v0, verify_lps.first().cloned());
        if a.finished {
            return;
        }
        a.last_token = v0;
        if let Err(e) = save_hidden(model, hidden, 0) {
            tracing::error!("save_hidden_for_mtp(0): {e:#}");
            return;
        }
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
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
        let propose_us = t_propose.elapsed().as_micros();
        tracing::debug!(
            "K4 REJECT: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k4_record_outcome(0, a.seq.seq_len);
    }
}
