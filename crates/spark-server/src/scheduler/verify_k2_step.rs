// SPDX-License-Identifier: AGPL-3.0-only

//! K=2 verify step.

use super::*;

/// K=2 verify: [last_token, draft] → [v0, v1]. Two outcomes: ACCEPT or REJECT.
pub fn step_verify_k2(model: &dyn Model, a: &mut ActiveSeq, drafts: &[u32], num_drafts: usize) {
    let t_sync = Instant::now();
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        a.finished = true;
        return;
    }
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

    let ep_us = t_ep.elapsed().as_micros();

    let t_verify = Instant::now();
    let result = match model.decode_verify_graphed(&tokens_k2, &mut a.seq, 0) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("decode_verify_graphed: {e:#}");
            a.finished = true;
            return;
        }
    };
    let verify_us = t_verify.elapsed().as_micros();
    a.last_token_time = Instant::now();
    let [v0_argmax, v1_argmax] = result;

    // Use argmax for speculative acceptance. Temperature-aware resampling
    // (verify_resample) requires blocking D2H copy (~0.8ms/step overhead).
    // TODO: implement GPU-side temperature sampling to avoid D2H penalty.
    let v0 = v0_argmax;
    let v1 = v1_argmax;
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

        // F62 (2026-04-27): SpecMamba commit. Full accept (num_accepted=k=2):
        // copy verify scratch → canonical state.
        if let Err(e) = model.commit_verify_state_async(&mut a.seq, 2, 2) {
            tracing::error!("commit_verify_state_async (accept): {e:#}");
            return;
        }
        if let Err(e) = model.save_hidden_for_mtp(1, 0) {
            tracing::error!("save_hidden_for_mtp(1): {e:#}");
            return;
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 1, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
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
        if a.seq.seq_len.is_multiple_of(50) {
            tracing::info!(
                "K2 ACCEPT: ep={ep_us}μs sync={sync_us}μs verify={verify_us}μs propose={propose_us}μs seq_len={}",
                a.seq.seq_len
            );
        }
    } else {
        // ── REJECTED ──
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();

        if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // F62 (2026-04-27): SpecMamba commit. K=2 reject means
        // num_accepted=1 (last_token is always accepted): copy
        // intermediate[0] → canonical. Verify scratch is discarded.
        if let Err(e) = model.commit_verify_state_async(&mut a.seq, 1, 2) {
            tracing::error!("commit_verify_state_async (reject): {e:#}");
            a.finished = true;
            return;
        }

        emit_token(a, v0, verify_lps.first().cloned());
        if a.finished {
            return;
        }
        a.last_token = v0;

        if let Err(e) = model.save_hidden_for_mtp(0, 0) {
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
        let new_draft = a.pending_drafts.first().copied().unwrap_or(0);
        tracing::info!(
            "K2 REJECT: ep={ep_us}μs sync={sync_us}μs verify={verify_us}μs propose={propose_us}μs seq_len={} last_tok={} prev_draft={} v0_verified={} new_draft={}",
            a.seq.seq_len,
            a.last_token,
            drafts[0],
            v0,
            new_draft,
        );
    }
}
