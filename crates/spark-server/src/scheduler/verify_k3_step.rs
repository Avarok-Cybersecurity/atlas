// SPDX-License-Identifier: AGPL-3.0-only

//! K=3 verify step.

use super::*;

/// K=3 verify: [last_token, draft1, draft2] → [v0, v1, v2]. Three outcomes.
pub fn step_verify_k3(model: &dyn Model, a: &mut ActiveSeq, drafts: &[u32], num_drafts: usize) {
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        a.finished = true;
        return;
    }

    // EP: broadcast verify K=3 command + 3 tokens so worker runs decode_verify_graphed_k3 in lockstep.
    let tokens_k3 = [a.last_token, drafts[0], drafts[1]];
    if let Err(e) = model.ep_broadcast_cmd(0xFFFFFFF3) {
        tracing::error!("EP broadcast verify_k3 cmd: {e:#}");
        a.finished = true;
        return;
    }
    for &t in &tokens_k3 {
        if let Err(e) = model.ep_broadcast_cmd(t) {
            tracing::error!("EP broadcast verify_k3 token: {e:#}");
            a.finished = true;
            return;
        }
    }

    let t_verify = Instant::now();
    let result = match model.decode_verify_graphed_k3(&tokens_k3, &mut a.seq, 0) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("decode_verify_graphed_k3: {e:#}");
            a.finished = true;
            return;
        }
    };
    let verify_us = t_verify.elapsed().as_micros();
    a.last_token_time = Instant::now();
    let [v0_argmax, v1_argmax, v2_argmax] = result;

    // Use argmax for speculative acceptance (see K=2 comment).
    let v0 = v0_argmax;
    let v1 = v1_argmax;
    let v2 = v2_argmax;

    let num_accepted = if drafts[0] != v0 {
        0
    } else if drafts[1] != v1 {
        1
    } else {
        2
    };

    // Extract logprobs from verify logits buffer (K=3 positions) when requested.
    let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
        extract_verify_logprobs(model, &[v0, v1, v2], top_logprobs)
    } else {
        Vec::new()
    };

    // EP: always broadcast num_accepted to worker (prevents deadlock on EOS).
    if let Err(e) = model.ep_broadcast_cmd(num_accepted as u32) {
        tracing::error!("EP broadcast verify_k3 result: {e:#}");
        a.finished = true;
        return;
    }

    tracing::info!(
        "K3 verify: tokens=[{},{},{}] → v=[{v0},{v1},{v2}] drafts=[{},{}] accepted={num_accepted} seq_len={}",
        tokens_k3[0],
        tokens_k3[1],
        tokens_k3[2],
        drafts[0],
        drafts[1],
        a.seq.seq_len
    );

    if num_accepted == 2 {
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

        // F62/F63 (2026-04-27): SpecMamba commit. K=3 full accept.
        if let Err(e) = model.commit_verify_state_async(&mut a.seq, 3, 3) {
            tracing::error!("commit_verify_state_async (K=3 accept-3): {e:#}");
            return;
        }
        if let Err(e) = model.save_hidden_for_mtp(2, 0) {
            tracing::error!("save_hidden_for_mtp(2): {e:#}");
            return;
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 2, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
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
        if a.seq.seq_len.is_multiple_of(50) {
            tracing::info!(
                "K3 ACCEPT-2: verify={verify_us}μs propose={propose_us}μs seq_len={}",
                a.seq.seq_len
            );
        }
    } else if num_accepted == 1 {
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 1, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // F62/F63 (2026-04-27): K=3 partial accept (2 of 3).
        if let Err(e) = model.commit_verify_state_async(&mut a.seq, 2, 3) {
            tracing::error!("commit_verify_state_async (K=3 accept-2): {e:#}");
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
        if let Err(e) = model.save_hidden_for_mtp(1, 0) {
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
        tracing::info!(
            "K3 ACCEPT-1: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
    } else {
        a.seq.seq_len -= 2;
        a.seq.tokens.pop();
        a.seq.tokens.pop();
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // F62/F63 (2026-04-27): K=3 partial accept (1 of 3).
        if let Err(e) = model.commit_verify_state_async(&mut a.seq, 1, 3) {
            tracing::error!("commit_verify_state_async (K=3 accept-1): {e:#}");
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
        tracing::info!(
            "K3 REJECT: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
    }
}
