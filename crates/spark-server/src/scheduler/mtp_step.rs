// SPDX-License-Identifier: AGPL-3.0-only

//! MTP speculative draft proposal step.

use super::*;

/// MTP-aware step: bootstrap sequences without drafts, then verify via CUDA graph.
/// Supports K=2 (num_drafts=1) and K=3 (num_drafts=2).
pub fn step_mtp(model: &dyn Model, active: &mut [ActiveSeq], num_drafts: usize) {
    let mut bootstrap_idxs: Vec<usize> = Vec::new();
    let mut verify_idxs: Vec<usize> = Vec::new();
    for (i, a) in active.iter().enumerate() {
        if !a.pending_drafts.is_empty() {
            verify_idxs.push(i);
        } else {
            bootstrap_idxs.push(i);
        }
    }

    // ── Phase A: Bootstrap decode for sequences without a draft ──
    for &idx in &bootstrap_idxs {
        let a = &mut active[idx];
        // EP: broadcast token to worker before decode (worker runs decode in lockstep).
        if let Err(e) = model.ep_broadcast_cmd(a.last_token) {
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
        let tok = match sample_token_with_grammar(
            model,
            logits,
            a.temperature,
            a.top_k,
            a.top_p,
            &[],
            a.grammar_state.as_mut(),
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

        emit_token(a, tok, lp);
        if a.finished {
            continue;
        }
        a.last_token = tok;

        if let Err(e) = model.save_hidden_for_mtp(0, 0) {
            tracing::error!("save_hidden_for_mtp: {e:#}");
            continue;
        }
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            tok,
            a.seq.seq_len,
            num_drafts,
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

        if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
            tracing::error!("bootstrap start_checkpoint_async: {e:#}");
        }
    }

    // ── Phase B: Verify with pipelined checkpoint ──
    for &idx in &verify_idxs {
        let a = &mut active[idx];
        let mut drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
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
            step_verify_dflash(model, a, &drafts, num_drafts);
        } else if num_drafts >= 3 && drafts.len() >= 3 {
            step_verify_k4(model, a, &drafts, num_drafts);
        } else if num_drafts >= 2 && drafts.len() >= 2 {
            step_verify_k3(model, a, &drafts, num_drafts);
        } else {
            step_verify_k2(model, a, &drafts, num_drafts);
        }
    }
}
