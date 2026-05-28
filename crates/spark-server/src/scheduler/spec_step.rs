// SPDX-License-Identifier: AGPL-3.0-only

//! Self-speculative + NGram speculative decoding step + grammar helpers.

use super::*;

/// Self-speculative step: draft via layer-skipping, verify with full model.
/// Combines bootstrap + verify in one step (no pipeline).
pub fn step_self_spec(model: &dyn Model, active: &mut [ActiveSeq], num_drafts: usize) {
    let a = &mut active[0];

    // 1. Full-model decode to get token_0
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, a.last_token) {
        tracing::error!("EP broadcast self-spec token: {e:#}");
        a.finished = true;
        return;
    }
    let logits = match model.decode(a.last_token, &mut a.seq, 0) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("self-spec decode error: {e:#}");
            a.finished = true;
            return;
        }
    };
    let token_0 = match model.argmax_on_device(logits, 0) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("self-spec argmax error: {e:#}");
            a.finished = true;
            return;
        }
    };

    // 2. Draft phase: layer-skipping for cheap predictions
    let seq_len_before_draft = a.seq.seq_len;
    let tokens_before_draft = a.seq.tokens.len();

    let mut draft_tokens = Vec::with_capacity(num_drafts);
    let mut draft_token = token_0;
    for _ in 0..num_drafts {
        let logits = match model.decode_draft(draft_token, &mut a.seq, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("self-spec draft error: {e:#}");
                break;
            }
        };
        draft_token = match model.argmax_on_device(logits, 0) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("self-spec draft argmax error: {e:#}");
                break;
            }
        };
        draft_tokens.push(draft_token);
    }

    // 3. Rewind to pre-draft state (SSM unchanged since we skipped SSM layers)
    a.seq.seq_len = seq_len_before_draft;
    a.seq.tokens.truncate(tokens_before_draft);

    if draft_tokens.is_empty() {
        // No drafts: emit token_0 and continue
        emit_token(a, token_0, None);
        if !a.finished {
            a.last_token = token_0;
        }
        return;
    }

    // 4. Checkpoint SSM states before verification
    if let Err(e) = model.checkpoint_ssm_states(&mut a.seq) {
        tracing::error!("self-spec checkpoint: {e:#}");
        a.finished = true;
        return;
    }
    let seq_len_before_verify = a.seq.seq_len;

    // 5. Verify: run full model on [token_0, d1, ..., dK]
    let mut verify_tokens = vec![token_0];
    verify_tokens.extend_from_slice(&draft_tokens);

    let verified = match model.decode_verify(&verify_tokens, &mut a.seq, 0) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("self-spec verify error: {e:#}");
            a.finished = true;
            return;
        }
    };

    // 6. Compare draft vs verified, count acceptances
    let n_drafts = draft_tokens.len();
    let mut num_accepted = 0;

    emit_token(a, token_0, None);
    if a.finished {
        return;
    }

    for i in 0..n_drafts {
        if draft_tokens[i] == verified[i] {
            emit_token(a, draft_tokens[i], None);
            if a.finished {
                return;
            }
            num_accepted += 1;
        } else {
            emit_token(a, verified[i], None);
            if a.finished {
                return;
            }
            a.last_token = verified[i];
            break;
        }
    }

    if num_accepted == n_drafts && n_drafts > 0 {
        emit_token(a, verified[n_drafts], None);
        if !a.finished {
            a.last_token = verified[n_drafts];
        }
    } else if num_accepted < n_drafts {
        // Already set a.last_token above in the break
    } else {
        a.last_token = token_0;
    }

    // 7. Rollback extra verify tokens
    // tokens_added = token_0 (always kept) + accepted drafts
    let tokens_added = 1 + num_accepted;
    let expected_seq_len = seq_len_before_verify + tokens_added;

    if a.seq.seq_len > expected_seq_len {
        let extra = a.seq.seq_len - expected_seq_len;
        for _ in 0..extra {
            a.seq.seq_len -= 1;
            a.seq.tokens.pop();
        }
        // +1 because token_0 is always accepted in the verify batch
        if let Err(e) = model.rollback_ssm_states(&mut a.seq, num_accepted + 1) {
            tracing::error!("self-spec rollback: {e:#}");
        }
    }
}

/// N-gram speculative step: CPU proposer + CUDA-graphed K=2 verify.
///
/// Two-phase pipeline (same as MTP but with N-gram proposer instead):
/// 1. Bootstrap: regular decode → argmax → N-gram propose → pending_drafts
/// 2. Verify: decode_verify_graphed(K=2) → accept/reject → SSM rollback
pub fn step_ngram(model: &dyn Model, active: &mut [ActiveSeq], proposer: &mut NgramProposer) {
    let a = &mut active[0];

    if !a.pending_drafts.is_empty() {
        // ── Phase B: Verify pending draft ──
        let drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
        step_ngram_verify(model, a, &drafts, proposer);
    } else {
        // ── Phase A: Bootstrap decode + N-gram propose ──
        if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, a.last_token) {
            tracing::error!("EP broadcast ngram bootstrap: {e:#}");
            a.finished = true;
            return;
        }
        let logits = match model.decode(a.last_token, &mut a.seq, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("ngram bootstrap decode error: {e:#}");
                a.finished = true;
                return;
            }
        };
        let tok = match model.argmax_on_device(logits, 0) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("ngram bootstrap argmax error: {e:#}");
                a.finished = true;
                return;
            }
        };

        // Observe the token for future predictions
        proposer.observe(&a.seq.tokens, tok);

        emit_token(a, tok, None);
        if a.finished {
            return;
        }
        a.last_token = tok;

        // N-gram propose (CPU-only, zero GPU cost)
        if let Some(draft) = proposer.propose(&a.seq.tokens) {
            a.pending_drafts = vec![draft];

            // Checkpoint SSM for potential rollback during verify
            if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
                tracing::error!("ngram start_checkpoint_async: {e:#}");
            }
        }
        // If no proposal: next iteration will be another bootstrap (regular decode)
    }
}

/// Verify a single N-gram draft via CUDA-graphed K=2 path.
pub fn step_ngram_verify(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    proposer: &mut NgramProposer,
) {
    let t_sync = Instant::now();
    if let Err(e) = model.sync_secondary() {
        tracing::error!("ngram sync_secondary: {e:#}");
        a.finished = true;
        return;
    }
    let sync_us = t_sync.elapsed().as_micros();

    // EP: broadcast verify K=2 command + tokens
    let tokens_k2 = [a.last_token, drafts[0]];
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, 0xFFFFFFF2) {
        tracing::error!("EP broadcast ngram verify cmd: {e:#}");
        a.finished = true;
        return;
    }
    for &t in &tokens_k2 {
        if let Err(e) = model.ep_broadcast_cmd(t) {
            tracing::error!("EP broadcast ngram verify token: {e:#}");
            a.finished = true;
            return;
        }
    }

    let t_verify = Instant::now();
    let result = match model.decode_verify_graphed(&tokens_k2, &mut a.seq, 0) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("ngram decode_verify_graphed: {e:#}");
            a.finished = true;
            return;
        }
    };
    let verify_us = t_verify.elapsed().as_micros();
    a.last_token_time = Instant::now();
    let [v0, v1] = result;
    let accepted = drafts[0] == v0;

    // EP: broadcast accept/reject to worker
    if let Err(e) = model.ep_broadcast_cmd(accepted as u32) {
        tracing::error!("EP broadcast ngram verify result: {e:#}");
        a.finished = true;
        return;
    }

    if accepted {
        // ── ACCEPTED: emit both tokens ──
        // After verify_graphed, a.seq.tokens has [.., last_token, drafts[0]] appended.
        // Observe: context ending with last_token → drafts[0] was correct
        // Observe: context ending with drafts[0] → v1 is the next prediction
        proposer.observe(&a.seq.tokens[..a.seq.tokens.len() - 1], drafts[0]);
        proposer.observe(&a.seq.tokens, v1);

        emit_token(a, drafts[0], None);
        if !a.finished {
            emit_token(a, v1, None);
        }
        if a.finished {
            return;
        }
        a.last_token = v1;

        // Checkpoint SSM for next verify
        if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
            tracing::error!("ngram accept checkpoint: {e:#}");
        }

        // Propose next draft
        if let Some(draft) = proposer.propose(&a.seq.tokens) {
            a.pending_drafts = vec![draft];
        }

        if a.seq.seq_len.is_multiple_of(50) {
            tracing::info!(
                "NGRAM K2 ACCEPT: sync={sync_us}μs verify={verify_us}μs cache={} seq_len={}",
                proposer.len(),
                a.seq.seq_len,
            );
        }
    } else {
        // ── REJECTED: rollback SSM, emit v0 only ──
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();

        if let Err(e) = model.start_rollback_and_checkpoint_async(&mut a.seq, 1) {
            tracing::error!("ngram rollback: {e:#}");
            a.finished = true;
            return;
        }

        // After pop, a.seq.tokens has [.., last_token].
        // Observe: context ending with last_token → v0 is the correct next token
        proposer.observe(&a.seq.tokens, v0);

        emit_token(a, v0, None);
        if a.finished {
            return;
        }
        a.last_token = v0;

        // Propose next draft
        if let Some(draft) = proposer.propose(&a.seq.tokens) {
            a.pending_drafts = vec![draft];
        }

        tracing::info!(
            "NGRAM K2 REJECT: sync={sync_us}μs verify={verify_us}μs cache={} seq_len={}",
            proposer.len(),
            a.seq.seq_len,
        );
    }
}

/// Fill the XGrammar bitmask for the current matcher position and clone it
/// into an owned `Vec<i32>` the caller can pass into MTP draft sampling.
///
/// Returns `None` when grammar is inactive, the sequence is currently inside
/// a `<think>` span (matcher is paused), the grammar has already terminated,
/// or `fill_bitmask` reported no constraint. In all those cases MTP should
/// fall back to its unconstrained GPU-argmax path.
///
/// The owned copy is small (~ceil(vocab/32)*4 bytes, ~32KB for 100k vocab)
/// and is necessary because the matcher is borrowed mutably by the scheduler
/// between `fill_bitmask` and the subsequent `accept_token` calls inside
/// `emit_token`, while the MTP propose call borrows the model immutably —
/// cloning sidesteps the lifetime overlap.
pub fn mtp_grammar_mask_for(a: &mut ActiveSeq) -> Option<Vec<i32>> {
    if a.inside_thinking {
        return None;
    }
    let gs = a.grammar_state.as_mut()?;
    if gs.is_terminated() {
        return None;
    }
    if !gs.fill_bitmask() {
        return None;
    }
    Some(gs.bitmask_data().to_vec())
}

/// Truncate a draft list at the first token the grammar would
/// reject *if it were the next emitted token at that draft position*.
///
/// Required for K=3+ MTP paths where `run_mtp_propose_multi` uses a
/// SINGLE bitmask snapshot (taken at the start of propose) for all N
/// drafts. The mask correctly constrains `drafts[0]` but does not
/// reflect the post-`drafts[0]` grammar state — so `drafts[1]` may
/// cross a structural boundary (e.g. `drafts[0] = </function>`
/// closing a tool body, then `drafts[1] = <parameter=` which is
/// invalid in the outer free-text grammar state).
///
/// Without this guard, the spec verifier accepts the cross-boundary
/// span (the model's actual sample matches whatever the in-tool
/// distribution happened to produce), `emit_token` advances the
/// grammar past `</function>`, and the next `accept_token` for
/// `drafts[1]` returns false silently — the token is already in
/// `output_tokens`, but the grammar is desync'd from the output
/// stream. Subsequent bitmasks are wrong.
///
/// Reference: arXiv:2512.15834 ("Speculative Tool Calls"). The
/// canonical fix is to re-run the grammar mask from a fresh outer
/// state for each draft; we approximate cheaply by simulating
/// `accept_token` per draft and truncating at the first rejection,
/// rolling the state back when done. The verifier then accepts at
/// most the validated prefix.
///
/// Returns the number of drafts that pass grammar validation.
/// Mutates `gs` transiently but restores it via `rollback`. K=2
/// (num_drafts=1) callers can skip this — a single draft uses its
/// own up-to-date mask.
pub fn truncate_drafts_at_grammar_boundary(gs: &mut GrammarState, drafts: &[u32]) -> usize {
    if drafts.len() < 2 || gs.is_terminated() {
        return drafts.len();
    }
    let mut accepted = 0usize;
    for &tok in drafts {
        if !gs.accept_token(tok) {
            break;
        }
        accepted += 1;
    }
    if accepted > 0 {
        gs.rollback(accepted);
    }
    if accepted < drafts.len() {
        tracing::warn!(
            kept = accepted,
            dropped = drafts.len() - accepted,
            "spec-decode boundary: truncated drafts crossing grammar transition"
        );
    }
    accepted
}
