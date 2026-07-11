// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash-based verify step (drafted token verification).

use super::*;

/// DFlash γ-token verify with accept-prefix.
///
/// Phase 3 minimal-viable implementation: routes `[last_token, drafts...]`
/// through the eager `decode_verify_dflash` path (which today defaults to
/// `decode_verify`) and finds the first index where draft ≠ verified
/// argmax. Tokens 0..first_mismatch are accepted; the verified token at
/// the mismatch position becomes the bonus token; subsequent drafts are
/// dropped.
///
/// Deferred to Phase 6 (full integration):
///   * EP=2 broadcast of verify-cmd + drafts (drafter currently runs only
///     on rank 0; verify on a single-rank target is correct, but EP=2 needs
///     the broadcast pattern from `step_verify_k2`).
///   * Per-position logprobs extraction.
///   * SSM `commit_verify_state_async(num_accepted, k)` loop. Without it,
///     hybrid models (Qwen3.6-A3B has GDN layers) will see SSM state drift
///     after γ-verify. Single-token decode unaffected; γ-verify only
///     correct on pure-attention targets until this is wired.
///   * `save_hidden_for_mtp` / `save_hidden_for_dflash` hook on the
///     accepted bonus token (the next propose() needs the latest hidden).
///   * Sliding-window state rollback for sliding-attention layers
///     (Gemma-4-style; not used by Qwen3.6 targets).
pub fn step_verify_dflash(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        a.finished = true;
        return;
    }

    // tokens = [last_verified, draft_0, draft_1, ..., draft_{γ-1}]
    let mut tokens = Vec::with_capacity(drafts.len() + 1);
    tokens.push(a.last_token);
    tokens.extend_from_slice(drafts);

    let verified_argmax = match model.decode_verify_dflash(&tokens, &mut a.seq, 0) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("decode_verify_dflash: {e:#}");
            a.finished = true;
            return;
        }
    };
    a.last_token_time = Instant::now();

    // DFlash drafter proposes on raw argmax; when dflash_verify_raw_argmax
    // is set (process-wide DFlash mode), skip the rep_pen/DRY pipeline so
    // verifier and drafter judge on the SAME (GOLD) basis.
    // ATLAS_DFLASH_MASKED_VERIFY=1 restores the masking stages for the
    // picks (special-token leak fix). For non-DFlash callers (unreachable
    // today since step_verify_dflash is only dispatched at drafts.len()>=4,
    // which only DFlash produces), apply the full pre-sample pipeline as in
    // K=2/3/4: `decode_verify_dflash` writes `[tokens.len(), vocab]` BF16
    // into `logits_buffer`; the helper reads it back, dequant + 8-stage
    // pipeline + argmax per slot. Fail-safe falls back to the raw GPU
    // argmax on D2H failure.
    let verified = if dflash_verify_raw_argmax
        && !crate::scheduler::verify_pipeline_helper::dflash_masked_verify_enabled()
    {
        verified_argmax
    } else {
        crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model,
            &verified_argmax,
            a,
            verify_ctx,
        )
    };

    // `decode_verify` already advanced `seq.seq_len` by `tokens.len()` and
    // pushed all γ+1 tokens into `seq.tokens`. The accept-prefix logic below
    // determines how many to keep — the rest must be rolled back so the
    // KV cache, SSM state, and emitted token sequence stay consistent.

    // Accept-prefix: drafts[i] is "accepted" iff drafts[i] == verified[i].
    // verified[i] is the target's argmax at position i (i.e. its
    // prediction for what should follow `tokens[i]`). drafts[i] was the
    // proposer's guess for the same slot. First mismatch terminates the
    // accepted prefix; verified[first_mismatch] becomes the bonus token.
    let mut num_accepted = 0usize;
    for i in 0..drafts.len() {
        if i + 1 >= verified.len() {
            break;
        }
        if drafts[i] == verified[i] {
            num_accepted += 1;
        } else {
            break;
        }
    }

    // Roll back the over-extended `seq_len` and `seq.tokens`. The verify
    // advanced both by `tokens.len() = γ+1` (all γ drafts + the prefix
    // bonus slot). We keep the original prefix + `num_accepted` drafts +
    // 1 bonus position. So the post-rollback target is
    // `pre_verify_len + num_accepted + 1` — note we do NOT push the bonus
    // again via emit_token's path (emit_token only updates the user-facing
    // output buffer, not seq.tokens), so the bonus stays in seq.tokens
    // exactly where decode_verify put it.
    let pre_verify_len = a.seq.seq_len.saturating_sub(tokens.len());
    let target_seq_len = pre_verify_len + num_accepted + 1;
    let to_drop = a.seq.seq_len.saturating_sub(target_seq_len);
    if to_drop > 0 {
        a.seq.seq_len = target_seq_len;
        let pop_n = to_drop.min(a.seq.tokens.len());
        for _ in 0..pop_n {
            a.seq.tokens.pop();
        }
    }

    // Emit accepted drafts.
    for i in 0..num_accepted {
        emit_token(a, drafts[i], None);
        if a.finished {
            return;
        }
    }

    // Bonus token = verified[num_accepted] (the one that "corrected" the draft
    // at the first mismatch, or the next-prediction past the full-accept case).
    let bonus_idx = num_accepted;
    if bonus_idx < verified.len() {
        let bonus = verified[bonus_idx];
        emit_token(a, bonus, None);
        if a.finished {
            return;
        }
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

    tracing::info!(
        "DFLASH K=γ verify: γ={} accepted={}/{} ({:.0}%) seq_len={}",
        drafts.len(),
        num_accepted,
        drafts.len(),
        100.0 * (num_accepted as f64) / (drafts.len() as f64),
        a.seq.seq_len,
    );

    // SSM commit / rollback. Hybrid models (Qwen3.6-A3B has 30 GDN layers)
    // advance recurrent SSM state per-position during verify; without this
    // commit, the canonical h_state stays at position+γ even if only a few
    // drafts were accepted, producing gibberish on subsequent decodes.
    //
    // Semantics (default trait impl):
    //  - num_accepted == k_verify (full accept): canonical = h_state
    //  - 0 < num_accepted < k_verify (partial): canonical = intermediate[num_accepted-1]
    //  - num_accepted == 0: canonical untouched (rollback to checkpoint)
    //
    // k_verify = drafts.len() + 1 (the prefix bonus position is also verified).
    let k_verify = drafts.len() + 1;
    let total_accepted = num_accepted + 1; // bonus is always "accepted"
    if let Err(e) = model.commit_verify_state_async(&mut a.seq, total_accepted, k_verify) {
        tracing::error!("commit_verify_state_async (dflash): {e:#}");
        a.finished = true;
        return;
    }

    // Save the latest hidden for the NEXT propose() call. Mirrors the
    // K=2 verify path's `save_hidden_for_mtp(1, 0)` after accept.
    let bonus_token_idx = total_accepted.saturating_sub(1);
    if let Err(e) = model.save_hidden_for_mtp(bonus_token_idx, 0) {
        tracing::error!("save_hidden_for_mtp (dflash): {e:#}");
    }

    if let Err(e) = model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
        tracing::error!("trim_proposer_state: {e:#}");
    }

    // Re-propose for next step.
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
        Err(e) => tracing::error!("run_mtp_propose_multi (dflash): {e:#}"),
    }
}
