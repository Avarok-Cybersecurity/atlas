// SPDX-License-Identifier: AGPL-3.0-only

//! Acceptance-aware single-sequence MTP verification for K=3 and K=4.
//! Only sampled prefix rows can mutate request policies. Real emission between
//! rows supplies the serial history, RNG seed, grammar and thinking state.

pub(super) mod grammar;

use super::ActiveSeq;
use super::decode_logits_seq::process_seq_logits;
use super::emit_step::emit_token;
use super::logit_processors::LogitsContext;
use super::sched_ctx::SchedCtx;

fn sample_and_emit(
    bytes: &[u8],
    vocab: usize,
    drafts: &[u32],
    seq: &mut ActiveSeq,
    sched: &SchedCtx,
    ctx: &LogitsContext,
) -> Vec<u32> {
    let forward_len = seq.seq.seq_len;
    let first_committed_len = forward_len - drafts.len();
    let mut emitted = Vec::new();
    for row in 0..=drafts.len() {
        // The model has executed the whole speculative batch, but emission's
        // context ceiling must see only the input prefix for this sample.
        seq.seq.seq_len = first_committed_len + row;
        let (token, lp) = process_seq_logits(seq, bytes, row, vocab, 2, false, ctx, false);
        emit_token(seq, token, lp, sched);
        // Preserve the full forward length for the common verdict rewind,
        // including rejection, EOS and context/budget finishes.
        seq.seq.seq_len = forward_len;
        emitted.push(token);
        if seq.finished || drafts.get(row) != Some(&token) {
            break;
        }
    }
    emitted
}

/// Finish a successful wide forward. The EP verdict must be sent even when
/// host copying or emission ends the sequence: worker ranks await it.
pub(super) fn finish(
    model: &dyn spark_model::traits::Model,
    seq: &mut ActiveSeq,
    sched: &SchedCtx,
    drafts: &[u32],
    num_drafts: usize,
    ctx: &LogitsContext,
) {
    let k = drafts.len() + 1;
    let vocab = model.vocab_size();
    let mut bytes = vec![0; k * vocab * 2];
    if let Err(e) = model.copy_logits_to_host(model.logits_buffer_ptr(), &mut bytes) {
        tracing::error!("copy K{k} MTP verify logits: {e:#}");
        if let Err(e) = model.ep_broadcast_cmd(0) {
            tracing::error!("EP broadcast failed MTP verify result: {e:#}");
        }
        seq.finished = true;
        return;
    }
    let picks = sample_and_emit(&bytes, vocab, drafts, seq, sched, ctx);
    // The final sample is the correction/bonus. If emission stops on an
    // accepted draft, it becomes the final token and needs no next forward.
    let na = picks.len() - 1;
    if let Err(e) = model.ep_broadcast_cmd(na as u32) {
        tracing::error!("EP broadcast K{k} MTP verify result: {e:#}");
        seq.finished = true;
        return;
    }

    if sched.levers.shadow_topk > 0 {
        let base = seq.seq.seq_len - k;
        tracing::info!("SHADOW_TGT base={base} v={picks:?} drafts={drafts:?}");
    }
    super::mtp_accept_debug::record(1, drafts.len(), picks[0] == drafts[0], na);
    tracing::debug!("K{k} MTP verify: sampled={picks:?} drafts={drafts:?} accepted={na}");

    if spark_model::speculative::mtp_refeed_accepted_enabled() {
        let base = seq.seq.seq_len - k;
        let shift = spark_model::speculative::mtp_refeed_shift();
        for row in 0..=na {
            let label = ((base + row + 1) as isize + shift).max(0) as usize;
            if let Err(e) = model.save_hidden_for_catchup(row, label) {
                tracing::debug!("save_hidden_for_catchup(K={k}, row={row}): {e:#}");
                break;
            }
        }
    }

    // The forward appended K input rows; only na+1 belong to the emitted
    // prefix. Restore all model state even on EOS/budget/cancellation.
    let rewind = k - (na + 1);
    seq.seq.seq_len -= rewind;
    seq.seq.tokens.truncate(seq.seq.tokens.len() - rewind);
    seq.last_token = picks[na];
    if let Err(e) = model.commit_accepted_prefix(&mut seq.seq, na + 1, k) {
        tracing::error!("commit_accepted_prefix(K={k}, prefix={}): {e:#}", na + 1);
        seq.finished = true;
        return;
    }
    if !super::verify_k2_step::commit_verify_aux_or_finish(model, seq, na + 1, k) {
        return;
    }
    if let Err(e) = model.trim_proposer_state(&mut seq.seq, na, 0) {
        tracing::error!("trim_proposer_state(K={k}): {e:#}");
        seq.finished = true;
        return;
    }
    match k {
        3 => super::verify_k3_step::k3_record_outcome(sched, na, seq.seq.seq_len),
        4 => super::verify_k4_step::stats::k4_record_outcome(sched, na, seq.seq.seq_len),
        _ => unreachable!("wide MTP only dispatches K=3/4"),
    }
    if seq.finished {
        return;
    }
    if let Err(e) = model.save_hidden_for_mtp(na, 0) {
        tracing::error!("save_hidden_for_mtp({na}): {e:#}");
        seq.finished = true;
        return;
    }
    let grammar_mask = super::mtp_grammar_mask_for(seq);
    match model.run_mtp_propose_multi(
        seq.last_token,
        seq.seq.seq_len,
        grammar::drafts(seq, num_drafts, false),
        &mut seq.seq,
        0,
        grammar_mask.as_deref(),
    ) {
        Ok(drafts) => seq.pending_drafts = drafts,
        Err(e) => tracing::error!("run_mtp_propose_multi(K={k}): {e:#}"),
    }
}

#[cfg(test)]
mod grammar_tests;
#[cfg(test)]
mod tests;
