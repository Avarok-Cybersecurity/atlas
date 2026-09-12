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

// provenance-id: 526f6e616c6420522e205374657369616b
/// The wide MTP verify takes the graphed verify's own GPU argmax for every row
/// the host pipeline provably cannot change, and copies the `[K, vocab]` logits
/// to the host only when some row needs them. ON by default;
/// `ATLAS_MTP_THINK_FAST_GREEDY=0` restores the unconditional D2H + host path
/// (the A/B and rollback switch, same convention as `ATLAS_NO_FAST_GREEDY_CHAT`).
fn mtp_fast_greedy_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_MTP_THINK_FAST_GREEDY").as_deref() != Ok("0"))
}

/// Why a row could not take the GPU argmax. Logged once per run per reason so
/// a serve log says which gate is paying for the D2H.
#[derive(Clone, Copy)]
enum SlowReason {
    Sampling,
    Grammar,
    Logprobs,
    Diagnostic,
    Penalties,
    F2Confidence,
    ThinkEndInjector,
    PinToolCall,
    Structural,
    NotImmune,
}

impl SlowReason {
    fn key(self) -> &'static str {
        match self {
            SlowReason::Sampling => "log:mtp_fast_greedy_slow:sampling",
            SlowReason::Grammar => "log:mtp_fast_greedy_slow:grammar",
            SlowReason::Logprobs => "log:mtp_fast_greedy_slow:logprobs",
            SlowReason::Diagnostic => "log:mtp_fast_greedy_slow:diagnostic",
            SlowReason::Penalties => "log:mtp_fast_greedy_slow:penalties",
            SlowReason::F2Confidence => "log:mtp_fast_greedy_slow:f2_confidence",
            SlowReason::ThinkEndInjector => "log:mtp_fast_greedy_slow:think_end_injector",
            SlowReason::PinToolCall => "log:mtp_fast_greedy_slow:pin_tool_call",
            SlowReason::Structural => "log:mtp_fast_greedy_slow:structural_argmax",
            SlowReason::NotImmune => "log:mtp_fast_greedy_slow:not_immune",
        }
    }
}

/// Proof that `process_seq_logits` for THIS row, in the sequence's CURRENT
/// state, returns exactly the raw argmax `tok`, so the row can be emitted
/// without host logits. Mirrors the gates of `fast_masked::try_chat_fast_path`
/// and the chat arm of `verify_pick_all_with_pipeline`, evaluated per row
/// because the wide path emits between rows and the state advances:
///  * greedy regime, no grammar, no logprobs, no diagnostic that reads the
///    host distribution (AdaDec, logit dumps);
///  * penalties + bias argmax-preserving (`fast_greedy::classify_penalties`
///    on the SAME `FinalDecode` params the slow path builds, so the A4
///    `</think>` floor classifies as Blocked while it is armed);
///  * no stage that reads the logits to mutate state or that rewrites the
///    distribution: F2 confidence early-stop, the forced-`</think>` injector
///    (armed or deferring), pin-to-tool-call;
///  * the argmax is not a maskable structural id (think_end, think_start,
///    tool_call_start), since every remaining stage only touches those.
/// Under a reduce-only penalty the argmax must also be immune (not in the
/// scoped history, raw logit > 0: one 2-byte D2H).
fn row_fast_greedy(
    model: &dyn spark_model::traits::Model,
    seq: &ActiveSeq,
    ctx: &LogitsContext,
    row: usize,
    tok: u32,
    vocab: usize,
) -> Result<(), SlowReason> {
    use crate::scheduler::confidence::{
        MAX_SENTENCE_DEFER_TOKENS, THINK_DEFER_ABS_CEILING, THINK_DEFER_BUDGET_FACTOR,
    };
    use crate::scheduler::fast_greedy::{self, PenaltyGate};
    if !(seq.temperature == 0.0 || ctx.sampling.force_temp_zero) {
        return Err(SlowReason::Sampling);
    }
    if seq.grammar_state.is_some() {
        return Err(SlowReason::Grammar);
    }
    if seq.top_logprobs.is_some() {
        return Err(SlowReason::Logprobs);
    }
    if ctx.sampling.adadec_diagnostic
        || ctx.dumps.logits.is_some()
        || ctx.dumps.adadec.is_some()
        || dump_logits_path_set()
    {
        return Err(SlowReason::Diagnostic);
    }
    let f2_active = !ctx.sampling.disable_watchdogs
        && seq.inside_thinking
        && !seq.force_end_thinking
        && seq.thinking_tokens >= 400
        && ctx.watchdog.confidence_early_stop;
    if f2_active {
        return Err(SlowReason::F2Confidence);
    }
    let defer_hard_override = match seq.thinking_budget {
        Some(b) => seq.thinking_tokens >= b.saturating_mul(THINK_DEFER_BUDGET_FACTOR),
        None => seq.thinking_tokens >= THINK_DEFER_ABS_CEILING,
    } || seq.sentence_defer_count >= MAX_SENTENCE_DEFER_TOKENS;
    if seq.inside_thinking && (seq.force_end_thinking || defer_hard_override) {
        return Err(SlowReason::ThinkEndInjector);
    }
    if seq.think_just_ended
        && seq.require_tool_call
        && !seq.tool_call_opened
        && !seq.inside_thinking
    {
        return Err(SlowReason::PinToolCall);
    }
    if Some(tok) == ctx.think_end_token
        || Some(tok) == seq.think_start_token
        || Some(tok) == ctx.tool_call_start_token
    {
        return Err(SlowReason::Structural);
    }
    // The force-temp-zero bypass returns the raw argmax before penalties.
    if ctx.sampling.force_temp_zero {
        return Ok(());
    }
    let params = crate::scheduler::sample_step::penalty_params_for(
        seq,
        crate::scheduler::sample_step::PositionKind::FinalDecode,
        0.0,
        None,
        seq.logit_bias.clone(),
    );
    match fast_greedy::classify_penalties(&params) {
        PenaltyGate::Blocked => Err(SlowReason::Penalties),
        PenaltyGate::Neutral => Ok(()),
        PenaltyGate::ReduceOnly => {
            let scoped = crate::scheduler::sample_step::penalty_history_scope(
                &seq.output_tokens,
                ctx.tool_call_end_token,
            );
            let immune = fast_greedy::argmax_immune(tok, scoped, || {
                fast_greedy::logit_is_positive(model, model.logits_buffer_ptr(), row, vocab, tok)
            });
            if immune {
                Ok(())
            } else {
                Err(SlowReason::NotImmune)
            }
        }
    }
}

fn dump_logits_path_set() -> bool {
    static SET: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("ATLAS_DUMP_LOGITS_PATH").is_some())
}

/// The fast-greedy sibling of `sample_and_emit`: same row loop, same emission
/// between rows, but a row whose pick is provably the GPU argmax
/// (`row_fast_greedy`) is emitted from `gpu_argmax` and the `[K, vocab]` D2H
/// happens lazily, on the first row that needs the host pipeline. Returns
/// `None` when that D2H fails (the caller then ends the sequence exactly as
/// the eager path does).
fn fast_greedy_and_emit(
    model: &dyn spark_model::traits::Model,
    gpu_argmax: &[u32],
    vocab: usize,
    drafts: &[u32],
    seq: &mut ActiveSeq,
    sched: &SchedCtx,
    ctx: &LogitsContext,
) -> Option<Vec<u32>> {
    let k = drafts.len() + 1;
    let forward_len = seq.seq.seq_len;
    let first_committed_len = forward_len - drafts.len();
    let mut bytes: Option<Vec<u8>> = None;
    let mut emitted = Vec::new();
    for row in 0..=drafts.len() {
        seq.seq.seq_len = first_committed_len + row;
        let tok = gpu_argmax[row];
        let (token, lp) = match row_fast_greedy(model, seq, ctx, row, tok, vocab) {
            Ok(()) => {
                if ctx.stats.once("log:mtp_fast_greedy") {
                    tracing::info!(
                        "MTP verify fast greedy ACTIVE: GPU argmax rows emitted without the \
                         [K, vocab] D2H (default on; ATLAS_MTP_THINK_FAST_GREEDY=0 disables)"
                    );
                }
                (tok, None)
            }
            Err(reason) => {
                if ctx.stats.once(reason.key()) {
                    tracing::info!(
                        "MTP verify fast greedy: row took the host pipeline ({})",
                        &reason.key()["log:mtp_fast_greedy_slow:".len()..]
                    );
                }
                if bytes.is_none() {
                    let mut b = vec![0; k * vocab * 2];
                    if let Err(e) = model.copy_logits_to_host(model.logits_buffer_ptr(), &mut b) {
                        tracing::error!("copy K{k} MTP verify logits: {e:#}");
                        seq.seq.seq_len = forward_len;
                        return None;
                    }
                    bytes = Some(b);
                }
                let b = bytes.as_deref().expect("fetched above");
                process_seq_logits(seq, b, row, vocab, 2, false, ctx, false)
            }
        };
        emit_token(seq, token, lp, sched);
        seq.seq.seq_len = forward_len;
        emitted.push(token);
        if seq.finished || drafts.get(row) != Some(&token) {
            break;
        }
    }
    Some(emitted)
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
    gpu_argmax: &[u32],
) {
    let k = drafts.len() + 1;
    // Lookup drafts (#974) wrote no drafter rows: the trim below is skipped
    // for them, and the index is scored on what the verify accepted.
    let from_lookup = std::mem::take(&mut seq.pending_drafts_lookup);
    let vocab = model.vocab_size();
    let picks = if mtp_fast_greedy_enabled() && gpu_argmax.len() == k {
        match fast_greedy_and_emit(model, gpu_argmax, vocab, drafts, seq, sched, ctx) {
            Some(p) => p,
            None => {
                if let Err(e) = model.ep_broadcast_cmd(0) {
                    tracing::error!("EP broadcast failed MTP verify result: {e:#}");
                }
                seq.finished = true;
                return;
            }
        }
    } else {
        // Borrow the run's staging buffer instead of allocating. `vec![0; ...]`
        // here was a fresh 1.49 MB (K=3, vocab 151936) allocation AND zero-fill on
        // EVERY verify step, immediately overwritten by the copy — the same waste
        // `decode_logits_step` already avoids with this exact borrow/restore. Only
        // grows, and every byte read below is written by `copy_logits_to_host`.
        let mut bytes = sched.scratch.host_bytes.borrow_mut().split_off(0);
        bytes.resize(k * vocab * 2, 0);
        if let Err(e) = model.copy_logits_to_host(model.logits_buffer_ptr(), &mut bytes) {
            tracing::error!("copy K{k} MTP verify logits: {e:#}");
            if let Err(e) = model.ep_broadcast_cmd(0) {
                tracing::error!("EP broadcast failed MTP verify result: {e:#}");
            }
            *sched.scratch.host_bytes.borrow_mut() = bytes;
            seq.finished = true;
            return;
        }
        let picks = sample_and_emit(&bytes, vocab, drafts, seq, sched, ctx);
        // Give the buffer back before any later early return; nothing below reads
        // `bytes`, and a path that skipped this would silently drop the reuse.
        *sched.scratch.host_bytes.borrow_mut() = bytes;
        picks
    };
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
    if from_lookup {
        sched.lookup.borrow_mut().record(drafts.len(), na);
    } else if let Err(e) = model.trim_proposer_state(&mut seq.seq, na, 0) {
        tracing::error!("trim_proposer_state(K={k}): {e:#}");
        seq.finished = true;
        return;
    }
    match k {
        3 => super::verify_k3_step::k3_record_outcome(sched, na, seq.seq.seq_len),
        // K=4 and every wider row count (the K=N step, #1060) share the K=4
        // outcome bucket: the ladder's stats are keyed by step shape, and the
        // wide shape is the same shape at more rows.
        _ => super::verify_k4_step::stats::k4_record_outcome(sched, na, seq.seq.seq_len),
    }
    if seq.finished {
        return;
    }
    if let Err(e) = model.save_hidden_for_mtp(na, 0) {
        tracing::error!("save_hidden_for_mtp({na}): {e:#}");
        seq.finished = true;
        return;
    }
    let capacity = model.mtp_slot_draft_capacity(seq.seq.slot_idx);
    if super::lookup_gate::take_lookup_drafts(seq, sched, num_drafts, capacity, false, model.is_ep()) {
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
