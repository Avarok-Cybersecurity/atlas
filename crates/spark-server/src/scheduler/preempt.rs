// SPDX-License-Identifier: AGPL-3.0-only

//! Decode-time KV preemption with RESUME instead of the kill it replaced.
//!
//! When batched decode exhausts the KV pool, a victim is preempted so the
//! rest of the batch keeps going. Until 2026-08 the victim was killed via
//! `send_error` — a client-facing correctness bug, not just lost work: the
//! stream got a mid-stream SSE error frame and then ended with no finish
//! chunk (an HTTP-200 "success" with silently truncated content), and on
//! the C=128 ladder (pool 102k tokens vs 157k demand) that shot 171
//! requests, discarding ~25% of all decode output. Now the victim is
//!   * SPILLED to the `--swap-space` pool when one is configured (KV pages
//!     saved verbatim; the existing swap-in loop restores it), or
//!   * REQUEUED with its context retained: GPU blocks are freed (offered to
//!     the prefix cache first, like `finish_sequence`) and the whole
//!     `ActiveSeq` is kept CPU-side; resume re-prefills the token history
//!     to rebuild KV + SSM state and decoding continues on the same sink.
//! Either way the stream contract holds: tokens already streamed stay
//! valid, no error frame is emitted, and generation resumes later.
//!
//! SSM state on resume (explicit choice): the requeue path REBUILDS the
//! recurrent state by re-prefilling every token of the history through the
//! SSM layers — that is the live path and is always correct. The prefill's
//! own prefix-cache / Marconi lookups run as usual, so when the
//! preempt-time `cache_sequence` blocks (or a decode-time Marconi
//! checkpoint) survive in the caches, the recompute is skipped for the
//! covered prefix; when they were reclaimed under pressure, the full
//! recompute happens. No separate SSM snapshot is carried across the
//! preemption — the GPU-resident rollback ring dies with the slot and is
//! re-created empty, exactly like the disk-swap path.

use super::*;

/// Starvation guard: a resumed victim may not be re-victimized until it has
/// generated this many NEW tokens. Without it, the freshly resumed (and
/// therefore recently *least*-progressed-per-block) sequence can be picked
/// again on the very next exhaustion, ping-ponging one request forever
/// while the rest of the batch never yields. 64 tokens ≈ several KV blocks
/// of real progress at block_size 16-32 — enough to amortize the re-prefill
/// — while still letting the pool rebalance within a few hundred steps.
pub(super) const PREEMPT_IMMUNITY_TOKENS: usize = 64;

/// Pick the decode-preemption victim.
///
/// Policy: LEAST PROGRESS (fewest generated tokens; ties broken by slot for
/// determinism). The pre-resume policy was "largest block_table" — maximize
/// blocks freed per kill — which was rational only while preemption was
/// PERMANENT. With resume, nothing is lost but *work to redo*: the resume
/// cost is a re-prefill of prompt + generated-so-far, so the cheapest
/// victim is precisely the one with the least progress; it also has the
/// least streamed output at stake if anything later goes wrong, and it
/// leaves the deepest sequences — the ones closest to finishing and
/// releasing ALL their blocks — running. This mirrors vLLM's last-in
/// preemption order.
///
/// Exclusions:
///   * grammar-active sequences (matcher state is not reconstructible);
///   * without spill, sequences whose tokens contain vision pads (their KV
///     came from image embeddings a token re-prefill cannot reproduce);
///   * immune sequences (just resumed — see [`PREEMPT_IMMUNITY_TOKENS`]),
///     unless every candidate is immune, in which case immunity yields:
///     preempting an immune victim (resumable) still beats erroring the
///     whole batch (fatal).
pub(super) fn choose_decode_victim(
    model: &dyn Model,
    active: &[ActiveSeq],
    can_spill: bool,
) -> Option<usize> {
    let eligible = |a: &ActiveSeq| {
        a.grammar_state.is_none() && (can_spill || !model.tokens_contain_vision_pad(&a.seq.tokens))
    };
    let pick = |immune_ok: bool| {
        active
            .iter()
            .enumerate()
            .filter(|(_, a)| {
                eligible(a) && (immune_ok || a.output_tokens.len() >= a.preempt_immune_until_tokens)
            })
            .min_by_key(|(_, a)| (a.output_tokens.len(), a.seq.slot_idx))
            .map(|(i, _)| i)
    };
    pick(false).or_else(|| pick(true))
}

/// Batched decode with KV-exhaustion preemption.
///
/// Runs `model.decode_batch` for the whole active set; on "KV cache
/// exhausted" it preempts one victim (spill when `spill` is available,
/// requeue otherwise) and retries so the rest of the batch makes progress.
/// Returns `Some(logits)` on success, `None` when the batch is gone (a
/// non-recoverable decode error already surfaced to every client).
pub(super) fn decode_batch_with_preemption(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    mut spill: Option<&mut KvSpillManager>,
    swapped: &mut Vec<SwappedSeq>,
    preempted: &mut Vec<PreemptedSeq>,
) -> Option<DevicePtr> {
    loop {
        let tokens: Vec<u32> = active.iter().map(|a| a.last_token).collect();
        let mut refs: Vec<&mut SequenceState> = active.iter_mut().map(|a| &mut a.seq).collect();
        match model.decode_batch(&tokens, &mut refs, 0) {
            Ok(l) => return Some(l),
            Err(e) => {
                drop(refs);
                let victim = if format!("{e:#}").contains("KV cache exhausted") && active.len() > 1
                {
                    choose_decode_victim(model, active, spill.is_some())
                } else {
                    None
                };
                let Some(vi) = victim else {
                    tracing::error!("decode_batch error: {e:#}");
                    for mut a in active.drain(..) {
                        send_error(model, &mut a, &format!("{e:#}"));
                    }
                    // A destroyed CUDA context (issue #429) surfaces here like
                    // any other decode error, but it is terminal: the next tick
                    // would admit the next request onto a dead context and fail
                    // it identically, forever. The backend has already probed
                    // and latched by this point, so the only thing left is to
                    // stop. `request` is idempotent, so the echoing failures of
                    // the remaining in-flight batches do not re-trigger it.
                    if let Some(reason) = atlas_core::fault::global().fault() {
                        crate::tui::shutdown::request(reason);
                    }
                    return None;
                };
                // `remove` (not `swap_remove`) keeps the ascending SSM-slot
                // order the caller established; a hole only costs the batched
                // path a fallback to the eager loop for this step, and the
                // next step re-sorts. The vacated SSM slot is re-compacted by
                // `retire_finished_sequences` later this same tick.
                let v = active.remove(vi);
                tracing::warn!(
                    "KV cache exhausted during decode: preempting slot={} \
                     ({} blocks, {} tokens generated) for later RESUME so the \
                     other {} sequence(s) can continue",
                    v.seq.slot_idx,
                    v.seq.block_table.len(),
                    v.output_tokens.len(),
                    active.len(),
                );
                match spill.as_deref_mut() {
                    Some(sp) => match spill_out_sequence(model, v, sp) {
                        Ok(s) => swapped.push(s),
                        Err((v, spill_err)) => {
                            tracing::warn!(
                                "decode-preempt spill failed ({spill_err:#}); \
                                 requeueing victim for re-prefill instead"
                            );
                            preempted.push(preempt_requeue(model, v));
                        }
                    },
                    None => preempted.push(preempt_requeue(model, v)),
                }
            }
        }
    }
}

/// Save a (already removed from `active`) sequence's KV+SSM image to the
/// spill pool, free its GPU resources and build the [`SwappedSeq`].
///
/// SSOT for the spill image — shared by admission-time `swap_out_sequence`
/// and decode-time preemption. On a save error the untouched `ActiveSeq` is
/// handed back so the caller can requeue it (decode path) or surface the
/// error (admission path) instead of silently dropping the request.
#[allow(clippy::result_large_err)]
pub(super) fn spill_out_sequence(
    model: &dyn Model,
    mut a: ActiveSeq,
    spill: &mut KvSpillManager,
) -> Result<SwappedSeq, (ActiveSeq, anyhow::Error)> {
    let (swap_id, mut writer) = match spill.create_file() {
        Ok(v) => v,
        Err(e) => return Err((a, e)),
    };
    if let Err(e) = model.save_sequence_state(&a.seq, &mut writer) {
        drop(writer);
        let _ = spill.remove_file(swap_id);
        return Err((a, e));
    }
    drop(writer);
    spill.record_usage(swap_id);

    let num_blocks = a.seq.block_table.len();
    let seq_len = a.seq.seq_len;
    let tokens = a.seq.tokens.clone();

    // Free GPU resources (KV blocks + SSM slot). A free failure after a
    // SUCCESSFUL save is logged, not fatal: the request's state is safely
    // on disk and the same log-and-continue contract already governs
    // `finish_sequence`/`send_error`.
    let slot_idx = a.seq.slot_idx as u32;
    if let Err(e) = model.free_sequence(&mut a.seq) {
        tracing::error!("spill_out_sequence: free_sequence: {e:#}");
    }
    let _ = model.ep_broadcast_cmd_for_seq(slot_idx, 0xFFFFFFF1);

    Ok(SwappedSeq {
        tokens,
        session_hash: a.session_hash,
        adapter_slot: a.seq.adapter_slot,
        adapter_id: a.seq.adapter_id,
        seq_len,
        num_blocks,
        last_token: a.last_token,
        output_tokens: a.output_tokens,
        remaining: a.remaining,
        min_tokens: a.min_tokens,
        eos_tokens: a.eos_tokens,
        sink: a.sink,
        temperature: a.temperature,
        top_k: a.top_k,
        top_p: a.top_p,
        top_n_sigma: a.top_n_sigma,
        min_p: a.min_p,
        repetition_penalty: a.repetition_penalty,
        presence_penalty: a.presence_penalty,
        frequency_penalty: a.frequency_penalty,
        repetition_penalty_window: 256,
        lz_penalty: DEFAULT_LZ_PENALTY,
        dry_multiplier: a.dry_multiplier,
        dry_base: a.dry_base,
        dry_allowed_length: a.dry_allowed_length,
        dry_sequence_breakers: a.dry_sequence_breakers,
        logit_bias: a.logit_bias,
        inside_thinking: a.inside_thinking,
        enable_thinking: a.enable_thinking,
        thinking_budget: a.thinking_budget,
        repetition_detection: a.repetition_detection,
        spontaneous_think_budget: a.spontaneous_think_budget,
        thinking_tokens: a.thinking_tokens,
        force_end_thinking: a.force_end_thinking,
        think_force_closed: a.think_force_closed,
        sentence_defer_count: a.sentence_defer_count,
        consecutive_confident: a.consecutive_confident,
        in_code_fence: a.in_code_fence,
        think_end_token: a.think_end_token,
        think_start_token: a.think_start_token,
        think_ended: a.think_ended,
        think_just_ended: a.think_just_ended,
        post_think_emitted: a.post_think_emitted,
        think_skip_count: a.think_skip_count,
        require_tool_call: a.require_tool_call,
        tool_request: a.tool_request,
        tools_present: a.tools_present,
        suppress_tool_call: a.suppress_tool_call,
        disable_mtp: a.disable_mtp,
        mtp_acct: a.mtp_acct,
        content_started: a.content_started,
        content_tokens: a.content_tokens,
        prose_tokens_since_last_tool: a.prose_tokens_since_last_tool,
        think_watchdog_fires: a.think_watchdog_fires,
        rollback_count: a.rollback_count,
        tool_call_start_token: a.tool_call_start_token,
        tool_call_opened: a.tool_call_opened,
        tool_call_end_token: a.tool_call_end_token,
        last_token_time: a.last_token_time,
        request_start: a.request_start,
        decode_start: a.decode_start,
        seed: a.seed,
        top_logprobs: a.top_logprobs,
        logprobs_data: a.logprobs_data,
        timeout_at: a.timeout_at,
        swap_id,
        cached_prompt_tokens: a.cached_prompt_tokens,
    })
}

/// Requeue a decode-preempted victim: free its GPU resources but retain the
/// whole `ActiveSeq` CPU-side for a later re-prefill resume.
///
/// The computed KV is offered to the prefix cache FIRST (exactly the
/// `finish_sequence` ordering), so the blocks become reclaimable-not-lost:
/// the retrying decode batch evicts only what it actually needs
/// (`alloc_block_evicting`), and whatever survives makes the resume
/// re-prefill a prefix-cache hit instead of a recompute.
pub(super) fn preempt_requeue(model: &dyn Model, mut a: ActiveSeq) -> PreemptedSeq {
    model.cache_sequence(&a.seq);
    let tokens = a.seq.tokens.clone();
    let slot_idx = a.seq.slot_idx as u32;
    if let Err(e) = model.free_sequence(&mut a.seq) {
        tracing::error!("preempt_requeue: free_sequence: {e:#}");
    }
    let _ = model.ep_broadcast_cmd_for_seq(slot_idx, 0xFFFFFFF1);
    // GPU-tied speculative state died with the slot.
    a.pending_drafts.clear();
    a.pending_draft_conf.clear();
    a.spec_adapt = Default::default();
    PreemptedSeq { a, tokens }
}

/// Resume a requeued victim by re-prefilling its token history.
///
/// Rebuilds KV and SSM state for `tokens` (prompt + processed output) on a
/// freshly allocated sequence; `a.last_token` — already streamed to the
/// client — stays the pending decode input, so nothing is re-sampled or
/// re-emitted and the stream continues exactly where it paused. The prefill
/// logits are discarded for the same reason. On error the client has
/// already been notified and freed — the caller just logs.
pub(super) fn resume_preempted_seq(model: &dyn Model, p: PreemptedSeq) -> Result<ActiveSeq> {
    let PreemptedSeq { mut a, tokens } = p;
    let mut seq = match model.alloc_sequence() {
        Ok(s) => s,
        Err(e) => {
            send_error_to_sink(&mut a.sink, &format!("preempt-resume alloc failed: {e:#}"));
            return Err(e);
        }
    };
    seq.session_hash = a.session_hash;
    seq.adapter_slot = a.seq.adapter_slot;
    // Task #24/#25 parity with the swap-in path: keep the STABLE adapter_id
    // stamped at the original prefill and re-acquire the slot ref released
    // by the preempt-time free. (Preempted seqs also hold the LoRA
    // quiescence gate in `run`, so the adapter cannot rotate in between.)
    seq.adapter_id = a.seq.adapter_id;
    seq.acquired_adapter_slot = model.acquire_adapter_slot(a.seq.adapter_slot);

    // EP: mirror the non-chunked prefill preamble so the worker mirrors the
    // re-prefill (no-ops on non-EP models).
    let prefill_result = (|| -> Result<()> {
        model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF0)?;
        model.ep_broadcast_cmd(tokens.len() as u32)?;
        model.ep_broadcast_cmd(0)?;
        model.ep_broadcast_cmd(tokens.len() as u32)?;
        model.ep_broadcast_tokens(&tokens)?;
        model.prefill(&tokens, &mut seq, 0)?;
        Ok(())
    })();
    if let Err(e) = prefill_result {
        a.seq = seq;
        send_error(
            model,
            &mut a,
            &format!("preempt-resume re-prefill failed: {e:#}"),
        );
        return Err(e);
    }
    // NOTE: `seq.prompt_len` now spans the WHOLE history — deliberately left
    // as prefill stamped it, because it must describe what prefill actually
    // ref-bumped for `cache_sequence`'s prompt/generated split; restoring
    // the original prompt length would double-bump the pre-preemption
    // output blocks at finish and pin them forever.
    a.seq = seq;
    // The rollback ring's GPU snapshots died with the old slot; start empty
    // (identical to the disk-swap resume path).
    a.ssm_rollback_ring = SsmDecodeRing::new(model.decode_rollback_ring_slots());
    a.preempt_immune_until_tokens = a.output_tokens.len() + PREEMPT_IMMUNITY_TOKENS;
    Ok(a)
}

/// Resume requeued victims while KV blocks and batch slots allow.
///
/// Cheapest-first (fewest blocks to re-prefill). Gated on the history
/// fitting with one block of growth headroom, reclaiming from the prefix
/// cache exactly like the swap-in loop — cached blocks never volunteer
/// themselves (see `reclaim_prefix_blocks` docs).
pub(super) fn resume_preempted_seqs(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    preempted: &mut Vec<PreemptedSeq>,
    max_batch_size: usize,
    block_size: usize,
) {
    while !preempted.is_empty() && active.len() < max_batch_size {
        let Some((idx, needed)) = preempted
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.tokens.len() / block_size.max(1) + 1))
            .min_by_key(|&(_, n)| n)
        else {
            return;
        };
        // +1 growth block of headroom: resuming into an exactly-full pool
        // would just re-preempt on the next decode step.
        let want = needed + 1;
        let total = model.num_total_blocks();
        if total > 0 && want > total {
            // Can never fit — erroring is honest; queueing forever is not.
            let mut p = preempted.remove(idx);
            send_error_to_sink(
                &mut p.a.sink,
                &format!(
                    "preempted sequence needs {want} KV blocks but the pool has {total}; \
                     cannot resume"
                ),
            );
            continue;
        }
        let mut free = model.num_free_blocks();
        while free < want {
            let got = model.reclaim_prefix_blocks(want - free);
            if got == 0 {
                break;
            }
            free = model.num_free_blocks();
        }
        if free < want {
            return;
        }
        let p = preempted.remove(idx);
        let n_tokens = p.tokens.len();
        match resume_preempted_seq(model, p) {
            Ok(a) => {
                tracing::info!(
                    "Preempt-resume: re-prefilled {n_tokens} tokens \
                     ({} generated so far), decode continues",
                    a.output_tokens.len(),
                );
                active.push(a);
            }
            Err(e) => tracing::error!("Preempt-resume failed: {e:#}"),
        }
    }
}
