// SPDX-License-Identifier: AGPL-3.0-only

//! Sequence lifecycle: finish, errors, swap-out, resume.

use super::*;

/// THE single mapping from a server-side guard (`ActiveSeq::guard_stop`)
/// to the wire `finish_reason`. Do not map guard names to wire reasons
/// anywhere else.
///
/// Every non-timeout guard reports `"length"`: the server ended the
/// response with the model UNFINISHED. Neither enum slot is literally
/// true — the budget was not hit either — so the choice is decided by
/// which lie clients handle safely.
///
/// `"stop"` asserts "the model said what it wanted". For a mid-sentence
/// repetition cut that is affirmatively false, and every client behaviour
/// keyed on `"stop"` is then the WRONG action: accept, validate, commit,
/// end the agent run. `"length"` asserts only "incomplete, serving side
/// cut it, do not treat as final" — true in every part clients act on,
/// and its handlers are all safe: `openai-python` raises
/// `LengthFinishReasonError` instead of parsing truncated JSON, aider
/// skips the auto-commit, Instructor raises `IncompleteOutputException`,
/// pydantic-ai raises rather than accepting a half tool call.
///
/// The failure modes are asymmetric. A false `"length"` costs a bounded,
/// VISIBLE retry (agents cap them). A false `"stop"` is unbounded and
/// INVISIBLE: degenerate output silently banked as a finished answer.
/// Measured — relabelling these guards to `"stop"` cost 2/10 then 6/10
/// episodes of the agentic gate, because the harness stopped recognising
/// truncation and ended runs at 3-10 turns instead of the 12-22 a
/// recovery takes.
///
/// This also matches the ecosystem. On every engine WITHOUT a
/// degeneration guard (SGLang, TGI, llama.cpp, vLLM by default) the same
/// loop simply runs to the budget and reports `"length"`; our guard is an
/// early, smarter budget, so `"length"` preserves parity. No engine maps a
/// quality cut to `"stop"` by design — vLLM minted a distinct value
/// (`"repetition"`) rather than overload it.
///
/// ★ And it removes an internal contradiction: `tool_loop_capped` already
/// ships `"length"` (see `api::chat_stream::handle_done`) on exactly this
/// reasoning — `"length"` is the OpenAI-spec slot for "forcibly truncated"
/// and gives agents a clean hook to break their outer loop. The same
/// argument covers `fuzzy_repetition`, `tool_envelope_stuck`,
/// `inter_tool_prose_budget` and the simhash trip.
///
/// Considered alternative: putting the guard name on the wire the way
/// vLLM ships "repetition"/"abort". Rejected — typed clients hard-fail on
/// unknown enum VALUES (Rust `async-openai` fails deserialization
/// outright; pydantic-ai raised on OpenRouter's non-standard "error"),
/// and our one deliberate exception, `"timeout"`, already carries that
/// documented risk (see `ir::FINISH_REASON_TIMEOUT`). The guard's NAME
/// reaches diagnostics via `StreamEvent::Done.guard_stop` and the --dump
/// body; the right home for it on the wire is an extension FIELD (unknown
/// fields are ignored by every SDK, cf. vLLM's `stop_reason`), which is
/// follow-up work, not a reason to keep lying in the enum.
fn guard_stop_wire_reason(guard: &'static str) -> &'static str {
    match guard {
        // Defensive: the deadline guard is intercepted before the
        // token-derived checks (see `derive_finish_reason`), so this arm
        // is normally unreachable — kept so the mapping is total.
        GUARD_STOP_REQUEST_TIMEOUT => crate::ir::FINISH_REASON_TIMEOUT,
        _ => "length",
    }
}

/// Derive the wire finish reason for a completed sequence.
///
/// INVARIANT: `"length"` means "the SERVER ended the response with the
/// model unfinished" — the token budget was exhausted (`hard_ceiling_hit`:
/// the `max_tokens` countdown or the served context ceiling) OR a guard
/// cut it short. `"stop"` means the MODEL finished: it sampled EOS or a
/// stop sequence. That is the distinction clients act on, and it is the
/// one worth keeping exact.
///
/// It is still NOT a catch-all. The bug this replaced derived `"length"`
/// from "the last token wasn't EOS", which swept in stop-string matches,
/// client cancels and early finalizes — none of which are truncations
/// (observed live: `Done: 573 tokens (length)` under max_new_tokens=1024).
/// Those now correctly report `"stop"`; only budget exhaustion and guard
/// cuts report `"length"`.
///
/// Precedence:
///   1. the server-side deadline → `"timeout"` — a truncation must never
///      be mistaken for a natural stop, even when the last token is EOS;
///   2. token-derived natural stops (EOS → `"stop"`, tool-call close →
///      `"tool_calls"`) — what the model actually sampled outranks any
///      other guard that tripped on the same step;
///   3. any other guard → `guard_stop_wire_reason` (single mapping above);
///   4. token budget exhausted → `"length"`;
///   5. otherwise `"stop"` — an early finalize with budget left and no
///      guard (client cancel via `cancel_flag`, dropped stream receiver,
///      server shutdown drain): generation stopped; it did not hit the
///      budget.
///
/// Pure so the precedence is unit-testable without a model or a GPU
/// (tests in `lifecycle_tests.rs`).
pub(super) fn derive_finish_reason(
    guard_stop: Option<&'static str>,
    last_tok: Option<u32>,
    eos_tokens: &[u32],
    tool_call_end_token: Option<u32>,
    remaining: usize,
    seq_len: usize,
    max_seq_len: usize,
) -> &'static str {
    if guard_stop == Some(GUARD_STOP_REQUEST_TIMEOUT) {
        return crate::ir::FINISH_REASON_TIMEOUT;
    }
    if let Some(t) = last_tok {
        if eos_tokens.contains(&t) {
            return "stop";
        }
        // Guarded on `Some(t)` so an EMPTY output (max_tokens==0 scoring
        // path) on a model with no tool-call end token configured cannot
        // satisfy `None == None` and misreport "tool_calls".
        if Some(t) == tool_call_end_token {
            return "tool_calls";
        }
    }
    if let Some(guard) = guard_stop {
        return guard_stop_wire_reason(guard);
    }
    if hard_ceiling_hit(remaining, seq_len, max_seq_len) {
        return "length";
    }
    "stop"
}

/// Send final response and free GPU resources for a completed sequence.
///
/// `max_seq_len` is the served context ceiling (`sched.limits.max_seq_len`,
/// 0 = unlimited) — needed so the `"length"` decision reuses the exact
/// stop predicate from `emit_step`/`decode_logits_step`.
pub fn finish_sequence(model: &dyn Model, a: &mut ActiveSeq, max_seq_len: usize) {
    let reason = derive_finish_reason(
        a.guard_stop,
        a.output_tokens.last().copied(),
        &a.eos_tokens,
        a.tool_call_end_token,
        a.remaining,
        a.seq.seq_len,
        max_seq_len,
    );
    // The model folded this request's routing into the sequence during
    // prefill; convert it once here, for whichever sink the request used.
    let expert_activation =
        a.seq.expert_activation.take().map(|acc| {
            Box::new(crate::ir::expert_activation::ExpertActivationReport::from_acc(&acc))
        });
    match &mut a.sink {
        ResponseSink::Streaming(tx) => {
            let ttft_ms = a.decode_start.duration_since(a.request_start).as_secs_f64() * 1000.0;
            let decode_ms = a.decode_start.elapsed().as_secs_f64() * 1000.0;
            // Terminal frame: detached from the scheduler thread — safe
            // because nothing follows Done on this channel and all earlier
            // events are already queued (see spawn_terminal_send).
            super::mod_helpers::spawn_terminal_send(
                tx,
                StreamEvent::Done {
                    finish_reason: reason.to_string(),
                    prompt_tokens: 0, // prompt_tokens tracked by API layer
                    completion_tokens: a.output_tokens.len(),
                    time_to_first_token_ms: ttft_ms,
                    decode_time_ms: decode_ms,
                    reasoning_tokens: a.thinking_tokens,
                    cached_prompt_tokens: a.cached_prompt_tokens,
                    accepted_prediction_tokens: a.mtp_acct.accepted_total() as usize,
                    expert_activation,
                    guard_stop: a.guard_stop,
                },
                "done frame",
            );
        }
        ResponseSink::Blocking(tx) => {
            if let Some(tx) = tx.take() {
                let ttft_ms = a.decode_start.duration_since(a.request_start).as_secs_f64() * 1000.0;
                let decode_ms = a.decode_start.elapsed().as_secs_f64() * 1000.0;
                if tx
                    .send(Ok(InferenceResponse {
                        output_tokens: a.output_tokens.clone(),
                        finish_reason: reason.to_string(),
                        time_to_first_token_ms: ttft_ms,
                        decode_time_ms: decode_ms,
                        logprobs: std::mem::take(&mut a.logprobs_data),
                        reasoning_tokens: a.thinking_tokens,
                        cached_prompt_tokens: a.cached_prompt_tokens,
                        accepted_prediction_tokens: a.mtp_acct.accepted_total() as usize,
                        expert_activation,
                        prompt_logprobs: std::mem::take(&mut a.seq.prompt_logprobs)
                            .into_iter()
                            .map(|p| crate::api::TokenLogprobs {
                                token_id: p.token_id,
                                logprob: p.logprob,
                                top: p.top,
                            })
                            .collect(),
                    }))
                    .is_err()
                {
                    tracing::warn!(
                        "finish_sequence: blocking response send failed (receiver dropped)"
                    );
                }
            }
        }
    }
    let decode_s = a.decode_start.elapsed().as_secs_f64();
    let n = a.output_tokens.len();
    let tps = if decode_s > 0.0 {
        n as f64 / decode_s
    } else {
        0.0
    };
    let ttft_ms = a.decode_start.duration_since(a.request_start).as_secs_f64() * 1000.0;
    super::mtp_accept_debug::RequestAccept::log_done(n, reason, tps, ttft_ms, &a.mtp_acct);
    // Cache the full sequence (prompt + generated) in the prefix cache.
    // Must happen BEFORE free_sequence() so block indices are still valid.
    // Enables multi-turn sessions to reuse KV cache for prior assistant responses.
    model.cache_sequence(&a.seq);
    if let Err(e) = model.free_sequence(&mut a.seq) {
        tracing::error!("free_sequence: {e:#}");
    }
    // EP: signal worker to free+realloc its mirrored sequence.
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, 0xFFFFFFF1) {
        tracing::error!("EP broadcast free+realloc: {e:#}");
    }
}

/// Send error to client and free GPU resources.
pub fn send_error(model: &dyn Model, a: &mut ActiveSeq, msg: &str) {
    match &mut a.sink {
        ResponseSink::Streaming(tx) => {
            super::mod_helpers::spawn_terminal_send(
                tx,
                StreamEvent::Error(msg.to_string()),
                "error frame",
            );
        }
        ResponseSink::Blocking(tx) => {
            if let Some(tx) = tx.take()
                && tx.send(Err(anyhow::anyhow!("{msg}"))).is_err()
            {
                tracing::warn!("send_error: blocking Error send failed (receiver dropped)");
            }
        }
    }
    if let Err(e) = model.free_sequence(&mut a.seq) {
        tracing::error!("send_error: free_sequence: {e:#}");
    }
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, 0xFFFFFFF1) {
        tracing::error!("send_error: ep_broadcast free+realloc: {e:#}");
    }
}

/// Send an error directly to a ResponseSink that hasn't been attached
/// to an ActiveSeq yet. Used by prefill_request when it fails AFTER
/// extracting the sink from the InferenceRequest but BEFORE building
/// an ActiveSeq. Without this the sender is silently dropped, producing
/// a misleading "Inference cancelled" error on the client side.
pub fn send_error_to_sink(sink: &mut ResponseSink, msg: &str) {
    match sink {
        ResponseSink::Streaming(tx) => {
            super::mod_helpers::spawn_terminal_send(
                tx,
                StreamEvent::Error(msg.to_string()),
                "pre-seq error frame",
            );
        }
        ResponseSink::Blocking(tx) => {
            if let Some(tx) = tx.take()
                && tx.send(Err(anyhow::anyhow!("{msg}"))).is_err()
            {
                tracing::warn!("send_error_to_sink: blocking Error send failed (receiver dropped)");
            }
        }
    }
}

/// Swap out an active sequence to disk, freeing its GPU blocks.
///
/// Removes the sequence at `victim_idx` from `active`, saves its state
/// to a swap file, frees GPU resources, and returns a `SwappedSeq`.
pub fn swap_out_sequence(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    victim_idx: usize,
    spill: &mut KvSpillManager,
) -> Result<SwappedSeq> {
    let mut a = active.swap_remove(victim_idx);

    // Compact the swapped-in sequence (same logic as retire path).
    if victim_idx < active.len() && active[victim_idx].seq.slot_idx != victim_idx {
        // NOT `?`. `a` is already OUT of `active` — this function holds the
        // only handle to it — so an early return here is the one owner
        // dropping the request, and the drop is silent twice over: the sink
        // goes with it, so the client reads a SERVER-side compaction failure
        // as `500 "Inference cancelled"`, the wording reserved for a client
        // abort (see `send_error_to_sink`); and `free_sequence` never runs,
        // so the victim's KV blocks and SSM slot leak for the life of the
        // serve. The `Err` arm ten lines down already surfaces the victim on
        // the identical failure one step later; nothing distinguishes the two
        // except which call happened to fail first.
        //
        // Note the comment below reasons only about the LATER `?`s inside
        // `spill_out_sequence` — this call sits above `detach_slot_for_reuse`,
        // so the RAII guard is still armed and `a`'s own slot is released on
        // drop. What is lost is the client and the KV blocks, not the slot.
        if let Err(e) = model.compact_sequence(&mut active[victim_idx].seq, victim_idx) {
            send_error(model, &mut a, &format!("swap-out failed: {e:#}"));
            return Err(e);
        }
        // Disown the victim's migrated slot BEFORE the fallible save below: sets
        // the reuse sentinel AND neutralizes the RAII guard so a `?`-early-
        // return (create_file/save_sequence_state error) that drops `a` cannot
        // double-release the slot now owned by the swapped-in sequence.
        model.detach_slot_for_reuse(&mut a.seq);
    }

    // Save + free + build moved to `preempt::spill_out_sequence` so the
    // decode-time preemption path (which must NOT reorder/compact the active
    // vec mid-step) can share it — SSOT for the spill image. On error the
    // victim is surfaced to its client and freed here rather than silently
    // dropped (the old path leaked the GPU blocks AND the client saw only
    // "Inference cancelled").
    match super::preempt::spill_out_sequence(model, a, spill) {
        Ok(s) => Ok(s),
        Err((mut a, e)) => {
            send_error(model, &mut a, &format!("swap-out failed: {e:#}"));
            Err(e)
        }
    }
}

/// Rebuild the GPU sequence for a parked request from its spill image.
///
/// Split out so [`resume_swapped_seq`] has exactly ONE fallible step to
/// handle. Every `?` in here used to be a `?` in that function, where the
/// caller's stack frame was the sole owner of the `SwappedSeq` — so each of
/// them dropped the request's `ResponseSink` on the floor. Keeping them
/// behind one boundary makes it impossible to add a fifth failure mode that
/// forgets the client.
///
/// A sequence allocated before a later step fails is freed here rather than
/// leaked, and the spill file is removed on every exit so a failed swap-in
/// cannot strand it on disk.
fn restore_swapped_image(
    model: &dyn Model,
    s: &SwappedSeq,
    spill: &mut KvSpillManager,
) -> Result<SequenceState> {
    let mut seq = model.alloc_sequence()?;
    // `reader` is dropped at the end of the closure, before `remove_file`.
    let restored = spill
        .open_file(s.swap_id)
        .and_then(|mut reader| model.restore_sequence_state(&mut seq, s.num_blocks, &mut reader));
    if let Err(e) = restored {
        if let Err(fe) = model.free_sequence(&mut seq) {
            tracing::error!("restore_swapped_image: free_sequence after failed restore: {fe:#}");
        }
        let _ = spill.remove_file(s.swap_id);
        return Err(e);
    }
    if let Err(e) = spill.remove_file(s.swap_id) {
        if let Err(fe) = model.free_sequence(&mut seq) {
            tracing::error!("restore_swapped_image: free_sequence after failed unlink: {fe:#}");
        }
        return Err(e);
    }
    Ok(seq)
}

/// Resume a swapped-out sequence by restoring its state from disk.
pub fn resume_swapped_seq(
    _think_end_token: Option<u32>,
    _think_start_token: Option<u32>,
    model: &dyn Model,
    mut s: SwappedSeq,
    spill: &mut KvSpillManager,
) -> Result<ActiveSeq> {
    // Starvation guard: a just-resumed sequence must not be the next KV
    // victim before it makes real progress (see `preempt` module docs).
    let immune_until = s.output_tokens.len() + super::preempt::PREEMPT_IMMUNITY_TOKENS;
    let mut seq = match restore_swapped_image(model, &s, spill) {
        Ok(seq) => seq,
        Err(e) => {
            // The caller (`scheduler::run`'s swap-in loop) has already taken
            // this request out of `swapped`, and it only logs the `Err` — so
            // returning without touching the sink drops the client's channel.
            // That is not a quiet failure: the blocking side turns the closed
            // oneshot into `500 "Inference cancelled"`, which is what the
            // server says when the CLIENT aborted, and the streaming side
            // just ends the SSE body under an HTTP 200 that has already been
            // committed and already carries partial output. A swap-in failure
            // is entirely server-side; the client has to be able to tell.
            send_error_to_sink(&mut s.sink, &format!("swap-in failed: {e:#}"));
            return Err(e);
        }
    };

    // Restore CPU-side metadata.
    seq.tokens = s.tokens;
    seq.seq_len = s.seq_len;
    seq.adapter_slot = s.adapter_slot;
    seq.adapter_id = s.adapter_id;
    // Task #25: swap-out released this seq's slot ref (via free_sequence); a
    // resumed seq re-enters ACTIVE decode WITHOUT re-running the prefill stamp,
    // so re-acquire here to balance that release and re-protect the slot for the
    // remainder of the decode. Stores the freshly resolved index (release keys
    // off it, so the acquire/release stay balanced regardless of any rotate).
    seq.acquired_adapter_slot = model.acquire_adapter_slot(s.adapter_slot);

    Ok(ActiveSeq {
        seq,
        session_hash: s.session_hash,
        last_token: s.last_token,
        output_tokens: s.output_tokens,
        remaining: s.remaining,
        min_tokens: s.min_tokens,
        eos_tokens: s.eos_tokens,
        finished: false,
        guard_stop: None,
        param_close_pending: 0,
        sink: s.sink,
        // cancel_flag isn't preserved across spill/restore — the
        // original stream is long gone by the time a swapped-out seq
        // resumes from disk, so the live guards don't apply here.
        cancel_flag: None,
        temperature: s.temperature,
        top_k: s.top_k,
        top_p: s.top_p,
        top_n_sigma: s.top_n_sigma,
        min_p: s.min_p,
        repetition_penalty: s.repetition_penalty,
        presence_penalty: s.presence_penalty,
        frequency_penalty: s.frequency_penalty,
        repetition_penalty_window: 256,
        lz_penalty: DEFAULT_LZ_PENALTY,
        dry_multiplier: s.dry_multiplier,
        dry_base: s.dry_base,
        dry_allowed_length: s.dry_allowed_length,
        dry_sequence_breakers: s.dry_sequence_breakers,
        logit_bias: s.logit_bias,
        inside_thinking: s.inside_thinking,
        enable_thinking: s.enable_thinking,
        thinking_budget: s.thinking_budget,
        repetition_detection: s.repetition_detection,
        spontaneous_think_budget: s.spontaneous_think_budget,
        thinking_tokens: s.thinking_tokens,
        force_end_thinking: s.force_end_thinking,
        sentence_defer_count: s.sentence_defer_count,
        consecutive_confident: s.consecutive_confident,
        in_code_fence: s.in_code_fence,
        think_end_token: s.think_end_token,
        think_start_token: s.think_start_token,
        think_ended: s.think_ended,
        think_just_ended: s.think_just_ended,
        post_think_emitted: s.post_think_emitted,
        spec_adapt: Default::default(),
        think_skip_count: s.think_skip_count,
        require_tool_call: s.require_tool_call,
        tool_request: s.tool_request,
        tools_present: s.tools_present,
        suppress_tool_call: s.suppress_tool_call,
        disable_mtp: s.disable_mtp,
        mtp_acct: s.mtp_acct,
        content_started: false,
        content_tokens: 0,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: s.think_watchdog_fires,
        think_force_closed: s.think_force_closed,
        rollback_count: s.rollback_count,
        // Decode-rollback SSM snapshots are GPU-resident and not part of
        // the disk swap image — a resumed sequence starts with an empty
        // ring. New boundary snapshots accrue as it decodes again; until
        // one exists, a hybrid-model rollback declines to the hard stop
        // (correct: there is no live snapshot to restore).
        ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
        tool_call_start_token: s.tool_call_start_token,
        tool_call_opened: s.tool_call_opened,
        // Resumed sequences re-enter outside any tool body — even if
        // the snapshot was mid-tool-call, the sample path needs a
        // safe default. Cleared at next emit if we re-cross a marker.
        inside_tool_body: false,
        tool_call_completed: false,
        post_completion_tool_opens: 0,
        tool_body_streak_tokens: 0,
        inside_parameter_body: false,
        param_body_chars_emitted: 0,
        tool_call_end_token: s.tool_call_end_token,
        // Grammar state is not serializable; resumed sequences use legacy fallback.
        grammar_state: None,
        pending_drafts: Vec::new(),
        pending_draft_conf: Vec::new(),
        last_token_time: Instant::now(),
        request_start: s.request_start,
        decode_start: s.decode_start,
        seed: s.seed,
        top_logprobs: s.top_logprobs,
        logprobs_data: s.logprobs_data,
        timeout_at: s.timeout_at,
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(s.temperature),
        cached_prompt_tokens: s.cached_prompt_tokens,
        preempt_immune_until_tokens: immune_until,
    })
}

// Tests live in `lifecycle_tests.rs` (sibling module registered in
// `scheduler/mod.rs`) to keep this file under the 500-line cap.
