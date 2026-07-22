// SPDX-License-Identifier: AGPL-3.0-only

//! Sequence lifecycle: finish, errors, swap-out, resume.

use super::*;

/// Send final response and free GPU resources for a completed sequence.
pub fn finish_sequence(model: &dyn Model, a: &mut ActiveSeq) {
    let last_tok = a.output_tokens.last().copied();
    let is_eos = last_tok.is_some_and(|t| a.eos_tokens.contains(&t));
    let is_tool_call_end = last_tok == a.tool_call_end_token;
    let reason = if is_eos {
        "stop"
    } else if is_tool_call_end {
        "tool_calls"
    } else {
        "length"
    };
    match &mut a.sink {
        ResponseSink::Streaming(tx) => {
            let ttft_ms = a.decode_start.duration_since(a.request_start).as_secs_f64() * 1000.0;
            let decode_ms = a.decode_start.elapsed().as_secs_f64() * 1000.0;
            if let Err(e) = tx.blocking_send(StreamEvent::Done {
                finish_reason: reason.to_string(),
                prompt_tokens: 0, // prompt_tokens tracked by API layer
                completion_tokens: a.output_tokens.len(),
                time_to_first_token_ms: ttft_ms,
                decode_time_ms: decode_ms,
                reasoning_tokens: a.thinking_tokens,
                cached_prompt_tokens: a.cached_prompt_tokens,
                guard_stop: a.guard_stop,
            }) {
                tracing::warn!(
                    "finish_sequence: streaming Done send failed (receiver dropped): {e}"
                );
            }
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
    tracing::info!("Done: {n} tokens ({reason}) {tps:.1} tok/s, TTFT={ttft_ms:.1}ms");
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
            if let Err(e) = tx.blocking_send(StreamEvent::Error(msg.to_string())) {
                tracing::warn!("send_error: streaming Error send failed (receiver dropped): {e}");
            }
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
            if let Err(e) = tx.blocking_send(StreamEvent::Error(msg.to_string())) {
                tracing::warn!(
                    "send_error_to_sink: streaming Error send failed (receiver dropped): {e}"
                );
            }
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
        model.compact_sequence(&mut active[victim_idx].seq, victim_idx)?;
        // Disown the victim's migrated slot BEFORE the fallible save below: sets
        // the reuse sentinel AND neutralizes the RAII guard so a `?`-early-
        // return (create_file/save_sequence_state error) that drops `a` cannot
        // double-release the slot now owned by the swapped-in sequence.
        model.detach_slot_for_reuse(&mut a.seq);
    }

    let (swap_id, mut writer) = spill.create_file()?;
    model.save_sequence_state(&a.seq, &mut writer)?;
    drop(writer);
    spill.record_usage(swap_id);

    let num_blocks = a.seq.block_table.len();
    let seq_len = a.seq.seq_len;
    let tokens = a.seq.tokens.clone();

    // Free GPU resources (KV blocks + SSM slot).
    let slot_idx = a.seq.slot_idx as u32;
    model.free_sequence(&mut a.seq)?;
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
        content_started: a.content_started,
        content_tokens: a.content_tokens,
        prose_tokens_since_last_tool: a.prose_tokens_since_last_tool,
        think_watchdog_fires: a.think_watchdog_fires,
        think_force_closed: a.think_force_closed,
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

/// Resume a swapped-out sequence by restoring its state from disk.
pub fn resume_swapped_seq(
    _think_end_token: Option<u32>,
    _think_start_token: Option<u32>,
    model: &dyn Model,
    s: SwappedSeq,
    spill: &mut KvSpillManager,
) -> Result<ActiveSeq> {
    let mut seq = model.alloc_sequence()?;
    let mut reader = spill.open_file(s.swap_id)?;
    model.restore_sequence_state(&mut seq, s.num_blocks, &mut reader)?;
    drop(reader);
    spill.remove_file(s.swap_id)?;

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
        last_token_time: Instant::now(),
        request_start: s.request_start,
        decode_start: s.decode_start,
        seed: s.seed,
        top_logprobs: s.top_logprobs,
        logprobs_data: s.logprobs_data,
        timeout_at: s.timeout_at,
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(s.temperature),
        cached_prompt_tokens: s.cached_prompt_tokens,
    })
}
