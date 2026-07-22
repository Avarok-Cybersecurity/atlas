// SPDX-License-Identifier: AGPL-3.0-only

//! prefill_request (single-shot prefill).

use super::*;

/// Prefill a new request and return an ActiveSeq ready for batched decode.
/// Returns None if the sequence completed during prefill (EOS on first token).
pub fn prefill_request(
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    model: &dyn Model,
    mut req: InferenceRequest,
    eos_tokens: &[u32],
    grammar_engine: &mut Option<GrammarEngine>,
    spontaneous_think_budget: u32,
    // Beam co-dispatch: the pre-pass's fused hypothesis for this request (see
    // `resolve_beam_hyp`); None ⇒ run the per-request search.
    precomputed_beam_hyp: Option<Vec<u32>>,
) -> Result<Option<ActiveSeq>> {
    // Merge user-supplied stop tokens with model EOS tokens.
    let stop_tokens = req.take_stop_tokens();
    let eos_tokens = if stop_tokens.is_empty() {
        eos_tokens.to_vec()
    } else {
        let mut merged = eos_tokens.to_vec();
        merged.extend(stop_tokens);
        merged.sort_unstable();
        merged.dedup();
        merged
    };
    let eos_tokens = &eos_tokens;

    let top_k = req.top_k();
    let top_p = req.top_p();
    let top_n_sigma = req.top_n_sigma();
    let min_p = req.min_p();
    let repetition_penalty = req.repetition_penalty();
    let presence_penalty = req.presence_penalty();
    let frequency_penalty = req.frequency_penalty();
    let _dry_multiplier = req.dry_multiplier();
    let _dry_base = req.dry_base();
    let _dry_allowed_length = req.dry_allowed_length();
    let req_lz_penalty = req.lz_penalty();
    let logit_bias = req.logit_bias().to_vec();
    let req_min_tokens = req.min_tokens();
    let req_session_hash = req.session_hash();
    let req_adapter_slot = req.adapter_slot(); // M2 per-request LoRA routing
    let req_src_lang = req.src_lang_id();
    let req_tgt_lang = req.tgt_lang_id();
    let req_num_beams = req.num_beams();
    let req_length_penalty = req.length_penalty();
    let req_early_stopping = req.early_stopping();
    let req_enable_thinking = req.enable_thinking();
    let req_thinking_budget = req.thinking_budget();
    let req_repetition_detection = req.repetition_detection();
    if req_enable_thinking {
        tracing::info!("Thinking enabled, budget={:?}", req_thinking_budget);
    }
    let req_require_tool_call = req.require_tool_call();
    let req_tools_present = req.tools_present();
    let req_suppress_tool_call = req.suppress_tool_call();
    let req_disable_mtp = req.disable_mtp();
    let req_seed = req.seed();
    let req_top_logprobs = req.top_logprobs();
    let req_timeout_at = req.timeout_at();
    let grammar_spec = req.take_grammar_spec();
    let mut grammar_state = compile_grammar_state(grammar_engine, &grammar_spec, eos_tokens);
    let (prompt_tokens, max_tokens, mut sink, image_pixels, temperature, cancel_flag) = match req {
        InferenceRequest::Streaming {
            prompt_tokens,
            max_tokens,
            temperature,
            token_tx,
            image_pixels,
            cancel_flag,
            ..
        } => (
            prompt_tokens,
            max_tokens,
            ResponseSink::Streaming(token_tx),
            image_pixels,
            temperature,
            Some(cancel_flag),
        ),
        InferenceRequest::Blocking {
            prompt_tokens,
            max_tokens,
            temperature,
            response_tx,
            image_pixels,
            ..
        } => (
            prompt_tokens,
            max_tokens,
            ResponseSink::Blocking(Some(response_tx)),
            image_pixels,
            temperature,
            None,
        ),
    };

    let request_start = Instant::now();
    tracing::info!(
        "Prefilling: {} prompt tokens, max_tokens={max_tokens}",
        prompt_tokens.len(),
    );
    // `sink` was extracted from `req` above. From here on, ANY error must
    // be surfaced to the client via `sink` BEFORE returning Err, otherwise
    // the client sees the misleading "Inference cancelled" error from the
    // API layer when the channel drops silently.
    let mut seq = match model.alloc_sequence() {
        Ok(s) => s,
        Err(e) => {
            let msg = format!("alloc_sequence failed: {e:#}");
            send_error_to_sink(&mut sink, &msg);
            return Err(e);
        }
    };
    seq.session_hash = req_session_hash;
    seq.adapter_slot = req_adapter_slot;
    seq.src_lang_id = req_src_lang;
    seq.tgt_lang_id = req_tgt_lang;
    seq.num_beams = req_num_beams;
    seq.length_penalty = req_length_penalty;
    seq.early_stopping = req_early_stopping;
    // Task #24: resolve the STABLE adapter_id NOW (model owns the authoritative
    // active slot for the `-1 = defer to active` case). Keys the KV/prefix cache
    // so this request reuses only same-adapter blocks.
    seq.adapter_id = model.adapter_id_for(req_adapter_slot);
    // Task #25: acquire a ref on the resolved LoRA slot so a swap into it is
    // refused while this seq is in-flight; released symmetrically in
    // free_sequence (normal finish + the error guard below both route there).
    seq.acquired_adapter_slot = model.acquire_adapter_slot(req_adapter_slot);

    // Beam search (NLLB): run the whole search to completion in the model and
    // build a FINISHED ActiveSeq carrying the winning hypothesis, bypassing the
    // token-by-token decode loop. Only reachable for blocking requests (streaming
    // + beam is rejected in the handler).
    if model.supports_beam() && seq.num_beams > 1 {
        let hyp = match resolve_beam_hyp(
            model,
            precomputed_beam_hyp,
            &seq,
            &prompt_tokens,
            max_tokens,
        ) {
            Ok(h) => h,
            Err(e) => {
                send_error_to_sink(&mut sink, &format!("beam search failed: {e:#}"));
                let _ = model.free_sequence(&mut seq);
                return Err(e);
            }
        };
        let last = hyp.last().copied().unwrap_or(0);
        let use_legacy_tool_call =
            req_require_tool_call && grammar_state.is_none() && tool_call_start_token.is_some();
        let tool_request = grammar_state.is_some() || use_legacy_tool_call;
        let now = Instant::now();
        let cached_prompt_tok = seq.cached_prefix_tokens as u32;
        let mut a = ActiveSeq {
            seq,
            session_hash: req_session_hash,
            last_token: last,
            output_tokens: hyp,
            remaining: 0,
            min_tokens: req_min_tokens,
            eos_tokens: eos_tokens.to_vec(),
            finished: true,
            guard_stop: None,
            param_close_pending: 0,
            sink,
            cancel_flag: cancel_flag.clone(),
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            repetition_penalty_window: 256,
            presence_penalty,
            frequency_penalty,
            lz_penalty: req_lz_penalty,
            dry_multiplier: DEFAULT_DRY_MULTIPLIER,
            dry_base: DEFAULT_DRY_BASE,
            dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
            dry_sequence_breakers: Vec::new(),
            logit_bias: logit_bias.clone(),
            pending_drafts: Vec::new(),
            inside_thinking: req_enable_thinking && think_end_token.is_some(),
            enable_thinking: req_enable_thinking,
            thinking_budget: req_thinking_budget,
            repetition_detection: req_repetition_detection,
            spontaneous_think_budget,
            thinking_tokens: 0,
            cached_prompt_tokens: cached_prompt_tok,
            force_end_thinking: false,
            think_force_closed: false,
            sentence_defer_count: 0,
            consecutive_confident: 0,
            in_code_fence: false,
            think_end_token,
            think_start_token,
            think_ended: !req_enable_thinking && think_end_token.is_some(),
            think_just_ended: false,
            post_think_emitted: 0,
            spec_adapt: Default::default(),
            think_skip_count: 0,
            require_tool_call: use_legacy_tool_call,
            tool_request,
            tools_present: req_tools_present,
            suppress_tool_call: req_suppress_tool_call,
            disable_mtp: req_disable_mtp,
            content_started: false,
            content_tokens: 0,
            prose_tokens_since_last_tool: 0,
            think_watchdog_fires: 0,
            rollback_count: 0,
            ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
            tool_call_start_token,
            tool_call_opened: false,
            inside_tool_body: false,
            tool_call_completed: false,
            post_completion_tool_opens: 0,
            tool_body_streak_tokens: 0,
            inside_parameter_body: false,
            param_body_chars_emitted: 0,
            tool_call_end_token,
            grammar_state,
            last_token_time: now,
            request_start,
            decode_start: now,
            seed: req_seed,
            top_logprobs: req_top_logprobs,
            logprobs_data: Vec::new(),
            timeout_at: req_timeout_at,
            adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(temperature),
        };
        finish_sequence(model, &mut a);
        return Ok(None);
    }

    // Guard: free SSM slot on any error after allocation (Bug #16).
    let prefill_result = (|| -> Result<u32> {
        // Vision: encode images and store embeddings for prefill token overwrite.
        if !image_pixels.is_empty() {
            model.prepare_vision_embed(&image_pixels)?;
        }

        // EP: broadcast prefill command + tokens to worker (bulk, single NCCL op).
        model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF0)?;
        model.ep_broadcast_cmd(prompt_tokens.len() as u32)?;
        model.ep_broadcast_cmd(0)?; // chunk_start = 0 (non-chunked)
        model.ep_broadcast_cmd(prompt_tokens.len() as u32)?; // full prompt length
        model.ep_broadcast_tokens(&prompt_tokens)?;

        let logits = model.prefill(&prompt_tokens, &mut seq, 0)?;
        // #131: constrain the FIRST token with the grammar too (and advance
        // the matcher). The plain decode loop only masks/accepts tokens 2..N,
        // so without this a leading prose token escapes before the grammar's
        // opening `{`. No-op vs `sample_token` when no grammar is active.
        // P1-4 (2026-07-09): thread the resolved `min_p` (request +
        // MODEL.toml floor via sampling_setup) — the first-token sample
        // previously ran with a hardcoded 0.0 min_p, bypassing the FP8
        // argmax-flip safety net. Kill-switch: ATLAS_NO_MTP_MINP=1.
        sample_first_token(
            model,
            logits,
            temperature,
            top_k,
            top_p,
            min_p,
            eos_tokens,
            grammar_state.as_mut(),
        )
    })();

    let first = match prefill_result {
        Ok(token) => token,
        Err(e) => {
            let msg = format!("prefill failed: {e:#}");
            send_error_to_sink(&mut sink, &msg);
            if let Err(free_err) = model.free_sequence(&mut seq) {
                tracing::error!(
                    "prefill_b_step: free_sequence (after prefill error): {free_err:#}"
                );
            }
            if let Err(bcast_err) = model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF1)
            {
                tracing::error!(
                    "prefill_b_step: ep_broadcast (after prefill error): {bcast_err:#}"
                );
            }
            return Err(e);
        }
    };

    // Spontaneous <think>: if the first token is <think> and thinking was not
    // requested, suppress it and enter thinking mode on the ActiveSeq.
    let spontaneous_think = !req_enable_thinking && think_start_token == Some(first);
    // Legacy echo+logprobs: prompt logprobs precede any token event.
    if seq.collect_prompt_logprobs.is_some()
        && let ResponseSink::Streaming(ref tx) = sink
    {
        let lps: Vec<crate::api::TokenLogprobs> = seq
            .prompt_logprobs
            .drain(..)
            .map(|p| crate::api::TokenLogprobs {
                token_id: p.token_id,
                logprob: p.logprob,
                top: p.top,
            })
            .collect();
        if let Err(e) = tx.blocking_send(StreamEvent::PromptLogprobs(lps)) {
            tracing::warn!("prefill_b_step: prompt-logprobs send failed: {e}");
        }
    }
    if !spontaneous_think
        && max_tokens > 0
        && let ResponseSink::Streaming(ref tx) = sink
        && let Err(e) = tx.blocking_send(StreamEvent::Token(first))
    {
        tracing::warn!("prefill_b_step: first-token send failed (receiver dropped): {e}");
    }

    let output_tokens = if spontaneous_think || max_tokens == 0 {
        vec![]
    } else {
        vec![first]
    };

    // When grammar is active, disable legacy require_tool_call (grammar handles EOS).
    let use_legacy_tool_call =
        req_require_tool_call && grammar_state.is_none() && tool_call_start_token.is_some();
    // F4: sticky tool-request flag — grammar attached OR legacy tool path.
    // Computed before `grammar_state` is moved into the ActiveSeq below.
    let tool_request = grammar_state.is_some() || use_legacy_tool_call;

    let now = Instant::now();
    let cached_prompt_tok = seq.cached_prefix_tokens as u32;

    if !spontaneous_think && (eos_tokens.contains(&first) || max_tokens <= 1) {
        let mut a = ActiveSeq {
            seq,
            session_hash: req_session_hash,
            last_token: first,
            output_tokens,
            remaining: 0,
            min_tokens: req_min_tokens,
            eos_tokens: eos_tokens.to_vec(),
            finished: true,
            guard_stop: None,
            param_close_pending: 0,
            sink,
            cancel_flag: cancel_flag.clone(),
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            repetition_penalty_window: 256,
            presence_penalty,
            frequency_penalty,
            lz_penalty: req_lz_penalty,
            dry_multiplier: DEFAULT_DRY_MULTIPLIER,
            dry_base: DEFAULT_DRY_BASE,
            dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
            dry_sequence_breakers: Vec::new(),
            logit_bias: logit_bias.clone(),
            pending_drafts: Vec::new(),
            inside_thinking: req_enable_thinking && think_end_token.is_some(),
            enable_thinking: req_enable_thinking,
            thinking_budget: req_thinking_budget,
            repetition_detection: req_repetition_detection,
            spontaneous_think_budget,
            thinking_tokens: 0,
            cached_prompt_tokens: cached_prompt_tok,
            force_end_thinking: false,
            think_force_closed: false,
            sentence_defer_count: 0,
            consecutive_confident: 0,
            in_code_fence: false,
            think_end_token,
            think_start_token,
            think_ended: !req_enable_thinking && think_end_token.is_some(),
            think_just_ended: false,
            post_think_emitted: 0,
            spec_adapt: Default::default(),
            think_skip_count: 0,
            require_tool_call: use_legacy_tool_call,
            tool_request,
            tools_present: req_tools_present,
            suppress_tool_call: req_suppress_tool_call,
            disable_mtp: req_disable_mtp,
            content_started: false,
            content_tokens: 0,
            prose_tokens_since_last_tool: 0,
            think_watchdog_fires: 0,
            rollback_count: 0,
            ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
            tool_call_start_token,
            tool_call_opened: false,
            inside_tool_body: false,
            tool_call_completed: false,
            post_completion_tool_opens: 0,
            tool_body_streak_tokens: 0,
            inside_parameter_body: false,
            param_body_chars_emitted: 0,
            tool_call_end_token,
            grammar_state,
            last_token_time: now,
            request_start,
            decode_start: now,
            seed: req_seed,
            top_logprobs: req_top_logprobs,
            logprobs_data: Vec::new(),
            timeout_at: req_timeout_at,
            adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(temperature),
        };
        finish_sequence(model, &mut a);
        return Ok(None);
    }

    Ok(Some(ActiveSeq {
        seq,
        session_hash: req_session_hash,
        last_token: first,
        output_tokens,
        remaining: max_tokens - 1,
        min_tokens: req_min_tokens,
        eos_tokens: eos_tokens.to_vec(),
        finished: false,
        guard_stop: None,
        param_close_pending: 0,
        sink,
        cancel_flag,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        repetition_penalty_window: 256,
        presence_penalty,
        frequency_penalty,
        lz_penalty: req_lz_penalty,
        dry_multiplier: DEFAULT_DRY_MULTIPLIER,
        dry_base: DEFAULT_DRY_BASE,
        dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
        dry_sequence_breakers: Vec::new(),
        logit_bias,
        pending_drafts: Vec::new(),
        inside_thinking: spontaneous_think || (req_enable_thinking && think_end_token.is_some()),
        enable_thinking: req_enable_thinking,
        thinking_budget: if spontaneous_think {
            Some(spontaneous_think_budget)
        } else {
            req_thinking_budget
        },
        repetition_detection: req_repetition_detection,
        spontaneous_think_budget,
        thinking_tokens: 0,
        cached_prompt_tokens: cached_prompt_tok,
        force_end_thinking: false,
        think_force_closed: false,
        sentence_defer_count: 0,
        consecutive_confident: 0,
        in_code_fence: false,
        think_end_token,
        think_start_token,
        think_ended: if spontaneous_think {
            false
        } else {
            !req_enable_thinking && think_end_token.is_some()
        },
        think_just_ended: false,
        post_think_emitted: 0,
        spec_adapt: Default::default(),
        think_skip_count: 0,
        require_tool_call: use_legacy_tool_call,
        tool_request,
        tools_present: req_tools_present,
        suppress_tool_call: req_suppress_tool_call,
        disable_mtp: req_disable_mtp,
        content_started: false,
        content_tokens: 0,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: 0,
        rollback_count: 0,
        ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
        tool_call_start_token,
        tool_call_opened: false,
        inside_tool_body: false,
        tool_call_completed: false,
        post_completion_tool_opens: 0,
        tool_body_streak_tokens: 0,
        inside_parameter_body: false,
        param_body_chars_emitted: 0,
        tool_call_end_token,
        grammar_state,
        last_token_time: now,
        request_start,
        decode_start: now,
        seed: req_seed,
        top_logprobs: req_top_logprobs,
        logprobs_data: Vec::new(),
        timeout_at: req_timeout_at,
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(temperature),
    }))
}
