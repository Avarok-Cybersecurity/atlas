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
    let req_enable_thinking = req.enable_thinking();
    let req_thinking_budget = req.thinking_budget();
    if req_enable_thinking {
        tracing::info!("Thinking enabled, budget={:?}", req_thinking_budget);
    }
    let req_require_tool_call = req.require_tool_call();
    let req_suppress_tool_call = req.suppress_tool_call();
    let req_disable_mtp = req.disable_mtp();
    let req_seed = req.seed();
    let req_top_logprobs = req.top_logprobs();
    let req_timeout_at = req.timeout_at();
    let grammar_spec = req.take_grammar_spec();
    let grammar_state = compile_grammar_state(grammar_engine, &grammar_spec);
    let (prompt_tokens, max_tokens, mut sink, image_pixels, temperature, cancel_flag) = match req
    {
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
        sample_token(model, logits, temperature, top_k, top_p, eos_tokens)
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
    if !spontaneous_think
        && let ResponseSink::Streaming(ref tx) = sink
        && let Err(e) = tx.blocking_send(StreamEvent::Token(first))
    {
        tracing::warn!("prefill_b_step: first-token send failed (receiver dropped): {e}");
    }

    let output_tokens = if spontaneous_think {
        vec![]
    } else {
        vec![first]
    };

    // When grammar is active, disable legacy require_tool_call (grammar handles EOS).
    let use_legacy_tool_call =
        req_require_tool_call && grammar_state.is_none() && tool_call_start_token.is_some();

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
            spontaneous_think_budget,
            thinking_tokens: 0,
            cached_prompt_tokens: cached_prompt_tok,
            force_end_thinking: false,
            consecutive_confident: 0,
            in_code_fence: false,
            think_end_token,
            think_start_token,
            think_ended: !req_enable_thinking && think_end_token.is_some(),
            think_just_ended: false,
            think_skip_count: 0,
            require_tool_call: use_legacy_tool_call,
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
        spontaneous_think_budget,
        thinking_tokens: 0,
        cached_prompt_tokens: cached_prompt_tok,
        force_end_thinking: false,
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
        think_skip_count: 0,
        require_tool_call: use_legacy_tool_call,
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
