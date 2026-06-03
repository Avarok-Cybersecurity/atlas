// SPDX-License-Identifier: AGPL-3.0-only

//! start_chunked_prefill (chunked prefill orchestration).

use super::*;

/// Start a chunked prefill: process chunk 0, return result.
pub fn start_chunked_prefill(
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    model: &dyn Model,
    mut req: InferenceRequest,
    eos_tokens: &[u32],
    max_prefill_tokens: usize,
    prefill_stream: u64,
    prefill_event: u64,
    grammar_engine: &mut Option<GrammarEngine>,
    spontaneous_think_budget: u32,
) -> Result<StartPrefillResult> {
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
    let dry_multiplier = req.dry_multiplier();
    let dry_base = req.dry_base();
    let dry_allowed_length = req.dry_allowed_length();
    let req_lz_penalty = req.lz_penalty();
    let logit_bias = req.logit_bias().to_vec();
    let req_min_tokens = req.min_tokens();
    let req_session_hash = req.session_hash();
    let req_enable_thinking = req.enable_thinking();
    let req_thinking_budget = req.thinking_budget();
    let req_repetition_detection = req.repetition_detection();
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
    let total = prompt_tokens.len();
    let chunk_len = total.min(max_prefill_tokens);
    let is_last = chunk_len >= total;

    tracing::info!(
        "Chunked prefill start: {} prompt tokens, chunk_size={}, max_tokens={max_tokens}",
        total,
        chunk_len,
    );

    // From here on, `sink` holds the client's response channel. Any
    // error MUST be reported via send_error_to_sink before returning,
    // otherwise the API layer will turn the dropped channel into a
    // misleading "Inference cancelled" error.
    let mut seq = match model.alloc_sequence() {
        Ok(s) => s,
        Err(e) => {
            let msg = format!("alloc_sequence failed: {e:#}");
            send_error_to_sink(&mut sink, &msg);
            return Err(e);
        }
    };
    seq.session_hash = req_session_hash;

    // Guard: free SSM slot on any error after allocation.
    let prefill_result = (|| -> Result<DevicePtr> {
        // Vision: encode images and store embeddings for chunk 0 token overwrite.
        if !image_pixels.is_empty() {
            model.prepare_vision_embed(&image_pixels)?;
            // prepare_vision_embed() runs the vision encoder asynchronously on
            // the default stream, writing this request's patch embeddings into
            // the encoder's buf_out. The chunk-0 embedding injection inside
            // prefill_chunk() runs on prefill_stream and reads buf_out. Without
            // ordering between the two streams, the injection reads buf_out
            // BEFORE this request's encode lands and overlays the PREVIOUS
            // request's image embeddings — lag-by-one cross-image contamination
            // (and torn reads / illegal access under interleaved load). Make
            // prefill_stream wait for the encode to complete before injecting.
            model.record_event(prefill_event, model.default_stream())?;
            model.stream_wait_event(prefill_stream, prefill_event)?;
        }

        // EP: broadcast chunk 0 tokens to worker.
        // Send full prompt length + all tokens so worker can do
        // identical Marconi prefix-cache lookups (bug #33 fix).
        // Uses bulk broadcast (single NCCL op) instead of per-token broadcast
        // which caused NCCL timeouts on long prompts (6K+ tokens = 6K+ broadcasts).
        model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF0)?;
        model.ep_broadcast_cmd(chunk_len as u32)?;
        model.ep_broadcast_cmd(0)?; // chunk_start
        model.ep_broadcast_cmd(prompt_tokens.len() as u32)?; // full prompt length
        model.ep_broadcast_tokens(&prompt_tokens)?;

        model.prefill_chunk(
            &prompt_tokens,
            &mut seq,
            0,
            chunk_len,
            is_last,
            prefill_stream,
        )
    })();

    let logits = match prefill_result {
        Ok(l) => l,
        Err(e) => {
            let msg = format!("prefill_chunk failed: {e:#}");
            send_error_to_sink(&mut sink, &msg);
            if let Err(free_err) = model.free_sequence(&mut seq) {
                tracing::error!(
                    "prefill_a_step: free_sequence (after prefill error): {free_err:#}"
                );
            }
            if let Err(bcast_err) = model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF1)
            {
                tracing::error!(
                    "prefill_a_step: ep_broadcast (after prefill error): {bcast_err:#}"
                );
            }
            return Err(e);
        }
    };

    // Sync prefill stream before sampling or returning to decode.
    // Record event on prefill stream, make default stream wait.
    if let Err(e) = model.record_event(prefill_event, prefill_stream) {
        tracing::error!("prefill_a_step: record_event(prefill_event): {e:#}");
    }
    if let Err(e) = model.stream_wait_event(model.default_stream(), prefill_event) {
        tracing::error!("prefill_a_step: stream_wait_event(default_stream, prefill_event): {e:#}");
    }

    if is_last {
        // Single chunk covered the entire prompt — get first token.
        // #131: constrain the FIRST token with the grammar (and advance the
        // matcher). Mirrors prefill_b_step; no-op when no grammar is active.
        let first = match sample_first_token(
            model,
            logits,
            temperature,
            top_k,
            top_p,
            eos_tokens,
            grammar_state.as_mut(),
        ) {
            Ok(t) => {
                tracing::info!("Prefill first token: {t}");
                t
            }
            Err(e) => {
                let msg = format!("sample_token failed: {e:#}");
                send_error_to_sink(&mut sink, &msg);
                if let Err(free_err) = model.free_sequence(&mut seq) {
                    tracing::error!(
                        "prefill_a_step: free_sequence (after sample error): {free_err:#}"
                    );
                }
                if let Err(bcast_err) =
                    model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF1)
                {
                    tracing::error!(
                        "prefill_a_step: ep_broadcast (after sample error): {bcast_err:#}"
                    );
                }
                return Err(e);
            }
        };

        let spontaneous_think = !req_enable_thinking && think_start_token == Some(first);
        if !spontaneous_think
            && let ResponseSink::Streaming(ref tx) = sink
            && let Err(e) = tx.blocking_send(StreamEvent::Token(first))
        {
            tracing::warn!("prefill_a_step: first-token send failed (receiver dropped): {e}");
        }

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
                output_tokens: vec![first],
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
                dry_multiplier,
                dry_base,
                dry_allowed_length,
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
                sentence_defer_count: 0,
                consecutive_confident: 0,
                in_code_fence: false,
                think_end_token,
                think_start_token,
                think_ended: !req_enable_thinking && think_end_token.is_some(),
                think_just_ended: false,
                think_skip_count: 0,
                require_tool_call: use_legacy_tool_call,
                tool_request,
                tool_call_start_token,
                tool_call_opened: false,
                inside_tool_body: false,
                tool_call_completed: false,
                tool_body_streak_tokens: 0,
                inside_parameter_body: false,
                param_body_chars_emitted: 0,
                suppress_tool_call: req_suppress_tool_call,
                disable_mtp: req_disable_mtp,
                content_started: false,
                content_tokens: 0,
                prose_tokens_since_last_tool: 0,
                think_watchdog_fires: 0,
                rollback_count: 0,
                ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
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
            Ok(StartPrefillResult::Finished)
        } else {
            Ok(StartPrefillResult::Active(ActiveSeq {
                seq,
                session_hash: req_session_hash,
                last_token: first,
                output_tokens: if spontaneous_think {
                    vec![]
                } else {
                    vec![first]
                },
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
                dry_multiplier,
                dry_base,
                dry_allowed_length,
                dry_sequence_breakers: Vec::new(),
                logit_bias: logit_bias.clone(),
                pending_drafts: Vec::new(),
                inside_thinking: spontaneous_think
                    || (req_enable_thinking && think_end_token.is_some()),
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
                think_skip_count: 0,
                require_tool_call: use_legacy_tool_call,
                tool_request,
                tool_call_start_token,
                tool_call_opened: false,
                inside_tool_body: false,
                tool_call_completed: false,
                tool_body_streak_tokens: 0,
                inside_parameter_body: false,
                param_body_chars_emitted: 0,
                suppress_tool_call: req_suppress_tool_call,
                disable_mtp: req_disable_mtp,
                content_started: false,
                content_tokens: 0,
                prose_tokens_since_last_tool: 0,
                think_watchdog_fires: 0,
                rollback_count: 0,
                ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
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
    } else {
        Ok(StartPrefillResult::InProgress(PrefillInProgress {
            prompt_tokens,
            session_hash: req_session_hash,
            seq,
            chunk_offset: chunk_len,
            max_tokens,
            min_tokens: req_min_tokens,
            eos_tokens: eos_tokens.to_vec(),
            sink,
            cancel_flag,
            request_start,
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
            // DRY comes from the request (populated at
            // api.rs from `sampling_presets.tools.dry_*` when tools
            // are active). 0.0 for other presets leaves DRY inert.
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            dry_sequence_breakers: Vec::new(),
            logit_bias,
            enable_thinking: req_enable_thinking,
            thinking_budget: req_thinking_budget,
            repetition_detection: req_repetition_detection,
            spontaneous_think_budget,
            require_tool_call: req_require_tool_call,
            suppress_tool_call: req_suppress_tool_call,
            disable_mtp: req_disable_mtp,
            grammar_state,
            seed: req_seed,
            top_logprobs: req_top_logprobs,
            timeout_at: req_timeout_at,
        }))
    }
}
