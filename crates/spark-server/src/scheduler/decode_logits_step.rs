// SPDX-License-Identifier: AGPL-3.0-only

//! process_decode_logits: post-decode logits processing.

use super::*;

/// Sample and process decode logits for all active sequences.
///
/// Factored out of `step_decode_only` so that `mixed_forward` can reuse
/// the same sampling + token-processing logic without duplication (SSOT).
/// `logits` must point to `[n, vocab_size]` BF16 on device where n = active.len().
pub fn process_decode_logits(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    logits: DevicePtr,
    t0: std::time::Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    reflection_suppress_ids: &[u32],
    adaptive_sampling: bool,
) {
    let n = active.len();

    // Grammar bitmask is CPU-side, so any sequence with active grammar forces
    // the host-side sampling path for its logits slice.
    let any_grammar = active.iter().any(|a| a.grammar_state.is_some());
    let any_logprobs = active.iter().any(|a| a.top_logprobs.is_some());
    // FP32 lm_head models (Gemma-4 dense) MUST use the host-side path —
    // `argmax_batch` assumes BF16 layout and would interpret 4-byte FP32
    // values as 2-byte BF16 pairs, returning garbage tokens.
    let model_logits_fp32 = model.decode_logits_fp32();
    let needs_host_logits = active
        .iter()
        .any(|a| a.inside_thinking || a.think_ended || a.grammar_state.is_some())
        || any_logprobs
        || model_logits_fp32;

    let new_tokens: Vec<(u32, Option<crate::api::TokenLogprobs>)> =
        if active.iter().all(|a| a.temperature == 0.0) && !any_grammar && !needs_host_logits {
            // Fast path: all greedy, no grammar, no thinking — GPU argmax for the full batch.
            match model.argmax_batch(logits, n, 0) {
                Ok(t) => t.into_iter().map(|tok| (tok, None)).collect(),
                Err(e) => {
                    tracing::error!("argmax_batch error: {e:#}");
                    for mut a in active.drain(..) {
                        send_error(model, &mut a, &format!("{e:#}"));
                    }
                    return;
                }
            }
        } else {
            // Host-side path: copy all batch logits to host, sample per-sequence.
            // Required when any sequence has temperature > 0 or grammar constraints.
            let vocab_size = model.vocab_size();
            // FP32 lm_head dispatch (Gemma-4 dense + ATLAS_GEMMA4_FP32_LMHEAD=1).
            // When the model writes FP32 logits to its decode-logits buffer, we
            // copy 4 bytes/element and skip the BF16→FP32 expansion. Earlier
            // bisection at model.rs:1192-1201 incorrectly concluded FP32 lm_head
            // had no effect on Gemma-4 because this dispatch was never wired —
            // the scheduler always read the (stale) BF16 logits buffer.
            // FP32 lm_head dispatch (Gemma-4 dense). When `use_fp32_logits` is
            // on, the per-token decode lm_head writes 4 bytes/element. The
            // passed `logits` pointer is whatever the most-recent forward
            // returned — that's already the correct buffer (prefill or decode).
            // We just need to read it with the matching width.
            let logits_fp32 = model.decode_logits_fp32();
            let elem_bytes = if logits_fp32 { 4 } else { 2 };
            let mut buf = vec![0u8; n * vocab_size * elem_bytes];
            if let Err(e) = model.copy_logits_to_host(logits, &mut buf) {
                tracing::error!("copy_logits_to_host error: {e:#}");
                for mut a in active.drain(..) {
                    send_error(model, &mut a, &format!("{e:#}"));
                }
                return;
            }
            active
                .iter_mut()
                .enumerate()
                .map(|(i, a)| {
                    process_seq_logits(
                        model,
                        a,
                        &buf,
                        i,
                        vocab_size,
                        elem_bytes,
                        logits_fp32,
                        think_end_token,
                        think_start_token,
                        tool_call_start_token,
                        tool_call_end_token,
                        reflection_suppress_ids,
                        adaptive_sampling,
                    )
                })
                .collect()
        };
    let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if tracing::enabled!(tracing::Level::DEBUG) {
        let token_ids: Vec<u32> = new_tokens.iter().map(|(t, _)| *t).collect();
        tracing::debug!(
            "DECODE: n={n} step={step_ms:.1}ms ({:.1} tok/s) tokens={:?}",
            1000.0 * n as f64 / step_ms,
            token_ids,
        );
    }

    let now = Instant::now();
    for (i, (tok, logprobs)) in new_tokens.into_iter().enumerate() {
        let a = &mut active[i];
        a.last_token = tok;
        a.last_token_time = now;

        // Spontaneous <think>: model generates <think> even when thinking
        // was not requested. Enter thinking mode so EOS is suppressed and
        // thinking content is stripped. Matches vLLM's behavior of always
        // parsing <think>...</think> regardless of enable_thinking setting.
        //
        // F9+F10 (2026-04-26): the sample-time logit mask at line ~1716
        // hard-blocks `<think>` when `think_ended=true`, so this branch
        // should not fire after a watchdog has force-closed thinking.
        // Defence-in-depth: if the model somehow still emits <think>
        // (e.g. the start token differs from the masked one in edge
        // cases), decay the budget by `>> watchdog_fires.min(4)` so
        // each successive re-entry has a tighter window. After 4+
        // fires, the budget is 1/16 of normal — the watchdog kills
        // re-entry within a handful of tokens.
        if !a.inside_thinking && think_start_token == Some(tok) {
            let decay_shift = a.think_watchdog_fires.min(4);
            let decayed = a.spontaneous_think_budget >> decay_shift;
            a.inside_thinking = true;
            a.think_ended = false; // reset so </think> detection path works
            a.think_skip_count = 0;
            a.thinking_budget = Some(decayed.max(8)); // floor to keep watchdog functional
            if a.think_watchdog_fires > 0 {
                tracing::debug!(
                    fires = a.think_watchdog_fires,
                    decayed_budget = decayed,
                    "Spontaneous <think> re-entry after watchdog; decayed budget"
                );
            } else {
                tracing::debug!("Spontaneous <think> detected, entering thinking mode");
            }
            continue; // don't emit <think> as content
        }

        // Silently skip </think> tokens outside thinking mode.
        // At long context (37k+), models degenerate into repeating </think>.
        // Skip up to 50 occurrences, then force-stop. This gives cached
        // prompts a chance to produce content while limiting degenerate loops.
        if !a.inside_thinking && think_end_token == Some(tok) {
            a.think_skip_count += 1;
            if a.think_skip_count >= 50 {
                a.finished = true;
            }
            continue;
        }
        // Reset skip counter when a real content token is generated.
        if a.think_ended {
            a.think_skip_count = 0;
        }

        // Advance grammar state with the sampled token — but only
        // once thinking is finished, because thinking tokens are
        // stripped from the API output and should not consume grammar
        // slots (matches the bitmask-skip in the sampler above).
        if !a.inside_thinking
            && let Some(ref mut gs) = a.grammar_state
        {
            gs.accept_token(tok);
        }

        // Thinking tokens don't count toward remaining (thinking is "free").
        if a.inside_thinking {
            if think_end_token == Some(tok) {
                a.inside_thinking = false;
                a.force_end_thinking = false;
                a.consecutive_confident = 0;
                a.think_ended = true;
                // One-shot: pin the next sampled token to the
                // tool-call-start token if the request requires a
                // tool call (Change 3b). Cleared in the `else`
                // branch below on the next emit.
                a.think_just_ended = true;
            } else {
                a.thinking_tokens += 1;
                // Set force_end_thinking when budget exhausted (picked up next iteration)
                if let Some(budget) = a.thinking_budget
                    && a.thinking_tokens >= budget
                    && !a.force_end_thinking
                {
                    a.force_end_thinking = true;
                    tracing::info!("Thinking budget exhausted ({budget} tokens), forcing </think>");
                }
                // Token-level fence-loop detection. Catches the Qwen3.5-35B
                // phrase attractor (`Running:\`\`\`bash cmd\`\`\`Executing:…`
                // cycling) within ~24-60 tokens of the loop starting,
                // instead of waiting for the 256-token thinking budget.
                if !a.force_end_thinking
                    && a.thinking_tokens >= THINK_LOOP_MIN_TOKENS
                    && a.thinking_tokens.is_multiple_of(THINK_LOOP_CHECK_STRIDE)
                    && detect_thinking_token_loop(&a.output_tokens)
                {
                    a.force_end_thinking = true;
                    a.think_watchdog_fires = a.think_watchdog_fires.saturating_add(1);
                    tracing::warn!(
                        thinking_tokens = a.thinking_tokens,
                        watchdog_fires = a.think_watchdog_fires,
                        "Thinking-loop watchdog fired (period-{}…{} repeat in tail); forcing </think> early",
                        THINK_LOOP_PERIOD_MIN,
                        THINK_LOOP_PERIOD_MAX,
                    );
                }
            }
        } else {
            a.remaining -= 1;
            a.content_started = true;
            a.content_tokens = a.content_tokens.saturating_add(1);
            // think_just_ended is a one-shot: it was set when the prior
            // token was `</think>`; clear it now that we've emitted the
            // first content token (which Change 3b's mask pinned to
            // tool_call_start_token when require_tool_call was set).
            a.think_just_ended = false;

            // Content-phase loop watchdog (2026-04-26 Claude Code
            // degeneration fix). Catches the agentic-failure mode
            // where the model emits the same sentence over and over
            // ("I see I've been creating Cargo.toml files but the
            // user hasn't given me a task. Let me wait for their
            // instructions." × 12). LZ penalty at strength 0.2 nudges
            // but cannot break the attractor once established — the
            // hard stop here lets the API path emit a clean
            // finish_reason="length"-equivalent instead of
            // streaming an unending repetition that wedges Claude
            // Code's display. Disabled inside grammar/tool-body
            // because structured JSON repeats are legitimate.
            if enable_loop_watchdog()
                && a.grammar_state.is_none()
                && !a.inside_tool_body
                && a.content_tokens >= CONTENT_LOOP_MIN_TOKENS
                && a.content_tokens.is_multiple_of(CONTENT_LOOP_CHECK_STRIDE)
                && detect_content_token_loop(&a.output_tokens)
            {
                tracing::warn!(
                    content_tokens = a.content_tokens,
                    output_len = a.output_tokens.len(),
                    "Content-loop watchdog fired (period-{}…{} repeat in tail); ending response early",
                    CONTENT_LOOP_PERIOD_MIN,
                    CONTENT_LOOP_PERIOD_MAX,
                );
                a.finished = true;
            }

            // F2 (2026-04-26): bounded inter-tool prose budget.
            // Counts only free-text tokens (not inside tool body,
            // not inside grammar-constrained emission). When the
            // budget trips we end the response cleanly so the next
            // turn can re-plan with fresh context, instead of
            // letting the model emit prose↔tool↔prose↔tool
            // forever (the `tool_choice="auto"` grammar never
            // self-terminates — see grammar.rs:461-462).
            if !a.inside_tool_body && a.grammar_state.is_some() {
                a.prose_tokens_since_last_tool = a.prose_tokens_since_last_tool.saturating_add(1);
                if a.prose_tokens_since_last_tool > MAX_INTER_TOOL_PROSE {
                    tracing::warn!(
                        prose_tokens = a.prose_tokens_since_last_tool,
                        max = MAX_INTER_TOOL_PROSE,
                        "Inter-tool prose budget exhausted, ending response"
                    );
                    a.finished = true;
                }
            }
        }

        // Track <tool_call> token: once seen, legacy tool call requirement is satisfied.
        // Guard with !inside_thinking — a <tool_call> inside thinking is spurious
        // and must not clear require_tool_call (which would allow premature EOS).
        if a.require_tool_call && tool_call_start_token == Some(tok) && !a.inside_thinking {
            a.require_tool_call = false;
            a.tool_call_opened = true;
        }
        // F2 (2026-04-26): reset the inter-tool prose budget on
        // every `<tool_call>` open. This keeps the budget scoped to
        // "free-text since the last tool call started" rather than
        // accumulating across the whole response.
        if tool_call_start_token == Some(tok) && !a.inside_thinking {
            a.prose_tokens_since_last_tool = 0;
        }
        // Safety: if require_tool_call is still set after 512 tokens, the model
        // isn't generating a tool call (grammar may have failed to compile).
        // Clear the flag so EOS is no longer suppressed — prevents infinite gen.
        if a.require_tool_call && a.output_tokens.len() > 512 {
            tracing::warn!(
                "require_tool_call safety: no <tool_call> after 512 tokens, clearing EOS suppression"
            );
            a.require_tool_call = false;
        }

        // Accumulate logprobs data for blocking responses.
        if let Some(lp) = logprobs {
            a.logprobs_data.push(lp);
        }

        // </tool_call> stop: in legacy mode (no grammar), stop after first tool call.
        // When grammar is active, allow the model to generate multiple tool calls —
        // the grammar controls when EOS is valid.
        if tool_call_end_token == Some(tok) && !a.inside_thinking {
            a.output_tokens.push(tok);
            if let ResponseSink::Streaming(ref tx) = a.sink {
                let event = if let Some(lp) = a.logprobs_data.last().cloned() {
                    StreamEvent::TokenWithLogprobs(tok, lp)
                } else {
                    StreamEvent::Token(tok)
                };
                match tx.try_send(event) {
                    Ok(()) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                        tracing::warn!(
                            "Streaming receiver dropped during tool_call_end, finishing sequence"
                        );
                        a.finished = true;
                        continue;
                    }
                    Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                        if let Err(e) = tx.blocking_send(event) {
                            tracing::error!(
                                "Streaming send failed during tool_call_end backpressure: {e}"
                            );
                            a.finished = true;
                            continue;
                        }
                    }
                }
            }
            if a.grammar_state.is_none() {
                // Legacy mode: one tool call per response
                a.finished = true;
            }
            // Mirror finish_sequence (lines ~3445-3448): keep
            // `inside_tool_body` and the grammar FSM in sync with the
            // emitted token stream. The `continue;` below skips the
            // `emit_token()` path that would normally do this, so
            // without these two lines the flag stays `true` for all
            // subsequent prose tokens — sampler penalties stay
            // disabled for the rest of the response, and the grammar
            // bitmask drifts out of sync with the actual emission.
            // Root-caused 2026-04-26 (8-agent sweep, F1).
            a.inside_tool_body = false;
            if let Some(ref mut gs) = a.grammar_state {
                gs.accept_token(tok);
            }
            // F9 companion (2026-04-26): clear `think_ended` at every
            // </tool_call> boundary so legitimate post-tool
            // re-thinking is allowed. F9 masks <think> when
            // `think_ended=true`, but between tool calls the model
            // SHOULD be allowed to re-think (MiniMax-M2 / Qwen3.6
            // pattern per project_minimax_m27_final.md). F10's
            // watchdog-fire counter still applies — repeated
            // re-thinking that loops will decay its budget.
            a.think_ended = false;
            continue;
        }

        // EOS handling: grammar-based, legacy, or min_tokens.
        // Grammar-based: grammar controls when EOS is allowed (is_terminated()).
        // Legacy: require_tool_call suppresses EOS until <tool_call> is seen.
        // min_tokens: suppress EOS until output_tokens.len() >= min_tokens.
        let grammar_suppresses_eos = a
            .grammar_state
            .as_ref()
            .is_some_and(|gs| !gs.is_terminated());
        let legacy_suppresses_eos = a.require_tool_call;
        let min_tokens_suppresses = a.output_tokens.len() < a.min_tokens;
        // Suppress EOS during thinking: <|im_end|> inside <think> is spurious.
        // Only </think> (think_end_token) should end the thinking phase.
        let thinking_suppresses_eos = a.inside_thinking;
        // Post-thinking EOS guard. Empirically (dump fix22b 2026-04-25
        // ses_23b4781f7ffebc7UgkKWedTmjd seq=43): when the thinking-loop
        // watchdog force-closes `</think>` mid-narration, the model can
        // emerge into content mode briefly (often emitting a bare
        // `<write>\n\n` opener) and immediately sample EOS — the
        // session ends with a partial tool-call shell but no real
        // call. We require at least POST_THINK_MIN_CONTENT non-thinking
        // tokens after `think_ended` before EOS is allowed, giving the
        // model the room to actually open a `<tool_call>` block. Same
        // shape as the existing `min_tokens` guard, but counted from
        // the `</think>` boundary so it doesn't penalise turns that
        // never entered thinking. 16 tokens is enough to start
        // `<tool_call>\n<function=NAME>\n<parameter=…` and is well
        // below typical real tool-call output sizes (>100 tokens).
        const POST_THINK_MIN_CONTENT: u32 = 16;
        let post_think_content_tokens =
            (a.output_tokens.len() as u32).saturating_sub(a.thinking_tokens);
        let post_think_suppresses_eos =
            a.think_ended && post_think_content_tokens < POST_THINK_MIN_CONTENT;
        let suppress_eos = grammar_suppresses_eos
            || legacy_suppresses_eos
            || min_tokens_suppresses
            || thinking_suppresses_eos
            || post_think_suppresses_eos;

        if a.eos_tokens.contains(&tok) && !suppress_eos {
            // Stop/EOS token: do NOT stream to client (OpenAI spec: returned text
            // must not contain the stop sequence). The token is still added to
            // output_tokens for correct token count; the API layer strips the
            // decoded text for blocking responses.
            a.output_tokens.push(tok);
            a.finished = true;
        } else if a.eos_tokens.contains(&tok) && suppress_eos {
            // EOS suppressed: grammar not terminated or legacy tool call not yet seen.
            // Don't stop, don't stream the EOS — the model must keep generating.
            // Don't add to output_tokens (EOS is discarded).
        } else {
            a.output_tokens.push(tok);
            // OPENCODE FIX: when the model spontaneously emits `<think>` even
            // though the request didn't ask for thinking (`enable_thinking=false`),
            // the `<think>` open token itself is suppressed (line ~1356), but
            // the thinking-content tokens that follow MUST also be kept off the
            // wire — otherwise opencode persists them as `assistant.content` and
            // on the next turn the model sees its own past garbage (fake
            // `<function=…>`, fake `<tool_response>`) as a "format example" and
            // continues the pattern. Tokens stay in `output_tokens` for the
            // blocking response path's reasoning_content extraction.
            let suppress_stream = a.inside_thinking && !a.enable_thinking;
            if let ResponseSink::Streaming(ref tx) = a.sink
                && !suppress_stream
            {
                let event = if let Some(lp) = a.logprobs_data.last().cloned() {
                    StreamEvent::TokenWithLogprobs(tok, lp)
                } else {
                    StreamEvent::Token(tok)
                };
                match tx.try_send(event) {
                    Ok(()) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                        tracing::debug!(
                            "Streaming receiver dropped (decode_logits), finishing seq"
                        );
                        a.finished = true;
                    }
                    Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                        if let Err(e) = tx.blocking_send(event) {
                            tracing::error!(
                                "Streaming send failed during backpressure (decode_logits): {e}"
                            );
                            a.finished = true;
                        }
                    }
                }
            }
            if a.remaining == 0 {
                tracing::info!(
                    "process_decode_logits: remaining=0, output_tokens={}, thinking_tokens={}",
                    a.output_tokens.len(),
                    a.thinking_tokens
                );
                a.finished = true;
            }
            // Grammar termination = end of sequence. With `stop_after_first=true`
            // (tool_choice="required"), the structural-tag matcher transitions
            // to its terminal state right after the single tool call closes.
            // The model's free distribution past that point can be degenerate
            // (Nemotron-Super-120B emits a `</parameter>` loop and never
            // samples EOS naturally). Finish here instead of letting it run.
            if a.grammar_state
                .as_ref()
                .is_some_and(|gs| gs.is_terminated())
            {
                a.finished = true;
            }

            // Intra-response fuzzy repetition detection: if the last 2*W tokens
            // approximately match the same W-token pattern, the model is looping.
            // Uses Hamming distance with ~12% tolerance to catch loops where the
            // model narrates the same plan with slight wording variations.
            // Skip during tool calls: XML parameter tags have natural repetition
            // (<parameter=..>...</parameter>) that triggers false positives.
            // Use last occurrence positions — completed tool calls shouldn't
            // disable the detector for subsequent text generation.
            let last_tc_start = a
                .tool_call_start_token
                .and_then(|t| a.output_tokens.iter().rposition(|&tok| tok == t));
            let last_tc_end = a
                .tool_call_end_token
                .and_then(|t| a.output_tokens.iter().rposition(|&tok| tok == t));
            let inside_tool_call = match (last_tc_start, last_tc_end) {
                (Some(start), Some(end)) => start > end,
                (Some(_), None) => true,
                _ => false,
            };
            if enable_loop_watchdog()
                && !a.finished
                && !a.inside_thinking
                && !inside_tool_call
                && let Some((pattern_len, mis_a, mis_b)) = detect_fuzzy_repetition(&a.output_tokens)
            {
                tracing::warn!(
                    "Fuzzy repetition: {pattern_len}-tok pattern x3 ({mis_a}+{mis_b} \
                     mismatches), stopping at {} tokens",
                    a.output_tokens.len()
                );
                a.finished = true;
            }

            // Check request timeout.
            if !a.finished
                && let Some(deadline) = a.timeout_at
                && Instant::now() >= deadline
            {
                tracing::warn!("Request timeout after {:?}", a.request_start.elapsed());
                a.finished = true;
            }
        }
    }
}
