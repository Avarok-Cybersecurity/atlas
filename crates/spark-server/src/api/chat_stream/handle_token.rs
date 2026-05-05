// SPDX-License-Identifier: AGPL-3.0-only
//
// `StreamEvent::Token` / `StreamEvent::TokenWithLogprobs` arm of the
// streaming `flat_map` closure (originally ~672 LoC at the top of the
// `chat_stream::chat_completions_stream` body).
//
// Returns the SSE events produced for this single token. Callers
// invoke `futures::stream::iter(...)` on the result to feed the
// `flat_map` output stream.

use axum::response::sse::Event;

use crate::openai::ChatCompletionChunk;
use crate::tool_parser;

use super::super::failures::{bump_f12_tool_call_count, check_loop_watchdog};
use super::super::sanitizer::sanitize_content_chunk;
use super::ctx::StreamCtx;
use super::state::StreamState;
use super::tool_handlers::{
    handle_complete_tool_call, handle_tool_call_delta, handle_tool_call_end, handle_tool_call_start,
};

type SseVec = Vec<Result<Event, std::convert::Infallible>>;

/// Process one token. Returns the SSE events to forward to the
/// client (empty `Vec` is valid).
pub(super) fn handle_token(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> SseVec {
    let mut sse_events: SseVec = Vec::new();
    state.all_toks.push(tok);

    // ── Thinking-phase: token-ID based </think> detection ────────────
    if !state.thinking_done {
        if let Some(end_id) = ctx.state.think_end_token_id
            && tok == end_id
        {
            state.thinking_done = true;
            // Emit only the residual reasoning delta not yet sent
            // by incremental streaming (e.g. trailing bytes held
            // back due to incomplete UTF-8 at prior token boundary).
            // The full reasoning has already been streamed
            // incrementally via reasoning_chunk deltas above —
            // re-emitting the full text here would double it.
            if ctx.enable_thinking && state.all_toks.len() > 1 {
                let full = ctx
                    .state
                    .tokenizer
                    .decode(&state.all_toks[..state.all_toks.len() - 1])
                    .unwrap_or_default();
                let stable = full.trim_end_matches('\u{FFFD}');
                if stable.len() > state.emitted {
                    let residual = &stable[state.emitted..];
                    if !residual.trim().is_empty() {
                        let chunk = ChatCompletionChunk::reasoning_chunk(
                            &ctx.model,
                            &ctx.id,
                            residual.to_string(),
                        );
                        let json = serde_json::to_string(&chunk).unwrap_or_default();
                        sse_events.push(Ok(Event::default().data(json)));
                    }
                }
            }
            // Reset tool detector to clear any thinking-era tag fragments.
            if let Some(ref mut det) = state.detector {
                det.reset();
            }
            state.emitted = 0; // Reset — next decode will be content-only
            state.all_toks.clear(); // Clear thinking tokens from accumulator
            return sse_events;
        }
        // Still in thinking — accumulate but don't emit as content
        if ctx.enable_thinking {
            // Open thinking: emit as reasoning_content
            let full = ctx
                .state
                .tokenizer
                .decode(&state.all_toks)
                .unwrap_or_default();
            let stable_end = full.trim_end_matches('\u{FFFD}').len();
            if stable_end > state.emitted {
                let mut cleaned = full[state.emitted..stable_end].to_string();
                state.emitted = stable_end;
                // Strip format tokens that shouldn't appear in thinking
                cleaned = cleaned.replace("<think>", "");
                if let Some(rest) = cleaned.strip_prefix("assistant\n") {
                    cleaned = rest.to_string();
                } else if let Some(rest) = cleaned.strip_prefix("assistant") {
                    cleaned = rest.to_string();
                }
                while let Some(start) = cleaned.find("<tool_call>") {
                    if let Some(end) = cleaned[start..].find("</tool_call>") {
                        cleaned = format!(
                            "{}{}",
                            &cleaned[..start],
                            &cleaned[start + end + "</tool_call>".len()..]
                        );
                    } else {
                        cleaned = cleaned[..start].to_string();
                        break;
                    }
                }
                if let Some(start) = cleaned.find("<function=") {
                    cleaned = cleaned[..start].to_string();
                }
                // Strip leaked tool-call closing tags from reasoning
                // (observed pattern: `</parameter></function>` right
                // before a role-word repetition loop — the model
                // emits them as BPE tokens after the real tool call
                // has already been structured by the detector).
                for tag in &["</parameter>", "</function>", "</tool_call>"] {
                    cleaned = cleaned.replace(tag, "");
                }
                // Collapse role-word repetition loops in reasoning
                // (Qwen3.5/3.6 post-tool-call hallucination). Pair-
                // collapse `userX...userX` → "" until no adjacent
                // pairs remain; then strip surviving line-bounded
                // standalones (`\nuser\n` → `\n`).
                for word in &["user", "assistant", "tool"] {
                    let pair = format!("{word}{word}");
                    while cleaned.contains(&pair) {
                        cleaned = cleaned.replace(&pair, "");
                    }
                    let nl_form = format!("\n{word}\n");
                    while cleaned.contains(&nl_form) {
                        cleaned = cleaned.replace(&nl_form, "\n");
                    }
                }
                // F19: final structured sanitisation pass catches
                // any leak markers the hand-rolled cleanups missed.
                let cleaned = sanitize_content_chunk(
                    &cleaned,
                    &mut state.reasoning_tag_scan_buf,
                    &mut state.reasoning_suppressing_leak,
                    &mut state.reasoning_inside_envelope,
                    &ctx.leak_markers,
                );
                if !cleaned.trim().is_empty() {
                    let chunk = ChatCompletionChunk::reasoning_chunk(&ctx.model, &ctx.id, cleaned);
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    sse_events.push(Ok(Event::default().data(json)));
                }
            }
        }
        return sse_events;
    }

    // ── Content phase: incremental decode via DecodeStream ───────────
    let decoder = state.content_decoder.get_or_insert_with(|| {
        // SAFETY: ctx.state (Arc<AppState>) is owned by the closure
        // and lives for its entire duration. The DecodeStream borrows
        // &Tokenizer from it. We extend the lifetime because the Arc
        // guarantees the tokenizer outlives the closure (and thus
        // the DecodeStream).
        let tokenizer_ref: &'static crate::tokenizer::ChatTokenizer =
            unsafe { &*(&ctx.state.tokenizer as *const crate::tokenizer::ChatTokenizer) };
        tokenizer_ref.streaming_decoder(true)
    });
    let mut delta = match decoder.step(tok) {
        Ok(Some(chunk)) => chunk,
        Ok(None) => return sse_events,
        Err(e) => {
            tracing::warn!("Streaming decoder error: {e:?}");
            return sse_events;
        }
    };

    // Strip residual think tags from content after thinking is done.
    if state.thinking_done {
        for tag in &[
            "</think>",
            "</thinking>",
            "<thinking>",
            "</analysis>",
            "<analysis>",
        ] {
            while let Some(pos) = delta.find(tag) {
                delta = format!("{}{}", &delta[..pos], delta[pos + tag.len()..].trim_start());
            }
        }
        // If model re-opens <think>, suppress content from <think> onward.
        if let Some(pos) = delta.find("<think>") {
            delta = delta[..pos].to_string();
            state.thinking_done = false;
            state.all_toks.clear();
            state.emitted = 0;
        }
    }

    // Bare role-literal leak (Qwen3.5/3.6) — companion to the
    // scheduler-side <|im_start|> hard-stop.
    {
        let trimmed = delta.trim();
        if delta.len() < 20 && matches!(trimmed, "user" | "assistant" | "tool") {
            tracing::debug!("role-literal strip: dropped bare '{trimmed}' delta");
            delta.clear();
        }
    }

    if delta.is_empty() {
        return sse_events;
    }

    // Multi-token stop sequences via string matching.
    if !ctx.stop_strings.is_empty() && !state.stop_string_triggered {
        state.accumulated_content.push_str(&delta);
        for stop_str in &ctx.stop_strings {
            if let Some(pos) = state.accumulated_content.find(stop_str.as_str()) {
                let content_before_stop = &state.accumulated_content[..pos];
                let already_emitted = state.accumulated_content.len() - delta.len();
                if pos > already_emitted {
                    delta = content_before_stop[already_emitted..].to_string();
                } else {
                    delta = String::new();
                }
                state.stop_string_triggered = true;
                break;
            }
        }
        if state.stop_string_triggered && delta.is_empty() {
            return sse_events;
        }
    }

    if state.stop_string_triggered {
        if !delta.is_empty() {
            let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, delta);
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            sse_events.push(Ok(Event::default().data(json)));
        }
        return sse_events;
    }

    // Fork: detector-active vs pure-content path.
    if state.detector.is_some() {
        // Drain the detector outputs into a local Vec so we can drop
        // the &mut borrow on `state.detector` before the helpers below
        // (which take other &mut state fields) run.
        let outputs = {
            let det = state.detector.as_mut().expect("detector is Some");
            det.process(&delta)
        };
        for output in outputs {
            match output {
                tool_parser::DetectorOutput::Content(text) => {
                    if let Some(events_out) = detector_content_arm(state, ctx, &text) {
                        sse_events.extend(events_out);
                        return sse_events;
                    }
                }
                tool_parser::DetectorOutput::ToolCall(mut tc, tc_idx) => {
                    handle_complete_tool_call(state, ctx, &mut tc, tc_idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallStart {
                    id: tc_id,
                    name,
                    idx,
                } => {
                    handle_tool_call_start(state, ctx, tc_id, name, idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallDelta { args, idx } => {
                    handle_tool_call_delta(state, ctx, args, idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                    handle_tool_call_end(state, ctx, idx);
                }
            }
        }
    } else {
        let sanitized = sanitize_content_chunk(
            &delta,
            &mut state.tag_scan_buf,
            &mut state.suppressing_param_leak,
            &mut state.inside_envelope,
            &ctx.leak_markers,
        );
        if let Some(events_out) = process_detector_content(state, ctx, &sanitized) {
            sse_events.extend(events_out);
            return sse_events;
        }
        // process_detector_content does NOT pre-sanitize when called
        // from the no-detector branch — but the sanitizer was already
        // run above, so the helper's branch handling matches.
    }

    sse_events
}

/// Common processing for a sanitized content chunk: SimHash semantic
/// guard, token-level loop watchdog, salvage on trip, otherwise
/// emit a `content_chunk`. Returns `Some(events)` when the watchdog
/// fired (caller must short-circuit), else `None` (caller continues).
///
/// Note: when called from the detector-active branch, `sanitized`
/// has already been routed through `sanitize_content_chunk`. When
/// called from the no-detector branch, the caller must pre-sanitize
/// (the no-detector path uses the same sanitizer state).
fn process_detector_content(
    state: &mut StreamState,
    ctx: &StreamCtx,
    sanitized_or_raw: &str,
) -> Option<SseVec> {
    // From the detector-active branch the input is the Content(text)
    // payload that still needs sanitization. From the no-detector
    // branch the input is already sanitized. Distinguish via a thin
    // wrapper: detector branch ALSO sanitizes; non-detector branch
    // skips by passing the already-sanitized text. To keep the call
    // site simple, we sanitize here only when the input contains the
    // hallmark of an unfiltered Content payload — which we can't
    // reliably detect. Solution: split into two paths.
    //
    // Inlining: this helper is only called once per branch with the
    // correct input type; it never re-sanitizes. The parameter is the
    // post-sanitizer text in both call sites.
    let sanitized = sanitized_or_raw;

    // F4 SimHash guard.
    let semantic_trip = if !state.loop_watchdog_triggered {
        state.simhash_pending.push_str(sanitized);
        let mut dup = false;
        if crate::loop_simhash::ends_at_sentence_boundary(&state.simhash_pending).is_some()
            || state.simhash_pending.len() >= 1024
        {
            dup = state.simhash_guard.check(&state.simhash_pending);
            state.simhash_pending.clear();
        }
        if state.simhash_pending.len() > 4096 {
            let drop_to = state.simhash_pending.len() / 2;
            state.simhash_pending.drain(..drop_to);
        }
        dup
    } else {
        false
    };

    let token_trip = check_loop_watchdog(
        sanitized,
        &mut state.loop_scan_buf,
        state.loop_watchdog_triggered,
    );

    if semantic_trip || token_trip {
        if semantic_trip {
            tracing::warn!(
                ring_len = state.simhash_guard.len(),
                "SimHash semantic-loop watchdog fired (paraphrased sentence repeat)"
            );
        }
        state.loop_watchdog_triggered = true;
        state.stop_string_triggered = true;

        let salvaged =
            crate::tool_salvage::salvage(&state.loop_scan_buf, &ctx.tool_defs_for_backfill);
        let mut events: SseVec = Vec::new();
        for (idx, tc) in salvaged.iter().enumerate() {
            tracing::warn!(
                tool = %tc.function.name,
                block_index = idx,
                "watchdog salvage: emitting synthetic tool_call",
            );
            bump_f12_tool_call_count(
                &mut state.tool_calls_emitted_count,
                ctx.max_tool_calls_per_response,
                &mut state.stop_string_triggered,
            );
            let start = ChatCompletionChunk::tool_call_start_chunk(&ctx.model, &ctx.id, tc, idx);
            events.push(Ok(
                Event::default().data(serde_json::to_string(&start).unwrap_or_default())
            ));
            let frag = ChatCompletionChunk::tool_call_args_fragment(
                &ctx.model,
                &ctx.id,
                idx,
                &tc.function.arguments,
            );
            events.push(Ok(
                Event::default().data(serde_json::to_string(&frag).unwrap_or_default())
            ));
        }
        if !salvaged.is_empty() {
            state.salvaged_tool_call = true;
        }
        return Some(events);
    }

    if !sanitized.is_empty() {
        if state.refusal_scan_buf.len() < 16_384 {
            state.refusal_scan_buf.push_str(sanitized);
        }
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, sanitized.to_string());
        let json = serde_json::to_string(&chunk).unwrap_or_default();
        let events: SseVec = vec![Ok(Event::default().data(json))];
        return Some(events);
    }
    None
}

/// Detector-active branch's `Content(text)` arm: sanitize first,
/// then run the shared semantic/token watchdog + emit pipeline.
fn detector_content_arm(state: &mut StreamState, ctx: &StreamCtx, text: &str) -> Option<SseVec> {
    let sanitized = sanitize_content_chunk(
        text,
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &mut state.inside_envelope,
        &ctx.leak_markers,
    );
    process_detector_content(state, ctx, &sanitized)
}
