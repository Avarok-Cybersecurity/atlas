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

use super::super::sanitizer::sanitize_content_chunk;
use super::ctx::StreamCtx;
use super::state::StreamState;
use super::strip::{
    maybe_log_decode_trace, strip_all_preserving_boundary, strip_preserving_boundary,
};
use super::tool_handlers::{
    handle_complete_tool_call, handle_tool_call_delta, handle_tool_call_end, handle_tool_call_start,
};

mod content;
mod stop_string;

use content::{detector_content_arm, process_detector_content};
use stop_string::apply_stop_string_holdback;

pub(super) type SseVec = Vec<Result<Event, std::convert::Infallible>>;

/// Maximum consecutive tokens the stream may spend with
/// `state.suppressing_param_leak == true` (sanitizer holding content
/// because of an orphan `<parameter=` / `<tool_call>` opener without
/// a matching close). When the model degenerates into a doom-loop of
/// partial-envelope leakage — observed 2026-05-24 on
/// opencode-hotfix.jsonl seq=10: 8192 tokens emitted after Atlas
/// rejected a `write({})` call, all suppressed by the sanitizer, no
/// content-loop watchdog fire (the period exceeded 64) — this
/// threshold ends the stream cleanly instead of burning to
/// `max_tokens=8192`. 256 tokens is enough headroom for legitimately
/// long tool-call bodies that take many tokens to close (long
/// `content` field on a `write` call) while bounding worst-case
/// wasted decode at ~10s @ 30 tok/s.
const MAX_SUPPRESS_STREAK_TOKENS: u32 = 256;

/// Process one token. Returns the SSE events to forward to the
/// client (empty `Vec` is valid).
///
/// Thin wrapper around [`handle_token_inner`] that runs the
/// orphan-suppression streak watchdog after every token regardless
/// of which early-return branch fired in the body. The watchdog
/// can't live inside `handle_token_inner` because that function has
/// many early returns (one per emission path) — putting the check
/// at the end of the body would only fire when the natural fall-
/// through is taken, leaving the doom-loop case (long suppressed
/// stream of orphan `<tool_call>` openers) uncaught.
pub(super) fn handle_token(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> SseVec {
    let result = handle_token_inner(state, ctx, tok);

    // Orphan-suppression streak watchdog. The sanitizer flips
    // `suppressing_param_leak=true` when it sees an orphan
    // `<tool_call>` / `<parameter=` opener without a matching close.
    // Suppressing forever (until max_tokens) burns the user's
    // patience and decode budget — observed live as an 8192-token
    // doom loop. If the streak exceeds the bound, end the stream.
    if state.suppressing_param_leak && !state.stop_string_triggered {
        state.suppress_streak_tokens = state.suppress_streak_tokens.saturating_add(1);
        if state.suppress_streak_tokens > MAX_SUPPRESS_STREAK_TOKENS {
            tracing::warn!(
                streak = state.suppress_streak_tokens,
                "orphan tool-call suppression streak exceeded {MAX_SUPPRESS_STREAK_TOKENS} tokens; ending stream",
            );
            state.loop_watchdog_triggered = true;
            state.stop_string_triggered = true;
            state
                .cancel_flag
                .store(true, std::sync::atomic::Ordering::Release);
        }
    } else if !state.suppressing_param_leak {
        state.suppress_streak_tokens = 0;
    }

    result
}

fn handle_token_inner(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> SseVec {
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
                    // Same fix as the in-loop emit: whitespace-only residuals
                    // are legitimate `\n   ` indents that the model emitted;
                    // dropping them would lose chars permanently.
                    if !residual.is_empty() {
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
            // Flush the reasoning sanitizer's tail buffer. Without this, up to
            // ~18 trailing bytes of the final thinking block (or anything held
            // back for partial-tag fusion) are silently dropped. Skip when
            // suppression is active (no close arrived during thinking) — those
            // bytes are intentionally not surfaced.
            if !state.reasoning_suppressing_leak && !state.reasoning_tag_scan_buf.is_empty() {
                let tail = std::mem::take(&mut state.reasoning_tag_scan_buf);
                // Whitespace-only tail can be a real trailing `\n   ` indent
                // — emit anything non-empty so byte boundaries align.
                if !tail.is_empty() {
                    let chunk = ChatCompletionChunk::reasoning_chunk(&ctx.model, &ctx.id, tail);
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    sse_events.push(Ok(axum::response::sse::Event::default().data(json)));
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
            // Layer-A one-shot guard: after the in-think tool-call leak
            // scanner has fired, suppress all subsequent reasoning
            // deltas for this stream. The scheduler's `cancel_flag`
            // (set when the scanner fired) finalises the sequence
            // within one token via `emit_step::emit_token`; this
            // guard catches the in-flight token race so the next
            // opener never reaches the client.
            if state.reasoning_xml_leak_detected {
                return sse_events;
            }
            // Open thinking: emit as reasoning_content
            let full = ctx
                .state
                .tokenizer
                .decode(&state.all_toks)
                .unwrap_or_default();
            let stable_end = full.trim_end_matches('\u{FFFD}').len();
            if stable_end > state.emitted {
                let raw = full[state.emitted..stable_end].to_string();
                let mut cleaned = raw.clone();
                state.emitted = stable_end;
                // Strip format tokens that shouldn't appear in thinking.
                // `<think>` only fires at the literal opener (always
                // whitespace-adjacent in the prompt), so a plain replace
                // is safe here.
                cleaned = cleaned.replace("<think>", "");
                if let Some(rest) = cleaned.strip_prefix("assistant\n") {
                    cleaned = rest.to_string();
                } else if let Some(rest) = cleaned.strip_prefix("assistant") {
                    cleaned = rest.to_string();
                }
                // Boundary-preserving strip: see `strip_preserving_boundary`
                // doc — prevents `the<tool_call>...</tool_call>project`
                // from collapsing to `theproject`.
                while let Some(start) = cleaned.find("<tool_call>") {
                    if let Some(end_rel) = cleaned[start..].find("</tool_call>") {
                        let end = start + end_rel + "</tool_call>".len();
                        cleaned = strip_preserving_boundary(&cleaned, start, end);
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
                // before a role-word repetition loop). Route through
                // `strip_all_preserving_boundary` (2026-05-23 sweep)
                // to avoid gluing words when a closing tag straddles
                // two reasoning sentences.
                for tag in &["</parameter>", "</function>", "</tool_call>"] {
                    cleaned = strip_all_preserving_boundary(&cleaned, tag);
                }
                // Collapse role-word repetition loops (Qwen3.5/3.6
                // post-tool-call hallucination): `userX...userX` →
                // "" until no adjacent pairs remain, then strip
                // line-bounded standalones (`\nuser\n` → `\n`).
                for word in &["user", "assistant", "tool"] {
                    let pair = format!("{word}{word}");
                    cleaned = strip_all_preserving_boundary(&cleaned, &pair);
                    let nl_form = format!("\n{word}\n");
                    while cleaned.contains(&nl_form) {
                        cleaned = cleaned.replace(&nl_form, "\n");
                    }
                }
                maybe_log_decode_trace(&raw, &cleaned, full.len(), stable_end - raw.len());
                // Layer-A in-think tool-call leak scanner. The per-
                // delta strippers above can miss boundary splits
                // (e.g. `<too` in delta N + `l_call>` in delta N+1)
                // and even when they strip, the model keeps emitting
                // the next repetition because its own KV already
                // contains the literal opener. This sliding-window
                // detector across deltas catches the opener on
                // arrival, drops the delta, sets the loop-cap flag
                // (→ finish_reason="length" via the PR #87 override)
                // and flips the scheduler cancel_flag so generation
                // terminates within one token via PR #89.
                let tools_active_request =
                    !ctx.tool_defs_for_backfill.is_empty() || state.detector.is_some();
                if tools_active_request {
                    state.reasoning_xml_scan_buf.push_str(&cleaned);
                    if state.reasoning_xml_scan_buf.len() > 256 {
                        let drop_to = state.reasoning_xml_scan_buf.len() - 256;
                        let cut = state
                            .reasoning_xml_scan_buf
                            .char_indices()
                            .find(|&(i, _)| i >= drop_to)
                            .map(|(i, _)| i)
                            .unwrap_or(state.reasoning_xml_scan_buf.len());
                        state.reasoning_xml_scan_buf.drain(..cut);
                    }
                    let opener = ["<tool_call>", "<function=", "<parameter=", "<invoke "]
                        .iter()
                        .copied()
                        .find(|m| state.reasoning_xml_scan_buf.contains(m));
                    if let Some(op) = opener {
                        state.reasoning_xml_leak_detected = true;
                        state.tool_loop_capped = true;
                        state.stop_string_triggered = true;
                        state
                            .cancel_flag
                            .store(true, std::sync::atomic::Ordering::Release);
                        let tail_start = state
                            .reasoning_xml_scan_buf
                            .char_indices()
                            .rev()
                            .nth(63)
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        let tail = &state.reasoning_xml_scan_buf[tail_start..];
                        tracing::warn!(
                            model = %ctx.model,
                            request_id = %ctx.id,
                            opener = op,
                            tail = %tail,
                            "in-think tool-call leak detected; cancelling sequence (finish_reason will be \"length\")"
                        );
                        return sse_events;
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
                // Emit whitespace-only chunks too. The `sanitize_content_chunk`
                // holdback can roll out runs of `\n   ` (newline + indent) as
                // a single committed chunk when the suffix exceeds tag_max
                // chars; dropping those via `trim().is_empty()` permanently
                // loses byte boundaries because `state.emitted` already
                // advanced past them. Symptom: streamed reasoning has
                // `**\n -Calculate` where the model actually emitted
                // `**\n   - Calculate` — verified byte-for-byte against the
                // non-streaming response on temp=0 seed=42 (live A/B
                // 2026-05-25). Drop only TRULY empty chunks.
                if !cleaned.is_empty() {
                    let chunk = ChatCompletionChunk::reasoning_chunk(&ctx.model, &ctx.id, cleaned);
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    sse_events.push(Ok(Event::default().data(json)));
                }
            }
        }
        return sse_events;
    }

    // ── Content phase: full-decode + slice (matches reasoning path) ──
    //
    // Previously this path used the HF `tokenizers` crate's
    // `DecodeStream` (`decoder.step(tok)`). That incremental decoder
    // drops the leading metaspace byte at certain BPE-token boundaries
    // for byte-level tokenizers like Qwen's GPT-2-style BPE — verified
    // live 2026-05-25 against the FP8 Qwen3.6 model, opencode session
    // `ses_1a0e59bc7ffeFKSvtvWqoswsll`: tool-call `<parameter=content>`
    // for a Cargo.toml emitted `name = test-rust-axum-v32version =
    // 0.1.0edition = 2021` (no newlines between fields, no quotes
    // around values). Non-streaming `tokenizer.decode(&all_toks)`
    // for the same tokens produces the correct multi-line TOML.
    //
    // The fix: mirror the reasoning path — keep `state.all_toks`
    // populated with content tokens (already done at line 86), decode
    // the cumulative list, and emit the byte slice that's stable past
    // `state.emitted`. `trim_end_matches('\u{FFFD}')` defers any
    // incomplete UTF-8 multi-byte sequence at the tail until the next
    // token completes it. `state.all_toks` and `state.emitted` are
    // reset at `</think>` (line 147), so this slice references the
    // post-thinking content only.
    let full = ctx
        .state
        .tokenizer
        .decode(&state.all_toks)
        .unwrap_or_default();
    let stable_end = full.trim_end_matches('\u{FFFD}').len();
    let _ = tok; // tok already in state.all_toks via line 86
    let mut delta = if stable_end > state.emitted {
        let raw = full[state.emitted..stable_end].to_string();
        state.emitted = stable_end;
        raw
    } else {
        return sse_events;
    };
    // Retire the lazy `content_decoder` field — kept in StreamState
    // only to avoid a wider state-struct migration. The HF decoder is
    // no longer the source of truth.
    let _ = &state.content_decoder;

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

    // Multi-token stop sequences via string matching, with a vLLM-style
    // hold-back buffer (see `vllm/v1/engine/detokenizer.py`
    // `IncrementalDetokenizer.update`). All the state mutation lives in
    // `apply_stop_string_holdback` so the algorithm can be unit-tested
    // without spinning up a full `StreamCtx`.
    if !ctx.stop_strings.is_empty() && !state.stop_string_triggered {
        delta = apply_stop_string_holdback(
            &delta,
            &ctx.stop_strings,
            ctx.stop_string_buffer_len,
            &mut state.accumulated_content,
            &mut state.stop_string_emitted_len,
            &mut state.stop_string_triggered,
        );
        if delta.is_empty() {
            // Either everything is sitting in the hold-back window
            // (waiting for the next chunk / stream close) or a match
            // already truncated the emittable bytes to nothing.
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
