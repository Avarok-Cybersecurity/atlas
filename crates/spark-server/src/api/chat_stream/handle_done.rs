// SPDX-License-Identifier: AGPL-3.0-only
//
// `StreamEvent::Done { ... }` arm of the streaming `flat_map`
// closure (originally ~396 LoC).

use axum::response::sse::Event;

use crate::openai::{ChatCompletionChunk, Usage};
use crate::tool_parser;

use super::super::sanitizer::sanitize_content_chunk;
use super::super::stream_guards::flush_content_sanitizer;
use super::ctx::StreamCtx;
use super::state::StreamState;
use super::tool_handlers::{
    handle_complete_tool_call, handle_tool_call_args_fragment, handle_tool_call_delta,
    handle_tool_call_end, handle_tool_call_start,
};

type SseVec = Vec<Result<Event, std::convert::Infallible>>;

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_done(
    state: &mut StreamState,
    ctx: &StreamCtx,
    finish_reason: String,
    completion_tokens: usize,
    time_to_first_token_ms: f64,
    decode_time_ms: f64,
    reasoning_tokens: u32,
    cached_prompt_tokens: u32,
) -> SseVec {
    let mut sse_events: SseVec = Vec::new();

    // ── Stop-string hold-back flush ─────────────────────────────────
    // vLLM's `IncrementalDetokenizer` releases any bytes still in the
    // hold-back window when the stream finalises (see
    // `vllm/v1/engine/detokenizer.py`). Mirror that here: if a match
    // never triggered (`stop_string_triggered == false`) the tail
    // bytes are legitimate output and must be forwarded. Route them
    // through the active detector / sanitizer so the same envelope
    // and leak-marker rules apply — without this, a sub-stop-string
    // suffix that happens to contain a tool-call fragment would
    // bypass the live pipeline.
    if !ctx.stop_strings.is_empty()
        && !state.stop_string_triggered
        && state.stop_string_emitted_len < state.accumulated_content.len()
    {
        let tail = state.accumulated_content[state.stop_string_emitted_len..].to_string();
        state.stop_string_emitted_len = state.accumulated_content.len();
        if !tail.is_empty() {
            if let Some(det) = state.detector.as_mut() {
                let outputs = det.process(&tail);
                for output in outputs {
                    match output {
                        tool_parser::DetectorOutput::Content(text) => {
                            let sanitized = sanitize_content_chunk(
                                &text,
                                &mut state.tag_scan_buf,
                                &mut state.suppressing_param_leak,
                                &mut state.inside_envelope,
                                &ctx.leak_markers,
                            );
                            if !sanitized.is_empty() {
                                let chunk = ChatCompletionChunk::content_chunk(
                                    &ctx.model, &ctx.id, sanitized,
                                );
                                sse_events.push(Ok(Event::default()
                                    .data(serde_json::to_string(&chunk).unwrap_or_default())));
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
                        tool_parser::DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                            handle_tool_call_args_fragment(
                                state,
                                ctx,
                                fragment,
                                idx,
                                &mut sse_events,
                            );
                        }
                        tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                            handle_tool_call_end(state, ctx, idx);
                        }
                    }
                }
            } else {
                let sanitized = sanitize_content_chunk(
                    &tail,
                    &mut state.tag_scan_buf,
                    &mut state.suppressing_param_leak,
                    &mut state.inside_envelope,
                    &ctx.leak_markers,
                );
                if !sanitized.is_empty() {
                    if state.refusal_scan_buf.len() < 16_384 {
                        state.refusal_scan_buf.push_str(&sanitized);
                    }
                    let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, sanitized);
                    sse_events
                        .push(Ok(Event::default()
                            .data(serde_json::to_string(&chunk).unwrap_or_default())));
                }
            }
        }
    }

    // ── Detector flush ──────────────────────────────────────────────
    if state.detector.is_some() {
        let outputs = {
            let det = state.detector.as_mut().expect("detector is Some");
            det.flush()
        };
        for output in outputs {
            match output {
                tool_parser::DetectorOutput::Content(text) => {
                    let sanitized = sanitize_content_chunk(
                        &text,
                        &mut state.tag_scan_buf,
                        &mut state.suppressing_param_leak,
                        &mut state.inside_envelope,
                        &ctx.leak_markers,
                    );
                    if !sanitized.is_empty() {
                        let chunk =
                            ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, sanitized)
                                .with_token_ids(state.take_ids_if(ctx.req_return_token_ids));
                        sse_events.push(Ok(Event::default()
                            .data(serde_json::to_string(&chunk).unwrap_or_default())));
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
                tool_parser::DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                    handle_tool_call_args_fragment(state, ctx, fragment, idx, &mut sse_events);
                }
                tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                    handle_tool_call_end(state, ctx, idx);
                }
            }
        }
    }

    // ── Sanitizer tail flush ────────────────────────────────────────
    let tail = flush_content_sanitizer(
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &ctx.leak_markers,
    );
    if !tail.is_empty() {
        if state.refusal_scan_buf.len() < 16_384 {
            state.refusal_scan_buf.push_str(&tail);
        }
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, tail)
            .with_token_ids(state.take_ids_if(ctx.req_return_token_ids));
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
        ));
    }

    // ── Usage block ─────────────────────────────────────────────────
    let tps = if decode_time_ms > 0.0 {
        completion_tokens.saturating_sub(1) as f64 / (decode_time_ms / 1000.0)
    } else {
        0.0
    };
    let usage = Usage {
        prompt_tokens: ctx.prompt_len,
        completion_tokens,
        total_tokens: ctx.prompt_len + completion_tokens,
        prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
            cached_tokens: cached_prompt_tokens as usize,
            audio_tokens: 0,
        }),
        completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
            reasoning_tokens: reasoning_tokens as usize,
            audio_tokens: 0,
            accepted_prediction_tokens: 0,
            rejected_prediction_tokens: 0,
        }),
        time_to_first_token_ms,
        response_tokens_per_second: tps,
    };

    let fr = if state.tool_loop_capped {
        // A tool-call loop guard (Bug-2 name-run cap, F11 within-dedup,
        // F5 cross-flush dedup, or F44 perm-fail) forcibly ended the
        // response. Signal "length" — OpenAI's slot for a truncated
        // response — so agent clients can break their outer retry
        // loop. Without this override the response otherwise looks like
        // a normal "tool_calls" completion (tool calls *were* emitted)
        // and agents (opencode, etc.) cheerfully run the tools and ask
        // the model to continue, perpetuating the loop one round at a
        // time.
        "length"
    } else if state.detector.as_ref().is_some_and(|d| d.has_tool_calls())
        || state.salvaged_tool_call
    {
        "tool_calls"
    } else {
        finish_reason.as_str()
    };

    // Refusal classification.
    let refusal_signal = if state.detector.as_ref().is_none_or(|d| !d.has_tool_calls()) {
        crate::refusal::detect(&state.refusal_scan_buf)
    } else {
        None
    };
    if let Some(ref r) = refusal_signal {
        let chunk = ChatCompletionChunk::refusal_chunk(&ctx.model, &ctx.id, r.clone());
        let json = serde_json::to_string(&chunk).unwrap_or_default();
        sse_events.push(Ok(Event::default().data(json)));
    }

    // Usage emission strategy.
    let emit_separate_usage = ctx.req_stream_include_usage;
    let usage_for_dump = usage.clone();
    if emit_separate_usage {
        let usage_chunk = ChatCompletionChunk::usage_only_chunk(&ctx.model, &ctx.id, usage.clone());
        let json = serde_json::to_string(&usage_chunk).unwrap_or_default();
        sse_events.push(Ok(Event::default().data(json)));
        // Residual: tokens whose decoded text was buffered/suppressed
        // and never rode a content chunk. Stamping them here keeps
        // Σ token_ids == completion_tokens exactly.
        let final_chunk = ChatCompletionChunk::final_chunk_no_usage(&ctx.model, &ctx.id, fr)
            .with_token_ids(state.take_ids_if(ctx.req_return_token_ids));
        let json = serde_json::to_string(&final_chunk).unwrap_or_default();
        sse_events.push(Ok(Event::default().data(json)));
    } else {
        let chunk = ChatCompletionChunk::done_chunk(&ctx.model, &ctx.id, fr, usage)
            .with_token_ids(state.take_ids_if(ctx.req_return_token_ids));
        let json = serde_json::to_string(&chunk).unwrap_or_default();
        sse_events.push(Ok(Event::default().data(json)));
    }

    // Metrics.
    crate::metrics::REQUESTS_ACTIVE.dec();
    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(ctx.prompt_len as u64);
    crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(completion_tokens as u64);
    crate::metrics::TTFT_SECONDS.observe(time_to_first_token_ms / 1000.0);

    // Rate-limit true-up.
    if let Some(ref rctx) = ctx.req_ctx {
        let actual = (ctx.prompt_len + completion_tokens) as u64;
        let refund = rctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            ctx.state.rate_limiter.refund_tokens(&rctx.identity, refund);
        }
    }

    // --dump synthesized response entry.
    if let (Some(seq), Some(dump)) = (ctx.dump_seq, ctx.state.dump_writer.as_ref()) {
        let has_tool_calls = state.detector.as_ref().is_some_and(|d| d.has_tool_calls());
        let body = serde_json::json!({
            "id": ctx.id,
            "model": ctx.model,
            "object": "chat.completion.synthesized",
            "finish_reason": fr,
            "content": state.refusal_scan_buf,
            "has_tool_calls": has_tool_calls,
            "usage": usage_for_dump,
            "stop_string_triggered": state.stop_string_triggered,
            "loop_watchdog_triggered": state.loop_watchdog_triggered,
            "tool_loop_capped": state.tool_loop_capped,
            "guard_stop": state.guard_stop,
            "_note": "Synthesized from post-sanitizer accumulators; \
                      per-chunk capture is a follow-up.",
        });
        dump.dump_response("/v1/chat/completions", seq, &body, true);
    }

    sse_events
}
