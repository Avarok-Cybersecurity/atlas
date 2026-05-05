// SPDX-License-Identifier: AGPL-3.0-only
//
// Helpers for the four `DetectorOutput` variants emitted by the
// streaming tool-call detector. Shared by both `handle_token` (mid-
// stream `process()` outputs) and `handle_done` (end-of-stream
// `flush()` outputs).

use axum::response::sse::Event;

use crate::openai::ChatCompletionChunk;
use crate::tool_parser;

use super::super::failures::{
    bump_f12_tool_call_count, f44_check_permanent_failure, flush_content_sanitizer,
};
use super::ctx::StreamCtx;
use super::state::StreamState;

type SseVec = Vec<Result<Event, std::convert::Infallible>>;

/// `DetectorOutput::ToolCall(tc, idx)`: complete tool call.
pub(super) fn handle_complete_tool_call(
    state: &mut StreamState,
    ctx: &StreamCtx,
    tc: &mut tool_parser::ToolCall,
    tc_idx: usize,
    sse_events: &mut SseVec,
) {
    // Content → Tool boundary: flush sanitiser tail.
    let pre_tool_tail = flush_content_sanitizer(
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &ctx.leak_markers,
    );
    if !pre_tool_tail.is_empty() {
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, pre_tool_tail);
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
        ));
    }
    tool_parser::backfill_required_params(std::slice::from_mut(tc), &ctx.tool_defs_for_backfill);
    if let Some(ref cwd) = ctx.cwd_for_normalize {
        tool_parser::normalize_paths(std::slice::from_mut(tc), cwd);
    }
    if let Err(e) = tool_parser::validate_single_tool_call(tc, &ctx.tool_defs_for_backfill) {
        tracing::warn!(
            tool = %tc.function.name,
            "tool call validation error: {e}; replacing with content and ending"
        );
        let msg = format!("[atlas] Tool call rejected: {e}");
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, msg);
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
        ));
        state.stop_string_triggered = true;
    } else if state
        .tool_arg_dedup
        .check(&tc.function.name, &tc.function.arguments)
    {
        tracing::warn!(
            tool = %tc.function.name,
            "tool-arg dedup tripped: refusing redundant tool_call and ending response"
        );
        state.stop_string_triggered = true;
    } else {
        bump_f12_tool_call_count(
            &mut state.tool_calls_emitted_count,
            ctx.max_tool_calls_per_response,
            &mut state.stop_string_triggered,
        );
        let start = ChatCompletionChunk::tool_call_start_chunk(&ctx.model, &ctx.id, tc, tc_idx);
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&start).unwrap_or_default())
        ));
        let frag = ChatCompletionChunk::tool_call_args_fragment(
            &ctx.model,
            &ctx.id,
            tc_idx,
            &tc.function.arguments,
        );
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&frag).unwrap_or_default())
        ));
    }
}

/// `DetectorOutput::ToolCallStart` — incremental: emit header now.
pub(super) fn handle_tool_call_start(
    state: &mut StreamState,
    ctx: &StreamCtx,
    tc_id: String,
    name: String,
    idx: usize,
    sse_events: &mut SseVec,
) {
    let pre_tool_tail = flush_content_sanitizer(
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &ctx.leak_markers,
    );
    if !pre_tool_tail.is_empty() {
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, pre_tool_tail);
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
        ));
    }
    state
        .streaming_tool_args
        .insert(idx, (name.clone(), String::new()));
    let tc = tool_parser::ToolCall {
        id: tc_id,
        call_type: "function".to_string(),
        function: tool_parser::FunctionCall {
            name,
            arguments: String::new(),
        },
    };
    bump_f12_tool_call_count(
        &mut state.tool_calls_emitted_count,
        ctx.max_tool_calls_per_response,
        &mut state.stop_string_triggered,
    );
    let start = ChatCompletionChunk::tool_call_start_chunk(&ctx.model, &ctx.id, &tc, idx);
    sse_events.push(Ok(
        Event::default().data(serde_json::to_string(&start).unwrap_or_default())
    ));
}

/// `DetectorOutput::ToolCallDelta` — incremental: append args.
pub(super) fn handle_tool_call_delta(
    state: &mut StreamState,
    ctx: &StreamCtx,
    args: String,
    idx: usize,
    sse_events: &mut SseVec,
) {
    if let Some(entry) = state.streaming_tool_args.get_mut(&idx) {
        entry.1.push_str(&args);
    }
    if !args.is_empty() {
        let frag = ChatCompletionChunk::tool_call_args_fragment(&ctx.model, &ctx.id, idx, &args);
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&frag).unwrap_or_default())
        ));
    }
}

/// `DetectorOutput::ToolCallEnd` — F11 within-response dedup +
/// F44 cross-turn permanent-failure check.
pub(super) fn handle_tool_call_end(state: &mut StreamState, ctx: &StreamCtx, idx: usize) {
    if let Some((name, args_json)) = state.streaming_tool_args.remove(&idx) {
        if state.tool_arg_dedup_within.check(&name, &args_json) {
            tracing::warn!(
                tool = %name,
                "F11 within-response dedup tripped: 2+ identical streaming tool calls; ending response"
            );
            state.stop_string_triggered = true;
        } else if ctx.f44_cache_active
            && f44_check_permanent_failure(&ctx.f44_cache, &name, &args_json)
        {
            tracing::warn!(
                tool = %name,
                "F44 streaming circuit-breaker tripped: tool_call matches a permanently-failed prior call; ending response"
            );
            state.stop_string_triggered = true;
        }
    }
}
