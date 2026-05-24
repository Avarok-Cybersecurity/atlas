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
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, pre_tool_tail)
            .with_token_ids(state.take_ids_if(ctx.req_return_token_ids));
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
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, msg)
            .with_token_ids(state.take_ids_if(ctx.req_return_token_ids));
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
        state.tool_loop_capped = true;
    } else {
        // Bug-2 name-run cap (mirrors handle_tool_call_end): catches
        // runaway loops in the complete-tool-call path that
        // tool_arg_dedup misses because of args drift.
        let run_len = match &state.name_run {
            Some((prev, n)) if prev == &tc.function.name => n + 1,
            _ => 1,
        };
        state.name_run = Some((tc.function.name.clone(), run_len));
        if run_len >= MAX_CONSEC_SAME_NAME_CALLS {
            tracing::warn!(
                tool = %tc.function.name,
                run = run_len,
                "Bug-2 name-run cap tripped (complete-call path): {run_len} successive `{}` tool calls; ending response",
                tc.function.name
            );
            state.stop_string_triggered = true;
            state.tool_loop_capped = true;
        }
        bump_f12_tool_call_count(
            &mut state.tool_calls_emitted_count,
            ctx.max_tool_calls_per_response,
            &mut state.stop_string_triggered,
        );
        // Successful complete-call path — log + metric to match the
        // blocking and incremental-streaming paths.
        let preview: String = tc.function.arguments.chars().take(120).collect();
        let s = if tc.function.arguments.len() > preview.len() {
            "…"
        } else {
            ""
        };
        tracing::info!("Tool call: {}({preview}{s})", tc.function.name);
        crate::metrics::TOOL_CALLS_TOTAL.inc();
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
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, pre_tool_tail)
            .with_token_ids(state.take_ids_if(ctx.req_return_token_ids));
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
///
/// For qwen3_coder XML the streaming detector emits a single Delta with
/// the full parsed-and-canonicalised JSON arguments at the `</tool_call>`
/// boundary (see `streaming_impl.rs::process` line ~67 — args can't be
/// streamed character-by-character because XML parameter blocks must
/// finish before they convert to JSON). This is the natural spot to run
/// the same `backfill_required_params` + `validate_single_tool_call`
/// chain that the complete-tool-call path runs at `handle_complete_tool_call`,
/// so that streaming and non-streaming responses behave identically.
///
/// Without this, a model that emits `<function=NAME></function>` with no
/// `<parameter=>` blocks (observed under qwen3_coder + multi-turn agentic
/// loops with 21 tools, OpenClaw 2026.5.7) streams literal `"{}"` to the
/// client even when required parameters are declared in the schema —
/// while the non-streaming path would have backfilled `{"required_key": ""}`
/// and at least logged a warning. Issue #40 (iromu) called out this
/// "Opencode breaks tool calling more often" symptom.
pub(super) fn handle_tool_call_delta(
    state: &mut StreamState,
    ctx: &StreamCtx,
    args: String,
    idx: usize,
    sse_events: &mut SseVec,
) {
    let mut emit_args = args.clone();
    if let Some(entry) = state.streaming_tool_args.get_mut(&idx) {
        let name = entry.0.clone();
        let mut tc = tool_parser::ToolCall {
            id: format!("call_{:016x}", idx),
            call_type: "function".into(),
            function: tool_parser::FunctionCall {
                name: name.clone(),
                arguments: args.clone(),
            },
        };
        tool_parser::backfill_required_params(
            std::slice::from_mut(&mut tc),
            &ctx.tool_defs_for_backfill,
        );
        if let Some(ref cwd) = ctx.cwd_for_normalize {
            tool_parser::normalize_paths(std::slice::from_mut(&mut tc), cwd);
        }
        if let Err(e) = tool_parser::validate_single_tool_call(&tc, &ctx.tool_defs_for_backfill) {
            tracing::warn!(
                tool = %name,
                "tool call validation error (stream Δ): {e}; replacing with content and ending"
            );
            let msg = format!("[atlas] Tool call rejected: {e}");
            // No `with_token_ids` here: `entry` holds a &mut borrow of
            // `state` through line 214, so `take_ids_if` can't run. This
            // is a rare synthetic-error edge (not the generation path);
            // the pending IDs are still flushed exactly once on the
            // finish chunk in handle_done, so Σ token_ids stays correct.
            let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, msg);
            sse_events.push(Ok(
                Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
            ));
            state.stop_string_triggered = true;
            entry.1.push_str(&args);
            return;
        }
        emit_args = tc.function.arguments.clone();
        entry.1.push_str(&emit_args);
    } else if !args.is_empty() {
        // No prior ToolCallStart for this idx — keep legacy passthrough.
    }
    if !emit_args.is_empty() {
        let frag =
            ChatCompletionChunk::tool_call_args_fragment(&ctx.model, &ctx.id, idx, &emit_args);
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&frag).unwrap_or_default())
        ));
    }
}

/// `DetectorOutput::ToolCallEnd` — F11 within-response dedup +
/// F44 cross-turn permanent-failure check + Bug-2 name-run cap.
///
/// Bug-2 cap (`MAX_CONSEC_SAME_NAME_CALLS`): trips when the same tool
/// name fires N times in a row regardless of args. F11 keys on
/// `(name, canonical_args)` and is defeated by runaway loops where
/// the model rolls a fresh timestamp / sequence number / id into the
/// payload each iteration; the F12 total cap (default 12) is the
/// only other server-side circuit, but a runaway can already have
/// flooded the SSE channel before F12 fires. The name-run cap is
/// strictly tighter than F11 and F12 for the runaway pattern.
const MAX_CONSEC_SAME_NAME_CALLS: u32 = 6;

pub(super) fn handle_tool_call_end(state: &mut StreamState, ctx: &StreamCtx, idx: usize) {
    if let Some((name, args_json)) = state.streaming_tool_args.remove(&idx) {
        if state.tool_arg_dedup_within.check(&name, &args_json) {
            tracing::warn!(
                tool = %name,
                "F11 within-response dedup tripped: 2+ identical streaming tool calls; ending response"
            );
            state.stop_string_triggered = true;
            state.tool_loop_capped = true;
        } else if ctx.f44_cache_active
            && f44_check_permanent_failure(&ctx.f44_cache, &name, &args_json)
        {
            tracing::warn!(
                tool = %name,
                "F44 streaming circuit-breaker tripped: tool_call matches a permanently-failed prior call; ending response"
            );
            state.stop_string_triggered = true;
            state.tool_loop_capped = true;
        }
        let run_len = match &state.name_run {
            Some((prev, n)) if prev == &name => n + 1,
            _ => 1,
        };
        state.name_run = Some((name.clone(), run_len));
        if run_len >= MAX_CONSEC_SAME_NAME_CALLS && !state.stop_string_triggered {
            tracing::warn!(
                tool = %name,
                run = run_len,
                "Bug-2 name-run cap tripped: {run_len} successive `{name}` tool calls; ending response (F11 missed because args drift)"
            );
            state.stop_string_triggered = true;
            state.tool_loop_capped = true;
        }
        if !state.stop_string_triggered {
            // Successful streaming tool call — log + metric to match the
            // blocking and complete-call paths.
            let preview: String = args_json.chars().take(120).collect();
            let s = if args_json.len() > preview.len() {
                "…"
            } else {
                ""
            };
            tracing::info!("Tool call: {name}({preview}{s})");
            crate::metrics::TOOL_CALLS_TOTAL.inc();
        }
    }
}
