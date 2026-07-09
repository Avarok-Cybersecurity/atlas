// SPDX-License-Identifier: AGPL-3.0-only
//
// Helpers for the four `DetectorOutput` variants emitted by the
// streaming tool-call detector. Shared by both `handle_token` (mid-
// stream `process()` outputs) and `handle_done` (end-of-stream
// `flush()` outputs).

use axum::response::sse::Event;

use crate::openai::ChatCompletionChunk;
use crate::tool_parser;

use super::super::stream_guards::{bump_f12_tool_call_count, flush_content_sanitizer};
use super::ctx::StreamCtx;
use super::state::{PendingRetry, StreamState};

type SseVec = Vec<Result<Event, std::convert::Infallible>>;

/// Tier 5c (2026-05-26): emit `chunk_json` to either the client SSE
/// stream OR a per-tool-call-index buffer in `StreamState`. When tool
/// retry is enabled we hold all tool_call SSE chunks until
/// `handle_tool_call_delta` runs validation; on pass the buffered chunks
/// flush to the client, on fail they're discarded and the retry fires
/// at `handle_done`. When tool retry is disabled this is a direct emit
/// (preserves the existing real-time streaming behaviour).
fn emit_or_buffer_tool_chunk(
    state: &mut StreamState,
    ctx: &StreamCtx,
    idx: usize,
    chunk_json: String,
    sse_events: &mut SseVec,
) {
    if ctx.tool_retry_enabled {
        state
            .buffered_tool_chunks
            .entry(idx)
            .or_default()
            .push(chunk_json);
    } else {
        sse_events.push(Ok(Event::default().data(chunk_json)));
    }
}

/// Flush all buffered SSE chunks for tool-call `idx` into `sse_events`.
/// No-op when retry is disabled (chunks were emitted directly).
fn flush_buffered_tool_chunks(state: &mut StreamState, idx: usize, sse_events: &mut SseVec) {
    if let Some(chunks) = state.buffered_tool_chunks.remove(&idx) {
        for chunk_json in chunks {
            sse_events.push(Ok(Event::default().data(chunk_json)));
        }
    }
}

/// Drop all buffered SSE chunks for tool-call `idx` without emitting.
/// Called when validation fails and we're going to fire a Tier 5c retry.
fn drop_buffered_tool_chunks(state: &mut StreamState, idx: usize) {
    state.buffered_tool_chunks.remove(&idx);
}

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
    if ctx.wants_typed_arguments {
        tool_parser::coerce_all(std::slice::from_mut(tc), &ctx.tool_defs_for_backfill);
    }
    if let Some(ref cwd) = ctx.cwd_for_normalize {
        tool_parser::normalize_paths(std::slice::from_mut(tc), cwd);
    }
    // Typed severity (was a fragile `contains("non-empty")` sniff): soft =
    // MissingParam (2026-07-03 ST-collapse class) + EmptyRequired (2026-05-25
    // disposition) pass through; Hard bails.
    let validation = tool_parser::assess_tool_call(tc, &ctx.tool_defs_for_backfill).map_err(|i| {
        (
            matches!(
                i,
                tool_parser::ToolCallIssue::MissingParam(_)
                    | tool_parser::ToolCallIssue::EmptyRequired(_)
            ),
            i.into_message(),
        )
    });
    let is_soft = validation.as_ref().err().is_some_and(|(soft, _)| *soft);
    if let Err((_, e)) = &validation
        && !is_soft
    {
        tracing::warn!(
            tool = %tc.function.name,
            "tool call validation error (hard): {e}; replacing with content and ending"
        );
        let msg = format!("[atlas] Tool call rejected: {e}");
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, msg)
            .with_token_ids(state.take_ids_if(ctx.req_return_token_ids));
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
        ));
        state.stop_string_triggered = true;
    } else if let Err((_, e)) = &validation {
        // Soft validation error (missing param / empty required string) —
        // emit the tool call as the model produced it and let the client's
        // per-tool schema surface its own actionable error. See
        // `handle_tool_call_delta` for the rationale.
        tracing::warn!(
            tool = %tc.function.name,
            "tool call validation error (soft): {e}; passing through to opencode"
        );
        bump_f12_tool_call_count(
            &mut state.tool_calls_emitted_count,
            ctx.max_tool_calls_per_response,
            &mut state.stop_string_triggered,
        );
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
        // P0-3 (2026-07-09): server-authored corrective feedback. The soft
        // pass-through is kept (ST-995 never-drop invariant), but a call
        // whose REQUIRED params are empty is guaranteed to fail client-side
        // with an opaque error (opencode: "BadResource: FileSystem.readFile")
        // that the model retries verbatim — the 45k doom loop. Name the
        // problem in a content chunk so the retry context actually changes.
        // Rate-limited to once per response.
        if !state.corrective_hint_sent {
            let empties = tool_parser::find_empty_required_params(tc, &ctx.tool_defs_for_backfill);
            let garbled = tc.function.arguments.contains("</parameter<parameter=");
            if !empties.is_empty() || garbled {
                state.corrective_hint_sent = true;
                let msg = format!(
                    "
[atlas] The {} call above has EMPTY required parameter(s): {}.                      It will fail. Re-issue the call with real values for every                      required parameter (do not repeat it unchanged).",
                    tc.function.name,
                    if empties.is_empty() {
                        "<garbled parameter boundary>".to_string()
                    } else {
                        empties.join(", ")
                    },
                );
                let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, msg)
                    .with_token_ids(state.take_ids_if(ctx.req_return_token_ids));
                sse_events.push(Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                ));
            }
        }
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
        state
            .cancel_flag
            .store(true, std::sync::atomic::Ordering::Release);
    } else {
        // Bug-2 name-run cap (mirrors handle_tool_call_end): catches
        // runaway loops in the complete-tool-call path that
        // tool_arg_dedup misses because of args drift.
        let run_len = advance_name_run(&mut state.name_run, &tc.function.name);
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
    let start_json = serde_json::to_string(&start).unwrap_or_default();
    emit_or_buffer_tool_chunk(state, ctx, idx, start_json, sse_events);
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
        if ctx.wants_typed_arguments {
            tool_parser::coerce_all(std::slice::from_mut(&mut tc), &ctx.tool_defs_for_backfill);
        }
        if let Some(ref cwd) = ctx.cwd_for_normalize {
            tool_parser::normalize_paths(std::slice::from_mut(&mut tc), cwd);
        }
        if let Err(issue) = tool_parser::assess_tool_call(&tc, &ctx.tool_defs_for_backfill) {
            // Mid-stream validation rejections used to emit a `[atlas] Tool
            // call rejected: …` content chunk and trip `stop_string_triggered`
            // — but `handle_tool_call_start` had already emitted the
            // `tool_calls[idx]` header to opencode, so suppressing the args
            // delta left opencode mid-call with no completion. opencode then
            // reported `SchemaError(Missing key)`, a less actionable error
            // than its own per-tool schema check (e.g. "The argument 'file'
            // cannot be empty. Received ''").
            //
            // Empty-required-string failures (most common: F78 path tools,
            // 2026-05-25 shell tools) and missing required params (2026-07-03
            // ST-collapse class) are recoverable: emit the args delta
            // as the model produced them and let opencode's per-tool schema
            // surface its own actionable error to the model on the next
            // turn. Hard failures (unknown tool name, args not valid JSON)
            // still bail with a content chunk because they cannot be made
            // into a complete tool call at all.
            let is_soft = matches!(
                issue,
                tool_parser::ToolCallIssue::MissingParam(_)
                    | tool_parser::ToolCallIssue::EmptyRequired(_)
            );
            let e = issue.into_message();
            if is_soft {
                tracing::warn!(
                    tool = %name,
                    "tool call validation error (stream Δ, soft): {e}; passing through so opencode can surface its own per-tool schema error"
                );
                emit_args = tc.function.arguments.clone();
                entry.1.push_str(&emit_args);
            } else if ctx.tool_retry_enabled {
                // Tier 5c (2026-05-26): drop the buffered start + args
                // chunks for this idx, record the failure context, and
                // signal the scheduler to stop. `handle_done` will see
                // `pending_retry` and fire the retry inference; if the
                // retry produces a valid call we emit it in place of
                // the failed call, so the client never sees the bad one.
                tracing::warn!(
                    tool = %name,
                    "tool call validation error (stream Δ, hard, retry pending): {e}"
                );
                // Release the `entry` borrow on `state.streaming_tool_args`
                // before mutating the buffered-chunks + pending_retry on
                // `state` (the borrow checker rejects two simultaneous
                // mutable borrows of `state`). Capture what we still need.
                entry.1.push_str(&args);
                let errors_summary = e.to_string();
                drop_buffered_tool_chunks(state, idx);
                state.pending_retry = Some(PendingRetry {
                    errors_summary,
                    failed_idx: idx,
                });
                state.stop_string_triggered = true;
                state
                    .cancel_flag
                    .store(true, std::sync::atomic::Ordering::Release);
                return;
            } else {
                tracing::warn!(
                    tool = %name,
                    "tool call validation error (stream Δ, hard): {e}; replacing with content and ending"
                );
                let msg = format!("[atlas] Tool call rejected: {e}");
                let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, msg);
                sse_events.push(Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                ));
                state.stop_string_triggered = true;
                entry.1.push_str(&args);
                return;
            }
        } else {
            emit_args = tc.function.arguments.clone();
            entry.1.push_str(&emit_args);
        }
    } else if !args.is_empty() {
        // No prior ToolCallStart for this idx — keep legacy passthrough.
    }
    if !emit_args.is_empty() {
        let frag =
            ChatCompletionChunk::tool_call_args_fragment(&ctx.model, &ctx.id, idx, &emit_args);
        let frag_json = serde_json::to_string(&frag).unwrap_or_default();
        // Either flush previously-buffered start + this args chunk
        // together (success path under retry), or emit directly (retry
        // disabled). When retry is disabled the start chunk was already
        // emitted in real time, so `emit_or_buffer_tool_chunk` just adds
        // the args chunk.
        emit_or_buffer_tool_chunk(state, ctx, idx, frag_json, sse_events);
        if ctx.tool_retry_enabled {
            flush_buffered_tool_chunks(state, idx, sse_events);
        }
    }
}

/// `DetectorOutput::ToolCallArgsFragment` — live-streaming: a ready-to-forward
/// slice of `function.arguments` the detector already coerced (XML) or sliced
/// (JSON). Append it verbatim to the accumulated args and emit it directly as an
/// OpenAI `tool_calls[idx].function.arguments` fragment — NO coercion or
/// validation (the detector did that per-field). If no prior `ToolCallStart`
/// created the accumulator entry for `idx`, the fragment is dropped (the header
/// must precede its arguments).
pub(super) fn handle_tool_call_args_fragment(
    state: &mut StreamState,
    ctx: &StreamCtx,
    fragment: String,
    idx: usize,
    sse_events: &mut SseVec,
) {
    let Some(entry) = state.streaming_tool_args.get_mut(&idx) else {
        return;
    };
    entry.1.push_str(&fragment);
    let frag = ChatCompletionChunk::tool_call_args_fragment(&ctx.model, &ctx.id, idx, &fragment);
    sse_events.push(Ok(
        Event::default().data(serde_json::to_string(&frag).unwrap_or_default())
    ));
}

/// Advance the same-name run counter for a completed call, returning the new
/// run length. Pure over the `Option` state so the #192 parallel fan-out
/// contract (N same-name calls with distinct args below the cap must NOT be
/// treated as a doom loop) is unit-testable without a `StreamState`.
fn advance_name_run(name_run: &mut Option<(String, u32)>, name: &str) -> u32 {
    let run_len = match name_run {
        Some((prev, n)) if prev == name => *n + 1,
        _ => 1,
    };
    *name_run = Some((name.to_string(), run_len));
    run_len
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
///
/// A3 (2026-05-26): tightened from 6 → 3 to match opencode's
/// `DOOM_LOOP_THRESHOLD = 3`. Live Wave-1/3 traces showed the model
/// emitting 4-6 same-name bash calls with drifted args before any
/// guard tripped, by which point ~MB-long degenerate commands had
/// already flooded the stream and the .git/ artifact pollution was
/// already created. Three same-name calls is the empirical threshold
/// at which opencode itself bails to the user for permission. Atlas
/// matching this means we end the response slightly before opencode
/// would surrender, giving the outer retry loop a clean signal.
///
/// #192 (2026-07-02): relaxed 3 → 8. A3's premise predates parallel tool
/// calls: back then one response could not legitimately contain 3 same-name
/// calls (generation hard-stopped at the first `</tool_call>`), so a run of
/// 3 was proof of degeneration. Post-#192 the same-name run IS the designed
/// shape of a parallel fan-out (BFCL `parallel`: get_weather × 3 cities —
/// live 2026-07-02 the cap flipped that clean turn to finish="length").
/// 8 mirrors the scheduler's own parallel bound
/// (`MAX_POST_COMPLETION_TOOL_OPENS = 8`, decode_logits_step.rs): a real
/// runaway still trips it an order of magnitude below the F12 total cap,
/// while any plausible legit fan-out stays under it. Identical-args loops
/// are still caught earlier by the F11 within-response dedup.
const MAX_CONSEC_SAME_NAME_CALLS: u32 = 8;

pub(super) fn handle_tool_call_end(state: &mut StreamState, _ctx: &StreamCtx, idx: usize) {
    if let Some((name, args_json)) = state.streaming_tool_args.remove(&idx) {
        if state.tool_arg_dedup_within.check(&name, &args_json) {
            tracing::warn!(
                tool = %name,
                "F11 within-response dedup tripped: 2+ identical streaming tool calls; ending response"
            );
            state.stop_string_triggered = true;
            state.tool_loop_capped = true;
        }
        let run_len = advance_name_run(&mut state.name_run, &name);
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

#[cfg(test)]
mod name_run_cap_tests {
    //! #192: the Bug-2 same-name run cap vs parallel fan-outs. A BFCL
    //! `parallel`-shape response (get_weather x 3 cities) must NOT be
    //! classified as a doom loop (live 2026-07-02: the old cap of 3 flipped a
    //! clean 3-call hermes turn to finish_reason="length"); a genuine
    //! same-name runaway must still trip the cap before the F12 total cap.
    use super::{MAX_CONSEC_SAME_NAME_CALLS, advance_name_run};

    #[test]
    fn three_call_parallel_fanout_stays_under_cap() {
        let mut run = None;
        for i in 1..=3u32 {
            let len = advance_name_run(&mut run, "get_weather");
            assert_eq!(len, i);
            assert!(
                len < MAX_CONSEC_SAME_NAME_CALLS,
                "a 3-call same-name parallel fan-out must not trip the doom-loop cap"
            );
        }
    }

    #[test]
    fn runaway_same_name_run_still_trips() {
        let mut run = None;
        let mut tripped_at = None;
        for i in 1..=12u32 {
            if advance_name_run(&mut run, "bash") >= MAX_CONSEC_SAME_NAME_CALLS {
                tripped_at = Some(i);
                break;
            }
        }
        assert_eq!(
            tripped_at,
            Some(MAX_CONSEC_SAME_NAME_CALLS),
            "cap must fire below the F12 total cap (12)"
        );
    }

    #[test]
    fn different_name_resets_run() {
        let mut run = None;
        assert_eq!(advance_name_run(&mut run, "get_weather"), 1);
        assert_eq!(advance_name_run(&mut run, "get_weather"), 2);
        assert_eq!(advance_name_run(&mut run, "get_time"), 1);
        assert_eq!(advance_name_run(&mut run, "get_weather"), 1);
    }
}
