// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use super::chat_stream::chat_completions_stream;
use super::responses_stream_finalize::{
    CloseOpenCtx, FinalizeCtx, close_open_items, emit_responses_prologue, finalize_responses_stream,
};
use super::responses_translate::{
    build_responses_usage, emit, find_frame_end, translate_chat_response_to_responses,
};
use super::stored::extract_assistant_incoming_message;
use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;
// Sibling-cluster items hoisted from the original `api.rs`.
use super::chat::chat_completions_inner;
use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
use super::completions::not_supported;
use super::failures::{
    F23ProgressMetrics, F29EnvironmentFact, F37FailureClass, F39FailureCache,
    F39PermanentFailureMatch, F49DuplicateWrite, append_f7_reminder_to_last_user,
    build_f7_stall_reminder, bump_f12_tool_call_count, check_loop_watchdog,
    collect_f7_stall_buckets, f23_build_reminder, f23_normalize_and_hash, f23_refuse_threshold,
    f23_score_progress, f23_warn_threshold, f28_text_looks_like_error,
    f29_extract_binary_from_error_line, f29_extract_environment_facts,
    f29_inject_environment_facts, f31_inject_hard_refusal, f32_reposition_failed_tool_result,
    f37_classify_failure, f39_build_circuit_breaker_banner, f39_build_failure_cache,
    f39_class_label, f39_detect_recent_retries, f39_extract_binary_name,
    f44_check_permanent_failure, f49_build_banner, f49_detect_duplicate_writes,
    f49_extract_write_path_and_content, f50_append_original_error, f60_disable_mtp_for_request,
    flush_content_sanitizer, prepend_reminder_to_system, recent_message_is_tool_error,
    strip_xml_leaks_from_assistant_content,
};
use super::inference_impl::{extract_thinking, strip_stop_sequences, tokenize_stop_sequences};
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};
use super::strip::strip_thinking_tags;
// Re-export sibling helpers via crate::api::* for short paths.
use super::failures::*;
use super::inference_types::*;
use super::sanitizer::*;

pub(super) async fn responses_endpoint_stream(
    state: State<Arc<AppState>>,
    mut chat_req: ChatCompletionRequest,
    metadata: Option<std::collections::HashMap<String, String>>,
    store_flag: bool,
    conversation_id: Option<String>,
) -> Response {
    use axum::response::sse::Event;
    use tokio::sync::mpsc;
    // Force streaming on the inner pipeline and disable the
    // stream_options usage sidecar — we want the `done_chunk` shape so
    // the transformer sees a single terminal `finish_reason` event.
    chat_req.stream = true;
    chat_req.stream_options = None;

    let model = chat_req.model.clone();
    let input_messages = chat_req.messages.clone();
    let state_arc = state.0.clone();

    // Capture the conversation-linked user items (the new turn's user
    // input) before handing chat_req off. We use them to append back
    // to the conversation after the stream finishes.
    let conv_new_user_items: Vec<serde_json::Value> = if let Some(cid) = conversation_id.as_ref() {
        let prior = state_arc
            .conversation_store
            .get(cid)
            .map(|s| s.items.len())
            .unwrap_or(0);
        input_messages
            .iter()
            .skip(prior)
            .map(|m| {
                serde_json::json!({
                    "type": "message",
                    "role": m.role,
                    "content": [{"type": "input_text", "text": m.content.text}],
                })
            })
            .collect()
    } else {
        Vec::new()
    };

    // Run the chat-completions streaming handler (re-using the full
    // pipeline: scheduler, tool detection, thinking, logprobs, ...).
    let chat_resp = chat_completions_inner(state.0, None, chat_req, None).await;
    let (parts, body) = chat_resp.into_parts();
    if !parts.status.is_success() {
        return Response::from_parts(parts, body);
    }

    // Channel the transformer pushes Responses SSE events into. Sized to
    // match chat_stream/mod.rs (~30s decode buffer at 50 tok/s).
    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(1024);
    let created_at = crate::openai::unix_timestamp();
    let resp_id = format!("resp_{}", crate::openai::uuid_v4());
    let metadata_for_done = metadata.clone();

    tokio::spawn(async move {
        let mut seq: u64 = 0;
        let mut content_text = String::new();
        let mut tool_args = String::new();
        let mut current_tool_name: Option<String> = None;
        let mut current_tool_call_id: Option<String> = None;
        let mut output_index: usize = 0;
        // Mint message item ids per output position so a text→fc→text
        // sequence doesn't reuse the same item_id (would collide with the
        // function_call still owning the prior output_index).
        let mut message_item_id = format!("msg_{}_{}", resp_id, output_index);
        let mut fc_item_id: Option<String> = None;
        let mut message_started = false;
        let mut fc_started = false;
        let mut fc_done = false;
        // Accumulated final output items (closed during streaming) so the
        // terminal `response.completed` payload reflects every emitted
        // item, not just the last live one. Required for stateful resume
        // via `previous_response_id` when the model emits text→fc→text.
        let mut completed_items: Vec<crate::openai::ResponsesOutputItem> = Vec::new();
        let mut final_usage: Option<serde_json::Value> = None;
        let mut finish_reason = "stop".to_string();
        let mut accumulator = Vec::<u8>::new();
        let mut refusal_text: Option<String> = None;

        seq = emit_responses_prologue(&tx, seq, &resp_id, created_at, &model, &metadata).await;

        let mut data_stream = body.into_data_stream();
        while let Some(next) = data_stream.next().await {
            let bytes = match next {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("responses stream read error: {e}");
                    break;
                }
            };
            accumulator.extend_from_slice(&bytes);

            // Parse SSE frames: look for "\n\n".
            while let Some(idx) = find_frame_end(&accumulator) {
                let frame_bytes: Vec<u8> = accumulator.drain(..idx + 2).collect();
                // Strip the trailing \n\n; find the data: line.
                let frame_str = std::str::from_utf8(&frame_bytes).unwrap_or("");
                let data_line = frame_str.lines().find_map(|l| l.strip_prefix("data: "));
                let Some(data) = data_line else { continue };
                if data == "[DONE]" {
                    continue;
                }
                // Parse as ChatCompletionChunk JSON.
                let Ok(chunk): Result<serde_json::Value, _> = serde_json::from_str(data) else {
                    continue;
                };
                // Chat chunks: { choices: [{ delta: {content|tool_calls|reasoning_content}, finish_reason, ... }], usage }
                if let Some(u) = chunk.get("usage")
                    && !u.is_null()
                {
                    final_usage = Some(u.clone());
                }
                let Some(choice) = chunk
                    .get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|a| a.first())
                else {
                    continue;
                };
                if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                    finish_reason = fr.to_string();
                }
                let delta = choice
                    .get("delta")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

                // Refusal delta (post-hoc: one chunk carries the full
                // refusal sentence, emitted by the inner chat streamer).
                if let Some(r) = delta.get("refusal").and_then(|v| v.as_str())
                    && !r.is_empty()
                {
                    refusal_text = Some(r.to_string());
                    let ev = crate::openai::ResponsesStreamEvent::RefusalDelta {
                        sequence_number: seq,
                        item_id: message_item_id.clone(),
                        output_index,
                        content_index: 0,
                        delta: r.to_string(),
                    };
                    emit(&tx, &ev).await;
                    seq += 1;
                }

                // Text content delta.
                if let Some(text) = delta.get("content").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    // If a function_call is currently open, close it
                    // before opening a fresh message item — otherwise
                    // the message would collide with the function_call
                    // on the same `output_index`.
                    if fc_started && !fc_done {
                        if let Some(fcid) = fc_item_id.clone() {
                            let ev =
                                crate::openai::ResponsesStreamEvent::FunctionCallArgumentsDone {
                                    sequence_number: seq,
                                    item_id: fcid.clone(),
                                    output_index,
                                    arguments: tool_args.clone(),
                                };
                            emit(&tx, &ev).await;
                            seq += 1;
                            let done = crate::openai::ResponsesOutputItem::FunctionCall {
                                id: fcid,
                                call_id: current_tool_call_id.clone().unwrap_or_default(),
                                name: current_tool_name.clone().unwrap_or_default(),
                                arguments: tool_args.clone(),
                                status: "completed",
                            };
                            completed_items.push(done.clone());
                            let ev = crate::openai::ResponsesStreamEvent::OutputItemDone {
                                sequence_number: seq,
                                output_index,
                                item: done,
                            };
                            emit(&tx, &ev).await;
                            seq += 1;
                        }
                        output_index += 1;
                        message_item_id = format!("msg_{}_{}", resp_id, output_index);
                        fc_done = true;
                        // Reset per-message text so the new message's
                        // OutputTextDone carries only the post-fc text.
                        content_text.clear();
                    }
                    if !message_started {
                        message_started = true;
                        // output_item.added for the message item.
                        let item = crate::openai::ResponsesOutputItem::Message {
                            id: message_item_id.clone(),
                            status: "in_progress",
                            role: "assistant",
                            content: vec![],
                        };
                        let ev = crate::openai::ResponsesStreamEvent::OutputItemAdded {
                            sequence_number: seq,
                            output_index,
                            item,
                        };
                        emit(&tx, &ev).await;
                        seq += 1;
                        // content_part.added.
                        let cp = crate::openai::ResponsesContentPart::OutputText {
                            text: String::new(),
                            annotations: None,
                        };
                        let ev = crate::openai::ResponsesStreamEvent::ContentPartAdded {
                            sequence_number: seq,
                            item_id: message_item_id.clone(),
                            output_index,
                            content_index: 0,
                            part: cp,
                        };
                        emit(&tx, &ev).await;
                        seq += 1;
                    }
                    content_text.push_str(text);
                    let ev = crate::openai::ResponsesStreamEvent::OutputTextDelta {
                        sequence_number: seq,
                        item_id: message_item_id.clone(),
                        output_index,
                        content_index: 0,
                        delta: text.to_string(),
                    };
                    emit(&tx, &ev).await;
                    seq += 1;
                }

                // Tool-call fragments.
                if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                    for tc in tool_calls {
                        let call_id = tc.get("id").and_then(|v| v.as_str()).map(str::to_string);
                        let name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|v| v.as_str())
                            .map(str::to_string);
                        let args_frag = tc
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if let Some(n) = name.clone() {
                            current_tool_name = Some(n.clone());
                            if let Some(id) = call_id.clone() {
                                current_tool_call_id = Some(id);
                            }
                            if !fc_started {
                                // Close any open message before starting function call.
                                if message_started {
                                    let ev = crate::openai::ResponsesStreamEvent::OutputTextDone {
                                        sequence_number: seq,
                                        item_id: message_item_id.clone(),
                                        output_index,
                                        content_index: 0,
                                        text: content_text.clone(),
                                    };
                                    emit(&tx, &ev).await;
                                    seq += 1;
                                    let done = crate::openai::ResponsesOutputItem::Message {
                                        id: message_item_id.clone(),
                                        status: "completed",
                                        role: "assistant",
                                        content: vec![
                                            crate::openai::ResponsesContentPart::OutputText {
                                                text: content_text.clone(),
                                                annotations: crate::citation::merged_annotations(
                                                    &content_text,
                                                ),
                                            },
                                        ],
                                    };
                                    completed_items.push(done.clone());
                                    let ev = crate::openai::ResponsesStreamEvent::OutputItemDone {
                                        sequence_number: seq,
                                        output_index,
                                        item: done,
                                    };
                                    emit(&tx, &ev).await;
                                    seq += 1;
                                    output_index += 1;
                                    message_started = false;
                                    content_text.clear();
                                }
                                let fcid = format!("fc_{}_{}", resp_id, output_index);
                                fc_item_id = Some(fcid.clone());
                                let item = crate::openai::ResponsesOutputItem::FunctionCall {
                                    id: fcid,
                                    call_id: current_tool_call_id.clone().unwrap_or_default(),
                                    name: n,
                                    arguments: String::new(),
                                    status: "in_progress",
                                };
                                let ev = crate::openai::ResponsesStreamEvent::OutputItemAdded {
                                    sequence_number: seq,
                                    output_index,
                                    item,
                                };
                                emit(&tx, &ev).await;
                                seq += 1;
                                fc_started = true;
                            }
                        }
                        if !args_frag.is_empty() {
                            tool_args.push_str(args_frag);
                            if let Some(fcid) = fc_item_id.clone() {
                                let ev = crate::openai::ResponsesStreamEvent::FunctionCallArgumentsDelta {
                                    sequence_number: seq,
                                    item_id: fcid,
                                    output_index,
                                    delta: args_frag.to_string(),
                                };
                                emit(&tx, &ev).await;
                                seq += 1;
                            }
                        }
                    }
                }
            }
        }

        seq = close_open_items(
            &tx,
            &mut completed_items,
            CloseOpenCtx {
                seq,
                message_started,
                message_item_id: &message_item_id,
                content_text: &content_text,
                fc_started,
                fc_done,
                fc_item_id: fc_item_id.clone(),
                current_tool_call_id: &current_tool_call_id,
                current_tool_name: &current_tool_name,
                tool_args: &tool_args,
                output_index,
            },
        )
        .await;

        finalize_responses_stream(
            &tx,
            state_arc.clone(),
            FinalizeCtx {
                seq,
                completed_items,
                final_usage,
                finish_reason,
                refusal_text,
                message_item_id,
                output_index,
                resp_id,
                created_at,
                model,
                metadata_for_done,
                store_flag,
                input_messages,
                conversation_id,
                conv_new_user_items,
            },
        )
        .await;
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
