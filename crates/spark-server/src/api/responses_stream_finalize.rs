// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

//! Post-stream finalization helpers for the Responses streaming endpoint.
//!
//! Hoisted out of `responses_stream.rs` to keep that file under the 500 LoC
//! cap. The single entry point here, [`finalize_responses_stream`], runs
//! after the inner chat stream drains: it builds the terminal
//! `response.completed` payload, persists for `previous_response_id` resume,
//! appends to a linked conversation, and emits the final SSE events.

use std::sync::Arc;

use axum::response::sse::Event;
use tokio::sync::mpsc;

use super::responses_translate::{build_responses_usage, emit};
use crate::AppState;

/// Emit the opening `response.created` and `response.in_progress` SSE
/// frames. Returns the next free sequence number.
pub(super) async fn emit_responses_prologue(
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    seq_start: u64,
    resp_id: &str,
    created_at: u64,
    model: &str,
    metadata: &Option<std::collections::HashMap<String, String>>,
) -> u64 {
    let mut seq = seq_start;
    // Emit response.created first (status: in_progress).
    let created = crate::openai::ResponsesStreamEvent::Created {
        sequence_number: seq,
        response: crate::openai::ResponsesStreamEnvelope {
            id: resp_id.to_string(),
            object: "response",
            created_at,
            model: model.to_string(),
            status: "in_progress",
            metadata: metadata.clone(),
        },
    };
    if let Ok(j) = serde_json::to_string(&created)
        && let Err(e) = tx
            .send(Ok(Event::default()
                .event(crate::openai::responses_event_name(&created))
                .data(j)))
            .await
    {
        tracing::warn!("responses_stream: response.created send failed (receiver dropped): {e}");
    }
    seq += 1;

    // Per the Responses spec, `response.in_progress` follows
    // `response.created` once generation actually starts. Emit it
    // immediately so spec-strict clients (e.g. the OpenAI Python SDK)
    // don't hang waiting for it.
    let in_progress = crate::openai::ResponsesStreamEvent::InProgress {
        sequence_number: seq,
        response: crate::openai::ResponsesStreamEnvelope {
            id: resp_id.to_string(),
            object: "response",
            created_at,
            model: model.to_string(),
            status: "in_progress",
            metadata: metadata.clone(),
        },
    };
    if let Ok(j) = serde_json::to_string(&in_progress)
        && let Err(e) = tx
            .send(Ok(Event::default()
                .event(crate::openai::responses_event_name(&in_progress))
                .data(j)))
            .await
    {
        tracing::warn!(
            "responses_stream: response.in_progress send failed (receiver dropped): {e}"
        );
    }
    seq += 1;
    seq
}

/// State for any "in-flight" output item that may need closing once the
/// upstream chat stream drains. Mirrors the locals in
/// `responses_endpoint_stream`'s tail block 1:1.
pub(super) struct CloseOpenCtx<'a> {
    pub seq: u64,
    pub message_started: bool,
    pub message_item_id: &'a str,
    pub content_text: &'a str,
    pub fc_started: bool,
    pub fc_done: bool,
    pub fc_item_id: Option<String>,
    pub current_tool_call_id: &'a Option<String>,
    pub current_tool_name: &'a Option<String>,
    pub tool_args: &'a str,
    pub output_index: usize,
}

/// Close any open message- or function-call output item left behind when
/// the inner chat stream ended. Pushes `OutputItemDone` events into
/// `completed_items` and returns the next free sequence number.
pub(super) async fn close_open_items(
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    completed_items: &mut Vec<crate::openai::ResponsesOutputItem>,
    ctx: CloseOpenCtx<'_>,
) -> u64 {
    let mut seq = ctx.seq;
    if ctx.message_started {
        let ev = crate::openai::ResponsesStreamEvent::OutputTextDone {
            sequence_number: seq,
            item_id: ctx.message_item_id.to_string(),
            output_index: ctx.output_index,
            content_index: 0,
            text: ctx.content_text.to_string(),
        };
        emit(tx, &ev).await;
        seq += 1;
        let done = crate::openai::ResponsesOutputItem::Message {
            id: ctx.message_item_id.to_string(),
            status: "completed",
            role: "assistant",
            content: vec![crate::openai::ResponsesContentPart::OutputText {
                text: ctx.content_text.to_string(),
                annotations: crate::openai::merged_annotations(ctx.content_text),
            }],
        };
        completed_items.push(done.clone());
        let ev = crate::openai::ResponsesStreamEvent::OutputItemDone {
            sequence_number: seq,
            output_index: ctx.output_index,
            item: done,
        };
        emit(tx, &ev).await;
        seq += 1;
    }
    if ctx.fc_started
        && !ctx.fc_done
        && let Some(fcid) = ctx.fc_item_id.clone()
    {
        let ev = crate::openai::ResponsesStreamEvent::FunctionCallArgumentsDone {
            sequence_number: seq,
            item_id: fcid.clone(),
            output_index: ctx.output_index,
            arguments: ctx.tool_args.to_string(),
        };
        emit(tx, &ev).await;
        seq += 1;
        let done = crate::openai::ResponsesOutputItem::FunctionCall {
            id: fcid,
            call_id: ctx.current_tool_call_id.clone().unwrap_or_default(),
            name: ctx.current_tool_name.clone().unwrap_or_default(),
            arguments: ctx.tool_args.to_string(),
            status: "completed",
        };
        completed_items.push(done.clone());
        let ev = crate::openai::ResponsesStreamEvent::OutputItemDone {
            sequence_number: seq,
            output_index: ctx.output_index,
            item: done,
        };
        emit(tx, &ev).await;
        seq += 1;
    }
    seq
}

/// Inputs collected during the streaming loop that the finalize stage
/// consumes. Bundled into a struct purely to keep the call site readable —
/// each field corresponds 1:1 to a local in `responses_endpoint_stream`'s
/// post-loop block.
pub(super) struct FinalizeCtx {
    pub seq: u64,
    pub completed_items: Vec<crate::openai::ResponsesOutputItem>,
    pub final_usage: Option<serde_json::Value>,
    pub finish_reason: String,
    pub refusal_text: Option<String>,
    pub message_item_id: String,
    pub output_index: usize,
    pub resp_id: String,
    pub created_at: u64,
    pub model: String,
    pub metadata_for_done: Option<std::collections::HashMap<String, String>>,
    pub store_flag: bool,
    pub input_messages: Vec<crate::openai::IncomingMessage>,
    pub conversation_id: Option<String>,
    pub conv_new_user_items: Vec<serde_json::Value>,
}

/// Emit `response.completed` (and the optional preceding `response.refusal.done`),
/// persist the response if `store_flag` is set, and append items to the
/// linked conversation when `conversation_id` is present.
pub(super) async fn finalize_responses_stream(
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    state_arc: Arc<AppState>,
    ctx: FinalizeCtx,
) {
    let FinalizeCtx {
        mut seq,
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
    } = ctx;

    let store_ref = state_arc.response_store.clone();

    // Build the final ResponsesResponse for the `completed` event from
    // the running `completed_items` log so multi-item streams (text→fc→text)
    // surface every emitted item, not just the last live one.
    let final_output = completed_items;
    let usage =
        final_usage
            .as_ref()
            .map(build_responses_usage)
            .unwrap_or(crate::openai::ResponsesUsage {
                input_tokens: 0,
                input_tokens_details: None,
                output_tokens: 0,
                output_tokens_details: None,
                total_tokens: 0,
            });
    let final_status: &'static str = if finish_reason == "error" {
        "failed"
    } else {
        "completed"
    };
    let final_resp = crate::openai::ResponsesResponse {
        id: resp_id.clone(),
        object: "response",
        created_at,
        model: model.clone(),
        status: final_status,
        error: None,
        output: final_output,
        reasoning: None,
        usage,
        metadata: metadata_for_done,
    };
    // Aggregate every emitted message's text so multi-item streams
    // (text→fc→text) round-trip cleanly through previous_response_id
    // resume and the Conversations API.
    let mut transcript_text = String::new();
    for item in &final_resp.output {
        if let crate::openai::ResponsesOutputItem::Message { content, .. } = item {
            for part in content {
                let crate::openai::ResponsesContentPart::OutputText { text, .. } = part;
                transcript_text.push_str(text);
            }
        }
    }
    // Persist for previous_response_id resume (Responses API defaults
    // to store=true; we respect the caller's store field).
    if store_flag && let Ok(body) = serde_json::to_value(&final_resp) {
        let mut transcript = input_messages;
        // Rebuild structured tool_calls from the accumulated
        // function-call output items so previous_response_id
        // resume sees the model's prior tool calls. Without
        // this, multi-turn streaming Responses chains lose
        // every function_call across turns and the model
        // re-issues them on resume.
        let stored_tool_calls: Vec<crate::tool_parser::IncomingToolCall> = final_resp
            .output
            .iter()
            .filter_map(|item| match item {
                crate::openai::ResponsesOutputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                    ..
                } => Some(crate::tool_parser::IncomingToolCall {
                    id: Some(call_id.clone()),
                    function: crate::tool_parser::IncomingFunction {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                }),
                _ => None,
            })
            .collect();
        if !transcript_text.is_empty() || !stored_tool_calls.is_empty() {
            transcript.push(crate::openai::IncomingMessage {
                role: "assistant".to_string(),
                content: crate::openai::ParsedContent {
                    text: transcript_text.clone(),
                    images: Vec::new(),
                },
                tool_calls: if stored_tool_calls.is_empty() {
                    None
                } else {
                    Some(stored_tool_calls)
                },
                tool_call_id: None,
                name: None,
                reasoning_content: None,
            });
        }
        store_ref.insert(crate::response_store::StoredEntry {
            id: resp_id.clone(),
            kind: crate::response_store::StoredKind::Response,
            model: model.clone(),
            created_at,
            messages: transcript,
            body,
            last_access: std::time::Instant::now(),
        });
    }
    // Append the new user items + assistant output to the linked
    // conversation (best-effort; silent on failure so the primary
    // SSE stream is never disrupted by conversation-store issues).
    if let Some(cid) = conversation_id.as_ref() {
        let mut batch = conv_new_user_items.clone();
        if !transcript_text.is_empty() {
            batch.push(serde_json::json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": transcript_text}],
            }));
        }
        if !batch.is_empty()
            && let Err(e) = state_arc.conversation_store.add_items(cid, batch)
        {
            tracing::warn!(
                "responses_stream_finalize: conversation_store.add_items failed for {cid}: {e:?}"
            );
        }
    }
    // Emit response.refusal.done before completed so safety-aware
    // clients see the final refusal text. The chat-side emitter
    // already sent RefusalDelta with the full sentence.
    if let Some(ref r) = refusal_text {
        let ev = crate::openai::ResponsesStreamEvent::RefusalDone {
            sequence_number: seq,
            item_id: message_item_id.clone(),
            output_index,
            content_index: 0,
            refusal: r.clone(),
        };
        emit(tx, &ev).await;
        seq += 1;
    }
    let completed = crate::openai::ResponsesStreamEvent::Completed {
        sequence_number: seq,
        response: final_resp,
    };
    emit(tx, &completed).await;
}
