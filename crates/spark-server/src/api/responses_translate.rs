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

use super::stored::extract_assistant_incoming_message;
use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

// Sibling-cluster items hoisted from the original `api.rs`. These uses
// give every sub-file access to helpers that the un-split file took for
// granted via single-module visibility.
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

pub(super) fn find_frame_end(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}

pub(super) async fn emit(
    tx: &tokio::sync::mpsc::Sender<Result<axum::response::sse::Event, std::convert::Infallible>>,
    ev: &crate::openai::ResponsesStreamEvent,
) {
    use axum::response::sse::Event;
    if let Ok(json) = serde_json::to_string(ev)
        && let Err(e) = tx
            .send(Ok(Event::default()
                .event(crate::openai::responses_event_name(ev))
                .data(json)))
            .await
    {
        tracing::warn!("responses_translate::emit: SSE send failed (receiver dropped): {e}");
    }
}

pub(super) fn build_responses_usage(u: &serde_json::Value) -> crate::openai::ResponsesUsage {
    crate::openai::ResponsesUsage {
        input_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        input_tokens_details: u
            .get("prompt_tokens_details")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        output_tokens: u
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize,
        output_tokens_details: u
            .get("completion_tokens_details")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        total_tokens: u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
    }
}

/// Translate a `ChatCompletionResponse` (already serialized in the Response)
/// into a `ResponsesResponse`. Only handles the JSON success path —
/// error responses are forwarded unchanged.
///
/// When `store` is provided and `store_flag` is true, the final
/// `{input_messages + assistant turn}` transcript is persisted under
/// `resp_<id>` so a follow-up `previous_response_id` lookup can resume.
pub(super) async fn translate_chat_response_to_responses(
    resp: Response,
    req_metadata: Option<std::collections::HashMap<String, String>>,
    store: Option<Arc<crate::response_store::ResponseStore>>,
    input_messages: Vec<crate::openai::IncomingMessage>,
    store_flag: bool,
    conversation: Option<(Arc<crate::conversation_store::ConversationStore>, String)>,
) -> Response {
    let (parts, body) = resp.into_parts();
    if !parts.status.is_success() {
        return Response::from_parts(parts, body);
    }
    let bytes = match axum::body::to_bytes(body, 16 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("adapter body read failed: {e}"),
            );
        }
    };
    let chat: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(c) => c,
        Err(_) => {
            // Not a chat-completion success body — pass through.
            return axum::response::Response::builder()
                .status(parts.status)
                .body(axum::body::Body::from(bytes))
                .unwrap_or_else(|_| {
                    openai_error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "adapter passthrough failed".into(),
                    )
                });
        }
    };
    let id = chat
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("chatcmpl-unknown")
        .to_string();
    let model = chat
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let created = chat
        .get("created")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(crate::openai::unix_timestamp);
    let mut output: Vec<crate::openai::ResponsesOutputItem> = Vec::new();
    if let Some(choice) = chat
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
    {
        let msg = choice
            .get("message")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
            for (i, tc) in tool_calls.iter().enumerate() {
                let call_id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                output.push(crate::openai::ResponsesOutputItem::FunctionCall {
                    id: format!("fc_{}_{}", id, i),
                    call_id,
                    name,
                    arguments,
                    status: "completed",
                });
            }
        }
        if let Some(text) = msg.get("content").and_then(|v| v.as_str()) {
            let annotations: Option<Vec<crate::openai::Annotation>> = msg
                .get("annotations")
                .and_then(|a| serde_json::from_value(a.clone()).ok());
            output.push(crate::openai::ResponsesOutputItem::Message {
                id: format!("msg_{}", id),
                status: "completed",
                role: "assistant",
                content: vec![crate::openai::ResponsesContentPart::OutputText {
                    annotations,
                    text: text.to_string(),
                }],
            });
        }
    }
    let u = chat
        .get("usage")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let usage = crate::openai::ResponsesUsage {
        input_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        input_tokens_details: u
            .get("prompt_tokens_details")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        output_tokens: u
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize,
        output_tokens_details: u
            .get("completion_tokens_details")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        total_tokens: u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
    };
    let resp_id = format!("resp_{}", id.trim_start_matches("chatcmpl-"));
    let resp = crate::openai::ResponsesResponse {
        id: resp_id.clone(),
        object: "response",
        created_at: created,
        model: model.clone(),
        status: "completed",
        error: None,
        output,
        reasoning: None,
        usage,
        metadata: req_metadata,
    };

    // Persist the full transcript for previous_response_id resume. We
    // serialize before returning so the stored body is byte-identical to
    // what we hand back to the caller.
    let body = match serde_json::to_value(&resp) {
        Ok(v) => v,
        Err(e) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("response serialization failed: {e}"),
            );
        }
    };
    if store_flag && let Some(store) = store {
        let mut transcript = input_messages.clone();
        // Append the assistant turn so subsequent resumes see it.
        if let Some(assistant_msg) = extract_assistant_incoming_message(&chat) {
            transcript.push(assistant_msg);
        }
        store.insert(crate::response_store::StoredEntry {
            id: resp_id,
            kind: crate::response_store::StoredKind::Response,
            model: model.clone(),
            created_at: created,
            messages: transcript,
            body: body.clone(),
            last_access: std::time::Instant::now(),
        });
    }

    // Conversation append: new user items + assistant reply.
    if let Some((conv_store, conv_id)) = conversation {
        let prior = conv_store.get(&conv_id).map(|s| s.items.len()).unwrap_or(0);
        let mut batch: Vec<serde_json::Value> = input_messages
            .iter()
            .skip(prior)
            .map(|m| {
                serde_json::json!({
                    "type": "message",
                    "role": m.role,
                    "content": [{"type": "input_text", "text": m.content.text}],
                })
            })
            .collect();
        let assistant_text = chat
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if !assistant_text.is_empty() {
            batch.push(serde_json::json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": assistant_text}],
            }));
        }
        if !batch.is_empty()
            && let Err(e) = conv_store.add_items(&conv_id, batch)
        {
            tracing::warn!(
                "responses_translate: conversation_store.add_items failed for {conv_id}: {e:?}"
            );
        }
    }

    Json(body).into_response()
}
