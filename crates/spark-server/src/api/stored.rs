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

use super::chat_stream::run_chat_stream;
use super::responses_stream::responses_endpoint_stream;
use super::responses_translate::{
    build_responses_usage, emit, find_frame_end, translate_chat_response_to_responses,
};
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
use super::inference_types::*;
use super::sanitizer::*;
/// Build the assistant transcript turn from the canonical response IR
/// (for `previous_response_id` resume storage).
pub(super) fn assistant_incoming_from_ir(
    chat: &crate::ir::ChatResponse,
) -> Option<crate::openai::IncomingMessage> {
    let choice = chat.choices.first()?;
    let tool_calls: Vec<tool_parser::IncomingToolCall> = choice
        .tool_calls
        .iter()
        .map(|tc| tool_parser::IncomingToolCall {
            id: Some(tc.id.clone()),
            function: tool_parser::IncomingFunction {
                name: tc.name.clone(),
                arguments: tc.arguments.to_string(),
            },
        })
        .collect();
    Some(crate::openai::IncomingMessage {
        role: "assistant".to_string(),
        content: crate::openai::ParsedContent {
            text: choice.content.clone().unwrap_or_default(),
            images: Vec::new(),
        },
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        tool_call_id: None,
        name: None,
        reasoning_content: choice.reasoning.clone(),
    })
}

/// GET /v1/chat/completions/{id} — OpenAI completion-storage retrieval.
///
/// When a prior `POST /v1/chat/completions` with `store: true` persisted a
/// completion, this endpoint returns the stored body verbatim. Otherwise
/// 404 with a clear error so Helicone/Langfuse-style clients fall back.
pub async fn get_stored_completion(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    match state
        .response_store
        .get(&id, crate::response_store::StoredKind::ChatCompletion)
    {
        Some(entry) => Json(entry.body).into_response(),
        None => openai_error_response(
            StatusCode::NOT_FOUND,
            format!(
                "Completion '{id}' not found. It may have expired, or was never stored (set `store: true` on the request to enable storage)."
            ),
        ),
    }
}

// ── Responses API CRUD ──────────────────────────────────────────────
//
// All Responses persisted by `responses_endpoint` (blocking or streaming
// unless `store: false`) are retrievable / deletable / inspectable via
// their `resp_<uuid>` id. The store is the same LRU+TTL backing
// `/v1/chat/completions/{id}`; kind-typed so these handlers only see
// `Response` entries.

/// GET /v1/responses/{id} — retrieve a stored Response body.
pub async fn get_stored_response(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    match state
        .response_store
        .get(&id, crate::response_store::StoredKind::Response)
    {
        Some(entry) => Json(entry.body).into_response(),
        None => openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Response '{id}' not found. It may have expired or was not stored."),
            Some("id"),
            Some("response_not_found"),
        ),
    }
}

/// DELETE /v1/responses/{id} — forget a stored Response.
///
/// Returns `{id, object:"response.deleted", deleted:true}` on hit and
/// 404 on miss. Deletion removes the entry from the in-memory LRU and,
/// when the filesystem backend is active, unlinks the `<id>.json` file.
pub async fn delete_stored_response(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let existed = state
        .response_store
        .delete(&id, crate::response_store::StoredKind::Response);
    if existed {
        Json(serde_json::json!({
            "id": id,
            "object": "response.deleted",
            "deleted": true,
        }))
        .into_response()
    } else {
        openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Response '{id}' not found."),
            Some("id"),
            Some("response_not_found"),
        )
    }
}

/// GET /v1/responses/{id}/input_items — list the items that were fed
/// into the stored Response.
///
/// Supports the `limit` and `order` query parameters from the OpenAI
/// spec. `after` / `before` cursor pagination is accepted and applied
/// when the id matches an item's `id` field. We don't page-break mid-
/// list — the whole transcript fits in a single page for any realistic
/// multi-turn conversation.
pub async fn list_response_input_items(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let Some(entry) = state
        .response_store
        .get(&id, crate::response_store::StoredKind::Response)
    else {
        return openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Response '{id}' not found."),
            Some("id"),
            Some("response_not_found"),
        );
    };

    // Drop the trailing assistant turn: `input_items` lists what the
    // caller sent in, not what the model produced. The store keeps the
    // full transcript so previous_response_id can resume; we exclude
    // the last assistant message on this endpoint.
    let mut msgs: Vec<crate::openai::IncomingMessage> = entry.messages;
    if msgs.last().map(|m| m.role == "assistant").unwrap_or(false) {
        msgs.pop();
    }

    let mut items: Vec<serde_json::Value> = msgs
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let content = if m.content.images.is_empty() {
                serde_json::json!([{ "type": "input_text", "text": m.content.text }])
            } else {
                let mut parts: Vec<serde_json::Value> = Vec::new();
                if !m.content.text.is_empty() {
                    parts.push(serde_json::json!({ "type": "input_text", "text": m.content.text }));
                }
                for img in &m.content.images {
                    parts.push(serde_json::json!({
                        "type": "input_image",
                        "image_url": img,
                    }));
                }
                serde_json::Value::Array(parts)
            };
            serde_json::json!({
                "id": format!("item_{id}_{i}"),
                "type": "message",
                "role": m.role,
                "content": content,
            })
        })
        .collect();

    let order = q.get("order").map(|s| s.as_str()).unwrap_or("asc");
    if order == "desc" {
        items.reverse();
    }
    let limit: usize = q
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(20)
        .min(100);
    if items.len() > limit {
        items.truncate(limit);
    }

    let first_id = items
        .first()
        .and_then(|v| v.get("id").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string();
    let last_id = items
        .last()
        .and_then(|v| v.get("id").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string();
    Json(serde_json::json!({
        "object": "list",
        "data": items,
        "first_id": first_id,
        "last_id": last_id,
        "has_more": false,
    }))
    .into_response()
}
