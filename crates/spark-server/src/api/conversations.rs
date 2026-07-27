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
use super::stored::assistant_incoming_from_ir;
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

#[derive(serde::Deserialize)]
pub struct CreateConversationRequest {
    #[serde(default)]
    pub items: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

#[derive(serde::Deserialize)]
pub struct UpdateConversationRequest {
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(serde::Deserialize)]
pub struct AddItemsRequest {
    pub items: Vec<serde_json::Value>,
}

/// Build the public JSON shape for a conversation snapshot.
pub(super) fn conversation_body(
    snap: &crate::conversation_store::ConversationSnapshot,
) -> serde_json::Value {
    serde_json::json!({
        "id": snap.id,
        "object": "conversation",
        "created_at": snap.created_at,
        "metadata": snap.metadata,
    })
}

/// POST /v1/conversations — create a conversation with optional
/// initial items + metadata.
pub async fn create_conversation(
    State(state): State<Arc<AppState>>,
    req: Result<Json<CreateConversationRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(_) => {
            // Body is optional per the OpenAI spec — empty body is OK.
            Json(CreateConversationRequest {
                items: None,
                metadata: None,
            })
        }
    };
    let items = req.items.unwrap_or_default();
    if items.len() > crate::conversation_store::MAX_ITEMS_PER_INSERT {
        return openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            format!(
                "`items` exceeds per-call cap of {} (got {}).",
                crate::conversation_store::MAX_ITEMS_PER_INSERT,
                items.len(),
            ),
            Some("items"),
            Some("items_too_many"),
        );
    }
    let id = state
        .conversation_store
        .create(items, req.metadata.unwrap_or_default());
    let snap = state.conversation_store.get(&id).expect("just created");
    Json(conversation_body(&snap)).into_response()
}

/// GET /v1/conversations/{id}
pub async fn get_conversation(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    match state.conversation_store.get(&id) {
        Some(snap) => Json(conversation_body(&snap)).into_response(),
        None => openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Conversation '{id}' not found."),
            Some("id"),
            Some("conversation_not_found"),
        ),
    }
}

/// POST /v1/conversations/{id} — update metadata.
pub async fn update_conversation(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
    req: Result<Json<UpdateConversationRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    match state.conversation_store.update_metadata(&id, req.metadata) {
        Some(snap) => Json(conversation_body(&snap)).into_response(),
        None => openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Conversation '{id}' not found."),
            Some("id"),
            Some("conversation_not_found"),
        ),
    }
}

/// DELETE /v1/conversations/{id}
pub async fn delete_conversation(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    if state.conversation_store.delete(&id) {
        Json(serde_json::json!({
            "id": id,
            "object": "conversation.deleted",
            "deleted": true,
        }))
        .into_response()
    } else {
        openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Conversation '{id}' not found."),
            Some("id"),
            Some("conversation_not_found"),
        )
    }
}

/// POST /v1/conversations/{id}/items — append items (≤20/call).
pub async fn add_conversation_items(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
    req: Result<Json<AddItemsRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    match state.conversation_store.add_items(&id, req.items) {
        Ok(items) => Json(serde_json::json!({
            "object": "list",
            "data": items,
        }))
        .into_response(),
        Err(crate::conversation_store::AddItemsError::NotFound) => {
            openai_error_response_with_param(
                StatusCode::NOT_FOUND,
                format!("Conversation '{id}' not found."),
                Some("id"),
                Some("conversation_not_found"),
            )
        }
        Err(crate::conversation_store::AddItemsError::TooMany(n)) => {
            openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                format!(
                    "`items` exceeds per-call cap of {} (got {n}).",
                    crate::conversation_store::MAX_ITEMS_PER_INSERT,
                ),
                Some("items"),
                Some("items_too_many"),
            )
        }
    }
}

/// GET /v1/conversations/{id}/items — list items with `limit` + `order`
/// query parameters (OpenAI spec: default 20, max 100, order=asc).
pub async fn list_conversation_items(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let Some(snap) = state.conversation_store.get(&id) else {
        return openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Conversation '{id}' not found."),
            Some("id"),
            Some("conversation_not_found"),
        );
    };
    let mut items = snap.items;
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

/// GET /v1/conversations/{id}/items/{item_id}
pub async fn get_conversation_item(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((id, item_id)): axum::extract::Path<(String, String)>,
) -> Response {
    let Some(snap) = state.conversation_store.get(&id) else {
        return openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Conversation '{id}' not found."),
            Some("id"),
            Some("conversation_not_found"),
        );
    };
    for it in &snap.items {
        if it.get("id").and_then(|v| v.as_str()) == Some(item_id.as_str()) {
            return Json(it.clone()).into_response();
        }
    }
    openai_error_response_with_param(
        StatusCode::NOT_FOUND,
        format!("Item '{item_id}' not found in conversation '{id}'."),
        Some("item_id"),
        Some("item_not_found"),
    )
}

/// DELETE /v1/conversations/{id}/items/{item_id}
pub async fn delete_conversation_item(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((id, item_id)): axum::extract::Path<(String, String)>,
) -> Response {
    if state.conversation_store.remove_item(&id, &item_id) {
        Json(serde_json::json!({
            "id": item_id,
            "object": "conversation.item.deleted",
            "deleted": true,
        }))
        .into_response()
    } else {
        openai_error_response_with_param(
            StatusCode::NOT_FOUND,
            format!("Item '{item_id}' not found in conversation '{id}'."),
            Some("item_id"),
            Some("item_not_found"),
        )
    }
}
