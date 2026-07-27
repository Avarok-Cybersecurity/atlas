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

pub async fn responses_endpoint(
    state: State<Arc<AppState>>,
    req: Result<Json<crate::openai::ResponsesRequest>, JsonRejection>,
) -> Response {
    let Json(r) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    let metadata = r.metadata.clone();
    let store_flag = r.store.unwrap_or(true); // Responses API defaults to store=true.
    let streaming = r.stream;

    // Resolve `conversation` field (2026): either a string id or
    // `{"id": "..."}`. When set, pre-seed the turn with the stored
    // items AND append the new turn's items back after completion.
    let conversation_id: Option<String> = match &r.conversation {
        None => None,
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Object(o)) => {
            o.get("id").and_then(|v| v.as_str()).map(|s| s.to_string())
        }
        Some(_) => {
            return openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                "`conversation` must be a string id or an object with an `id` field.".into(),
                Some("conversation"),
                None,
            );
        }
    };
    let conversation_prefix: Vec<crate::openai::IncomingMessage> = match &conversation_id {
        None => Vec::new(),
        Some(cid) => match state.conversation_store.get(cid) {
            Some(snap) => snap
                .items
                .iter()
                .filter_map(crate::openai::IncomingMessage::from_conversation_item)
                .collect(),
            None => {
                return openai_error_response_with_param(
                    StatusCode::NOT_FOUND,
                    format!("Conversation '{cid}' not found."),
                    Some("conversation"),
                    Some("conversation_not_found"),
                );
            }
        },
    };

    // Resolve previous_response_id against the in-memory store. Wrapped in
    // a closure so lower_responses_to_chat can report a clean 400 when the
    // id is unknown without us leaking the store ref into openai.rs.
    let store = state.response_store.clone();
    let resolve = |prior_id: &str| -> Option<Vec<crate::openai::IncomingMessage>> {
        store
            .get(prior_id, crate::response_store::StoredKind::Response)
            .map(|e| e.messages)
    };

    let mut chat_req = match crate::openai::lower_responses_to_chat(r, resolve) {
        Ok(c) => c,
        Err(crate::openai::LowerResponsesError::BadRequest(m)) => {
            return openai_error_response(StatusCode::BAD_REQUEST, m);
        }
        Err(crate::openai::LowerResponsesError::PriorNotFound(m)) => {
            return openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                m,
                Some("previous_response_id"),
                Some("response_not_found"),
            );
        }
    };

    // Prepend conversation items (if any) before the lowered input so
    // the turn sees the full prior history.
    if !conversation_prefix.is_empty() {
        let mut combined = conversation_prefix;
        combined.append(&mut chat_req.messages);
        chat_req.messages = combined;
    }

    if streaming {
        return responses_endpoint_stream(state, chat_req, metadata, store_flag, conversation_id)
            .await;
    }

    // Capture the input transcript BEFORE moving chat_req into the handler
    // — we need it for the stored-turn's `messages` field.
    let input_messages = chat_req.messages.clone();

    // Re-enter the blocking chat-completions handler with the lowered
    // request. Use the _inner variant because we already have a parsed
    // struct (no raw bytes available to dump at this layer; the Responses
    // handler dumps at its own entry point if --dump is enabled).
    let resp = chat_completions_inner(state.0.clone(), None, chat_req.into(), None).await;
    let conv_pair = conversation_id.map(|cid| (state.conversation_store.clone(), cid));
    translate_chat_response_to_responses(
        resp,
        metadata,
        Some(state.response_store.clone()),
        input_messages,
        store_flag,
        conv_pair,
    )
    .await
}
