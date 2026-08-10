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

/// Encode the pipeline outcome for the Responses API surface. A
/// blocking success arrives as the canonical response IR (typed — no
/// serialized-body re-parse); error envelopes pass through unchanged.
///
/// When `store` is provided and `store_flag` is true, the final
/// `{input_messages + assistant turn}` transcript is persisted under
/// `resp_<id>` so a follow-up `previous_response_id` lookup can resume.
pub(super) async fn translate_chat_response_to_responses(
    outcome: super::chat::ChatOutcome,
    req_metadata: Option<std::collections::HashMap<String, String>>,
    store: Option<Arc<crate::response_store::ResponseStore>>,
    input_messages: Vec<crate::openai::IncomingMessage>,
    store_flag: bool,
    conversation: Option<(Arc<crate::conversation_store::ConversationStore>, String)>,
) -> Response {
    let chat = match outcome {
        super::chat::ChatOutcome::Http(r) => return r,
        // Unreachable: this encoder serves the non-streaming endpoint.
        super::chat::ChatOutcome::Streaming(_) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal: expected blocking outcome".to_string(),
            );
        }
        super::chat::ChatOutcome::Blocking(ir) => *ir,
    };
    // Wire-compatible ids: the historical path embedded the full
    // `chatcmpl-<uuid>` id inside the item ids.
    let full_id = format!("chatcmpl-{}", chat.id);
    let mut output: Vec<crate::openai::ResponsesOutputItem> = Vec::new();
    if let Some(choice) = chat.choices.first() {
        for (i, tc) in choice.tool_calls.iter().enumerate() {
            output.push(crate::openai::ResponsesOutputItem::FunctionCall {
                id: format!("fc_{}_{}", full_id, i),
                call_id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: tc.arguments.to_string(),
                status: "completed",
            });
        }
        if let Some(text) = choice.content.as_deref() {
            // URL-citation annotations derived from the final content,
            // matching the chat surface's encoder.
            let annotations = crate::openai::merged_annotations(text);
            output.push(crate::openai::ResponsesOutputItem::Message {
                id: format!("msg_{}", full_id),
                status: "completed",
                role: "assistant",
                content: vec![crate::openai::ResponsesContentPart::OutputText {
                    annotations,
                    text: text.to_string(),
                }],
            });
        }
    }
    let usage = crate::openai::ResponsesUsage {
        input_tokens: chat.usage.prompt_tokens,
        input_tokens_details: Some(crate::openai::PromptTokensDetails {
            cached_tokens: chat.usage.cached_prompt_tokens,
            audio_tokens: 0,
        }),
        output_tokens: chat.usage.completion_tokens,
        output_tokens_details: Some(crate::openai::CompletionTokensDetails {
            reasoning_tokens: chat.usage.reasoning_tokens,
            audio_tokens: 0,
            accepted_prediction_tokens: 0,
            rejected_prediction_tokens: 0,
        }),
        total_tokens: chat.usage.prompt_tokens + chat.usage.completion_tokens,
    };
    let resp_id = format!("resp_{}", chat.id);
    let resp = crate::openai::ResponsesResponse {
        id: resp_id.clone(),
        object: "response",
        created_at: chat.created,
        model: chat.model.clone(),
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
        if let Some(assistant_msg) = assistant_incoming_from_ir(&chat) {
            transcript.push(assistant_msg);
        }
        store.insert(crate::response_store::StoredEntry {
            id: resp_id,
            kind: crate::response_store::StoredKind::Response,
            model: chat.model.clone(),
            created_at: chat.created,
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
            .choices
            .first()
            .and_then(|c| c.content.as_deref())
            .unwrap_or("");
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
