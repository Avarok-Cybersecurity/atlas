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

pub async fn cancel_response(axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    openai_error_response_with_param(
        StatusCode::BAD_REQUEST,
        format!(
            "Response '{id}' cannot be cancelled: Atlas completes responses synchronously. Cancel only applies when the request was created with `background: true`, which this server does not support."
        ),
        Some("id"),
        Some("response_not_cancellable"),
    )
}

/// GET /metrics — Prometheus metrics endpoint.
pub async fn metrics_handler() -> impl IntoResponse {
    use prometheus::Encoder;
    use std::fmt::Write;

    let encoder = prometheus::TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    let mut text = String::from_utf8(buffer).unwrap_or_default();

    // Prefix cache counters (global atomics from spark-runtime)
    let hits = spark_runtime::prefix_cache::cache_hit_count();
    let misses = spark_runtime::prefix_cache::cache_miss_count();
    let hit_tokens = spark_runtime::prefix_cache::cache_hit_tokens_total();
    let total = hits + misses;
    let hit_rate = if total > 0 {
        hits as f64 / total as f64
    } else {
        0.0
    };

    let _ = write!(
        text,
        "\
        # HELP atlas_prefix_cache_hits_total Prefix cache lookups that found cached blocks\n\
        # TYPE atlas_prefix_cache_hits_total counter\n\
        atlas_prefix_cache_hits_total {hits}\n\
        # HELP atlas_prefix_cache_misses_total Prefix cache lookups with no match\n\
        # TYPE atlas_prefix_cache_misses_total counter\n\
        atlas_prefix_cache_misses_total {misses}\n\
        # HELP atlas_prefix_cache_hit_tokens_total Tokens reused from prefix cache\n\
        # TYPE atlas_prefix_cache_hit_tokens_total counter\n\
        atlas_prefix_cache_hit_tokens_total {hit_tokens}\n\
        # HELP atlas_prefix_cache_hit_rate Prefix cache hit rate (0-1)\n\
        # TYPE atlas_prefix_cache_hit_rate gauge\n\
        atlas_prefix_cache_hit_rate {hit_rate:.4}\n"
    );

    // Entropy monitoring (global atomics from spark-runtime sampler)
    let entropy = spark_runtime::sampler::last_entropy();
    let low_entropy = spark_runtime::sampler::low_entropy_token_count();
    let total_sampled = spark_runtime::sampler::total_sampled_token_count();
    let low_ratio = if total_sampled > 0 {
        low_entropy as f64 / total_sampled as f64
    } else {
        0.0
    };

    let _ = write!(
        text,
        "\
        # HELP atlas_token_entropy_last Most recent per-token entropy (nats)\n\
        # TYPE atlas_token_entropy_last gauge\n\
        atlas_token_entropy_last {entropy:.4}\n\
        # HELP atlas_low_entropy_tokens_total Tokens with entropy below 0.3\n\
        # TYPE atlas_low_entropy_tokens_total counter\n\
        atlas_low_entropy_tokens_total {low_entropy}\n\
        # HELP atlas_low_entropy_ratio Fraction of tokens with entropy below 0.3\n\
        # TYPE atlas_low_entropy_ratio gauge\n\
        atlas_low_entropy_ratio {low_ratio:.4}\n"
    );

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        text,
    )
}

/// GET /health — readiness probe (503 while model is loading).
pub async fn health(State(state): State<Arc<AppState>>) -> Response {
    if state.model_ready.load(std::sync::atomic::Ordering::Relaxed) {
        Json(serde_json::json!({"status": "ready", "model": &state.model_name})).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "loading"})),
        )
            .into_response()
    }
}

/// GET /health/live — liveness probe (always 200).
pub async fn health_live() -> &'static str {
    "ok"
}

/// POST /tokenize — tokenize text or chat messages, return token IDs and count.
pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    req: Result<Json<crate::openai::TokenizeRequest>, JsonRejection>,
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

    let tokens = if let Some(ref prompt) = req.prompt {
        match state.tokenizer.encode(prompt) {
            Ok(t) => t,
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Tokenization error: {e}"),
                );
            }
        }
    } else if let Some(ref messages) = req.messages {
        let json_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content.text}))
            .collect();
        match state.tokenizer.apply_chat_template_jinja(
            &json_messages,
            None,
            false,
            state.behavior.disable_tool_steering,
        ) {
            Ok(t) => t,
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Tokenization error: {e}"),
                );
            }
        }
    } else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "Either 'prompt' or 'messages' is required".to_string(),
        );
    };

    let count = tokens.len();
    Json(crate::openai::TokenizeResponse { tokens, count }).into_response()
}

/// Request body for POST /detokenize.
#[derive(serde::Deserialize)]
pub struct DetokenizeRequest {
    tokens: Vec<u32>,
}

/// POST /detokenize — decode token IDs back to text.
pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    req: Result<Json<DetokenizeRequest>, JsonRejection>,
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
    match state.tokenizer.decode(&req.tokens) {
        Ok(text) => Json(serde_json::json!({"text": text})).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}
