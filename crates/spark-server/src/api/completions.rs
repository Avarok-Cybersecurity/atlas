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

pub async fn completions(
    State(state): State<Arc<AppState>>,
    req: Result<Json<CompletionRequest>, JsonRejection>,
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
    // For thinking models, prepend <think></think>\n\n to suppress think-tag
    // leakage in raw completions mode (the model expects this prefix after
    // training). Users who construct their own think tokens can include them
    // in the prompt — we only add the prefix if the prompt doesn't already
    // contain a </think> token.
    let raw_prompt = if state.tokenizer.supports_thinking() && !req.prompt.contains("</think>") {
        format!("<think></think>\n\n{}", req.prompt)
    } else {
        req.prompt.clone()
    };
    let prompt_tokens = match state.tokenizer.encode(&raw_prompt) {
        Ok(t) => t,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Tokenization error: {e}"),
            );
        }
    };

    let prompt_len = prompt_tokens.len();
    if prompt_len >= state.max_seq_len {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "Prompt too long: {prompt_len} tokens exceeds max_seq_len {}",
                state.max_seq_len
            ),
        );
    }

    let temperature = req.temperature.unwrap_or(state.default_temperature);
    let top_k = req.top_k.unwrap_or(state.default_top_k);
    let top_p = req.top_p.unwrap_or(state.default_top_p);
    let top_n_sigma = req.top_n_sigma.unwrap_or(state.default_top_n_sigma);
    let min_p = req.min_p.unwrap_or(state.default_min_p);
    let repetition_penalty = req
        .repetition_penalty
        .unwrap_or(state.sampling_presets.non_thinking.repetition_penalty);
    let presence_penalty = req.presence_penalty.unwrap_or(0.0);
    let frequency_penalty = req.frequency_penalty.unwrap_or(0.0);
    // Convert logit_bias from OpenAI format (string keys) to Vec<(u32, f32)>
    let logit_bias: Vec<(u32, f32)> = req.logit_bias.as_ref().map_or(Vec::new(), |map| {
        map.iter()
            .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
            .collect()
    });
    let stop_tokens = tokenize_stop_sequences(&state.tokenizer, &req.stop);

    if req.stream {
        return match completions_stream(
            state,
            prompt_tokens,
            req.max_tokens,
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            logit_bias.clone(),
            stop_tokens,
            req.seed,
        )
        .await
        {
            Ok(r) => r,
            Err((status, msg)) => openai_error_response(status, msg),
        };
    }

    // ── Blocking path ──
    let (tx, rx) = tokio::sync::oneshot::channel();
    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let request = InferenceRequest::Blocking {
        prompt_tokens,
        session_hash,
        image_pixels: Vec::new(),
        max_tokens: req.max_tokens,
        min_tokens: 0,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        // Legacy /v1/completions path doesn't have tool semantics, so
        // no DRY. (DRY on raw completion would dampen legitimate
        // long-repeated prose.)
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        lz_penalty: 0.0,
        logit_bias,
        stop_tokens,
        enable_thinking: false,
        thinking_budget: None,
        require_tool_call: false,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_spec: None,
        seed: req.seed,
        top_logprobs: None,
        timeout_at: {
            let secs = state.request_timeout as f32;
            if secs > 0.0 {
                Some(std::time::Instant::now() + std::time::Duration::from_secs_f32(secs))
            } else {
                None
            }
        },
        response_tx: tx,
    };

    if state.request_tx.send(request).await.is_err() {
        return openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "Scheduler queue full".to_string(),
        );
    }

    let response = match rx.await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Inference error: {e}"),
            );
        }
        Err(_) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Inference cancelled".to_string(),
            );
        }
    };

    let output_text = match state.tokenizer.decode(&response.output_tokens) {
        Ok(t) => t,
        Err(e) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Decode error: {e}"),
            );
        }
    };
    let output_text = strip_stop_sequences(output_text, &req.stop);
    let output_text = strip_thinking_tags(&output_text);

    let num_completion = response.output_tokens.len();
    let tokens_per_second = if response.decode_time_ms > 0.0 {
        (num_completion.saturating_sub(1)) as f64 / (response.decode_time_ms / 1000.0)
    } else {
        0.0
    };
    let usage = Usage {
        prompt_tokens: prompt_len,
        completion_tokens: num_completion,
        total_tokens: prompt_len + num_completion,
        prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
            cached_tokens: response.cached_prompt_tokens as usize,
            audio_tokens: 0,
        }),
        completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
            reasoning_tokens: response.reasoning_tokens as usize,
            audio_tokens: 0,
            accepted_prediction_tokens: 0,
            rejected_prediction_tokens: 0,
        }),
        time_to_first_token_ms: response.time_to_first_token_ms,
        response_tokens_per_second: tokens_per_second,
    };

    Json(CompletionResponse::new(
        &state.model_name,
        output_text,
        usage,
        &response.finish_reason,
    ))
    .into_response()
}

/// SSE streaming path for legacy completions.
pub(super) async fn completions_stream(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    top_n_sigma: f32,
    min_p: f32,
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    logit_bias: Vec<(u32, f32)>,
    stop_tokens: Vec<u32>,
    seed: Option<u64>,
) -> Result<Response, (StatusCode, String)> {
    // Match chat_stream/mod.rs sizing; see comment there.
    let (token_tx, token_rx) = tokio::sync::mpsc::channel::<StreamEvent>(1024);
    let prompt_len = prompt_tokens.len();

    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let request = InferenceRequest::Streaming {
        prompt_tokens,
        session_hash,
        image_pixels: Vec::new(),
        max_tokens,
        min_tokens: 0,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        // Legacy /v1/completions path doesn't have tool semantics.
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        lz_penalty: 0.0,
        logit_bias,
        stop_tokens,
        enable_thinking: false,
        thinking_budget: None,
        require_tool_call: false,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_spec: None,
        seed,
        top_logprobs: None,
        timeout_at: None,
        token_tx,
    };

    state.request_tx.send(request).await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Scheduler queue full".to_string(),
        )
    })?;

    let chunk_id = crate::openai::new_completion_id();
    let model_name = state.model_name.clone();

    let model = model_name.clone();
    let id = chunk_id.clone();
    let mut all_toks: Vec<u32> = Vec::new();
    let mut emitted: usize = 0;
    let token_stream = ReceiverStream::new(token_rx).map(move |event| match event {
        StreamEvent::Token(tok) | StreamEvent::TokenWithLogprobs(tok, _) => {
            all_toks.push(tok);
            let full = state.tokenizer.decode(&all_toks).unwrap_or_default();
            let stable_end = full.trim_end_matches('\u{FFFD}').len();
            if stable_end <= emitted {
                let chunk = CompletionChunk::text_chunk(&model, &id, String::new());
                let json = serde_json::to_string(&chunk).unwrap_or_default();
                return Ok::<_, std::convert::Infallible>(Event::default().data(json));
            }
            let delta = full[emitted..stable_end].to_string();
            emitted = stable_end;
            let chunk = CompletionChunk::text_chunk(&model, &id, delta);
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            Ok::<_, std::convert::Infallible>(Event::default().data(json))
        }
        StreamEvent::Done {
            finish_reason,
            prompt_tokens: _,
            completion_tokens,
            time_to_first_token_ms,
            decode_time_ms,
            reasoning_tokens,
            cached_prompt_tokens,
        } => {
            let tps = if decode_time_ms > 0.0 {
                completion_tokens.saturating_sub(1) as f64 / (decode_time_ms / 1000.0)
            } else {
                0.0
            };
            let usage = Usage {
                prompt_tokens: prompt_len,
                completion_tokens,
                total_tokens: prompt_len + completion_tokens,
                prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
                    cached_tokens: cached_prompt_tokens as usize,
                    audio_tokens: 0,
                }),
                completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
                    reasoning_tokens: reasoning_tokens as usize,
                    audio_tokens: 0,
                    accepted_prediction_tokens: 0,
                    rejected_prediction_tokens: 0,
                }),
                time_to_first_token_ms,
                response_tokens_per_second: tps,
            };
            let chunk = CompletionChunk::done_chunk(&model, &id, &finish_reason, usage);
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            Ok(Event::default().data(json))
        }
        StreamEvent::Error(msg) => Ok(Event::default().data(format!(r#"{{"error":"{msg}"}}"#))),
    });

    let done_event = futures::stream::once(async {
        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
    });
    let full_stream = token_stream.chain(done_event);

    Ok(Sse::new(full_stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

/// GET /v1/models
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: crate::openai::unix_timestamp(),
            owned_by: "atlas-spark".to_string(),
        }],
    })
}

/// GET /v1/models/{model_id} — retrieve a single model (OpenAI SDK `client.models.retrieve()`).
pub async fn get_model(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Response {
    if model_id == state.model_name {
        Json(serde_json::json!({
            "id": state.model_name,
            "object": "model",
            "created": crate::openai::unix_timestamp(),
            "owned_by": "atlas-spark",
        }))
        .into_response()
    } else {
        openai_error_response(
            StatusCode::NOT_FOUND,
            format!("The model '{model_id}' does not exist"),
        )
    }
}

/// POST /v1/embeddings — stub for clients that probe this endpoint during auto-detection.
pub async fn embeddings_stub() -> Response {
    openai_error_response(
        StatusCode::NOT_IMPLEMENTED,
        "Embeddings are not supported by this model. Atlas serves generative (chat/completion) models only.".into(),
    )
}

/// Generic 501 "not supported" response used by the auto-probe stubs
/// below. OpenAI-SDK auto-detection and observability wrappers expect a
/// 501 + `error.type = server_error`; returning 404 would be interpreted
/// as "wrong URL".
pub(super) fn not_supported(message: &'static str) -> Response {
    openai_error_response(StatusCode::NOT_IMPLEMENTED, message.into())
}
