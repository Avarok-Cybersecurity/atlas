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

pub async fn batches_stub() -> Response {
    not_supported(
        "Batch API is not supported. Submit requests directly to /v1/chat/completions; Atlas serves them synchronously.",
    )
}

/// GET /v1/batches/{id} — Atlas has no batch store.
pub async fn batch_get_stub() -> Response {
    not_supported("Batch API is not supported. No batches are tracked on this server.")
}

/// GET /v1/batches and DELETE /v1/batches/{id} share the same 501 shape.
pub async fn batch_list_stub() -> Response {
    not_supported("Batch API is not supported. No batches are tracked on this server.")
}

/// POST/GET/DELETE /v1/files* — Atlas has no file-upload store.
pub async fn files_stub() -> Response {
    not_supported(
        "File storage API is not supported. Atlas is an inference-only server; upload-then-reference workflows (batches, vision by file_id) are not available.",
    )
}

/// POST /v1/audio/* — Atlas has no ASR/TTS model loaded.
pub async fn audio_stub() -> Response {
    not_supported("Audio API is not supported. Atlas serves text chat/completion models only.")
}

/// POST /v1/images/* — Atlas has no image-generation model loaded.
pub async fn images_stub() -> Response {
    not_supported("Image API is not supported. Atlas serves text chat/completion models only.")
}

/// POST /v1/moderations — Atlas does not run a safety-classifier model.
pub async fn moderations_stub() -> Response {
    not_supported(
        "Moderations API is not supported. Atlas does not classify inputs for safety; run your own moderation pass upstream if needed.",
    )
}
