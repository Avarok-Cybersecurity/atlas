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
use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
use super::completions::not_supported;
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};

use super::chat::chat_completions_inner;
use super::strip::strip_thinking_tags;

// Re-export sibling helpers via crate::api::* for short paths.
use super::inference_types::*;
use super::sanitizer::*;

impl InferenceRequest {
    /// Number of prompt tokens in this request.
    pub fn prompt_len(&self) -> usize {
        match self {
            InferenceRequest::Blocking { prompt_tokens, .. } => prompt_tokens.len(),
            InferenceRequest::Streaming { prompt_tokens, .. } => prompt_tokens.len(),
        }
    }

    /// Whether this request carries preprocessed image embeddings.
    pub fn has_image_pixels(&self) -> bool {
        match self {
            InferenceRequest::Blocking { image_pixels, .. } => !image_pixels.is_empty(),
            InferenceRequest::Streaming { image_pixels, .. } => !image_pixels.is_empty(),
        }
    }

    /// Preprocessed image data, consumed by the scheduler before prefill.
    pub fn take_image_pixels(&mut self) -> Vec<(Vec<f32>, usize, usize)> {
        match self {
            InferenceRequest::Blocking { image_pixels, .. } => std::mem::take(image_pixels),
            InferenceRequest::Streaming { image_pixels, .. } => std::mem::take(image_pixels),
        }
    }

    /// Borrow the preprocessed image data (non-consuming) — used by the vision
    /// co-dispatch pre-pass to batch-encode across requests before the admit
    /// loop consumes each request.
    pub fn image_pixels_ref(&self) -> &[(Vec<f32>, usize, usize)] {
        match self {
            InferenceRequest::Blocking { image_pixels, .. } => image_pixels.as_slice(),
            InferenceRequest::Streaming { image_pixels, .. } => image_pixels.as_slice(),
        }
    }

    /// Per-request stop tokens, consumed by the scheduler.
    pub fn take_stop_tokens(&mut self) -> Vec<u32> {
        match self {
            InferenceRequest::Blocking { stop_tokens, .. } => std::mem::take(stop_tokens),
            InferenceRequest::Streaming { stop_tokens, .. } => std::mem::take(stop_tokens),
        }
    }

    /// Top-k sampling parameter.
    pub fn top_k(&self) -> u32 {
        match self {
            InferenceRequest::Blocking { top_k, .. } => *top_k,
            InferenceRequest::Streaming { top_k, .. } => *top_k,
        }
    }

    /// Top-p sampling parameter.
    pub fn top_p(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { top_p, .. } => *top_p,
            InferenceRequest::Streaming { top_p, .. } => *top_p,
        }
    }

    /// Top-n-sigma sampling parameter.
    pub fn top_n_sigma(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { top_n_sigma, .. } => *top_n_sigma,
            InferenceRequest::Streaming { top_n_sigma, .. } => *top_n_sigma,
        }
    }

    /// Min-p sampling parameter.
    pub fn min_p(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { min_p, .. } => *min_p,
            InferenceRequest::Streaming { min_p, .. } => *min_p,
        }
    }

    /// Repetition penalty parameter.
    pub fn repetition_penalty(&self) -> f32 {
        match self {
            InferenceRequest::Blocking {
                repetition_penalty, ..
            } => *repetition_penalty,
            InferenceRequest::Streaming {
                repetition_penalty, ..
            } => *repetition_penalty,
        }
    }

    /// Presence penalty (OpenAI-style additive).
    pub fn presence_penalty(&self) -> f32 {
        match self {
            InferenceRequest::Blocking {
                presence_penalty, ..
            } => *presence_penalty,
            InferenceRequest::Streaming {
                presence_penalty, ..
            } => *presence_penalty,
        }
    }

    /// Frequency penalty (OpenAI-style additive).
    pub fn frequency_penalty(&self) -> f32 {
        match self {
            InferenceRequest::Blocking {
                frequency_penalty, ..
            } => *frequency_penalty,
            InferenceRequest::Streaming {
                frequency_penalty, ..
            } => *frequency_penalty,
        }
    }

    /// DRY (Don't-Repeat-Yourself) penalty multiplier. 0.0 = disabled.
    pub fn dry_multiplier(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { dry_multiplier, .. } => *dry_multiplier,
            InferenceRequest::Streaming { dry_multiplier, .. } => *dry_multiplier,
        }
    }

    /// LZ penalty (A.1, arXiv:2504.20131). 0.0 = disabled.
    pub fn lz_penalty(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { lz_penalty, .. } => *lz_penalty,
            InferenceRequest::Streaming { lz_penalty, .. } => *lz_penalty,
        }
    }

    /// DRY penalty exponential base.
    pub fn dry_base(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { dry_base, .. } => *dry_base,
            InferenceRequest::Streaming { dry_base, .. } => *dry_base,
        }
    }

    /// DRY minimum match length before penalty applies.
    pub fn dry_allowed_length(&self) -> u32 {
        match self {
            InferenceRequest::Blocking {
                dry_allowed_length, ..
            } => *dry_allowed_length,
            InferenceRequest::Streaming {
                dry_allowed_length, ..
            } => *dry_allowed_length,
        }
    }

    /// Per-token logit bias.
    pub fn logit_bias(&self) -> &[(u32, f32)] {
        match self {
            InferenceRequest::Blocking { logit_bias, .. } => logit_bias,
            InferenceRequest::Streaming { logit_bias, .. } => logit_bias,
        }
    }

    /// Session hash for SSM snapshot isolation.
    pub fn session_hash(&self) -> u64 {
        match self {
            InferenceRequest::Blocking { session_hash, .. } => *session_hash,
            InferenceRequest::Streaming { session_hash, .. } => *session_hash,
        }
    }

    /// Whether thinking mode is enabled for this request.
    pub fn enable_thinking(&self) -> bool {
        match self {
            InferenceRequest::Blocking {
                enable_thinking, ..
            } => *enable_thinking,
            InferenceRequest::Streaming {
                enable_thinking, ..
            } => *enable_thinking,
        }
    }

    /// Thinking token budget (None = unlimited).
    pub fn thinking_budget(&self) -> Option<u32> {
        match self {
            InferenceRequest::Blocking {
                thinking_budget, ..
            } => *thinking_budget,
            InferenceRequest::Streaming {
                thinking_budget, ..
            } => *thinking_budget,
        }
    }

    /// Per-request override for the vLLM-anchored token-loop detector.
    /// `None` = use the boot-global watchdog parameters.
    pub fn repetition_detection(&self) -> Option<crate::openai::RepetitionDetectionParams> {
        match self {
            InferenceRequest::Blocking {
                repetition_detection,
                ..
            } => *repetition_detection,
            InferenceRequest::Streaming {
                repetition_detection,
                ..
            } => *repetition_detection,
        }
    }

    /// Whether a tool call is required for this request.
    pub fn require_tool_call(&self) -> bool {
        match self {
            InferenceRequest::Blocking {
                require_tool_call, ..
            } => *require_tool_call,
            InferenceRequest::Streaming {
                require_tool_call, ..
            } => *require_tool_call,
        }
    }

    /// #192: whether the request declared tools (gates multi-tool-call
    /// continuation — `</tool_call>` is not a hard stop when true).
    pub fn tools_present(&self) -> bool {
        match self {
            InferenceRequest::Blocking { tools_present, .. } => *tools_present,
            InferenceRequest::Streaming { tools_present, .. } => *tools_present,
        }
    }

    /// Whether `<tool_call>` should be suppressed (loop detected).
    pub fn suppress_tool_call(&self) -> bool {
        match self {
            InferenceRequest::Blocking {
                suppress_tool_call, ..
            } => *suppress_tool_call,
            InferenceRequest::Streaming {
                suppress_tool_call, ..
            } => *suppress_tool_call,
        }
    }

    /// F60 (2026-04-27): whether MTP speculative decoding should be
    /// disabled for this request (set when tools are active and the
    /// env gate is on).
    pub fn disable_mtp(&self) -> bool {
        match self {
            InferenceRequest::Blocking { disable_mtp, .. } => *disable_mtp,
            InferenceRequest::Streaming { disable_mtp, .. } => *disable_mtp,
        }
    }

    /// Seed for deterministic sampling (None = non-deterministic).
    pub fn seed(&self) -> Option<u64> {
        match self {
            InferenceRequest::Blocking { seed, .. } => *seed,
            InferenceRequest::Streaming { seed, .. } => *seed,
        }
    }

    /// Take the grammar specification for constrained decoding.
    pub fn take_grammar_spec(&mut self) -> Option<GrammarSpec> {
        match self {
            InferenceRequest::Blocking { grammar_spec, .. } => grammar_spec.take(),
            InferenceRequest::Streaming { grammar_spec, .. } => grammar_spec.take(),
        }
    }

    /// Minimum tokens before allowing EOS/stop (0 = no minimum).
    pub fn min_tokens(&self) -> usize {
        match self {
            InferenceRequest::Blocking { min_tokens, .. } => *min_tokens,
            InferenceRequest::Streaming { min_tokens, .. } => *min_tokens,
        }
    }

    /// Number of top logprobs to return per token. None = disabled.
    pub fn top_logprobs(&self) -> Option<u8> {
        match self {
            InferenceRequest::Blocking { top_logprobs, .. } => *top_logprobs,
            InferenceRequest::Streaming { top_logprobs, .. } => *top_logprobs,
        }
    }

    /// Legacy /v1/completions prompt-token logprobs (echo scoring).
    pub fn prompt_logprobs(&self) -> Option<u8> {
        match self {
            InferenceRequest::Blocking {
                prompt_logprobs, ..
            } => *prompt_logprobs,
            InferenceRequest::Streaming {
                prompt_logprobs, ..
            } => *prompt_logprobs,
        }
    }

    /// Request timeout deadline. None = no timeout.
    pub fn timeout_at(&self) -> Option<std::time::Instant> {
        match self {
            InferenceRequest::Blocking { timeout_at, .. } => *timeout_at,
            InferenceRequest::Streaming { timeout_at, .. } => *timeout_at,
        }
    }
}

/// Tokenize stop sequence strings into single-token stop IDs.
/// Multi-token stop sequences are logged but excluded (require string matching).
pub(crate) fn tokenize_stop_sequences(
    tokenizer: &crate::tokenizer::ChatTokenizer,
    stops: &[String],
) -> Vec<u32> {
    let mut tokens = Vec::new();
    for s in stops {
        match tokenizer.encode(s) {
            Ok(ids) if ids.len() == 1 => tokens.push(ids[0]),
            Ok(ids) if ids.len() > 1 => {
                tracing::info!(
                    "Multi-token stop '{}' ({} tokens) — use string matching",
                    s,
                    ids.len()
                );
            }
            Ok(_) => {} // Empty encoding
            Err(e) => tracing::warn!("Failed to tokenize stop '{}': {e}", s),
        }
    }
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

/// Strip any matching stop sequence from the end of the output text.
/// Per OpenAI spec, returned text must not contain the stop sequence.
pub(crate) fn strip_stop_sequences(mut text: String, stops: &[String]) -> String {
    // Try longest-first so overlapping prefixes (`["</answer", "</answer>"]`)
    // don't truncate at the shorter (wrong) match boundary. strip_suffix
    // is end-anchored, so usually only one of the two end-matches at a
    // time, but defensive ordering handles the cases where it doesn't.
    let mut sorted: Vec<&String> = stops.iter().collect();
    sorted.sort_by_key(|s| std::cmp::Reverse(s.len()));
    for s in sorted {
        if let Some(stripped) = text.strip_suffix(s.as_str()) {
            text.truncate(stripped.len());
            break;
        }
    }
    text
}

/// Strip `<think>...</think>` reasoning content from model output.
///
/// Qwen3.5 models generate internal reasoning between `<think>` and `</think>` tags.
/// This must be removed from the API response so that:
/// 1. Clients don't see internal reasoning in the content field
/// 2. Multi-turn conversations aren't corrupted when clients echo assistant content back
///
/// Returns only the text after the final `</think>` tag (the actual response),
/// trimmed of leading whitespace.
/// Extract `<think>...</think>` reasoning from model output.
///
/// Returns `(reasoning_content, response_content)`.
/// - `enable_thinking=true`: reasoning extracted into first element, response in second.
/// - `enable_thinking=false`: reasoning discarded (None), only response returned.
pub(crate) fn extract_thinking(
    text: &str,
    enable_thinking: bool,
    parser: Option<&dyn crate::reasoning_parser::ReasoningParser>,
) -> (Option<String>, String) {
    if let Some(p) = parser {
        p.extract_thinking(text, enable_thinking)
    } else {
        (None, text.to_string())
    }
}
