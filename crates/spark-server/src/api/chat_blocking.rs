// SPDX-License-Identifier: AGPL-3.0-only

//! Blocking (non-streaming) `/v1/chat/completions` path. Extracted from
//! `chat_completions_inner` (refactor wave-4e) to keep `chat.rs` under
//! the 500 LoC cap. Supports `n >= 1` (multiple choices per request) by
//! looping the scheduler send + decode + tool-parse pipeline once per
//! choice index.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use crate::AppState;
use crate::openai::{ChatCompletionRequest, ChatCompletionResponse, Usage};
use crate::tool_parser;

use super::compact::openai_error_response;
use super::inference_impl::strip_stop_sequences;
use super::inference_types::{GrammarSpec, InferenceRequest};

// Per-choice decoding / message-building and final-response assembly were
// moved to the `choice` / `finalize` submodules to keep this file ≤500 LoC.
mod choice;
mod finalize;

use choice::{build_choice_message, decode_response_text};
use finalize::{build_logprobs, finalize_response};

pub(super) struct BlockingPathArgs {
    pub state: Arc<AppState>,
    pub req: ChatCompletionRequest,
    pub req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    pub dump_seq: Option<u64>,
    pub prompt_tokens: Vec<u32>,
    pub session_hash: u64,
    pub image_pixels: Vec<(Vec<f32>, usize, usize)>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub top_n_sigma: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    pub lz_penalty: f32,
    pub logit_bias: Vec<(u32, f32)>,
    pub stop_tokens: Vec<u32>,
    pub enable_thinking: bool,
    pub thinking_budget: Option<u32>,
    pub tools_active: bool,
    pub tool_choice_required: bool,
    pub suppress_tool_call: bool,
    pub grammar_spec: Option<GrammarSpec>,
    pub top_logprobs: Option<u8>,
    pub timeout_at: Option<std::time::Instant>,
    pub cwd_hint: Option<String>,
    pub prompt_len: usize,
}

pub(super) async fn run_blocking_path(args: BlockingPathArgs) -> Response {
    let BlockingPathArgs {
        state,
        req,
        req_ctx,
        dump_seq,
        prompt_tokens,
        session_hash,
        image_pixels,
        max_tokens,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        lz_penalty,
        logit_bias,
        stop_tokens,
        enable_thinking,
        thinking_budget,
        tools_active,
        tool_choice_required,
        suppress_tool_call,
        grammar_spec,
        top_logprobs,
        timeout_at,
        cwd_hint,
        prompt_len,
    } = args;

    let n = req.n.max(1);
    let mut all_choices: Vec<crate::openai::ChatChoice> = Vec::with_capacity(n);
    let mut total_completion_tokens = 0usize;
    let mut first_ttft = 0.0f64;
    let mut last_decode_time_ms = 0.0f64;
    let mut total_reasoning_tokens = 0u32;
    let mut total_cached_prompt_tokens = 0u32;

    // Arc-wrap the prompt tokens ONCE. Per-choice scheduler requests
    // and the Tier 5c retry path all share the same Arc — no Vec<u32>
    // deep clones (~40 KB on a typical long-context opencode prompt).
    let prompt_tokens = std::sync::Arc::new(prompt_tokens);

    for choice_idx in 0..n {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let request = InferenceRequest::Blocking {
            prompt_tokens: prompt_tokens.clone(),
            session_hash,
            image_pixels: if choice_idx == 0 {
                image_pixels.clone()
            } else {
                Vec::new()
            },
            max_tokens,
            min_tokens: req.min_tokens,
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            lz_penalty,
            logit_bias: logit_bias.clone(),
            stop_tokens: stop_tokens.clone(),
            enable_thinking,
            thinking_budget,
            repetition_detection: req.repetition_detection(),
            require_tool_call: tool_choice_required,
            suppress_tool_call,
            disable_mtp: false,
            grammar_spec: grammar_spec.clone(),
            seed: req.seed.map(|s| s.wrapping_add(choice_idx as u64)),
            top_logprobs,
            timeout_at,
            response_tx: tx,
        };

        if state.request_tx.send(request).await.is_err() {
            crate::metrics::REQUESTS_ACTIVE.dec();
            return openai_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "Scheduler queue full".to_string(),
            );
        }

        let response = match rx.await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                crate::metrics::REQUESTS_ACTIVE.dec();
                return openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Inference error: {e}"),
                );
            }
            Err(_) => {
                crate::metrics::REQUESTS_ACTIVE.dec();
                return openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Inference cancelled".to_string(),
                );
            }
        };

        if choice_idx == 0 {
            first_ttft = response.time_to_first_token_ms;
        }
        last_decode_time_ms = response.decode_time_ms;

        let num_completion = response.output_tokens.len();
        total_completion_tokens += num_completion;
        total_reasoning_tokens += response.reasoning_tokens;
        // cached_prompt_tokens is a per-request prefix-cache hit count; for
        // n>1 we only charge once (same prompt reused).
        total_cached_prompt_tokens = total_cached_prompt_tokens.max(response.cached_prompt_tokens);

        let (reasoning_content_i, output_text_i) =
            decode_response_text(&state, &response, enable_thinking);
        let output_text_i = strip_stop_sequences(output_text_i, &req.stop);

        let (message, finish_reason_i) = build_choice_message(
            &state,
            &req,
            &response,
            reasoning_content_i,
            output_text_i,
            tools_active,
            cwd_hint.as_deref(),
            choice_idx,
        )
        .await;

        all_choices.push(crate::openai::ChatChoice {
            index: choice_idx,
            message,
            finish_reason: finish_reason_i,
            logprobs: build_logprobs(&state, &response),
        });
    }

    finalize_response(
        state,
        req,
        req_ctx,
        dump_seq,
        all_choices,
        total_completion_tokens,
        first_ttft,
        last_decode_time_ms,
        total_reasoning_tokens,
        total_cached_prompt_tokens,
        prompt_len,
    )
}
