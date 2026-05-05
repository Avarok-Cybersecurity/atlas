// SPDX-License-Identifier: AGPL-3.0-only

//! Streaming-path dispatch from `chat_completions_inner` to
//! `chat_completions_stream`. Extracted (refactor wave-4e) to reduce
//! the chat.rs LoC footprint.

#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::Response;

use crate::AppState;
use crate::openai::ChatCompletionRequest;
use crate::tool_parser;

use super::chat_stream::chat_completions_stream;
use super::compact::openai_error_response;
use super::failures::f39_build_failure_cache;
use super::inference_types::GrammarSpec;

pub(super) async fn dispatch_streaming(
    state: Arc<AppState>,
    req: &ChatCompletionRequest,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    dump_seq: Option<u64>,
    prompt_tokens: Vec<u32>,
    session_hash: u64,
    image_pixels: Vec<(Vec<f32>, usize, usize)>,
    max_tokens: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    top_n_sigma: f32,
    min_p: f32,
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: u32,
    lz_penalty: f32,
    logit_bias: Vec<(u32, f32)>,
    enable_thinking: bool,
    thinking_budget: Option<u32>,
    tools_active: bool,
    tool_choice_required: bool,
    suppress_tool_call: bool,
    cwd_hint: Option<String>,
    stop_tokens: Vec<u32>,
    grammar_spec: Option<GrammarSpec>,
    top_logprobs: Option<u8>,
    timeout_at: Option<std::time::Instant>,
) -> Response {
    if req.n > 1 {
        crate::metrics::REQUESTS_ACTIVE.dec();
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "n > 1 is not supported in streaming mode".to_string(),
        );
    }
    let tool_defs: Vec<tool_parser::ToolDefinition> = req.tools.clone().unwrap_or_default();
    // Sort by length descending so the streaming stop-string scan
    // matches the longest overlapping prefix first (e.g. when the
    // caller provides ["</answer", "</answer>"], `find` would
    // otherwise truncate at the shorter, wrong boundary).
    let mut stop_strings = req.stop.clone();
    stop_strings.sort_by_key(|s| std::cmp::Reverse(s.len()));
    let stream_include_usage = req.stream_options.map(|o| o.include_usage).unwrap_or(false);
    let req_service_tier = req.service_tier.clone();
    let req_metadata = req.metadata.clone();
    let ctx_for_stream = req_ctx.as_ref().map(|e| e.0.clone());
    // F44 (2026-04-27): build the cross-turn permanent-failure cache
    // here (where `req.messages` is in scope) and pass it into the
    // streaming function for the ToolCallEnd hook.
    let f44_cache = f39_build_failure_cache(&req.messages);
    match chat_completions_stream(
        state,
        prompt_tokens,
        session_hash,
        image_pixels,
        max_tokens,
        req.min_tokens,
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
        enable_thinking,
        thinking_budget,
        tools_active,
        tool_choice_required,
        suppress_tool_call,
        tool_defs,
        cwd_hint,
        stop_tokens,
        grammar_spec,
        req.seed,
        top_logprobs,
        timeout_at,
        stop_strings,
        stream_include_usage,
        req_service_tier,
        req_metadata,
        ctx_for_stream,
        dump_seq,
        f44_cache,
    )
    .await
    {
        Ok(r) => r,
        Err((status, msg)) => openai_error_response(status, msg),
    }
}
