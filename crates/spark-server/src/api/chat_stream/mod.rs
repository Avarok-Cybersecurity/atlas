// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

//! Streaming `/v1/chat/completions` SSE handler.
//!
//! Wave-4g extraction (2026-05-03): the original 1484-LoC
//! `chat_stream.rs` was a single async fn whose body terminated in
//! one `flat_map(move |event| { ... })` closure with three deeply
//! coupled `StreamEvent` arms (`Token | TokenWithLogprobs`, `Done`,
//! `Error`) and ~24 captured mutable locals plus ~15 read-only
//! captures.
//!
//! Sub-files:
//! - `state`        — `StreamState`: every captured-mutable local
//! - `ctx`          — `StreamCtx`: every captured-read-only value
//! - `handle_token` — Token / TokenWithLogprobs arm + tool-call
//!                    helpers shared with the Done arm's flush
//! - `handle_done`  — Done arm (flush, salvage, usage, dump, metrics)
//! - `handle_error` — Error arm

mod ctx;
mod handle_done;
mod handle_error;
mod handle_token;
mod state;
mod token_ids;
mod tool_handlers;

use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::openai::ChatCompletionChunk;
use crate::tool_parser;

use super::failures::{F39FailureCache, f60_disable_mtp_for_request};
use super::inference_types::{GrammarSpec, InferenceRequest, StreamEvent};

use ctx::StreamCtx;
use state::StreamState;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn chat_completions_stream(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    session_hash: u64,
    image_pixels: Vec<(Vec<f32>, usize, usize)>,
    max_tokens: usize,
    min_tokens: usize,
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
    tool_defs: Vec<tool_parser::ToolDefinition>,
    cwd_hint: Option<String>,
    stop_tokens: Vec<u32>,
    grammar_spec: Option<GrammarSpec>,
    seed: Option<u64>,
    top_logprobs: Option<u8>,
    timeout_at: Option<std::time::Instant>,
    stop_strings: Vec<String>,
    req_stream_include_usage: bool,
    req_return_token_ids: bool,
    req_service_tier: Option<String>,
    req_metadata: Option<std::collections::HashMap<String, String>>,
    req_ctx: Option<crate::rate_limiter::RequestContext>,
    dump_seq: Option<u64>,
    f44_cache: F39FailureCache,
) -> Result<Response, (StatusCode, String)> {
    // service_tier + metadata are request echoes only; the chat-completion-
    // chunk schema doesn't carry them, but we surface them via the final
    // usage block when `include_usage=true`. Accept and retain for future
    // wiring — see gap #18/#19 in the OpenAI compat plan.
    let _ = (&req_service_tier, &req_metadata);
    // Channel capacity sized for ~30s of decode at 50 tok/s. The previous
    // 64-slot buffer would fill in <2s under any HTTP-flush stall and silently
    // drop the seq via emit_step.rs's `try_send().is_err()` (now fixed to
    // discriminate Full from Closed). Larger capacity = fewer Full→blocking_send
    // round-trips in the steady state.
    let (token_tx, token_rx) = tokio::sync::mpsc::channel::<StreamEvent>(1024);
    let prompt_len = prompt_tokens.len();

    // Scheduler tracks thinking only when the template actually opens it.
    // When enable_thinking=false, the template inserts closed
    // `<think>\n\n</think>\n\n` and the model generates no thinking tokens —
    // no need for scheduler tracking.
    let scheduler_thinking = enable_thinking;
    let request = InferenceRequest::Streaming {
        prompt_tokens,
        session_hash,
        image_pixels,
        max_tokens,
        min_tokens,
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
        enable_thinking: scheduler_thinking,
        thinking_budget,
        require_tool_call: tool_choice_required,
        suppress_tool_call,
        disable_mtp: f60_disable_mtp_for_request(tools_active),
        grammar_spec,
        seed,
        top_logprobs,
        timeout_at,
        token_tx,
    };

    state.request_tx.send(request).await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Scheduler queue full".to_string(),
        )
    })?;

    let chunk_id = crate::openai::new_chunk_id();
    let model_name = state.model_name.clone();

    // First chunk: role announcement
    let role_chunk = ChatCompletionChunk::role_chunk(&model_name, &chunk_id);
    let role_json = serde_json::to_string(&role_chunk).unwrap_or_default();

    // Resolve the active parser's leak-marker vocabulary once, at request
    // setup. `'static` slices so we borrow by reference throughout the
    // stream without cloning. If no parser is active, the sanitizer runs
    // in pass-through mode via the fast-path in `sanitize_content_chunk`.
    let leak_markers: tool_parser::LeakMarkers = state
        .tool_call_parser
        .as_ref()
        .map(|p| p.leak_markers())
        .unwrap_or(tool_parser::LeakMarkers::EMPTY);

    let max_tool_calls_per_response: usize = std::env::var("ATLAS_MAX_TOOL_CALLS_PER_RESPONSE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    // F44/F55: log cache state at stream entry.
    let f44_cache_active =
        !f44_cache.direct.is_empty() || !f44_cache.missing_bins_by_tool.is_empty();
    if f44_cache_active {
        tracing::info!(
            direct_entries = f44_cache.direct.len(),
            missing_bin_tools = f44_cache.missing_bins_by_tool.len(),
            sample_direct = ?f44_cache.direct.keys().take(3).collect::<Vec<_>>(),
            "F44/F55: streaming closure entered with non-empty failure cache"
        );
    }

    let ctx = StreamCtx {
        state: state.clone(),
        model: model_name.clone(),
        id: chunk_id.clone(),
        prompt_len,
        enable_thinking,
        tool_defs_for_backfill: tool_defs,
        cwd_for_normalize: cwd_hint,
        stop_strings,
        leak_markers,
        max_tool_calls_per_response,
        req_stream_include_usage,
        req_return_token_ids,
        req_ctx,
        dump_seq,
        f44_cache,
        f44_cache_active,
    };

    let mut stream_state = StreamState::new(tools_active, enable_thinking);

    let token_stream = ReceiverStream::new(token_rx).flat_map(move |event| {
        let events = match event {
            StreamEvent::Token(tok) | StreamEvent::TokenWithLogprobs(tok, _) => {
                handle_token::handle_token(&mut stream_state, &ctx, tok)
            }
            StreamEvent::Done {
                finish_reason,
                prompt_tokens: _,
                completion_tokens,
                time_to_first_token_ms,
                decode_time_ms,
                reasoning_tokens,
                cached_prompt_tokens,
            } => handle_done::handle_done(
                &mut stream_state,
                &ctx,
                finish_reason,
                completion_tokens,
                time_to_first_token_ms,
                decode_time_ms,
                reasoning_tokens,
                cached_prompt_tokens,
            ),
            StreamEvent::Error(msg) => handle_error::handle_error(&ctx, msg),
        };
        futures::stream::iter(events)
    });

    // Prepend role chunk, append [DONE] sentinel
    let role_event = futures::stream::once(async move { Ok(Event::default().data(role_json)) });
    let done_event = futures::stream::once(async {
        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
    });
    let full_stream = role_event.chain(token_stream).chain(done_event);

    Ok(Sse::new(full_stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}
