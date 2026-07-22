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
mod strip;
mod token_ids;
mod tool_handlers;

use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::tool_parser;

use super::inference_types::{GrammarSpec, InferenceRequest, StreamEvent};

use ctx::StreamCtx;
use state::StreamState;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_chat_stream(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    session_hash: u64,
    // M2 per-request LoRA routing: resolved adapter slot (-1 = defer to active).
    adapter_slot: i32,
    // Resolved source-language token id (0 = deployment default).
    src_lang_id: u32,
    // Resolved target-language token id (0 = deployment default).
    tgt_lang_id: u32,
    // NLLB beam search params (1/1.0/false = greedy). Streaming + beam is
    // rejected in the handler; threaded here only to set the Streaming defaults.
    num_beams: u32,
    length_penalty: f32,
    early_stopping: bool,
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
    repetition_detection: Option<crate::api::inference_types::RepetitionDetectionParams>,
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
    req_return_token_ids: bool,
    req_ctx: Option<crate::rate_limiter::RequestContext>,
    dump_seq: Option<u64>,
) -> Result<crate::ir::DeltaStream, (StatusCode, String)> {
    // Channel capacity sized for ~30s of decode at 50 tok/s. The previous
    // 64-slot buffer would fill in <2s under any HTTP-flush stall and silently
    // drop the seq via emit_step.rs's `try_send().is_err()` (now fixed to
    // discriminate Full from Closed). Larger capacity = fewer Full→blocking_send
    // round-trips in the steady state.
    let (token_tx, token_rx) = tokio::sync::mpsc::channel::<StreamEvent>(1024);
    let prompt_len = prompt_tokens.len();
    // Cooperative cancellation flag shared with the scheduler. Flipped
    // by stream-side loop guards (Bug-2 name-run cap, F11/F44 dedup,
    // loop-watchdog) so the scheduler stops generating instead of just
    // having its output suppressed — without it a degenerate-loop
    // response keeps generating until max_tokens (or hangs on a
    // channel-full blocking_send) while the client waits for `[DONE]`.
    let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Scheduler tracks thinking only when the template actually opens it.
    // When enable_thinking=false, the template inserts closed
    // `<think>\n\n</think>\n\n` and the model generates no thinking tokens —
    // no need for scheduler tracking.
    let scheduler_thinking = enable_thinking;
    // Wrap prompt tokens in Arc ONCE — the scheduler request, the
    // streaming context, and the Tier 5c retry path all share the
    // same Arc. No deep clones of the ~40 KB Vec<u32>.
    let prompt_tokens = std::sync::Arc::new(prompt_tokens);
    let prompt_tokens_for_retry = prompt_tokens.clone();
    let grammar_spec_for_retry = grammar_spec.clone();
    let request = InferenceRequest::Streaming {
        prompt_tokens,
        session_hash,
        adapter_slot,
        src_lang_id,
        tgt_lang_id,
        num_beams,
        length_penalty,
        early_stopping,
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
        repetition_detection,
        require_tool_call: tool_choice_required,
        tools_present: tools_active,
        suppress_tool_call,
        disable_mtp: false,
        grammar_spec,
        seed,
        top_logprobs,
        prompt_logprobs: None,
        echo: false,
        timeout_at,
        token_tx,
        cancel_flag: cancel_flag.clone(),
    };

    state.request_tx.send(request).await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Scheduler queue full".to_string(),
        )
    })?;

    // `ctx.id`/`ctx.model` feed the --dump synthesis; each surface
    // encoder mints its own wire ids.
    let chunk_id = format!("chatcmpl-{}", crate::ids::uuid_v4());
    let model_name = state.model_name.clone();

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

    // Cache the hold-back window length once. vLLM's
    // `IncrementalDetokenizer.update` uses
    // `max(len(s) for s in stop_strings) - 1` so a stop string that
    // straddles two decoded chunks cannot leak its prefix to the
    // client before the suffix arrives. Zero when no stop strings are
    // configured (preserves the existing pass-through behaviour for
    // requests without `stop`).
    let stop_string_buffer_len: usize = stop_strings
        .iter()
        .map(|s| s.len())
        .max()
        .map(|m| m.saturating_sub(1))
        .unwrap_or(0);

    let prompt_vocab: Arc<std::collections::HashSet<String>> =
        Arc::new(std::collections::HashSet::new());

    let ctx = StreamCtx {
        state: state.clone(),
        model: model_name.clone(),
        id: chunk_id.clone(),
        prompt_len,
        enable_thinking,
        tool_defs_for_backfill: tool_defs,
        cwd_for_normalize: cwd_hint,
        stop_strings,
        stop_string_buffer_len,
        leak_markers,
        wants_typed_arguments: state
            .tool_call_parser
            .as_ref()
            .is_some_and(|p| p.wants_typed_arguments()),
        max_tool_calls_per_response,
        req_return_token_ids,
        req_ctx,
        dump_seq,
        tool_retry_enabled: false,
        prompt_tokens: prompt_tokens_for_retry,
        prompt_vocab,
        grammar_spec: grammar_spec_for_retry,
        max_tokens,
        timeout_at,
    };

    let mut stream_state = StreamState::new(
        tools_active,
        enable_thinking,
        cancel_flag.clone(),
        ctx.tool_defs_for_backfill.clone(),
    );

    let token_stream = ReceiverStream::new(token_rx).flat_map(move |event| {
        use futures::StreamExt;
        // The handlers emit provider-neutral deltas (`ir::StreamDelta`);
        // they never touch the wire format. Encoding happens in the
        // caller's surface encoder.
        let deltas = match event {
            StreamEvent::Token(tok) | StreamEvent::TokenWithLogprobs(tok, _) => {
                handle_token::handle_token(&mut stream_state, &ctx, tok)
            }
            // Legacy /v1/completions-only event; chat never requests
            // prompt_logprobs, so nothing to emit here.
            StreamEvent::PromptLogprobs(_) => Vec::new(),
            StreamEvent::Done {
                finish_reason,
                prompt_tokens: _,
                completion_tokens,
                time_to_first_token_ms,
                decode_time_ms,
                reasoning_tokens,
                cached_prompt_tokens,
                guard_stop,
            } => {
                stream_state.guard_stop = guard_stop;
                handle_done::handle_done(
                    &mut stream_state,
                    &ctx,
                    finish_reason,
                    completion_tokens,
                    time_to_first_token_ms,
                    decode_time_ms,
                    reasoning_tokens,
                    cached_prompt_tokens,
                )
            }
            StreamEvent::Error(msg) => handle_error::handle_error(&ctx, msg),
        };

        futures::stream::iter(deltas).boxed()
    });

    Ok(Box::pin(token_stream))
}
