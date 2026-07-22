// SPDX-License-Identifier: AGPL-3.0-only

//! Streaming-path dispatch from `chat_completions_inner` to
//! `chat_completions_stream`. Extracted (refactor wave-4e) to reduce
//! the chat.rs LoC footprint.

#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use axum::http::StatusCode;

use crate::AppState;
use crate::ir::ChatRequest;
use crate::tool_parser;

use super::chat_stream::run_chat_stream;
use super::compact::openai_error_response;
use super::inference_types::GrammarSpec;

pub(super) async fn dispatch_streaming(
    state: Arc<AppState>,
    req: &ChatRequest,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    dump_seq: Option<u64>,
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
) -> super::chat::ChatOutcome {
    if req.n > 1 {
        crate::metrics::REQUESTS_ACTIVE.dec();
        return super::chat::ChatOutcome::Http(openai_error_response(
            StatusCode::BAD_REQUEST,
            "n > 1 is not supported in streaming mode".to_string(),
        ));
    }
    let tool_defs: Vec<tool_parser::ToolDefinition> = req.tools.clone();
    // Sort by length descending so the streaming stop-string scan
    // matches the longest overlapping prefix first (e.g. when the
    // caller provides ["</answer", "</answer>"], `find` would
    // otherwise truncate at the shorter, wrong boundary).
    let mut stop_strings = req.stop.clone();
    stop_strings.sort_by_key(|s| std::cmp::Reverse(s.len()));
    let req_return_token_ids = req.return_token_ids;
    let ctx_for_stream = req_ctx.as_ref().map(|e| e.0.clone());
    let repetition_detection = req.repetition_detection;
    match run_chat_stream(
        state,
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
        repetition_detection,
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
        req_return_token_ids,
        ctx_for_stream,
        dump_seq,
    )
    .await
    {
        Ok(deltas) => super::chat::ChatOutcome::Streaming(deltas),
        Err((status, msg)) => super::chat::ChatOutcome::Http(openai_error_response(status, msg)),
    }
}
