// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

//! `/v1/chat/completions` orchestrator.
//!
//! Wave-4g extraction (2026-05-03): the original 1121-LoC `chat.rs`
//! held one async fn (`chat_completions_inner`) where every phase
//! shared a function-local `MsgEntry` struct + ~25 carry-through
//! locals. This module now coordinates:
//!
//! - `msg_entry`      — `MsgEntry` + `build_msg_entries` (req →
//!                      tokenisable shape, image preprocessing,
//!                      cwd extraction)
//! - `loop_detect`    — generic loop / spinning detection +
//!                      task-pin re-anchor
//! - `thinking`       — `(enable_thinking, thinking_budget)`
//!                      resolution
//! - `template`       — JSON-message build, auto-compact,
//!                      Jinja apply, image-pad expand,
//!                      template-forced-thinking detection
//! - `sampling_setup` — preset / penalty / stop-token / grammar /
//!                      timeout / logprobs resolution

pub(crate) mod echo;
mod loop_detect;
mod msg_entry;
pub(crate) mod prepare;
mod sampling_setup;
mod template;
mod thinking;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;

use crate::AppState;

pub(crate) use echo::ResponseEcho;

/// Result of the shared chat pipeline. A non-streaming success carries
/// the canonical response IR for the caller's surface encoder;
/// streaming SSE and error envelopes are already-complete HTTP
/// responses (streaming moves onto the delta IR next).
pub(crate) enum ChatOutcome {
    Blocking(Box<crate::ir::ChatResponse>),
    /// Streaming success: the neutral delta stream; each surface runs
    /// its own SSE encoder over it.
    Streaming(crate::ir::DeltaStream),
    Http(Response),
}

/// Test-only accessors: cross-module tests (the Anthropic adapter's
/// rendered-prompt golden) drive the IR → MsgEntry → template-JSON
/// path without an AppState.
#[cfg(test)]
#[allow(clippy::result_large_err)]
pub(crate) fn test_build_msg_entries(
    input: &[crate::ir::Message],
    tools_active: bool,
) -> Result<Vec<msg_entry::MsgEntry>, axum::response::Response> {
    msg_entry::build_msg_entries(None, None, input, tools_active).map(|o| o.messages)
}

#[cfg(test)]
pub(crate) fn test_build_json_messages(entries: &[msg_entry::MsgEntry]) -> Vec<serde_json::Value> {
    template::build_json_messages(entries)
}

use super::compact::openai_error_response;

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    body: axum::body::Bytes,
) -> Response {
    // Parse the body ourselves (instead of using axum's `Json`
    // extractor) so the same bytes can feed both the deserialized
    // handler path and the `--dump` raw-capture path without
    // cloning the struct or cascading `Serialize` through every
    // request type.
    let req: crate::openai::ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };

    // --dump: record the incoming request body verbatim.
    let dump_seq = state.dump_writer.as_ref().and_then(|d| {
        match serde_json::from_slice::<serde_json::Value>(&body) {
            Ok(v) => {
                let seq = d.next_seq();
                d.dump_request("/v1/chat/completions", seq, &v);
                Some(seq)
            }
            Err(_) => None,
        }
    });

    // Wire → IR at the edge: echo-only fields peel off beside the
    // envelope; everything downstream reads only the IR.
    let echo = ResponseEcho::from(&req);
    match chat_completions_inner(state.clone(), req_ctx, req.into(), dump_seq).await {
        ChatOutcome::Blocking(ir) => {
            crate::openai::encode_chat_response(&state, *ir, &echo, dump_seq)
        }
        ChatOutcome::Streaming(deltas) => {
            crate::openai::encode_sse_response(deltas, state.model_name.clone(), echo.include_usage)
        }
        ChatOutcome::Http(r) => r,
    }
}

/// Internal entry for the IR-request path. Called by
/// [`chat_completions`] after body capture and wire→IR lowering, and
/// by the Responses / Anthropic adapters (which lower their own wire
/// formats and skip HTTP body bytes). `dump_seq` is `Some` only on
/// the public handler path.
pub(crate) async fn chat_completions_inner(
    state: Arc<AppState>,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    mut req: crate::ir::ChatRequest,
    dump_seq: Option<u64>,
) -> ChatOutcome {
    crate::metrics::REQUESTS_TOTAL.inc();
    crate::metrics::REQUESTS_ACTIVE.inc();

    // ── Input validation + cross-turn F-feature guards ──
    if let Err(resp) = super::chat_phases::validate_input(&req) {
        // Balance the REQUESTS_ACTIVE gauge incremented above: every other
        // terminal path decrements it, but this fail-fast 400 returns before
        // reaching a dispatch handler.
        crate::metrics::REQUESTS_ACTIVE.dec();
        return ChatOutcome::Http(resp);
    }

    // Tool-parser behavioral system prompt REMOVED again (2026-05-25 PM).
    //
    // Re-injecting the qwen3_coder `system_prompt` (with its
    // `<parameter=content>[package]\nname = "x"</parameter>` example
    // and `For 'Write'/'Edit' tools specifically: ...` guidance) was a
    // mid-day attempt to give the model better multi-line content
    // hints. Live opencode v39 session showed the opposite effect:
    // the model emitted LITERAL `<tool_call><bash><command>` XML as
    // CONTENT (with HTML-entity escaping like `&amp;`) because TWO
    // tool-format guidances were competing — the chat template's
    // `tools` argument AND my injected prompt — combined with PR 73's
    // `qwen3_xml` parser. The model got confused which format to use
    // and emitted free-form XML that the parser couldn't recognise.
    //
    // Per user's recall: the "MUCH better" state had `thinking_in_tools=true`
    // and the chat template alone (no injection). Reverting matches
    // that state. PR 73's qwen3_xml + native FP8 SSM + streaming
    // byte-exact + gate-BF16 + thinking_in_tools=true is the live
    // combination.

    // M2 per-request LoRA routing: resolve the optional `adapter` name to a
    // pool slot ONCE here (both dispatch paths inherit it). Unset defers to the
    // installed active adapter (`-1`, byte-identical to today); an unknown name
    // is a hard 400, a STAGEABLE name triggers the #27 on-miss RDMA promotion.
    let adapter_slot = match super::lora_control::resolve_request_adapter_slot(
        &state,
        req.adapter.as_deref(),
        &req.model,
    )
    .await
    {
        Ok(slot) => slot,
        Err(resp) => return ChatOutcome::Http(resp),
    };

    // Resolve optional per-request source/target language token NAMES to token
    // ids via the server tokenizer. Absent = deployment default (0); an unknown
    // token is a hard 400 (mirrors the adapter-name resolution convention).
    let resolve_lang = |name: &Option<String>| -> Result<u32, Response> {
        match name {
            None => Ok(0),
            Some(s) => state.tokenizer.inner().token_to_id(s).ok_or_else(|| {
                openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("unknown language token '{s}'"),
                )
            }),
        }
    };
    let src_lang_id = match resolve_lang(&req.src_lang) {
        Ok(v) => v,
        Err(resp) => return ChatOutcome::Http(resp),
    };
    let tgt_lang_id = match resolve_lang(&req.tgt_lang) {
        Ok(v) => v,
        Err(resp) => return ChatOutcome::Http(resp),
    };

    // NLLB beam search params (mirrors src/tgt lang resolution). Streaming +
    // beam is unsupported (the beam path emits a single completed hypothesis,
    // not an incremental token stream) — reject up front like n>1.
    let num_beams = req.num_beams.unwrap_or(1);
    let length_penalty = req.length_penalty.unwrap_or(1.0);
    let early_stopping = req.early_stopping.unwrap_or(false);
    if num_beams > 1 && req.stream {
        return ChatOutcome::Http(openai_error_response(
            StatusCode::BAD_REQUEST,
            "num_beams > 1 is not supported in streaming mode".to_string(),
        ));
    }

    // ── Phases 1-5 (prompt-affecting): shared with count_tokens ──
    let prepare::PreparedChat {
        tools_active,
        cwd_hint,
        image_pixels,
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    } = match prepare::prepare_chat_prompt(&state, &mut req) {
        Ok(p) => p,
        Err(resp) => return ChatOutcome::Http(resp),
    };

    // ── Phase 4: generic loop / spinning detection + task pin ───
    let loop_detect::LoopDetectOut {
        suppress_tool_call,
        tool_call_repeat_count,
    } = loop_detect::check_loops(&req.messages, tools_active);

    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let tools_count = req.tools.len();
    tracing::info!(
        "Session {session_hash:#x}: {prompt_tokens} prompt tokens, tools={tools_active} ({tools_count} defined)",
        prompt_tokens = prompt_tokens.len()
    );
    let prompt_len = prompt_tokens.len();
    if prompt_len >= state.max_seq_len {
        return ChatOutcome::Http(openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "Prompt too long: {prompt_len} tokens exceeds max_seq_len {} (leave room for output tokens)",
                state.max_seq_len
            ),
        ));
    }

    // ── Phase 6: sampling preset / stop / grammar / timeout ─────
    let sampling_setup::SamplingSetup {
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
        max_tokens,
        stop_tokens,
        tool_choice_required,
        grammar_spec,
        timeout_at,
        top_logprobs,
    } = match sampling_setup::build_sampling(
        &state,
        &req,
        enable_thinking,
        tools_active,
        suppress_tool_call,
        tool_call_repeat_count,
    ) {
        Ok(s) => s,
        Err(resp) => return ChatOutcome::Http(resp),
    };

    // ── Phase 7: dispatch streaming or blocking ─────────────────
    if req.stream {
        return super::chat_stream_dispatch::dispatch_streaming(
            state,
            &req,
            req_ctx,
            dump_seq,
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
            logit_bias.clone(),
            enable_thinking,
            thinking_budget,
            tools_active,
            tool_choice_required,
            suppress_tool_call,
            cwd_hint.clone(),
            stop_tokens,
            grammar_spec.clone(),
            top_logprobs,
            timeout_at,
        )
        .await;
    }

    super::chat_blocking::run_blocking_path(super::chat_blocking::BlockingPathArgs {
        state,
        req,
        req_ctx,
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
    })
    .await
}
