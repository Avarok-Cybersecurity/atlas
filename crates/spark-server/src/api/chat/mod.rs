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

mod loop_detect;
mod msg_entry;
pub(super) mod repair_json;
mod sampling_setup;
mod template;
mod thinking;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;

use crate::AppState;
use crate::openai::ChatCompletionRequest;

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
    let req: ChatCompletionRequest = match serde_json::from_slice(&body) {
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

    chat_completions_inner(state, req_ctx, req, dump_seq).await
}

/// Internal entry for the parsed-request path. Called by
/// [`chat_completions`] after body capture, and by the Responses
/// API adapter (which builds a `ChatCompletionRequest` in-memory
/// and skips HTTP body bytes). `dump_seq` is `Some` only on the
/// public handler path.
pub(crate) async fn chat_completions_inner(
    state: Arc<AppState>,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    mut req: ChatCompletionRequest,
    dump_seq: Option<u64>,
) -> Response {
    crate::metrics::REQUESTS_TOTAL.inc();
    crate::metrics::REQUESTS_ACTIVE.inc();

    // ── Input validation + cross-turn F-feature guards ──
    if let Err(resp) = super::chat_phases::validate_input(&req) {
        return resp;
    }
    let f23_metrics = super::chat_phases::apply_failure_guards(&mut req);
    let _ = f23_metrics; // kept available for downstream consumers

    // Tool-active gating.
    let tools_active = state.tool_call_parser.is_some()
        && req.tools.as_ref().is_some_and(|t| !t.is_empty())
        && !req.tool_choice.as_ref().is_some_and(|tc| tc.is_none());

    // Inject parser-specific behavioral system prompt when tools are active.
    // Each ToolCallParser defines guardrails (e.g. "emit <tool_call> immediately,
    // do not narrate") that the Jinja chat template alone does not enforce.
    if tools_active && let Some(ref parser) = state.tool_call_parser {
        let default_choice = crate::tool_parser::ToolChoice::Mode("auto".to_string());
        let tool_choice = req.tool_choice.as_ref().unwrap_or(&default_choice);
        let tool_prompt = parser.system_prompt(req.tools.as_deref().unwrap_or(&[]), tool_choice);
        if let Some(first) = req.messages.first_mut().filter(|m| m.role == "system") {
            first.content.text = format!("{}\n\n{}", tool_prompt, first.content.text);
        } else {
            req.messages.insert(
                0,
                crate::openai::IncomingMessage::synthetic_system(tool_prompt),
            );
        }
    }

    tracing::info!(
        "Request: model={}, messages={}, tools={}, tools_active={}, tool_choice={:?}, stream={}, temp={:?}, max_tokens={}, freq_pen={:?}, rep_pen={:?}",
        req.model,
        req.messages.len(),
        req.tools.as_ref().map_or(0, |t| t.len()),
        tools_active,
        req.tool_choice,
        req.stream,
        req.temperature,
        req.max_tokens,
        req.frequency_penalty,
        req.repetition_penalty,
    );

    // ── Phase 1: build MsgEntry vec + image preprocess + cwd ────
    let msg_entry::BuildOut {
        mut messages,
        cwd_hint,
        image_pixels,
        image_pad_counts,
        consecutive_tool_errors,
    } = match msg_entry::build_msg_entries(&state, &req, tools_active) {
        Ok(o) => o,
        Err(resp) => return resp,
    };

    // ── Phase 1.5: merge server-level chat_template_kwargs default ─
    // When the client sends no thinking parameters and the server has a
    // --default-chat-template-kwargs flag set, inject those kwargs into
    // the request so the existing resolve_thinking() chain sees them as
    // normal request-body fields. We don't mutate the resolution logic —
    // we just pre-populate the field it already checks.
    if let Some(ref default_kw) = state.default_chat_template_kwargs
        && !req.thinking_explicitly_requested()
    {
        req.chat_template_kwargs = Some(default_kw.clone());
    }

    // ── Phase 2: thinking resolution (pre-template) ─────────────
    let (enable_thinking, thinking_budget) = thinking::resolve_thinking(&state, &req, tools_active);

    // ── Phase 3: stale-failure observation masking ──────────────
    {
        let bodies: Vec<(&str, &str)> = messages
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_str()))
            .collect();
        let mask = crate::observation_mask::compute_masking(&bodies, 2);
        let mut masked_count = 0usize;
        for (i, replacement) in mask.into_iter().enumerate() {
            if let Some(new_body) = replacement {
                messages[i].content = new_body;
                masked_count += 1;
            }
        }
        if masked_count > 0 {
            tracing::info!(
                masked_count,
                "observation_mask: elided {masked_count} stale tool-failure bodies"
            );
            crate::metrics::OBSERVATION_MASK_ELIDED_BODIES.inc_by(masked_count as u64);
        }
    }

    // ── Phase 4: generic loop / spinning detection + task pin ───
    let loop_detect::LoopDetectOut {
        suppress_tool_call,
        tool_call_repeat_count,
    } = loop_detect::check_loops(&req, &mut messages, consecutive_tool_errors, tools_active);

    // ── Phase 5: render Jinja template + image-pad expansion ────
    let template::TemplateOut {
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    } = match template::render_template(
        &state,
        &req,
        &messages,
        &image_pad_counts,
        enable_thinking,
        thinking_budget,
        tools_active,
    ) {
        Ok(o) => o,
        Err(resp) => return resp,
    };

    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let tools_count = req.tools.as_ref().map_or(0, |t| t.len());
    tracing::info!(
        "Session {session_hash:#x}: {prompt_tokens} prompt tokens, tools={tools_active} ({tools_count} defined)",
        prompt_tokens = prompt_tokens.len()
    );
    let prompt_len = prompt_tokens.len();
    if prompt_len >= state.max_seq_len {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "Prompt too long: {prompt_len} tokens exceeds max_seq_len {} (leave room for output tokens)",
                state.max_seq_len
            ),
        );
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
        Err(resp) => return resp,
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
    })
    .await
}
