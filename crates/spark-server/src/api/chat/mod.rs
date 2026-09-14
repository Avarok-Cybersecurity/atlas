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
pub(crate) mod levers;
mod loop_detect;
mod msg_entry;
pub(crate) mod prepare;
pub(crate) mod remote_image;
mod sampling_setup;
mod template;
mod thinking;

use crate::main_modules::model_host::CurrentModel;
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
    msg_entry::build_msg_entries(
        None,
        None,
        &remote_image::RemoteImagePolicy::default(),
        &msg_entry::VideoDecode {
            ffmpeg: &spark_model::video_decode_ffmpeg::FfmpegPolicy {
                enabled: false,
                ..Default::default()
            },
            fps: 2.0,
        },
        input,
        tools_active,
        &levers::ChatLevers::OFF,
        false,
    )
    .map(|o| o.messages)
}

#[cfg(test)]
pub(crate) fn test_build_json_messages(entries: &[msg_entry::MsgEntry]) -> Vec<serde_json::Value> {
    template::build_json_messages(entries)
}

use super::compact::openai_error_response;

pub async fn chat_completions(
    // The HOST, not `CurrentModel`: this is the one handler that can CREATE the
    // model. Extractors run before the body is read, so `CurrentModel` would
    // hand back whatever was loaded BEFORE the request said which model it
    // wanted — exactly the thing the auto-swap has to decide on.
    axum::extract::State(host): axum::extract::State<
        std::sync::Arc<crate::main_modules::model_host::ModelHost>,
    >,
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

    // Unknown reasoning_effort spellings 400 here — the raw string does
    // not survive wire→IR lowering, and a typo must never silently buy a
    // default tier (see `chat_request::validate_reasoning_effort`).
    if let Err(msg) = req.validate_reasoning_effort() {
        return openai_error_response(StatusCode::BAD_REQUEST, msg);
    }

    // Ollama-style: a request naming a different KNOWN model loads it first.
    // Off unless `--auto-swap`, and `--no-auto-swap` overrides that; every
    // other case (absent, unknown, already live) falls through untouched, which
    // is byte-identical to the behaviour before this existed.
    if host.auto_swap_enabled() {
        let live = host.live_model().unwrap_or_default();
        // Short-circuit the overwhelmingly common case — a client naming the
        // model that is already loaded — before touching the disk. Reading the
        // recipe index means an `ArtifactStore::discover` plus a JSON parse,
        // and doing that per request put blocking I/O on a runtime worker in
        // the hot path of every completion.
        if !req.model.is_empty() && req.model != live {
            let requested = req.model.clone();
            let swap_host = host.clone();
            // Both the catalogue read and the load are blocking, so BOTH belong
            // off the runtime — the decision needs the index, and the index is
            // on disk.
            let outcome = tokio::task::spawn_blocking(move || {
                let catalogue = atlas_plugin::ArtifactStore::discover()
                    .ok()
                    .map(|s| crate::recipe::fetch::cached(s.root()).recipes)
                    .unwrap_or_default();
                match crate::main_modules::auto_swap::decide(&requested, &live, &catalogue) {
                    crate::main_modules::auto_swap::Decision::SwapTo(recipe_id) => {
                        crate::main_modules::auto_swap::ensure_loaded(
                            &swap_host, &recipe_id, &requested, &catalogue,
                        )
                    }
                    // Unknown to the catalogue, or already live: serve as-is.
                    crate::main_modules::auto_swap::Decision::ServeCurrent => Ok(()),
                }
            })
            .await;
            match outcome {
                Ok(Ok(())) => {}
                // A failed load already restored the previous model where it
                // could; say so and serve on whatever is actually live rather
                // than failing a request the old model could have answered.
                Ok(Err(e)) => tracing::warn!("auto-swap to {:?} failed: {e:#}", req.model),
                Err(e) => tracing::warn!("auto-swap task failed: {e}"),
            }
        }
    }

    // Resolve AFTER any swap, so the request is served by the model it asked
    // for rather than the one that happened to be loaded when it arrived.
    let Some(state) = host.current() else {
        // `_typed`, not the plain form: the plain one derives `error.type` from
        // the status and would label this `server_error`, which is both wrong
        // (nothing failed) and inconsistent with the `CurrentModel` extractor,
        // which reports `model_not_loaded` for the identical condition.
        return crate::api::compact::openai_error_response_typed(
            StatusCode::SERVICE_UNAVAILABLE,
            "no model is loaded".to_string(),
            "model_not_loaded",
        );
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
    // RAII: decrements on EVERY exit path, including this future being dropped
    // when the client disconnects (see ActiveRequestGuard / atlas#368). Moved
    // into the SSE stream for streaming requests so it outlives this function.
    let active_guard = crate::metrics::ActiveRequestGuard::new();

    // ── Input validation + cross-turn F-feature guards ──
    if let Err(resp) = super::chat_phases::validate_input(&req) {
        return ChatOutcome::Http(resp);
    }

    // Parser/native-template ownership is resolved by prepare_chat_prompt.
    // Native Qwen templates provide their own instructions; fallback,
    // custom-template, Hermes, and TSCG paths retain parser contributions.

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
    //
    // Runs on the BLOCKING pool, not the async worker. Phase timing measured
    // this chain at 1.4-3.2 ms on a short prompt and 13.0 ms at 12931 prompt
    // tokens — nearly all of it the Jinja render + tokenize — and it used to run
    // inline in this async fn, so a worker thread was held for the duration and
    // could poll no other request.
    //
    // `spawn_blocking` rather than `block_in_place`: `block_in_place` evicts the
    // other tasks from the current worker and may spawn a replacement thread, so
    // it suits OCCASIONAL blocking. Here every request would do it, which turns
    // into continuous worker cannibalisation as concurrency rises. The blocking
    // pool exists for exactly this and keeps all async workers pollable.
    //
    // `'static` is satisfied by ownership, not by a scoped-spawn crate: `state`
    // is already an `Arc` (clone = refcount bump) and `req` is MOVED in and
    // handed back, which also preserves the tool-system-prompt mutation
    // `prepare_chat_prompt` applies to it via `&mut`.
    //
    // NOTE: this is a CONCURRENCY fix. At --max-batch-size 1 single-stream there
    // is no second request to unblock, so it buys nothing there and costs one
    // task handoff; it is worth it for concurrent serving.
    let _t_seg = std::time::Instant::now();
    let state_for_prepare = state.clone();
    let (prepared, moved_req) = match tokio::task::spawn_blocking(move || {
        let out = prepare::prepare_chat_prompt(&state_for_prepare, &mut req);
        (out, req)
    })
    .await
    {
        Ok(pair) => pair,
        Err(join_err) => {
            // The blocking task panicked. Surface a 500 rather than letting the
            // JoinError unwind through the handler.
            tracing::error!("prepare_chat_prompt panicked: {join_err}");
            return ChatOutcome::Http(openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal error preparing the chat prompt".to_string(),
            ));
        }
    };
    req = moved_req;
    let prepare::PreparedChat {
        tools_active,
        cwd_hint,
        image_pixels,
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    } = match prepared {
        Ok(p) => p,
        Err(resp) => return ChatOutcome::Http(resp),
    };

    let us_prepare = _t_seg.elapsed().as_micros();

    // ── Phase 4: generic loop / spinning detection + task pin ───
    let loop_detect::LoopDetectOut {
        suppress_tool_call,
        tool_call_repeat_count,
    } = loop_detect::check_loops(&req.messages, tools_active);
    let us_loop_detect = _t_seg.elapsed().as_micros() - us_prepare;

    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let us_session_hash = _t_seg.elapsed().as_micros() - us_prepare - us_loop_detect;
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
    if state.chat.phase_timing {
        let us_sampling =
            _t_seg.elapsed().as_micros() - us_prepare - us_loop_detect - us_session_hash;
        tracing::info!(
            "CHAT_PHASE handler: prepare={us_prepare}us loop_detect={us_loop_detect}us \
             session_hash={us_session_hash}us sampling_and_grammar={us_sampling}us \
             total_pre_dispatch={}us",
            _t_seg.elapsed().as_micros()
        );
    }

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
            active_guard,
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
