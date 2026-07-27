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
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, PromptInput, Usage,
};
use crate::tool_parser;

// Sibling-cluster items hoisted from the original `api.rs`. These uses
// give every sub-file access to helpers that the un-split file took for
// granted via single-module visibility.
use super::chat::chat_completions_inner;
use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
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
use super::inference_types::*;
use super::sanitizer::*;

/// Resolve an OpenAI `prompt` field into the scheduler's prompt tokens. Text
/// tokenizes with `add_special_tokens=false` (no BOS); token-ID forms are used
/// verbatim, range-checked against the vocab (out-of-range → 400).
fn resolve_prompts(
    state: &AppState,
    prompt: &PromptInput,
) -> Result<Vec<Vec<u32>>, (StatusCode, String)> {
    match prompt {
        PromptInput::Text(s) => {
            Ok(vec![state.tokenizer.encode(s).map_err(|e| {
                (StatusCode::BAD_REQUEST, format!("Tokenization error: {e}"))
            })?])
        }
        PromptInput::TextArray(parts) => {
            // OpenAI spec: each array element is an INDEPENDENT prompt
            // yielding its own choice. (Earlier Atlas joined the array
            // into one prompt — that silently corrupted batched eval
            // harnesses like lm-eval at batch_size > 1.)
            parts
                .iter()
                .map(|part| {
                    state
                        .tokenizer
                        .encode(part)
                        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Tokenization error: {e}")))
                })
                .collect()
        }
        PromptInput::TokenIds(ids) => {
            validate_token_ids(state, ids)?;
            Ok(vec![ids.clone()])
        }
        PromptInput::TokenIdBatch(batch) => {
            // Same spec rule for pre-tokenized batches: one prompt per
            // sub-array (was: flattened into a single sequence).
            for ids in batch {
                validate_token_ids(state, ids)?;
            }
            Ok(batch.clone())
        }
    }
}

/// Fail-fast validation that every supplied token ID is within the model
/// vocabulary. The tokenizer is the authoritative source of vocab size
/// (SSOT); an OOB ID would index past the embedding table.
fn validate_token_ids(state: &AppState, ids: &[u32]) -> Result<(), (StatusCode, String)> {
    let vocab_size = state.tokenizer.inner().get_vocab_size(true) as u32;
    if let Some(&bad) = ids.iter().find(|&&id| id >= vocab_size) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Token ID {bad} out of range: vocab_size is {vocab_size}"),
        ));
    }
    Ok(())
}

pub async fn completions(
    State(state): State<Arc<AppState>>,
    req: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    let prompts = match resolve_prompts(&state, &req.prompt) {
        Ok(t) => t,
        Err((status, msg)) => return openai_error_response(status, msg),
    };
    if prompts.is_empty() {
        return openai_error_response(StatusCode::BAD_REQUEST, "Empty prompt".to_string());
    }
    for prompt_tokens in &prompts {
        let prompt_len = prompt_tokens.len();
        if prompt_len >= state.max_seq_len {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "Prompt too long: {prompt_len} tokens exceeds max_seq_len {}",
                    state.max_seq_len
                ),
            );
        }
    }

    // Range-validate sampling params, mirroring the chat path (which returns
    // 400 for out-of-spec values). Without this, a negative temperature is
    // silently reinterpreted as greedy decoding and out-of-range penalties are
    // applied verbatim, both diverging from OpenAI (and Atlas's own chat
    // endpoint), which reject with HTTP 400.
    if let Some(t) = req.temperature
        && !(0.0..=2.0).contains(&t)
    {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("temperature must be between 0 and 2, got {t}"),
        );
    }
    if let Some(pp) = req.presence_penalty
        && !(-2.0..=2.0).contains(&pp)
    {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("presence_penalty must be between -2 and 2, got {pp}"),
        );
    }
    if let Some(fp) = req.frequency_penalty
        && !(-2.0..=2.0).contains(&fp)
    {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("frequency_penalty must be between -2 and 2, got {fp}"),
        );
    }

    let temperature = req.temperature.unwrap_or(state.default_temperature);
    let top_k = req.top_k.unwrap_or(state.default_top_k);
    let top_p = req.top_p.unwrap_or(state.default_top_p);
    let top_n_sigma = req.top_n_sigma.unwrap_or(state.default_top_n_sigma);
    let min_p = req.min_p.unwrap_or(state.default_min_p);
    let repetition_penalty = req
        .repetition_penalty
        .unwrap_or(state.sampling_presets.non_thinking.repetition_penalty);
    let presence_penalty = req.presence_penalty.unwrap_or(0.0);
    let frequency_penalty = req.frequency_penalty.unwrap_or(0.0);
    // Convert logit_bias from OpenAI format (string keys) to Vec<(u32, f32)>
    let logit_bias: Vec<(u32, f32)> = req.logit_bias.as_ref().map_or(Vec::new(), |map| {
        map.iter()
            .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
            .collect()
    });
    // OpenAI spec bounds n to 1-128; an unbounded n would drive both an
    // attacker-controlled allocation and an unbounded sequential-inference
    // loop (CodeQL: uncontrolled allocation size). Fail fast per spec.
    if req.n == 0 || req.n > 128 {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("n must be between 1 and 128, got {}", req.n),
        );
    }
    let stop_tokens = tokenize_stop_sequences(&state.tokenizer, &req.stop);
    // OpenAI clamps chat top_logprobs to 20; same bound here (spec says
    // 5 for legacy — being more permissive, never less).
    let logprobs_k = req.logprobs.map(|k| k.min(20));

    // M2 per-request LoRA routing: resolve the optional `adapter` name to a
    // pool slot (`-1` = defer to installed active; unknown = 400), with the #27
    // on-miss RDMA promotion for stageable adapters.
    let adapter_slot = match super::lora_control::resolve_request_adapter_slot(
        &state,
        req.adapter.as_deref(),
        &req.model,
    )
    .await
    {
        Ok(slot) => slot,
        Err(resp) => return resp,
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
        Err(resp) => return resp,
    };
    let tgt_lang_id = match resolve_lang(&req.tgt_lang) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    // NLLB beam search params (mirrors src/tgt lang resolution).
    let num_beams = req.num_beams.unwrap_or(1);
    let length_penalty = req.length_penalty.unwrap_or(1.0);
    let early_stopping = req.early_stopping.unwrap_or(false);

    let params = super::completions_exec::CompletionParams {
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        logit_bias,
        stop_tokens,
        repetition_detection: req.repetition_detection,
        logprobs_k,
        adapter_slot,
        src_lang_id,
        tgt_lang_id,
        num_beams,
        length_penalty,
        early_stopping,
    };

    if req.stream {
        if prompts.len() > 1 || req.n > 1 {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                "stream=true supports a single prompt with n=1".to_string(),
            );
        }
        if num_beams > 1 {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                "num_beams > 1 is not supported in streaming mode".to_string(),
            );
        }
        let prompt_tokens = prompts.into_iter().next().expect("checked non-empty");
        return match completions_stream(state, prompt_tokens, req, params).await {
            Ok(r) => r,
            Err((status, msg)) => openai_error_response(status, msg),
        };
    }

    super::completions_exec::run_blocking(state, &req, prompts, params).await
}

/// SSE streaming path for legacy completions (single prompt, n=1, handler-
/// guarded). Echo: prompt text (+ logprobs when set) is the FIRST chunk; with
/// `stream_options.include_usage` a `choices: []` usage chunk precedes `[DONE]`.
pub(super) async fn completions_stream(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    req: CompletionRequest,
    p: super::completions_exec::CompletionParams,
) -> Result<Response, (StatusCode, String)> {
    // Match chat_stream/mod.rs sizing; see comment there.
    let (token_tx, token_rx) = tokio::sync::mpsc::channel::<StreamEvent>(1024);
    let prompt_len = prompt_tokens.len();
    let echo = req.echo;
    let logprobs_k = p.logprobs_k;
    let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
    // Echo needs the prompt tokens after the request consumes them.
    let echo_prompt = if echo {
        Some(prompt_tokens.clone())
    } else {
        None
    };
    // Echo WITHOUT logprobs has no PromptLogprobs event to hook — the
    // prompt text chunk is prepended client-side; decode it now, before
    // the request takes ownership of the tokens.
    let echo_only_text = if echo && logprobs_k.is_none() {
        Some(state.tokenizer.decode(&prompt_tokens).unwrap_or_default())
    } else {
        None
    };

    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let request = InferenceRequest::Streaming {
        prompt_tokens: std::sync::Arc::new(prompt_tokens),
        session_hash,
        adapter_slot: p.adapter_slot,
        src_lang_id: p.src_lang_id,
        tgt_lang_id: p.tgt_lang_id,
        num_beams: p.num_beams,
        length_penalty: p.length_penalty,
        early_stopping: p.early_stopping,
        image_pixels: Vec::new(),
        max_tokens: req.max_tokens,
        min_tokens: 0,
        temperature: p.temperature,
        top_k: p.top_k,
        top_p: p.top_p,
        top_n_sigma: p.top_n_sigma,
        min_p: p.min_p,
        repetition_penalty: p.repetition_penalty,
        presence_penalty: p.presence_penalty,
        frequency_penalty: p.frequency_penalty,
        // Legacy /v1/completions path doesn't have tool semantics.
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        lz_penalty: 0.0,
        logit_bias: p.logit_bias,
        stop_tokens: p.stop_tokens,
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: p.repetition_detection,
        require_tool_call: false,
        // Completions API defines no tools — multi-tool-call continuation off.
        tools_present: false,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_spec: None,
        seed: req.seed,
        top_logprobs: logprobs_k,
        prompt_logprobs: if echo { logprobs_k } else { None },
        echo,
        timeout_at: None,
        token_tx,
        // /v1/completions has no guard pipeline yet — the flag is
        // created so the scheduler's emit_step type-checks cleanly,
        // but never flipped.
        cancel_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };

    state.request_tx.send(request).await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Scheduler queue full".to_string(),
        )
    })?;

    let chunk_id = crate::openai::new_completion_id();
    let model_name = state.model_name.clone();

    let model = model_name.clone();
    let id = chunk_id.clone();
    let mut all_toks: Vec<u32> = Vec::new();
    let mut emitted: usize = 0;
    // Incremental-detokenizer state (see `ChatTokenizer::incremental_decode`):
    // extends `content_decoded` a bounded suffix at a time instead of
    // re-decoding all_toks every token (O(n²) → O(n)).
    let mut content_decoded = String::new();
    let mut detok_prefix_offset: usize = 0;
    let mut detok_read_offset: usize = 0;
    let token_stream = ReceiverStream::new(token_rx).flat_map(move |event| {
        let events: Vec<Result<Event, std::convert::Infallible>> = match event {
            // Echo + logprobs: prompt text and its logprobs, before any
            // generated token (the scheduler emits this exactly once).
            StreamEvent::PromptLogprobs(lps) => {
                let prompt_toks = echo_prompt.clone().unwrap_or_default();
                let text = state.tokenizer.decode(&prompt_toks).unwrap_or_default();
                let decode = |tid: u32| state.tokenizer.decode(&[tid]).unwrap_or_default();
                let lp = super::completions_logprobs::build_completion_logprobs(
                    &decode,
                    true,
                    &prompt_toks,
                    &lps,
                    &[],
                    &[],
                );
                let chunk = CompletionChunk::echo_chunk(&model, &id, text, Some(lp));
                vec![Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                )]
            }
            StreamEvent::Token(tok) | StreamEvent::TokenWithLogprobs(tok, _) => {
                all_toks.push(tok);
                content_decoded.push_str(&state.tokenizer.incremental_decode(
                    &all_toks,
                    &mut detok_prefix_offset,
                    &mut detok_read_offset,
                ));
                let stable_end = content_decoded.len();
                let delta = if stable_end <= emitted {
                    String::new()
                } else {
                    let d = content_decoded[emitted..stable_end].to_string();
                    emitted = stable_end;
                    d
                };
                let chunk = CompletionChunk::text_chunk(&model, &id, delta);
                vec![Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                )]
            }
            StreamEvent::Done {
                finish_reason,
                prompt_tokens: _,
                completion_tokens,
                time_to_first_token_ms,
                decode_time_ms,
                reasoning_tokens,
                cached_prompt_tokens,
                guard_stop: _,
            } => {
                let tps = if decode_time_ms > 0.0 {
                    completion_tokens.saturating_sub(1) as f64 / (decode_time_ms / 1000.0)
                } else {
                    0.0
                };
                let usage = Usage {
                    prompt_tokens: prompt_len,
                    completion_tokens,
                    total_tokens: prompt_len + completion_tokens,
                    prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
                        cached_tokens: cached_prompt_tokens as usize,
                        audio_tokens: 0,
                    }),
                    completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
                        reasoning_tokens: reasoning_tokens as usize,
                        audio_tokens: 0,
                        accepted_prediction_tokens: 0,
                        rejected_prediction_tokens: 0,
                    }),
                    time_to_first_token_ms,
                    response_tokens_per_second: tps,
                };
                if include_usage {
                    // Chat parity: finish chunk without usage, then a
                    // choices:[] usage-only chunk.
                    let fin = CompletionChunk::finish_chunk_no_usage(&model, &id, &finish_reason);
                    let usage_chunk = CompletionChunk::usage_only_chunk(&model, &id, usage);
                    vec![
                        Ok(Event::default().data(serde_json::to_string(&fin).unwrap_or_default())),
                        Ok(Event::default()
                            .data(serde_json::to_string(&usage_chunk).unwrap_or_default())),
                    ]
                } else {
                    let chunk = CompletionChunk::done_chunk(&model, &id, &finish_reason, usage);
                    vec![Ok(
                        Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
                    )]
                }
            }
            StreamEvent::Error(msg) => {
                vec![Ok(Event::default().data(format!(r#"{{"error":"{msg}"}}"#)))]
            }
        };
        futures::stream::iter(events)
    });

    // Echo WITHOUT logprobs: the scheduler emits no PromptLogprobs event
    // (nothing to collect), so prepend the prompt text chunk directly.
    let echo_prefix: Option<Event> = echo_only_text.map(|text| {
        let chunk = CompletionChunk::echo_chunk(&model_name, &chunk_id, text, None);
        Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
    });

    let done_event = futures::stream::once(async {
        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
    });
    let prefix = futures::stream::iter(
        echo_prefix
            .into_iter()
            .map(Ok::<_, std::convert::Infallible>),
    );
    let full_stream = prefix.chain(token_stream).chain(done_event);

    Ok(Sse::new(full_stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

/// Generic 501 "not supported" response used by the auto-probe stubs
/// below. OpenAI-SDK auto-detection and observability wrappers expect a
/// 501 + `error.type = server_error`; returning 404 would be interpreted
/// as "wrong URL".
pub(super) fn not_supported(message: &'static str) -> Response {
    openai_error_response(StatusCode::NOT_IMPLEMENTED, message.into())
}
