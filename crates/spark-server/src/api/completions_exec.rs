// SPDX-License-Identifier: AGPL-3.0-only

//! Blocking execution loop for legacy `/v1/completions`: batched prompts
//! × `n` choices, sequential scheduler requests (mirrors the chat n>1
//! loop in `chat_blocking.rs`), prompt-major choice indices
//! (`index = prompt_i * n + n_i`), summed usage.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;

use crate::AppState;
use crate::openai::{
    CompletionChoice, CompletionRequest, CompletionResponse, RepetitionDetectionParams, Usage,
};

use super::compact::openai_error_response;
use super::completions_logprobs::build_completion_logprobs;
use super::inference_impl::strip_stop_sequences;
use super::inference_types::InferenceRequest;
use super::strip::strip_thinking_tags;

/// Sampling/request parameters resolved once by the handler and shared
/// by every choice in the request.
pub(super) struct CompletionParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub top_n_sigma: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub logit_bias: Vec<(u32, f32)>,
    pub stop_tokens: Vec<u32>,
    pub repetition_detection: Option<RepetitionDetectionParams>,
    /// Clamped `logprobs` (OpenAI integer form). Drives generated-token
    /// logprobs always, and prompt logprobs when `echo` is set.
    pub logprobs_k: Option<u8>,
}

/// Run every (prompt × n) choice sequentially and assemble the response.
pub(super) async fn run_blocking(
    state: Arc<AppState>,
    req: &CompletionRequest,
    prompts: Vec<Vec<u32>>,
    p: CompletionParams,
) -> Response {
    // n is validated to 1..=128 at the handler entry. The capacity is a
    // HINT clamped independently of user input (the vec grows amortized
    // past it) so no request can size an allocation directly.
    let n = req.n.clamp(1, 128);
    let mut choices: Vec<CompletionChoice> =
        Vec::with_capacity(prompts.len().saturating_mul(n).min(1024));
    let mut sum_prompt = 0usize;
    let mut sum_completion = 0usize;
    let mut sum_cached = 0usize;
    let mut sum_reasoning = 0usize;
    let mut last_ttft = 0.0f64;
    let mut last_tps = 0.0f64;

    for (prompt_i, prompt_tokens) in prompts.iter().enumerate() {
        for n_i in 0..n {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let session_hash = crate::session_manager::compute_session_hash(prompt_tokens);
            let request = InferenceRequest::Blocking {
                prompt_tokens: Arc::new(prompt_tokens.clone()),
                session_hash,
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
                // Legacy /v1/completions path doesn't have tool semantics,
                // so no DRY (would dampen legitimate long-repeated prose).
                dry_multiplier: 0.0,
                dry_base: 1.75,
                dry_allowed_length: 2,
                lz_penalty: 0.0,
                logit_bias: p.logit_bias.clone(),
                stop_tokens: p.stop_tokens.clone(),
                enable_thinking: false,
                thinking_budget: None,
                repetition_detection: p.repetition_detection,
                require_tool_call: false,
                tools_present: false,
                suppress_tool_call: false,
                disable_mtp: false,
                grammar_spec: None,
                // Distinct choices need distinct sampling streams; same
                // offset scheme as chat_blocking's n>1 loop.
                seed: req
                    .seed
                    .map(|s| s.wrapping_add((prompt_i * n + n_i) as u64)),
                top_logprobs: p.logprobs_k,
                prompt_logprobs: if req.echo { p.logprobs_k } else { None },
                echo: req.echo,
                timeout_at: {
                    let secs = state.request_timeout as f32;
                    if secs > 0.0 {
                        Some(std::time::Instant::now() + std::time::Duration::from_secs_f32(secs))
                    } else {
                        None
                    }
                },
                response_tx: tx,
            };

            if state.request_tx.send(request).await.is_err() {
                return openai_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Scheduler queue full".to_string(),
                );
            }
            let response = match rx.await {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => {
                    return openai_error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Inference error: {e}"),
                    );
                }
                Err(_) => {
                    return openai_error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Inference cancelled".to_string(),
                    );
                }
            };

            let completion_text = match state.tokenizer.decode(&response.output_tokens) {
                Ok(t) => t,
                Err(e) => {
                    return openai_error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Decode error: {e}"),
                    );
                }
            };
            // Strips apply to the COMPLETION only; echoed prompt text is
            // returned verbatim per the legacy spec.
            let completion_text = strip_stop_sequences(completion_text, &req.stop);
            let completion_text = strip_thinking_tags(&completion_text);
            let text = if req.echo {
                let prompt_text = state.tokenizer.decode(prompt_tokens).unwrap_or_default();
                format!("{prompt_text}{completion_text}")
            } else {
                completion_text
            };

            let logprobs = p.logprobs_k.map(|_| {
                let decode = |id: u32| state.tokenizer.decode(&[id]).unwrap_or_default();
                build_completion_logprobs(
                    &decode,
                    req.echo,
                    prompt_tokens,
                    &response.prompt_logprobs,
                    &response.output_tokens,
                    &response.logprobs,
                )
            });

            sum_prompt += prompt_tokens.len();
            sum_completion += response.output_tokens.len();
            sum_cached += response.cached_prompt_tokens as usize;
            sum_reasoning += response.reasoning_tokens as usize;
            last_ttft = response.time_to_first_token_ms;
            last_tps = if response.decode_time_ms > 0.0 {
                (response.output_tokens.len().saturating_sub(1)) as f64
                    / (response.decode_time_ms / 1000.0)
            } else {
                0.0
            };

            choices.push(CompletionChoice {
                index: prompt_i * n + n_i,
                text,
                finish_reason: response.finish_reason,
                logprobs,
            });
        }
    }

    let usage = Usage {
        prompt_tokens: sum_prompt,
        completion_tokens: sum_completion,
        total_tokens: sum_prompt + sum_completion,
        prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
            cached_tokens: sum_cached,
            audio_tokens: 0,
        }),
        completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
            reasoning_tokens: sum_reasoning,
            audio_tokens: 0,
            accepted_prediction_tokens: 0,
            rejected_prediction_tokens: 0,
        }),
        time_to_first_token_ms: last_ttft,
        response_tokens_per_second: last_tps,
    };

    Json(CompletionResponse::from_choices(
        &state.model_name,
        choices,
        usage,
    ))
    .into_response()
}
