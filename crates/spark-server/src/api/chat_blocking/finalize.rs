// SPDX-License-Identifier: AGPL-3.0-only

//! Logprobs conversion + final `ChatCompletionResponse` assembly (metrics,
//! store, rate-limit refund), split out of `chat_blocking.rs` to keep that
//! file ≤500 LoC.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use std::sync::Arc;

use axum::response::{IntoResponse, Json, Response};

use crate::AppState;
use crate::openai::{ChatCompletionRequest, ChatCompletionResponse, Usage};

/// Convert internal logprobs to OpenAI `ChoiceLogprobs` format.
pub(super) fn build_logprobs(
    state: &AppState,
    response: &super::super::inference_types::InferenceResponse,
) -> Option<crate::openai::ChoiceLogprobs> {
    if response.logprobs.is_empty() {
        return None;
    }
    Some(crate::openai::ChoiceLogprobs {
        content: response
            .logprobs
            .iter()
            .map(|lp| {
                let token_str = state.tokenizer.decode(&[lp.token_id]).unwrap_or_default();
                crate::openai::TokenLogprobInfo {
                    token: token_str,
                    logprob: lp.logprob,
                    bytes: None,
                    top_logprobs: lp
                        .top
                        .iter()
                        .map(|&(tid, lp_val)| crate::openai::TopLogprob {
                            token: state.tokenizer.decode(&[tid]).unwrap_or_default(),
                            logprob: lp_val,
                            bytes: None,
                        })
                        .collect(),
                }
            })
            .collect(),
    })
}

/// Build the final `ChatCompletionResponse` plus metrics, store, and
/// rate-limit refund. Returns the JSON-encoded HTTP response.
#[allow(clippy::too_many_arguments)]
pub(super) fn finalize_response(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    dump_seq: Option<u64>,
    all_choices: Vec<crate::openai::ChatChoice>,
    total_completion_tokens: usize,
    first_ttft: f64,
    last_decode_time_ms: f64,
    total_reasoning_tokens: u32,
    total_cached_prompt_tokens: u32,
    prompt_len: usize,
) -> Response {
    let tokens_per_second = if last_decode_time_ms > 0.0 && total_completion_tokens > 0 {
        (total_completion_tokens.saturating_sub(1)) as f64 / (last_decode_time_ms / 1000.0)
    } else {
        0.0
    };
    let usage = Usage {
        prompt_tokens: prompt_len,
        completion_tokens: total_completion_tokens,
        total_tokens: prompt_len + total_completion_tokens,
        prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
            cached_tokens: total_cached_prompt_tokens as usize,
            audio_tokens: 0,
        }),
        completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
            reasoning_tokens: total_reasoning_tokens as usize,
            audio_tokens: 0,
            accepted_prediction_tokens: 0,
            rejected_prediction_tokens: 0,
        }),
        time_to_first_token_ms: first_ttft,
        response_tokens_per_second: tokens_per_second,
    };

    let completion_id = format!("chatcmpl-{}", crate::openai::uuid_v4());
    let created_at = crate::openai::unix_timestamp();
    let completion = ChatCompletionResponse {
        id: completion_id.clone(),
        object: "chat.completion".to_string(),
        created: created_at,
        model: state.model_name.clone(),
        system_fingerprint: Some("fp_atlas".to_string()),
        choices: all_choices,
        usage: usage.clone(),
        service_tier: req.service_tier.clone(),
        metadata: req.metadata.clone(),
    };

    crate::metrics::REQUESTS_ACTIVE.dec();
    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(prompt_len as u64);
    crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(total_completion_tokens as u64);
    crate::metrics::TTFT_SECONDS.observe(first_ttft / 1000.0);

    // Completion-storage backend: when `store: true`, persist the
    // serialized body so a subsequent GET /v1/chat/completions/{id}
    // can return it. Bounded LRU + TTL in response_store.
    if req.store.unwrap_or(false)
        && let Ok(body) = serde_json::to_value(&completion)
    {
        state
            .response_store
            .insert(crate::response_store::StoredEntry {
                id: completion_id,
                kind: crate::response_store::StoredKind::ChatCompletion,
                model: state.model_name.clone(),
                created_at,
                messages: Vec::new(),
                body,
                last_access: std::time::Instant::now(),
            });
    }

    // Rate-limit true-up. Middleware admitted with a conservative
    // reservation of `max_seq_len` tokens; refund the difference.
    if let Some(axum::extract::Extension(ref ctx)) = req_ctx {
        let actual = (prompt_len + total_completion_tokens) as u64;
        let refund = ctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            state.rate_limiter.refund_tokens(&ctx.identity, refund);
        }
    }

    // --dump: record the non-streaming response body, correlated with
    // the request via the shared seq number.
    if let (Some(seq), Some(dump)) = (dump_seq, state.dump_writer.as_ref()) {
        dump.dump_response("/v1/chat/completions", seq, &completion, false);
    }

    Json(completion).into_response()
}
