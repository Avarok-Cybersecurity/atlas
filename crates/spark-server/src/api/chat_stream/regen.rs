// SPDX-License-Identifier: AGPL-3.0-only
//
// Layer C — streaming Answer-Regeneration (arXiv 2510.14773).
//
// When a streaming turn finishes with empty/degenerate visible content but
// substantial `<think>` reasoning, splice ONE phase-2 generation —
// `original prompt ++ phase-1 reasoning ++ </think> bridge`, thinking OFF,
// tools/grammar OFF — and stream ITS content as the answer, then a single
// combined terminal. Driver-task + output-channel architecture: the SSE
// body becomes a `ReceiverStream`; the driver drains phase-1, intercepts
// `Done`, and either runs the existing `handle_done` verbatim (Close — the
// common case) or performs the phase-2 splice (Regenerate). Entirely behind
// `answer_regen_enabled()`: when off, `chat_stream::mod` keeps the original
// `role.chain(token_stream).chain(done)` path → byte-identical to today.

use std::convert::Infallible;

use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::openai::{ChatCompletionChunk, Usage};

use super::super::inference_types::{InferenceRequest, StreamEvent};
use super::ctx::StreamCtx;
use super::handle_done::handle_done;
use super::handle_error::handle_error;
use super::handle_token::handle_token;
use super::state::StreamState;

/// Reusable phase-1 request params needed to construct the phase-2
/// `InferenceRequest::Streaming` (captured in `chat_stream::mod` before the
/// phase-1 request is moved into the scheduler queue).
pub(super) struct Phase2Tpl {
    pub(super) prompt_tokens: Vec<u32>,
    pub(super) session_hash: u64,
    pub(super) max_tokens: usize,
    pub(super) min_tokens: usize,
    pub(super) min_content_tokens: usize,
    pub(super) temperature: f32,
    pub(super) top_k: u32,
    pub(super) top_p: f32,
    pub(super) top_n_sigma: f32,
    pub(super) min_p: f32,
    pub(super) repetition_penalty: f32,
    pub(super) presence_penalty: f32,
    pub(super) frequency_penalty: f32,
    pub(super) dry_multiplier: f32,
    pub(super) dry_base: f32,
    pub(super) dry_allowed_length: u32,
    pub(super) lz_penalty: f32,
    pub(super) xtc_probability: f32,
    pub(super) xtc_threshold: f32,
    pub(super) logit_bias: Vec<(u32, f32)>,
    pub(super) stop_tokens: Vec<u32>,
    pub(super) thinking_budget: Option<u32>,
    pub(super) disable_mtp: bool,
    pub(super) seed: Option<u64>,
    pub(super) timeout_at: Option<std::time::Instant>,
}

fn data_event<T: serde::Serialize>(v: &T) -> Result<Event, Infallible> {
    Ok(Event::default().data(serde_json::to_string(v).unwrap_or_default()))
}

/// Decide whether to regenerate. Returns true iff Answer-Regen is enabled,
/// not already used, the phase-1 reasoning is substantial, this is NOT a
/// tool-call turn, and the visible content is empty or degenerate. The
/// degeneracy markers MIRROR bench/reasoning_eval.py::score (SSOT).
fn should_regenerate(state: &StreamState, ctx: &StreamCtx, finish_reason: &str) -> bool {
    if !crate::scheduler::answer_regen_enabled() || state.regen_used {
        return false;
    }
    if finish_reason == "tool_calls"
        || state.salvaged_tool_call
        || state.detector.as_ref().is_some_and(|d| d.has_tool_calls())
    {
        return false;
    }
    if state.regen_reasoning_acc.len() < crate::scheduler::answer_regen_min_reasoning_bytes() {
        return false;
    }
    let c = &state.refusal_scan_buf; // post-sanitizer visible content (capped)
    let low = c.to_ascii_lowercase();
    let doctype = low.matches("<!doctype").count();
    let role_leak = c.contains("\nassistant") || c.contains("\nuser") || c.contains("\ntool");
    let degenerate = c.trim().is_empty()
        || doctype >= 2
        || (doctype >= 1 && !low.contains("</html>"))
        || role_leak;
    let _ = ctx;
    degenerate
}

/// Phase-2 prompt = original prompt (which already ends at the thinking
/// generation prompt `…assistant\n<think>\n`) ++ the phase-1 reasoning ++
/// a `</think>` close + blank line. The model is now post-think and emits
/// the final answer directly as content (phase-2 runs with thinking OFF).
/// On any tokenizer error, returns `None` (caller falls back to Close).
fn build_phase2_prompt(
    orig: &[u32],
    reasoning: &str,
    tokenizer: &crate::tokenizer::ChatTokenizer,
) -> Option<Vec<u32>> {
    let mut out = orig.to_vec();
    let r = reasoning.trim();
    if !r.is_empty() {
        out.extend(tokenizer.encode(r).ok()?);
    }
    out.extend(tokenizer.encode("\n</think>\n\n").ok()?);
    Some(out)
}

fn build_phase2_request(
    tpl: &Phase2Tpl,
    prompt_tokens: Vec<u32>,
    token_tx: mpsc::Sender<StreamEvent>,
    cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> InferenceRequest {
    InferenceRequest::Streaming {
        prompt_tokens,
        session_hash: tpl.session_hash,
        image_pixels: Vec::new(),
        max_tokens: tpl.max_tokens,
        min_tokens: tpl.min_tokens,
        min_content_tokens: tpl.min_content_tokens,
        temperature: tpl.temperature,
        top_k: tpl.top_k,
        top_p: tpl.top_p,
        top_n_sigma: tpl.top_n_sigma,
        min_p: tpl.min_p,
        repetition_penalty: tpl.repetition_penalty,
        presence_penalty: tpl.presence_penalty,
        frequency_penalty: tpl.frequency_penalty,
        dry_multiplier: tpl.dry_multiplier,
        dry_base: tpl.dry_base,
        dry_allowed_length: tpl.dry_allowed_length,
        lz_penalty: tpl.lz_penalty,
        xtc_probability: tpl.xtc_probability,
        xtc_threshold: tpl.xtc_threshold,
        logit_bias: tpl.logit_bias.clone(),
        stop_tokens: tpl.stop_tokens.clone(),
        // Phase-2 is the ANSWER pass: no thinking, no tools, no grammar.
        enable_thinking: false,
        thinking_budget: tpl.thinking_budget,
        require_tool_call: false,
        suppress_tool_call: false,
        disable_mtp: tpl.disable_mtp,
        grammar_spec: None,
        seed: tpl.seed,
        top_logprobs: None,
        timeout_at: tpl.timeout_at,
        token_tx,
        cancel_flag,
    }
}

#[allow(clippy::too_many_arguments)]
async fn regen_finalize(
    ctx: &StreamCtx,
    fr: &str,
    completion_tokens: usize,
    time_to_first_token_ms: f64,
    decode_time_ms: f64,
    reasoning_tokens: u32,
    cached_prompt_tokens: u32,
    out_tx: &mpsc::Sender<Result<Event, Infallible>>,
) {
    let tps = if decode_time_ms > 0.0 {
        completion_tokens.saturating_sub(1) as f64 / (decode_time_ms / 1000.0)
    } else {
        0.0
    };
    let usage = Usage {
        prompt_tokens: ctx.prompt_len,
        completion_tokens,
        total_tokens: ctx.prompt_len + completion_tokens,
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
    if ctx.req_stream_include_usage {
        let uc = ChatCompletionChunk::usage_only_chunk(&ctx.model, &ctx.id, usage);
        let _ = out_tx.send(data_event(&uc)).await;
        let fc = ChatCompletionChunk::final_chunk_no_usage(&ctx.model, &ctx.id, fr);
        let _ = out_tx.send(data_event(&fc)).await;
    } else {
        let dc = ChatCompletionChunk::done_chunk(&ctx.model, &ctx.id, fr, usage);
        let _ = out_tx.send(data_event(&dc)).await;
    }
    crate::metrics::REQUESTS_ACTIVE.dec();
    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(ctx.prompt_len as u64);
    crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(completion_tokens as u64);
    crate::metrics::TTFT_SECONDS.observe(time_to_first_token_ms / 1000.0);
    if let Some(ref rctx) = ctx.req_ctx {
        let actual = (ctx.prompt_len + completion_tokens) as u64;
        let refund = rctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            ctx.state.rate_limiter.refund_tokens(&rctx.identity, refund);
        }
    }
    let _ = out_tx.send(Ok(Event::default().data("[DONE]"))).await;
}

/// Build the Answer-Regen SSE response: a spawned driver task feeding a
/// `ReceiverStream`. Used only when `answer_regen_enabled()`.
pub(super) fn answer_regen_response(
    ctx: StreamCtx,
    mut state: StreamState,
    mut token_rx: mpsc::Receiver<StreamEvent>,
    role_json: String,
    tpl: Phase2Tpl,
) -> Response {
    let (out_tx, out_rx) = mpsc::channel::<Result<Event, Infallible>>(1024);
    let request_tx = ctx.state.request_tx.clone();

    tokio::spawn(async move {
        if out_tx
            .send(Ok(Event::default().data(role_json)))
            .await
            .is_err()
        {
            return;
        }
        while let Some(ev) = token_rx.recv().await {
            match ev {
                StreamEvent::Token(t) | StreamEvent::TokenWithLogprobs(t, _) => {
                    for e in handle_token(&mut state, &ctx, t) {
                        if out_tx.send(e).await.is_err() {
                            return;
                        }
                    }
                }
                StreamEvent::Error(m) => {
                    for e in handle_error(&ctx, m) {
                        let _ = out_tx.send(e).await;
                    }
                    let _ = out_tx.send(Ok(Event::default().data("[DONE]"))).await;
                    return;
                }
                StreamEvent::Done {
                    finish_reason,
                    prompt_tokens: _,
                    completion_tokens,
                    time_to_first_token_ms,
                    decode_time_ms,
                    reasoning_tokens,
                    cached_prompt_tokens,
                } => {
                    if !should_regenerate(&state, &ctx, &finish_reason) {
                        // Close — the common path: existing terminal verbatim.
                        for e in handle_done(
                            &mut state,
                            &ctx,
                            finish_reason,
                            completion_tokens,
                            time_to_first_token_ms,
                            decode_time_ms,
                            reasoning_tokens,
                            cached_prompt_tokens,
                        ) {
                            if out_tx.send(e).await.is_err() {
                                return;
                            }
                        }
                        let _ = out_tx.send(Ok(Event::default().data("[DONE]"))).await;
                        return;
                    }
                    // Regenerate (max once).
                    state.regen_used = true;
                    let prompt2 = build_phase2_prompt(
                        &tpl.prompt_tokens,
                        &state.regen_reasoning_acc,
                        &ctx.state.tokenizer,
                    );
                    let fallback = |reason: &str| {
                        tracing::warn!("Answer-Regen: {reason}; falling back to phase-1 close");
                    };
                    let Some(prompt2) = prompt2 else {
                        fallback("phase-2 prompt tokenize failed");
                        for e in handle_done(
                            &mut state, &ctx, finish_reason, completion_tokens,
                            time_to_first_token_ms, decode_time_ms, reasoning_tokens,
                            cached_prompt_tokens,
                        ) {
                            if out_tx.send(e).await.is_err() {
                                return;
                            }
                        }
                        let _ = out_tx.send(Ok(Event::default().data("[DONE]"))).await;
                        return;
                    };
                    tracing::info!(
                        "Answer-Regen: phase-1 content degenerate ({} reasoning B); \
                         splicing phase-2 (thinking off, {} prompt tokens)",
                        state.regen_reasoning_acc.len(),
                        prompt2.len(),
                    );
                    let (tx2, mut rx2) = mpsc::channel::<StreamEvent>(1024);
                    let cancel2 = ctx.cancel_flag.clone();
                    let req2 = build_phase2_request(&tpl, prompt2, tx2, cancel2);
                    if request_tx.send(req2).await.is_err() {
                        fallback("scheduler queue full");
                        for e in handle_done(
                            &mut state, &ctx, finish_reason, completion_tokens,
                            time_to_first_token_ms, decode_time_ms, reasoning_tokens,
                            cached_prompt_tokens,
                        ) {
                            if out_tx.send(e).await.is_err() {
                                return;
                            }
                        }
                        let _ = out_tx.send(Ok(Event::default().data("[DONE]"))).await;
                        return;
                    }
                    let mut dec = ctx.state.tokenizer.streaming_decoder(true);
                    let mut p2_ct = 0usize;
                    let mut p2_fr = finish_reason.clone();
                    while let Some(e2) = rx2.recv().await {
                        match e2 {
                            StreamEvent::Token(t) | StreamEvent::TokenWithLogprobs(t, _) => {
                                p2_ct += 1;
                                if let Ok(Some(s)) = dec.step(t) {
                                    if !s.is_empty() {
                                        let ch = ChatCompletionChunk::content_chunk(
                                            &ctx.model, &ctx.id, s,
                                        );
                                        if out_tx.send(data_event(&ch)).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                            }
                            StreamEvent::Done {
                                finish_reason: fr2,
                                completion_tokens: c2,
                                ..
                            } => {
                                p2_fr = fr2;
                                p2_ct = c2.max(p2_ct);
                                break;
                            }
                            StreamEvent::Error(_) => break,
                        }
                    }
                    regen_finalize(
                        &ctx,
                        &p2_fr,
                        completion_tokens + p2_ct,
                        time_to_first_token_ms,
                        decode_time_ms,
                        reasoning_tokens,
                        cached_prompt_tokens,
                        &out_tx,
                    )
                    .await;
                    return;
                }
            }
        }
        // token_rx closed without Done: just end the stream.
        let _ = out_tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    Sse::new(ReceiverStream::new(out_rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}
