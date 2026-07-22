// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::Arc;

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use crate::AppState;

use super::handlers_stream::*;
use super::helpers::*;
use super::types::*;

// ── Handler ──

/// POST /v1/messages — Anthropic Messages API.
///
/// Lowers the request directly into the canonical chat IR
/// (`From<MessagesRequest> for ir::ChatRequest`), dispatches through
/// `api::chat_completions_inner` (which runs every fix12-23
/// sanitization, salvage, watchdog, and dump path), and translates the
/// response back into Anthropic format. The Anthropic-specific surface
/// is strictly format conversion — no policy or sampling decisions are
/// made here.
pub async fn messages(State(state): State<Arc<AppState>>, body: axum::body::Bytes) -> Response {
    // 1. Parse the Anthropic request.
    let req: MessagesRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Invalid request JSON: {e}"),
            );
        }
    };

    tracing::info!(
        "Anthropic request: max_tokens={}, thinking={:?}, tools={}, model={}, stream={}",
        req.max_tokens,
        req.thinking
            .as_ref()
            .map(|t| format!("type={} budget={:?}", t.thinking_type, t.budget_tokens)),
        req.tools.as_ref().map_or(0, |t| t.len()),
        req.model,
        req.stream,
    );

    let stream = req.stream;
    let model_echo = req.model.clone();

    // 2. --dump: capture the raw Anthropic body. We mint our own seq so
    //    the entry shows endpoint="/v1/messages"; chat_completions_inner
    //    is invoked with dump_seq=None so it doesn't double-dump as
    //    "/v1/chat/completions".
    let dump_seq = state.dump_writer.as_ref().and_then(|d| {
        match serde_json::from_slice::<serde_json::Value>(&body) {
            Ok(v) => {
                let seq = d.next_seq();
                d.dump_request("/v1/messages", seq, &v);
                Some(seq)
            }
            Err(_) => None,
        }
    });

    // 3. Lower the Anthropic wire request straight into the IR envelope
    //    (typed peer adapter — no OpenAI-wire-JSON intermediary). The
    //    old count-level translation-drift audit is gone with the JSON
    //    hop: a typed constructor makes that failure class
    //    unrepresentable at runtime; the adapter unit tests own it.
    //
    // 4. Run the shared pipeline. All sanitization, salvage, watchdog,
    //    sampling preset, and prompt mutation logic lives there.
    //    Anthropic has no echo-only wire fields (service_tier / store /
    //    stream_options), so the echo context stays default.
    let outcome = crate::api::chat_completions_inner(state.clone(), None, req.into(), None).await;

    // 5. Encode back to Anthropic shape. Non-streaming success arrives
    //    as the response IR — no body re-parse; a malformed inner body
    //    is now structurally impossible.
    let chat_resp = match outcome {
        crate::api::ChatOutcome::Blocking(ir) => {
            let messages_resp = MessagesResponse::from(*ir);
            if let (Some(seq), Some(dump)) = (dump_seq, state.dump_writer.as_ref()) {
                dump.dump_response("/v1/messages", seq, &messages_resp, false);
            }
            return Json(messages_resp).into_response();
        }
        crate::api::ChatOutcome::Streaming(deltas) => {
            // --dump: the encoder aggregates the typed Anthropic events
            // and writes them under the same seq as the request entry.
            let dump =
                dump_seq.and_then(|seq| state.dump_writer.as_ref().map(|d| (seq, d.clone())));
            return anthropic_sse_from_deltas(deltas, model_echo, dump);
        }
        crate::api::ChatOutcome::Http(r) => r,
    };

    if !chat_resp.status().is_success() {
        // Forward the error envelope. Translate the JSON body into
        // Anthropic's error shape if it's an OpenAI-style envelope; else
        // pass bytes through.
        let (parts, body) = chat_resp.into_parts();
        let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
            Ok(b) => b,
            Err(e) => {
                return anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    format!("Error body collect: {e}"),
                );
            }
        };
        let err_msg = serde_json::from_slice::<serde_json::Value>(&body_bytes)
            .ok()
            .and_then(|v| {
                v.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| String::from_utf8_lossy(&body_bytes).into_owned());
        return anthropic_error(parts.status, "api_error", err_msg);
    }

    // Unreachable in practice: Http outcomes are error envelopes (both
    // success shapes returned above), and the error branch returned too.
    // Fall back to forwarding whatever arrived.
    let _ = stream;
    chat_resp
}

// ── Count tokens endpoint ──

/// POST /v1/messages/count_tokens — returns input token count.
///
/// Claude Code calls this to validate the model and estimate token usage.
pub async fn count_tokens(
    State(state): State<Arc<AppState>>,
    req: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Invalid request JSON: {e}"),
            );
        }
    };

    // Count against the EXACT prompt the serving path renders: same
    // adapter (the IR `From` impl), same tool-prompt injection / hint /
    // cwd / thinking resolution / Jinja variant (prepare_chat_prompt). The
    // old path was a third, divergent lowering — it dropped images and
    // thinking blocks, force-prepended an empty `<think>` wrapper, and
    // rendered through the non-openai Jinja variant, so counts drifted
    // from real usage.
    let mut ir_req = crate::ir::ChatRequest::from(req);
    // Counting must not require the vision encoder: strip image parts
    // (they contributed 0 tokens in the old count too — pad expansion
    // needs real pixel grids, which a count endpoint can't produce).
    for m in &mut ir_req.messages {
        m.content
            .retain(|p| !matches!(p, crate::ir::ContentPart::Image(_)));
    }
    let prepared = match crate::api::chat::prepare::prepare_chat_prompt(&state, &mut ir_req) {
        Ok(p) => p,
        Err(resp) => return openai_error_to_anthropic(resp).await,
    };

    let body = serde_json::json!({
        "input_tokens": prepared.prompt_tokens.len()
    });
    Json(body).into_response()
}

/// Re-shape an OpenAI-envelope error `Response` (produced by the shared
/// pipeline) into Anthropic's error body, preserving the status code.
async fn openai_error_to_anthropic(resp: Response) -> Response {
    let (parts, body) = resp.into_parts();
    let message = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| {
                v.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned()),
        Err(e) => format!("Error body collect: {e}"),
    };
    let error_type = if parts.status.is_client_error() {
        "invalid_request_error"
    } else {
        "api_error"
    };
    anthropic_error(parts.status, error_type, message)
}
