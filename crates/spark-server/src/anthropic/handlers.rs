// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::Arc;

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use crate::AppState;
use crate::openai;

use super::convert::*;
use super::handlers_stream::*;
use super::helpers::*;
use super::translate::*;
use super::types::*;

// ── Handler ──

/// POST /v1/messages — Anthropic Messages API.
///
/// Translates the request into an OpenAI `ChatCompletionRequest`, dispatches
/// through `api::chat_completions_inner` (which runs every fix12-23
/// sanitization, salvage, watchdog, and dump path), and translates the
/// response back into Anthropic format. The Anthropic-specific surface is
/// strictly format conversion — no policy or sampling decisions are made
/// here.
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

    // 3. Translate to a ChatCompletionRequest.
    let chat_json = anthropic_to_chat_request_json(&req);

    // Translation-drift telemetry (P5.1, 2026-04-25). Detects whether
    // structural information from the Anthropic request was preserved
    // in the translated OpenAI shape — a count-level audit cheap
    // enough to run on every request. The metric counts mismatches;
    // verbose diff logging is gated behind ATLAS_DEBUG_TRANSLATION_DRIFT
    // (an opt-in for forensics, not a hot-path log).
    audit_translation_drift(&req, &chat_json);

    let chat_req: openai::ChatCompletionRequest = match serde_json::from_value(chat_json) {
        Ok(r) => r,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Translation error: {e}"),
            );
        }
    };

    // Drop `req` here — anthropic_to_chat_request_json already cloned what
    // it needed, and we've finished extracting `stream` + `model_echo`.

    // 4. Run the shared OpenAI pipeline. All sanitization, salvage,
    //    watchdog, sampling preset, and prompt mutation logic lives there.
    let chat_resp = crate::api::chat_completions_inner(state.clone(), None, chat_req, None).await;

    // 5. Translate the response back to Anthropic shape.
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

    if stream {
        let resp = wrap_chat_sse_for_anthropic(chat_resp, model_echo).await;
        // Note: streaming dump from the Anthropic side is best-effort;
        // chat_completions_inner already wrote a synthesized OpenAI-shape
        // response dump entry under endpoint="/v1/chat/completions" if
        // we'd passed a dump_seq. Anthropic-shape response capture is a
        // follow-up.
        let _ = dump_seq;
        resp
    } else {
        let (parts, body) = chat_resp.into_parts();
        let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
            Ok(b) => b,
            Err(e) => {
                return anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    format!("Body collect error: {e}"),
                );
            }
        };
        let chat_value: serde_json::Value = match serde_json::from_slice(&body_bytes) {
            Ok(v) => v,
            Err(e) => {
                return anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    format!("Inner response decode error: {e}"),
                );
            }
        };
        let messages_resp = chat_to_anthropic_response(&chat_value, model_echo);

        if let (Some(seq), Some(dump)) = (dump_seq, state.dump_writer.as_ref()) {
            dump.dump_response("/v1/messages", seq, &messages_resp, false);
        }

        // Preserve status code/headers from chat_completions_inner and
        // serialize the translated body.
        let json_bytes = serde_json::to_vec(&messages_resp).unwrap_or_default();
        Response::from_parts(parts, axum::body::Body::from(json_bytes))
    }
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

    let tools_active =
        state.tool_call_parser.is_some() && req.tools.as_ref().is_some_and(|t| !t.is_empty());
    let enable_thinking = if state.disable_thinking {
        false
    } else {
        let from_body = req.thinking.as_ref().map(|t| t.thinking_type.as_str());
        let explicit = from_body.is_some();
        let base = match from_body {
            Some("enabled") => true,
            Some("disabled") => false,
            _ => state.behavior.thinking_default,
        };
        // MODEL.toml `thinking_in_tools=false` is the DEFAULT for tool-active
        // turns — an explicit `thinking.type="enabled"` from the client still
        // opts in. See api.rs for rationale.
        if tools_active && !state.behavior.thinking_in_tools && !explicit {
            false
        } else {
            base
        }
    };

    struct CountMsgEntry {
        role: String,
        content: String,
        tool_calls: Option<Vec<serde_json::Value>>,
    }

    let mut messages: Vec<CountMsgEntry> = Vec::with_capacity(req.messages.len() + 1);
    if let Some(ref sys) = req.system {
        let sys_text = match sys {
            SystemContent::Blocks(blocks) => blocks
                .iter()
                .filter(|b| {
                    b.block_type == "text"
                        && !b.text.as_deref().unwrap_or("").starts_with("x-anthropic-")
                })
                .filter_map(|b| b.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
            SystemContent::Text(s) => s.clone(),
        };
        messages.push(CountMsgEntry {
            role: "system".into(),
            content: sys_text,
            tool_calls: None,
        });
    }
    for m in &req.messages {
        let role = match m.role.as_str() {
            "user" => "user",
            "assistant" => "assistant",
            _ => "user",
        };
        let (mut text, incoming_tool_calls) = flatten_content(&m.content);

        // Mirror the same transformations as the real /v1/messages path
        // so the token count matches what actually gets sent to the model.
        if role == "assistant" && state.tokenizer.supports_thinking() {
            text = format!("<think>\n\n</think>\n\n{text}");
        }
        let tool_calls_json =
            if tools_active && role == "assistant" && !incoming_tool_calls.is_empty() {
                let parsed: Vec<serde_json::Value> = incoming_tool_calls
                    .iter()
                    .map(|tc| {
                        let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::Value::Object(Default::default()));
                        serde_json::json!({
                            "id": tc.id.as_deref().unwrap_or(""),
                            "type": "function",
                            "function": { "name": tc.function.name, "arguments": args }
                        })
                    })
                    .collect();
                Some(parsed)
            } else {
                None
            };
        if tools_active
            && role == "user"
            && let AnthropicContent::Blocks(blocks) = &m.content
        {
            let has_tool_result = blocks
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolResult { .. }));
            if has_tool_result {
                for block in blocks {
                    match block {
                        ContentBlock::ToolResult { content, .. } => {
                            let result_text =
                                content.as_ref().map(|c| c.to_text()).unwrap_or_default();
                            messages.push(CountMsgEntry {
                                role: "tool".into(),
                                content: result_text,
                                tool_calls: None,
                            });
                        }
                        ContentBlock::Text { text: t } if !t.is_empty() => {
                            messages.push(CountMsgEntry {
                                role: "user".into(),
                                content: t.clone(),
                                tool_calls: None,
                            });
                        }
                        _ => {}
                    }
                }
                continue;
            }
        }

        messages.push(CountMsgEntry {
            role: role.to_string(),
            content: text,
            tool_calls: tool_calls_json,
        });
    }

    // Build JSON messages for token counting (mirrors real /v1/messages path).
    let json_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            let mut msg = serde_json::json!({"role": m.role, "content": m.content});
            if let Some(ref tcs) = m.tool_calls {
                msg["tool_calls"] = serde_json::Value::Array(tcs.clone());
            }
            msg
        })
        .collect();
    let jinja_tools: Option<Vec<serde_json::Value>> = if tools_active {
        req.tools.as_ref().map(|ts| {
            let oai = convert_tools(ts);
            oai.iter().map(|t| serde_json::json!({
                "type": "function",
                "function": { "name": t.function.name, "description": t.function.description, "parameters": t.function.parameters }
            })).collect()
        })
    } else {
        None
    };

    let input_tokens = match state.tokenizer.apply_chat_template_jinja(
        &json_messages,
        jinja_tools.as_deref(),
        enable_thinking,
        state.behavior.disable_tool_steering,
    ) {
        Ok(t) => t.len(),
        Err(_) => 0,
    };

    let body = serde_json::json!({
        "input_tokens": input_tokens
    });
    Json(body).into_response()
}
