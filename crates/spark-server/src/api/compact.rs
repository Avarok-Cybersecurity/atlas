// SPDX-License-Identifier: AGPL-3.0-only

//! Progressive context compaction + shared OpenAI-compatible error helpers
//! (extracted from `api.rs`, lines 20-225).

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

/// OpenAI-compatible JSON error response.
/// Coding agents (OpenCode, Cline, nanobot) expect this exact structure.
/// Progressive context compaction (5 stages, per arXiv:2603.05344 OpenDev).
///
/// Uses actual prompt_tokens (from trial tokenization) to select the
/// appropriate compaction stage. Always keeps system message + last N messages.
///
/// Stage 2 (80%): Truncate middle tool responses to first+last 3 lines
/// Stage 3 (85%): Replace middle tool responses with `"[truncated]"` pointers
/// Stage 4 (90%): Drop oldest middle message pairs (keep last 6)
/// Stage 5 (95%): Trim system prompt + keep only last 4 messages
pub fn compact_messages(
    msgs: &[serde_json::Value],
    prompt_tokens: usize,
    max_seq_len: usize,
) -> Vec<serde_json::Value> {
    let ratio = prompt_tokens as f32 / max_seq_len as f32;

    let (result, stage) = if ratio < 0.80 {
        // Stage 2: truncate long tool responses in middle messages
        let keep_tail = 6.min(msgs.len());
        let tail_start = msgs.len().saturating_sub(keep_tail);
        let mut out = Vec::with_capacity(msgs.len());
        for (i, msg) in msgs.iter().enumerate() {
            if i == 0 || i >= tail_start {
                out.push(msg.clone());
            } else {
                let content = msg["content"].as_str().unwrap_or("");
                if content.len() > 500 {
                    let lines: Vec<&str> = content.lines().collect();
                    let truncated = if lines.len() > 6 {
                        format!(
                            "{}\n... [{} lines truncated] ...\n{}",
                            lines[..3].join("\n"),
                            lines.len() - 6,
                            lines[lines.len() - 3..].join("\n")
                        )
                    } else {
                        content.to_string()
                    };
                    let mut m = msg.clone();
                    m["content"] = serde_json::Value::String(truncated);
                    out.push(m);
                } else {
                    out.push(msg.clone());
                }
            }
        }
        (out, 2)
    } else if ratio < 0.85 {
        // Stage 3: mask observations — replace tool response content with pointer
        let keep_tail = 6.min(msgs.len());
        let tail_start = msgs.len().saturating_sub(keep_tail);
        let mut out = Vec::with_capacity(msgs.len());
        for (i, msg) in msgs.iter().enumerate() {
            if i == 0 || i >= tail_start {
                out.push(msg.clone());
            } else {
                let role = msg["role"].as_str().unwrap_or("");
                let content = msg["content"].as_str().unwrap_or("");
                if (role == "tool" || role == "user") && content.len() > 200 {
                    let mut m = msg.clone();
                    m["content"] = serde_json::Value::String(format!(
                        "[Tool output truncated — {} chars]",
                        content.len()
                    ));
                    out.push(m);
                } else {
                    out.push(msg.clone());
                }
            }
        }
        (out, 3)
    } else if ratio < 0.95 {
        // Stage 4: drop oldest middle messages, keep system + last 6
        // Ensure tail starts on a user message (not tool/assistant) to avoid
        // Jinja "No user query found" error and orphaned tool_response messages.
        let keep_tail = 6.min(msgs.len().saturating_sub(1));
        let mut tail_start = msgs.len().saturating_sub(keep_tail);
        // Walk backward to find a real user message in the tail
        let has_user_query = (tail_start..msgs.len()).any(|i| {
            let role = msgs[i]["role"].as_str().unwrap_or("");
            let content = msgs[i]["content"].as_str().unwrap_or("");
            role == "user" && !content.starts_with("<tool_response>")
        });
        if !has_user_query {
            // Expand tail backwards until we find a real user message
            while tail_start > 1 {
                tail_start -= 1;
                let role = msgs[tail_start]["role"].as_str().unwrap_or("");
                let content = msgs[tail_start]["content"].as_str().unwrap_or("");
                if role == "user" && !content.starts_with("<tool_response>") {
                    break;
                }
            }
        }
        // Don't start tail on a "tool" message — it needs a preceding assistant
        while tail_start < msgs.len() && msgs[tail_start]["role"].as_str() == Some("tool") {
            tail_start += 1;
        }
        let mut out = Vec::with_capacity(msgs.len() - tail_start + 1);
        out.push(msgs[0].clone()); // system
        for msg in &msgs[tail_start..] {
            out.push(msg.clone());
        }
        (out, 4)
    } else {
        // Stage 5: trim system prompt + keep only last 4 messages
        // Same safety: ensure a real user message is present and no orphaned tool messages.
        let keep_tail = 4.min(msgs.len().saturating_sub(1));
        let mut tail_start = msgs.len().saturating_sub(keep_tail);
        let has_user_query = (tail_start..msgs.len()).any(|i| {
            let role = msgs[i]["role"].as_str().unwrap_or("");
            let content = msgs[i]["content"].as_str().unwrap_or("");
            role == "user" && !content.starts_with("<tool_response>")
        });
        if !has_user_query {
            while tail_start > 1 {
                tail_start -= 1;
                let role = msgs[tail_start]["role"].as_str().unwrap_or("");
                let content = msgs[tail_start]["content"].as_str().unwrap_or("");
                if role == "user" && !content.starts_with("<tool_response>") {
                    break;
                }
            }
        }
        while tail_start < msgs.len() && msgs[tail_start]["role"].as_str() == Some("tool") {
            tail_start += 1;
        }
        let mut out = Vec::with_capacity(msgs.len() - tail_start + 1);
        // Trim system prompt: keep first ~2000 + last ~1000 chars.
        // Use floor/ceil_char_boundary to avoid panics on multi-byte UTF-8.
        let sys_content = msgs[0]["content"].as_str().unwrap_or("");
        let trimmed_sys = if sys_content.len() > 4000 {
            let head_end = sys_content.floor_char_boundary(2000);
            let tail_start = sys_content.ceil_char_boundary(sys_content.len().saturating_sub(1000));
            format!(
                "{}...\n[System prompt truncated — {} chars removed]\n...{}",
                &sys_content[..head_end],
                sys_content.len() - head_end - (sys_content.len() - tail_start),
                &sys_content[tail_start..]
            )
        } else {
            sys_content.to_string()
        };
        let mut sys = msgs[0].clone();
        sys["content"] = serde_json::Value::String(trimmed_sys);
        out.push(sys);
        for msg in &msgs[tail_start..] {
            out.push(msg.clone());
        }
        (out, 5)
    };

    tracing::info!(
        "Auto-compact stage {}: {} → {} messages (was {:.0}% of {})",
        stage,
        msgs.len(),
        result.len(),
        ratio * 100.0,
        max_seq_len,
    );
    result
}

pub(super) fn openai_error_response(status: StatusCode, message: String) -> Response {
    openai_error_response_with_param(status, message, None, None)
}

/// OpenAI-compatible error with optional `param` (field path like
/// `messages[0].role`) and `code` (e.g. `"context_length_exceeded"`).
pub(super) fn openai_error_response_with_param(
    status: StatusCode,
    message: String,
    param: Option<&str>,
    code: Option<&str>,
) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": match status {
                StatusCode::BAD_REQUEST => "invalid_request_error",
                StatusCode::UNAUTHORIZED => "authentication_error",
                StatusCode::FORBIDDEN => "permission_error",
                StatusCode::NOT_FOUND => "not_found_error",
                StatusCode::TOO_MANY_REQUESTS => "rate_limit_exceeded",
                StatusCode::SERVICE_UNAVAILABLE => "server_error",
                _ => "server_error",
            },
            "param": param,
            "code": code,
        }
    });
    (status, Json(body)).into_response()
}
