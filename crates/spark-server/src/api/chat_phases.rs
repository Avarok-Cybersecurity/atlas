// SPDX-License-Identifier: AGPL-3.0-only

//! Helper functions extracted from `chat::chat_completions_inner`. These
//! are the F1..F60 cross-turn agentic guards — each one inspects and
//! mutates `req.messages` (and occasionally returns a flag the caller
//! uses for log-line correlation). Keeping them in a sibling file makes
//! `chat.rs` fit the 500-LoC cap and gives each guard a clear top-level
//! seam for future testing.

use axum::http::StatusCode;
use axum::response::Response;

use crate::ir::ChatRequest;

use super::compact::{openai_error_response, openai_error_response_with_param};

/// Validate the input contract on the IR envelope: messages length,
/// max_tokens > 0, temperature/top_p ranges, tool_choice mode/required
/// compatibility. Returns `Err(Response)` for fail-fast 400 paths so
/// the caller can `?` directly into a Response.
#[allow(clippy::result_large_err)]
pub(super) fn validate_input(req: &ChatRequest) -> Result<(), Response> {
    if req.messages.is_empty() {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "messages must contain at least one message".into(),
            Some("messages"),
            None,
        ));
    }
    if req.messages.len() > 2048 {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "messages array exceeds maximum length (2048)".into(),
            Some("messages"),
            None,
        ));
    }
    if let Some(t) = req.sampling.temperature
        && (!(0.0..=2.0).contains(&t))
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "temperature must be between 0 and 2".into(),
            Some("temperature"),
            None,
        ));
    }
    if let Some(p) = req.sampling.top_p
        && (p <= 0.0 || p > 1.0)
    {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "top_p must be between 0 (exclusive) and 1".into(),
            Some("top_p"),
            None,
        ));
    }
    if req.max_tokens == 0 {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            "max_tokens must be at least 1".into(),
            Some("max_tokens"),
            None,
        ));
    }
    // OpenAI spec bounds n to 1-128; an unbounded n would drive both an
    // attacker-controlled allocation (Vec::with_capacity(n)) and an
    // unbounded sequential-inference loop in the blocking path. Fail fast
    // per spec — mirrors the /v1/completions guard.
    if req.n == 0 || req.n > 128 {
        return Err(openai_error_response_with_param(
            StatusCode::BAD_REQUEST,
            format!("n must be between 1 and 128, got {}", req.n),
            Some("n"),
            None,
        ));
    }
    if let Some(crate::tool_parser::ToolChoice::Mode(ref s)) = req.tool_choice {
        if !["auto", "none", "required"].contains(&s.as_str()) {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "Invalid tool_choice value: '{s}'. Must be 'auto', 'none', 'required', or a function object."
                ),
            ));
        }
        if s == "required" && req.tools.is_empty() {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                "tool_choice is 'required' but no tools were provided".into(),
            ));
        }
    }
    Ok(())
}
