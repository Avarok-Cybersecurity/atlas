// SPDX-License-Identifier: AGPL-3.0-only

#[allow(unused_imports)]
use std::sync::Arc;

#[allow(unused_imports)]
use axum::extract::State;
#[allow(unused_imports)]
use axum::extract::rejection::JsonRejection;
#[allow(unused_imports)]
use axum::http::StatusCode;
#[allow(unused_imports)]
use axum::response::sse::{Event, KeepAlive};
#[allow(unused_imports)]
use axum::response::{IntoResponse, Json, Response, Sse};
#[allow(unused_imports)]
use futures::StreamExt;
#[allow(unused_imports)]
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::AppState;
#[allow(unused_imports)]
use crate::openai;
#[allow(unused_imports)]
use crate::tool_parser;

#[allow(unused_imports)]
use super::convert::*;
#[allow(unused_imports)]
use super::handlers_stream::*;
#[allow(unused_imports)]
use super::translate::*;
#[allow(unused_imports)]
use super::translator::*;
#[allow(unused_imports)]
use super::types::*;

pub(super) fn anthropic_error(status: StatusCode, error_type: &str, message: String) -> Response {
    let body = serde_json::json!({
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        }
    });
    (status, Json(body)).into_response()
}

// ── Conversion helpers ──

/// Convert Anthropic tools to OpenAI-compatible tool definitions.
pub(super) fn convert_tools(tools: &[AnthropicTool]) -> Vec<tool_parser::ToolDefinition> {
    tools
        .iter()
        .map(|t| tool_parser::ToolDefinition {
            tool_type: "function".to_string(),
            function: tool_parser::FunctionDefinition {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: Some(t.input_schema.clone()),
            },
        })
        .collect()
}

/// Convert Anthropic tool_choice to OpenAI-compatible tool_choice.
pub(super) fn convert_tool_choice(tc: &AnthropicToolChoice) -> tool_parser::ToolChoice {
    match tc.choice_type.as_str() {
        "any" => tool_parser::ToolChoice::Mode("required".to_string()),
        "auto" => tool_parser::ToolChoice::Mode("auto".to_string()),
        "none" => tool_parser::ToolChoice::Mode("none".to_string()),
        "tool" => {
            if let Some(ref name) = tc.name {
                tool_parser::ToolChoice::Specific {
                    function: tool_parser::ToolChoiceFunction { name: name.clone() },
                }
            } else {
                tool_parser::ToolChoice::Mode("auto".to_string())
            }
        }
        _ => tool_parser::ToolChoice::Mode("auto".to_string()),
    }
}

/// Convert Anthropic finish reason to stop_reason string.
pub(super) fn convert_stop_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "stop" => "end_turn",
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}
