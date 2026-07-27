// SPDX-License-Identifier: AGPL-3.0-only

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use crate::tool_parser;

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

impl From<&AnthropicTool> for tool_parser::ToolDefinition {
    /// Convert an Anthropic tool to an OpenAI-compatible tool definition.
    fn from(t: &AnthropicTool) -> Self {
        tool_parser::ToolDefinition {
            tool_type: "function".to_string(),
            function: tool_parser::FunctionDefinition {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: Some(t.input_schema.clone()),
            },
        }
    }
}

impl From<&AnthropicToolChoice> for tool_parser::ToolChoice {
    /// Convert an Anthropic tool_choice to an OpenAI-compatible tool_choice.
    fn from(tc: &AnthropicToolChoice) -> Self {
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
}

/// Convert an OpenAI finish reason to Anthropic's stop_reason string.
pub(super) fn convert_stop_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "stop" => "end_turn",
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        // Safety-filtered output maps to Anthropic's dedicated refusal
        // stop reason (2025-05 API), not a normal end_turn — clients
        // branch on this to avoid retrying verbatim.
        "content_filter" => "refusal",
        _ => "end_turn",
    }
}
