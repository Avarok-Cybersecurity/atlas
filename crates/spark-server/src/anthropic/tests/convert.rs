// SPDX-License-Identifier: AGPL-3.0-only
//
// Wire-type deserialization and the small conversion helpers that sit
// beside the `MessagesRequest -> ir::ChatRequest` adapter.
//
// These replace the `types_convert` suite, which tested a
// `flatten_content` / `convert_tools` / `convert_tool_choice` free-function
// layer that the IR migration replaced with `From` impls. The behaviours
// below still exist; the entry points changed.

use crate::tool_parser;

use super::super::helpers::convert_stop_reason;
use super::super::types::{
    AnthropicTool, AnthropicToolChoice, ContentBlock, MessagesRequest, SystemContent,
};

// ── finish reason → Anthropic stop_reason ──

#[test]
fn stop_reason_maps_the_whole_wire_vocabulary() {
    assert_eq!(convert_stop_reason("stop"), "end_turn");
    assert_eq!(convert_stop_reason("tool_calls"), "tool_use");
    assert_eq!(convert_stop_reason("length"), "max_tokens");
    // Safety-filtered output has a dedicated Anthropic reason (2025-05
    // API); clients branch on it to avoid retrying the same prompt.
    assert_eq!(convert_stop_reason("content_filter"), "refusal");
    // A deadline cut is TRUNCATION. Anthropic has no deadline reason and
    // the default "end_turn" would tell the client the turn completed —
    // the exact silent-truncation bug this arm exists to prevent.
    assert_eq!(
        convert_stop_reason(crate::ir::FINISH_REASON_TIMEOUT),
        "max_tokens"
    );
    assert_eq!(convert_stop_reason("something_new"), "end_turn");
}

#[test]
fn stop_reason_covers_every_finish_reason_wire_string() {
    // The streaming translator feeds `FinishReason::as_wire()` straight
    // into `convert_stop_reason`. Any variant whose wire string falls
    // through to the `_` arm silently becomes "end_turn", so pin the
    // pairing rather than the arms in isolation.
    use crate::ir::FinishReason;
    for (reason, expected) in [
        (FinishReason::Stop, "end_turn"),
        (FinishReason::Length, "max_tokens"),
        (FinishReason::ToolCalls, "tool_use"),
        (FinishReason::ContentFilter, "refusal"),
        (
            FinishReason::Other(crate::ir::FINISH_REASON_TIMEOUT.to_string()),
            "max_tokens",
        ),
    ] {
        assert_eq!(
            convert_stop_reason(reason.as_wire()),
            expected,
            "finish reason {reason:?}"
        );
    }
}

// ── tool_choice ──

fn choice(kind: &str, name: Option<&str>) -> AnthropicToolChoice {
    AnthropicToolChoice {
        choice_type: kind.to_string(),
        name: name.map(str::to_string),
    }
}

#[test]
fn tool_choice_modes_map_to_openai_vocabulary() {
    let cases = [("any", "required"), ("auto", "auto"), ("none", "none")];
    for (anthropic, openai) in cases {
        match tool_parser::ToolChoice::from(&choice(anthropic, None)) {
            tool_parser::ToolChoice::Mode(m) => assert_eq!(m, openai, "{anthropic}"),
            other => panic!("{anthropic}: expected Mode, got {other:?}"),
        }
    }
}

#[test]
fn tool_choice_specific_tool_carries_the_name() {
    match tool_parser::ToolChoice::from(&choice("tool", Some("get_weather"))) {
        tool_parser::ToolChoice::Specific { function } => {
            assert_eq!(function.name, "get_weather");
        }
        other => panic!("expected Specific, got {other:?}"),
    }
}

#[test]
fn tool_choice_tool_without_a_name_degrades_to_auto() {
    // `{"type":"tool"}` with no `name` is malformed. Falling back to
    // "auto" keeps the request servable; forcing a nameless tool would
    // hand the grammar an empty name and fail the whole turn.
    match tool_parser::ToolChoice::from(&choice("tool", None)) {
        tool_parser::ToolChoice::Mode(m) => assert_eq!(m, "auto"),
        other => panic!("expected Mode(auto), got {other:?}"),
    }
    // Unknown types take the same path.
    match tool_parser::ToolChoice::from(&choice("wat", None)) {
        tool_parser::ToolChoice::Mode(m) => assert_eq!(m, "auto"),
        other => panic!("expected Mode(auto), got {other:?}"),
    }
}

// ── tool definitions ──

#[test]
fn tool_definition_carries_description_and_schema() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    });
    let tool = AnthropicTool {
        name: "get_weather".to_string(),
        description: Some("Get weather".to_string()),
        input_schema: schema.clone(),
    };
    let def = tool_parser::ToolDefinition::from(&tool);
    assert_eq!(def.tool_type, "function");
    assert_eq!(def.function.name, "get_weather");
    assert_eq!(def.function.description.as_deref(), Some("Get weather"));
    // The schema travels verbatim — the prompt renderer serializes it
    // into the <tools> block, so any reshaping here changes the prefix.
    assert_eq!(def.function.parameters.as_ref(), Some(&schema));
}

#[test]
fn tool_definition_without_a_description_keeps_none() {
    let tool = AnthropicTool {
        name: "x".to_string(),
        description: None,
        input_schema: serde_json::json!({"type": "object"}),
    };
    let def = tool_parser::ToolDefinition::from(&tool);
    assert!(def.function.description.is_none());
}

// ── wire deserialization ──

#[test]
fn request_deserializes_string_system_and_defaults() {
    let req: MessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "qwen3-80b",
        "max_tokens": 1024,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "Hello!"}],
    }))
    .expect("MessagesRequest");
    assert_eq!(req.model, "qwen3-80b");
    assert_eq!(req.max_tokens, 1024);
    assert!(matches!(req.system, Some(SystemContent::Text(ref s)) if s == "You are helpful."));
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.messages[0].role, "user");
    // Everything optional is absent, not defaulted to a sampling value —
    // the served preset owns those knobs when the client stays quiet.
    assert!(req.temperature.is_none());
    assert!(req.top_k.is_none());
    assert!(req.top_p.is_none());
    assert!(req.tools.is_none());
    assert!(req.tool_choice.is_none());
    assert!(req.thinking.is_none());
    assert!(req.stop_sequences.is_empty());
    assert!(!req.stream);
}

#[test]
fn thinking_config_deserializes_type_and_budget() {
    let req: MessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "qwen3-80b",
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 4096},
        "messages": [{"role": "user", "content": "Think hard."}],
    }))
    .expect("MessagesRequest");
    let t = req.thinking.expect("thinking config");
    assert_eq!(t.thinking_type, "enabled");
    assert_eq!(t.budget_tokens, Some(4096));
}

#[test]
fn tool_result_is_error_defaults_to_none_when_absent() {
    // The success path is "field absent", not "field false" — a missing
    // `is_error` must not deserialize into an error-flagged result.
    let with_err: ContentBlock = serde_json::from_str(
        r#"{"type":"tool_result","tool_use_id":"x","content":"oops","is_error":true}"#,
    )
    .expect("tool_result with is_error");
    match with_err {
        ContentBlock::ToolResult { is_error, .. } => assert_eq!(is_error, Some(true)),
        other => panic!("wrong variant: {other:?}"),
    }

    let no_field: ContentBlock =
        serde_json::from_str(r#"{"type":"tool_result","tool_use_id":"x","content":"ok"}"#)
            .expect("tool_result without is_error");
    match no_field {
        ContentBlock::ToolResult { is_error, .. } => assert_eq!(is_error, None),
        other => panic!("wrong variant: {other:?}"),
    }
}

#[test]
fn unrecognised_block_types_deserialize_to_unknown_instead_of_failing() {
    // Anthropic ships new block types (`redacted_thinking`,
    // `server_tool_use`, …) without a version bump. Rejecting the body
    // would 400 a whole conversation over one block Atlas ignores.
    let block: ContentBlock = serde_json::from_str(r#"{"type":"redacted_thinking","data":"AAAA"}"#)
        .expect("unknown block type must parse");
    assert!(matches!(block, ContentBlock::Unknown));
}
