// SPDX-License-Identifier: AGPL-3.0-only

use super::super::*;

fn write_file_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "write_file".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            })),
        },
    }
}

fn call(arguments: serde_json::Value) -> ToolCall {
    ToolCall {
        id: "call_test".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "write_file".to_string(),
            arguments: arguments.to_string(),
        },
    }
}

#[test]
fn poolside_parser_does_not_reinject_tool_prompt() {
    let prompt = PoolsideV1Parser.system_prompt(
        &[write_file_tool()],
        &ToolChoice::Mode("auto".to_string()),
        &crate::tool_parser::PromptLevers::OFF,
    );

    assert!(prompt.is_empty());
}

#[test]
fn missing_write_path_is_not_executable_after_backfill() {
    let mut calls = vec![call(serde_json::json!({"content": "hello"}))];
    let tools = [write_file_tool()];
    backfill_required_params(&mut calls, &tools);
    let validated = validate_tool_calls(calls, &tools);

    assert!(validated.valid.is_empty());
    assert_eq!(validated.errors.len(), 1);
    assert!(validated.errors[0].contains("path"));
}

#[test]
fn empty_write_path_is_not_executable() {
    let validated = validate_tool_calls(
        vec![call(serde_json::json!({
            "path": "   ",
            "content": "hello"
        }))],
        &[write_file_tool()],
    );

    assert!(validated.valid.is_empty());
    assert_eq!(validated.errors.len(), 1);
    assert!(validated.errors[0].contains("path"));
}

#[test]
fn empty_content_remains_schema_valid() {
    let validated = validate_tool_calls(
        vec![call(serde_json::json!({
            "path": "/tmp/a",
            "content": ""
        }))],
        &[write_file_tool()],
    );

    assert_eq!(validated.valid.len(), 1);
    assert!(validated.errors.is_empty());
}
