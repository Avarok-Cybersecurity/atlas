// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::super::*;

// ────────────────────────────────────────────────────────────────────────
// Backfill / validation repair for delegation tools (opencode / Claude Code
// `task`). These exercise `backfill_required_params` +
// `validate_single_tool_call` directly. They live here (not in `group_a`,
// the historical backfill home) because `group_a` is disabled (commented
// out of `tests.rs` pending the `parse_minimax_xml_call` cleanup). A new
// group keeps the regression COMPILED and RUNNING.
// ────────────────────────────────────────────────────────────────────────

#[test]
fn backfill_fills_omitted_subagent_type_with_valid_agent() {
    // Regression for the opencode `task` failure: the model emits `task`
    // with description + prompt but OMITS the required free-form
    // `subagent_type`. The missing-required backfill used to insert "",
    // and opencode rejected it with "Unknown agent type:  is not a valid
    // agent type" → identical retry → delegation abandoned. Backfill must
    // instead pick a VALID agent name parsed from the tool description.
    let input = "<tool_call>\n\
        <function=task>\n\
        <parameter=description>\nFind hot spots\n</parameter>\n\
        <parameter=prompt>\nexplore the repo\n</parameter>\n\
        </function>\n\
        </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);

    let tool = ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "task".to_string(),
            description: Some(
                "Launch a new agent to handle complex tasks.\n\
                 Available agent types and the tools they have access to:\n\
                 - explore: Fast agent specialized for exploring codebases.\n\
                 - general: General-purpose agent for researching questions."
                    .to_string(),
            ),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "prompt": {"type": "string"},
                    "subagent_type": {"type": "string"}
                },
                "required": ["description", "prompt", "subagent_type"]
            })),
        },
    };
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["subagent_type"], "general",
        "omitted subagent_type must be filled with a valid agent, not empty"
    );
    assert!(
        validate_single_tool_call(&calls[0], std::slice::from_ref(&tool)).is_ok(),
        "the repaired call must validate"
    );
}

#[test]
fn backfill_subagent_type_prefers_general_purpose_variant() {
    // Claude Code names its general agent `general-purpose`; the
    // "contains general" preference must pick it over earlier-listed
    // specialized agents even when the call body is completely empty.
    let input = "<tool_call>\n\
        <function=Task>\n\
        </function>\n\
        </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.arguments, "{}");

    let tool = ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "Task".to_string(),
            description: Some(
                "Launch a new agent.\n\
                 - claude-code-guide: Use this agent for Claude Code questions.\n\
                 - Explore: Fast codebase exploration agent.\n\
                 - general-purpose: General-purpose agent for complex questions.\n\
                 - statusline-setup: Configure the status line."
                    .to_string(),
            ),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"subagent_type": {"type": "string"}},
                "required": ["subagent_type"]
            })),
        },
    };
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["subagent_type"], "general-purpose");
}

// ────────────────────────────────────────────────────────────────────────
// Phantom namespaced-name guard (PR #171 follow-up). The bare-identifier
// scanners admit `:` so `ns:tool{...}` parses, but prose like `json:{...}`
// used to become a phantom call literally named `json:` because
// `normalize_tool_name` only strips a namespace with a non-empty tail.
// Names that keep a `:` after normalization must be rejected WITHOUT
// consuming the text, so later fallbacks (or content) still see it.
// ────────────────────────────────────────────────────────────────────────

#[test]
fn phantom_json_colon_prose_is_not_a_tool_call() {
    let input = r#"Here is the payload as json:{"a":1}"#;
    let (content, calls) = parse_tool_calls(input);
    assert!(
        calls.is_empty(),
        "prose `json:{{...}}` must not become a phantom call — got {calls:#?}"
    );
    let content = content.expect("original text must be preserved as content");
    assert!(
        content.contains(r#"json:{"a":1}"#),
        "content must keep the unconsumed text, got: {content}"
    );
}

#[test]
fn namespaced_bare_identifier_still_parses_as_tool() {
    let input = r#"ns:tool{"query":"rust"}"#;
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "tool");
}

#[test]
fn tool_call_colon_prefix_falls_through_to_embedded_json_name() {
    // The `tool_call:` prefix must NOT be scanned as a call named
    // `tool_call:` — the rejected candidate falls through to the JSON
    // fallback, which extracts the embedded `"name"` correctly.
    let input = r#"tool_call:{"name":"get_weather","arguments":{"city":"Paris"}}"#;
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1, "expected one call — got {calls:#?}");
    assert_eq!(calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Paris");
}
