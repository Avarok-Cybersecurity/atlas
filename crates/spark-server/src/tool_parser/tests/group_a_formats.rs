// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

//! The non-Hermes halves of the group-A backfill: MiniMax's XML envelope and
//! Qwen3-Coder's block form. Split out of `group_a.rs` when reviving these
//! tests pushed that file past the 500-LoC cap — the seam is the one the file
//! already had, one section per tool-call format, so nothing was regrouped to
//! make the numbers work.

use super::super::*;

use super::*;

// ── MiniMax XML format ──

#[test]
fn parse_minimax_xml_single_param() {
    let input = "<minimax:tool_call>\n\
            <invoke name=\"get_weather\">\n\
            <parameter name=\"location\">Paris</parameter>\n\
            </invoke>\n\
            </minimax:tool_call>";
    let (content, calls) = parse_tool_calls(input);
    assert!(
        content.is_none(),
        "expected no leading content, got {content:?}"
    );
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Paris");
}

#[test]
fn parse_minimax_xml_multiple_params() {
    let input = "<minimax:tool_call>\n\
            <invoke name=\"search\">\n\
            <parameter name=\"query\">rust async</parameter>\n\
            <parameter name=\"limit\">10</parameter>\n\
            </invoke>\n\
            </minimax:tool_call>";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["query"], "rust async");
    assert_eq!(args["limit"], "10");
}

#[test]
fn parse_minimax_xml_with_content_prefix() {
    let input = "Let me check. <minimax:tool_call>\n\
            <invoke name=\"ls\">\n\
            <parameter name=\"path\">/tmp</parameter>\n\
            </invoke>\n\
            </minimax:tool_call>";
    let (content, calls) = parse_tool_calls(input);
    assert_eq!(content.as_deref(), Some("Let me check."));
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "ls");
}

#[test]
fn minimax_xml_format_tool_calls_roundtrip() {
    let parser = MinimaxXmlParser;
    let call = IncomingToolCall {
        id: None,
        function: IncomingFunction {
            name: "get_weather".into(),
            arguments: "{\"location\":\"Tokyo\"}".into(),
        },
    };
    let formatted = parser.format_tool_calls(&[call]);
    let (_, parsed) = parse_tool_calls(&formatted);
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&parsed[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Tokyo");
}

// ── Qwen3-Coder format ──

#[test]
fn parse_qwen3_coder_empty_body_then_backfill() {
    // Repro of OpenClaw 2026.5.7 + Qwen3.6-35B-A3B-NVFP4 + 21 tools
    // multi-turn agentic regression (issue #40 / Discord #bugs
    // 2026-05-08 universe06608): the model emits the `exec`
    // function with NO `<parameter=>` blocks under long-context
    // tool-saturation pressure. The parser correctly returns
    // arguments=`{}`. The streaming path (path B in
    // chat_stream/tool_handlers.rs) was emitting that `{}` directly
    // to the client without running backfill_required_params, so
    // tools that declare `required: [command]` reached OpenClaw as
    // bare `{}` and were rejected ("must have required properties
    // command"). The non-streaming path always ran backfill, so the
    // two code paths diverged.
    //
    // This test verifies the recovery semantics: parse → empty
    // args → backfill adds the required string field with empty
    // value (mirroring path A). The chat_stream::tool_handlers fix
    // calls this same chain inside handle_tool_call_delta so
    // streaming behaviour matches.
    //
    // The validator then REJECTS the backfilled-empty `exec`: the
    // SHELL_FAMILY rule in validation.rs mirrors F78 for shell
    // tools, because opencode's bash handler answers an empty
    // command with "The argument 'file' cannot be empty" and the
    // model burns to max_tokens retrying it. Rejecting turns the
    // call into a no-op so the reply falls through to text.
    // (This assertion once expected `is_ok()` on the theory that
    // only WRITE_FAMILY rejects empty values — that predates
    // SHELL_FAMILY, which covers `exec`.)
    let input = "<tool_call>\n\
            <function=exec>\n\
            </function>\n\
            </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    assert_eq!(
        calls.len(),
        1,
        "parser must yield the named call even with no params"
    );
    assert_eq!(calls[0].function.name, "exec");
    assert_eq!(
        calls[0].function.arguments, "{}",
        "no params → empty JSON object"
    );

    let tool = ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "exec".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            })),
        },
    };
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["command"], "",
        "backfill must add the required string key with an empty default"
    );
    let err = validate_single_tool_call(&calls[0], std::slice::from_ref(&tool))
        .expect_err("SHELL_FAMILY rejects `exec` with an empty command");
    assert!(
        err.contains("non-empty 'command'"),
        "rejection must name the offending key so the model can recover; got {err:?}"
    );
}

#[test]
fn parse_qwen3_coder_single_param() {
    let input = "<tool_call>\n\
            <function=get_weather>\n\
            <parameter=location>\nParis\n</parameter>\n\
            </function>\n\
            </tool_call>";
    let (c, calls) = parse_tool_calls(input);
    assert!(c.is_none());
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Paris");
}
