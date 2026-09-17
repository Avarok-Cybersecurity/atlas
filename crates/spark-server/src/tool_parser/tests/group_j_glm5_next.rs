// SPDX-License-Identifier: AGPL-3.0-only
//!
//! GLM-5.3-Flash (`model_type = "glm5_next"`) tool calling.
//!
//! The contract under test is quoted verbatim from the checkpoint's own
//! `chat_template.jinja`, which tells the model exactly what to emit:
//!
//! ```text
//! For each function call, output the function name and arguments within the
//! following XML format:
//! <tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>
//! ```
//!
//! That is byte-for-byte the `poolside_v1` envelope, so GLM is mapped onto that
//! parser in `tool_defaults.toml` rather than getting a duplicate
//! implementation. These tests pin the mapping AND the wire contract, so if a
//! future GLM revision diverges the mapping fails here instead of at serve time.
//!
//! 🔴 The bug these guard: with no `[model_type]` entry,
//! `resolve_tool_call_parser` returns `None`, which makes `tools_active` false
//! in `api/chat/prepare.rs` — Atlas then silently DROPS the caller's `tools`
//! before rendering the chat template. The model is never told the tools exist
//! and replies "I don't actually have any tools available in this
//! conversation." No parse error, no warning, a 200, and a plausible answer.

use super::super::*;

/// The literal format line from GLM-5.3-Flash's `chat_template.jinja`, with the
/// placeholders filled in. If this stops parsing, GLM tool calling is broken.
const GLM_TWO_ARG: &str = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value><arg_key>unit</arg_key><arg_value>celsius</arg_value></tool_call>";

fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_weather".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string"},
                    "days": {"type": "integer"},
                    "verbose": {"type": "boolean"}
                },
                "required": ["location"]
            })),
        },
    }
}

fn args_of(call: &ToolCall) -> serde_json::Value {
    serde_json::from_str(&call.function.arguments).expect("arguments are valid JSON")
}

/// The mapping itself. This is the entire fix — without the entry, every
/// assertion below is unreachable in production because `tools` never ship.
#[test]
fn glm5_next_is_registered_to_the_poolside_v1_wire_format() {
    let defaults: toml::Value =
        toml::from_str(include_str!("../../../tool_defaults.toml")).expect("tool_defaults parses");
    let fmt_str = defaults
        .get("model_type")
        .and_then(|t| t.get("glm5_next"))
        .and_then(|s| s.as_str())
        .expect("tool_defaults [model_type] must register glm5_next");
    assert_eq!(
        fmt_str, "poolside_v1",
        "GLM-5.3 emits the poolside_v1 envelope; see chat_template.jinja"
    );
    let fmt: ToolCallFormat = fmt_str.parse().expect("glm5_next tool format parses");
    assert_eq!(fmt.into_parser().name(), "poolside_v1");
}

/// The exact string from the template's format line round-trips to structure.
#[test]
fn glm_template_format_line_parses_to_a_structured_call() {
    let (content, calls) = parse_tool_calls_promoting_bare_names(GLM_TWO_ARG);
    assert_eq!(calls.len(), 1, "one call from the template's own example");
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(
        args_of(&calls[0]),
        serde_json::json!({"location": "Paris", "unit": "celsius"}),
        "both arg_key/arg_value pairs, in order, as strings"
    );
    assert!(
        content.as_deref().unwrap_or("").trim().is_empty(),
        "the envelope is entirely consumed, leaving no stray content"
    );
}

/// Prose before the call is normal GLM output and must survive as content
/// while the call is still extracted.
#[test]
fn leading_prose_is_preserved_as_content() {
    let text = format!("Let me check that for you.{GLM_TWO_ARG}");
    let (content, calls) = parse_tool_calls_promoting_bare_names(&text);
    assert_eq!(calls.len(), 1);
    assert_eq!(
        content.as_deref().unwrap().trim(),
        "Let me check that for you."
    );
}

/// GLM emits reasoning in `<think>` tags; a call deliberated inside thinking is
/// NOT an invocation, and only the post-`</think>` call counts.
#[test]
fn calls_inside_thinking_are_not_invocations() {
    let text = format!(
        "<think>I could call <tool_call>get_weather<arg_key>location</arg_key><arg_value>Berlin</arg_value></tool_call> but Paris was asked.</think>{GLM_TWO_ARG}"
    );
    let (_, calls) = parse_tool_calls_promoting_bare_names(&text);
    assert_eq!(calls.len(), 1, "only the call after </think> is real");
    assert_eq!(args_of(&calls[0])["location"], "Paris");
}

/// A zero-argument call is a bare name inside the envelope — what the template
/// produces when `tc.arguments` is empty. This is why GLM must use the
/// bare-name-promoting entry point, exactly as poolside does.
#[test]
fn zero_argument_call_is_a_bare_name_in_the_envelope() {
    let (_, calls) = parse_tool_calls_promoting_bare_names("<tool_call>get_status</tool_call>");
    assert_eq!(calls.len(), 1, "zero-arg call must not be dropped");
    assert_eq!(calls[0].function.name, "get_status");
    assert_eq!(args_of(&calls[0]), serde_json::json!({}));
}

/// Two calls in one assistant turn — the template's `{% for tc in m.tool_calls %}`
/// loop concatenates envelopes with no separator.
#[test]
fn two_calls_in_one_turn_are_both_extracted() {
    let text = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>Lyon</arg_value></tool_call>";
    let (_, calls) = parse_tool_calls_promoting_bare_names(text);
    assert_eq!(calls.len(), 2);
    assert_eq!(args_of(&calls[0])["location"], "Paris");
    assert_eq!(args_of(&calls[1])["location"], "Lyon");
}

/// `arg_value` is untyped on the wire: every value arrives as text. The schema
/// is what says `days` is an integer and `verbose` a boolean, so the typed
/// coercion must fire or the tool receives `"3"` where it demands `3`.
#[test]
fn untyped_wire_values_are_coerced_to_the_schema() {
    let text = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value><arg_key>verbose</arg_key><arg_value>true</arg_value></tool_call>";
    let (_, mut calls) = parse_tool_calls_promoting_bare_names(text);
    assert_eq!(calls.len(), 1);
    assert!(
        PoolsideV1Parser.wants_typed_arguments(),
        "GLM's wire format carries no types; coercion is mandatory"
    );
    coerce_all(&mut calls, &[weather_tool()]);
    let args = args_of(&calls[0]);
    assert_eq!(
        args["location"],
        serde_json::json!("Paris"),
        "stays a string"
    );
    assert_eq!(args["days"], serde_json::json!(3), "integer, not \"3\"");
    assert_eq!(
        args["verbose"],
        serde_json::json!(true),
        "bool, not \"true\""
    );
}

/// Replay parity: what Atlas writes back into assistant history must be what
/// the template itself would have written, or the second turn of every
/// multi-turn tool scenario diverges from training. The template's rule
/// (line 129 of chat_template.jinja) is: a string value is emitted raw, any
/// other value goes through `tojson`.
#[test]
fn formatted_history_matches_the_templates_own_rendering_rule() {
    let calls = vec![IncomingToolCall {
        id: Some("call_1".to_string()),
        function: IncomingFunction {
            name: "get_weather".to_string(),
            arguments: r#"{"location":"Paris","days":3}"#.to_string(),
        },
    }];
    let rendered = PoolsideV1Parser.format_tool_calls(&calls);
    assert_eq!(
        rendered,
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>",
        "string raw, non-string via tojson — matches chat_template.jinja"
    );
    // And it survives a round trip back to the same structure.
    let (_, reparsed) = parse_tool_calls_promoting_bare_names(&rendered);
    assert_eq!(reparsed.len(), 1);
    assert_eq!(reparsed[0].function.name, "get_weather");
    assert_eq!(args_of(&reparsed[0])["location"], "Paris");
}
