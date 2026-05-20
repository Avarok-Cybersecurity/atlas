// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::super::*;

fn make_tool(name: &str, props: serde_json::Value) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: None,
            parameters: Some(serde_json::json!({ "type": "object", "properties": props })),
        },
    }
}

fn make_call(name: &str, args: &str) -> ToolCall {
    ToolCall {
        id: "call_test".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: args.to_string(),
        },
    }
}

// ── type_coerce unit tests ──

#[test]
fn coerce_integer_string() {
    let tools = vec![make_tool(
        "search",
        serde_json::json!({ "limit": { "type": "integer" } }),
    )];
    let mut calls = vec![make_call("search", r#"{"limit":"10"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["limit"], 10);
}

#[test]
fn coerce_number_float() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "score": { "type": "number" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"score":"3.14"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    let got = args["score"].as_f64().unwrap();
    assert!((got - 3.14).abs() < 1e-9, "expected 3.14, got {got}");
}

#[test]
fn coerce_boolean_lower() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "flag": { "type": "boolean" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"flag":"true"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["flag"], true);
}

#[test]
fn coerce_boolean_capitalized() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "flag": { "type": "boolean" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"flag":"False"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["flag"], false);
}

#[test]
fn coerce_array_string() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "items": { "type": "array" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"items":"[1,2,3]"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["items"], serde_json::json!([1, 2, 3]));
}

#[test]
fn coerce_object_string() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "opts": { "type": "object" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"opts":"{\"a\":1}"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["opts"]["a"], 1);
}

#[test]
fn coerce_null_string() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "cursor": { "type": "null" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"cursor":"null"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert!(args["cursor"].is_null());
}

#[test]
fn no_coerce_already_number() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "limit": { "type": "integer" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"limit":42}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["limit"], 42);
}

#[test]
fn no_coerce_unparseable() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "limit": { "type": "integer" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"limit":"notanumber"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["limit"], "notanumber");
}

#[test]
fn empty_arg_preserved() {
    // Empty string with integer schema: can't parse, left as "".
    // Pins the contract: coerce_all doesn't auto-fix absent values.
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "limit": { "type": "integer" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"limit":""}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["limit"], "");
}

#[test]
fn coerce_all_multi_call() {
    let tools = vec![
        make_tool(
            "search",
            serde_json::json!({ "limit": { "type": "integer" } }),
        ),
        make_tool("toggle", serde_json::json!({ "on": { "type": "boolean" } })),
    ];
    let mut calls = vec![
        make_call("search", r#"{"limit":"5"}"#),
        make_call("toggle", r#"{"on":"true"}"#),
    ];
    coerce_all(&mut calls, &tools);
    let a0: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    let a1: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
    assert_eq!(a0["limit"], 5);
    assert_eq!(a1["on"], true);
}

#[test]
fn coerce_ignores_missing_tool() {
    let tools: Vec<ToolDefinition> = vec![];
    let mut calls = vec![make_call("unknown", r#"{"x":"1"}"#)];
    coerce_all(&mut calls, &tools);
    assert_eq!(calls[0].function.arguments, r#"{"x":"1"}"#);
}

// ── Qwen3XmlParser trait tests ──

#[test]
fn qwen3_xml_name() {
    assert_eq!(Qwen3XmlParser.name(), "qwen3_xml");
}

#[test]
fn qwen3_xml_wants_typed() {
    assert!(Qwen3XmlParser.wants_typed_arguments());
}

#[test]
fn qwen3_coder_not_typed() {
    assert!(!Qwen3CoderParser.wants_typed_arguments());
}

#[test]
fn qwen3_xml_has_grammar() {
    assert!(Qwen3XmlParser.has_tool_grammar());
}

#[test]
fn qwen3_xml_system_prompt_contains_markers() {
    let tools: Vec<ToolDefinition> = vec![];
    let tc = ToolChoice::Mode("auto".to_string());
    let prompt = Qwen3XmlParser.system_prompt(&tools, &tc);
    assert!(prompt.contains("<tool_call>"), "missing <tool_call>");
    assert!(prompt.contains("<function="), "missing <function=");
    assert!(prompt.contains("<parameter="), "missing <parameter=");
}

// ── End-to-end: parse + coerce ──

#[test]
fn qwen3_xml_coerced_via_parse_and_coerce_all() {
    // Full non-streaming path: parse raw model output, then coerce types.
    let raw = "<tool_call>\n\
        <function=search>\n\
        <parameter=query>\nrust async\n</parameter>\n\
        <parameter=limit>\n5\n</parameter>\n\
        </function>\n\
        </tool_call>";
    let (_, mut calls) = parse_tool_calls(raw);
    assert_eq!(calls.len(), 1);

    let tools = vec![make_tool(
        "search",
        serde_json::json!({
            "query": { "type": "string" },
            "limit": { "type": "integer" }
        }),
    )];
    coerce_all(&mut calls, &tools);

    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["limit"], 5, "limit must be integer 5 after coercion");
    assert_eq!(args["query"], "rust async");
}

#[test]
fn coerce_boolean_false_lower() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "flag": { "type": "boolean" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"flag":"false"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["flag"], false);
}

#[test]
fn coerce_boolean_true_capitalized() {
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "flag": { "type": "boolean" } }),
    )];
    let mut calls = vec![make_call("fn", r#"{"flag":"True"}"#)];
    coerce_all(&mut calls, &tools);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["flag"], true);
}

#[test]
fn coerce_invalid_arguments_json_is_noop() {
    // Malformed JSON (e.g., truncated streaming output) must not panic or mutate.
    let tools = vec![make_tool(
        "fn",
        serde_json::json!({ "limit": { "type": "integer" } }),
    )];
    let raw = r#"{"limit":"10""#; // missing closing }
    let mut calls = vec![make_call("fn", raw)];
    coerce_all(&mut calls, &tools);
    assert_eq!(calls[0].function.arguments, raw);
}

#[test]
fn tool_call_format_from_str_qwen3_xml() {
    let fmt = "qwen3_xml".parse::<ToolCallFormat>().unwrap();
    assert_eq!(fmt.name(), "qwen3_xml");
}
