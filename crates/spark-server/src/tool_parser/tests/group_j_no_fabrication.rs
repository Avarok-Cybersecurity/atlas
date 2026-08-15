// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::super::*;

// ────────────────────────────────────────────────────────────────────────
// "Atlas must not author the model's arguments" (2026-08-13).
//
// Live repro, Nemotron-3.5 Lightning 30B, thinking ON + tools armed,
// temperature 0: `"What's the weather in Santiago? Use the tool."` came
// back as `get_weather({"city": ""})`, deterministically, 5/5. The model
// wrote its `<tool_call>` INSIDE `<think>`; a `</think>` force-close cut it
// between the function name and its arguments; the blocking path hoisted
// the half-call out of the reasoning trace (F7); and the missing-required
// backfill then completed it with `""`.
//
// `""` is the damaging part. It is a VALID instance of
// `{"type": "string"}`, so the resulting call satisfies both Atlas's
// `assess_tool_call` and the client's own JSON-Schema `required` check —
// the tool ran, with an argument no model ever produced. An ABSENT key
// fails `required` loudly and the model gets a correctable error.
// ────────────────────────────────────────────────────────────────────────

fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "get_weather".into(),
            description: Some("Get the current weather for a city.".into()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": { "city": {"type": "string"} },
                "required": ["city"]
            })),
        },
    }
}

fn name_only_call() -> Vec<ToolCall> {
    // Exactly what the truncated `<tool_call><function=get_weather>` shape
    // parses to: correct name, zero arguments.
    vec![ToolCall {
        id: "call_0".into(),
        call_type: "function".into(),
        function: FunctionCall {
            name: "get_weather".into(),
            arguments: "{}".into(),
        },
    }]
}

#[test]
fn truncated_call_keeps_required_param_absent() {
    let tool = weather_tool();
    let mut calls = name_only_call();
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert!(
        args.get("city").is_none(),
        "backfill must not author `city`; got {args}"
    );
}

#[test]
fn truncated_call_is_reported_as_missing_param() {
    // The honest failure the client can see. Before the fix this returned
    // Ok(()) because `city` was present as "".
    let tool = weather_tool();
    let mut calls = name_only_call();
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let issue = assess_tool_call(&calls[0], std::slice::from_ref(&tool))
        .expect_err("a call with no `city` cannot satisfy its required schema");
    assert!(
        matches!(issue, ToolCallIssue::MissingParam(_)),
        "expected MissingParam, got {issue:?}"
    );
    assert!(
        issue.message().contains("city"),
        "the error must name the parameter: {}",
        issue.message()
    );
}

#[test]
fn model_authored_empty_string_is_preserved() {
    // The converse guard. When the MODEL emits `city=""`, that is its
    // output and Atlas neither erases nor "repairs" it — Theia's
    // `getWorkspaceFileList` legitimately sends `path=""`, and
    // `assess_tool_call` documents that empty-string policing is the
    // client's concern.
    let tool = weather_tool();
    let mut calls = vec![ToolCall {
        id: "call_0".into(),
        call_type: "function".into(),
        function: FunctionCall {
            name: "get_weather".into(),
            arguments: r#"{"city":""}"#.into(),
        },
    }];
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["city"], "",
        "model-authored empty must survive verbatim"
    );
}

#[test]
fn derivable_required_params_are_still_filled() {
    // The fix removes FABRICATION, not derivation: `description` restates
    // the `command` the model did produce, and `subagent_type`'s legal
    // values are readable off the tool description. Both must keep working
    // (group_f covers subagent_type end-to-end; this pins `description`).
    let tool = ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "bash".into(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["command", "description"]
            })),
        },
    };
    let mut calls = vec![ToolCall {
        id: "call_0".into(),
        call_type: "function".into(),
        function: FunctionCall {
            name: "bash".into(),
            arguments: r#"{"command":"ls /tmp"}"#.into(),
        },
    }];
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["description"], "Run: ls /tmp");
}

#[test]
fn undecidable_required_param_stays_absent_across_the_family() {
    // Every tool whose required param carries real intent — a path, a
    // search string, a city — must come back absent, not blank.
    for (tool_name, param) in [
        ("write", "filePath"),
        ("edit", "old_string"),
        ("grep", "pattern"),
        ("get_weather", "city"),
    ] {
        let tool = ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: tool_name.into(),
                description: None,
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": { param: {"type": "string"} },
                    "required": [param]
                })),
            },
        };
        let mut calls = vec![ToolCall {
            id: "call_0".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: tool_name.into(),
                arguments: "{}".into(),
            },
        }];
        backfill_required_params(&mut calls, std::slice::from_ref(&tool));
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(
            args.get(param).is_none(),
            "{tool_name}.{param} must stay absent, got {args}"
        );
    }
}

// ────────────────────────────────────────────────────────────────────────
// XML-attribute drift (2026-08-13). Same live sweep as above: under
// `tool_choice="auto"` the qwen3_coder grammar is a LATE trigger, so a
// model that opens with a bare `<function=NAME>` (no `<tool_call>`
// prefix) decodes unconstrained and writes its parameter as an XML
// attribute. 4 of 10 Lightning requests took that shape and reached the
// client with no arguments at all.
// ────────────────────────────────────────────────────────────────────────

fn args_of(input: &str) -> serde_json::Value {
    let (_c, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1, "expected one call from {input:?}");
    serde_json::from_str(&calls[0].function.arguments).unwrap()
}

#[test]
fn attribute_drift_recovers_the_parameter() {
    // Verbatim from the live log. The attribute carries the value; the
    // element body ("Chile") is drift continuation and is dropped.
    let args = args_of(
        "<function=get_weather>\n<parameter city=\"Santiago\">\nChile\n</parameter>\n\
         </function>\n</tool_call>\n",
    );
    assert_eq!(args["city"], "Santiago");
}

#[test]
fn attribute_drift_with_echoed_body() {
    // The other live shape: attribute and body agree.
    let args = args_of(
        "<function=get_weather>\n<parameter city=\"Nairobi\">Nairobi</parameter>\n\
         </function>\n</tool_call>\n",
    );
    assert_eq!(args["city"], "Nairobi");
}

#[test]
fn name_attribute_dialect_takes_the_body_as_the_value() {
    // `<parameter name="KEY">VALUE</parameter>` is the OTHER XML dialect;
    // reading the attribute as the value there would yield `city: "city"`.
    let args = args_of(
        "<function=get_weather>\n<parameter name=\"city\">Santiago</parameter>\n\
         </function>\n</tool_call>\n",
    );
    assert_eq!(args["city"], "Santiago");
}

#[test]
fn strict_form_still_wins_over_the_salvage() {
    // A well-formed call must be byte-identical to before the salvage
    // existed, even when a stray `<parameter …>` follows it.
    let args = args_of(
        "<tool_call>\n<function=get_weather>\n<parameter=city>Kyoto</parameter>\n\
         <parameter city=\"Osaka\">x</parameter>\n</function>\n</tool_call>",
    );
    assert_eq!(args["city"], "Kyoto");
}

#[test]
fn salvage_ignores_shapes_it_cannot_read_confidently() {
    // Unquoted value, multi-attribute tag, and a non-identifier key are
    // all left alone rather than guessed at.
    for body in [
        "<parameter city=Santiago>x</parameter>",
        "<parameter city=\"Santiago\" units=\"c\">x</parameter>",
        "<parameter 9city=\"Santiago\">x</parameter>",
    ] {
        let args = args_of(&format!(
            "<function=get_weather>\n{body}\n</function>\n</tool_call>\n"
        ));
        assert!(
            args.as_object().unwrap().is_empty(),
            "must not guess from {body:?}, got {args}"
        );
    }
}

#[test]
fn salvage_does_not_cross_a_function_boundary() {
    // A sibling `<function=…>` block's attribute parameter must not be
    // merged into this call's args (the 2026-04-25 opencode arg-mixing
    // class, guarded the same way as the strict loop).
    let (_c, calls) = parse_tool_calls(
        "<tool_call>\n<function=get_weather>\n<parameter=city>Kyoto</parameter>\n\
         </function>\n</tool_call>\n\
         <tool_call>\n<function=get_time>\n<parameter zone=\"JST\">x</parameter>\n\
         </function>\n</tool_call>",
    );
    assert_eq!(calls.len(), 2);
    let a0: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert!(
        a0.get("zone").is_none(),
        "arg leaked across functions: {a0}"
    );
    assert_eq!(a0["city"], "Kyoto");
    let a1: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
    assert_eq!(a1["zone"], "JST");
}

// ────────────────────────────────────────────────────────────────────────
// Disposition of an OMITTED parameter, by tool class. Dropping the `""`
// fabrication moved the mutation/shell tools from "empty string" to
// "absent" — both must stay OUT of `valid`, because a write with no path
// and a shell call with no command are equally unexecutable.
// ────────────────────────────────────────────────────────────────────────

fn one_string_tool(name: &str, param: &str) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: name.into(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": { param: {"type": "string"} },
                "required": [param]
            })),
        },
    }
}

fn empty_call(name: &str) -> Vec<ToolCall> {
    vec![ToolCall {
        id: "call_0".into(),
        call_type: "function".into(),
        function: FunctionCall {
            name: name.into(),
            arguments: "{}".into(),
        },
    }]
}

#[test]
fn omitted_mutation_path_is_not_executable() {
    for (tool_name, param) in [("write", "file_path"), ("edit", "path")] {
        let tools = [one_string_tool(tool_name, param)];
        let mut calls = empty_call(tool_name);
        backfill_required_params(&mut calls, &tools);
        let validated = validate_tool_calls(calls, &tools);
        assert!(
            validated.valid.is_empty(),
            "{tool_name} with no {param} must not be attached as a valid call"
        );
        assert_eq!(validated.errors.len(), 1);
        assert!(validated.errors[0].contains(param));
    }
}

#[test]
fn omitted_shell_command_is_not_executable() {
    let tools = [one_string_tool("bash", "command")];
    let mut calls = empty_call("bash");
    backfill_required_params(&mut calls, &tools);
    let validated = validate_tool_calls(calls, &tools);
    assert!(validated.valid.is_empty());
    assert!(validated.errors[0].contains("command"));
}

#[test]
fn omitted_ordinary_param_stays_attached_with_an_error() {
    // The ST-995 rule: a read-only tool missing a parameter is still
    // delivered as the model produced it, so the client's schema check —
    // not an empty response — is what the model sees.
    let tools = [one_string_tool("get_weather", "city")];
    let mut calls = empty_call("get_weather");
    backfill_required_params(&mut calls, &tools);
    let validated = validate_tool_calls(calls, &tools);
    assert_eq!(validated.valid.len(), 1);
    assert_eq!(validated.valid[0].function.arguments, "{}");
    assert_eq!(validated.errors.len(), 1);
    assert!(validated.errors[0].contains("city"));
}
