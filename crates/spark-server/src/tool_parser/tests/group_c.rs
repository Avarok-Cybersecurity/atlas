// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::super::*;

#[test]
fn parse_mistral_nested_json_args() {
    let input =
        r#"[TOOL_CALLS]search[ARGS]{"filters":{"year":2025,"tags":["rust","cuda"]},"limit":10}"#;
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["filters"]["year"], 2025);
    assert_eq!(args["filters"]["tags"][1], "cuda");
    assert_eq!(args["limit"], 10);
}

#[test]
fn parse_mistral_with_whitespace() {
    let input = "[TOOL_CALLS] get_weather [ARGS]\n{\n  \"location\": \"Paris\"\n}";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
}

#[test]
fn streaming_detector_mistral_single() {
    let mut det = StreamingToolDetector::new();
    let out = det.process("Let me check.[TOOL_CALLS]get_weather[ARGS]{\"location\":\"Paris\"}");
    // Expect: Content("Let me check.") + ToolCallStart + ToolCallDelta + ToolCallEnd
    assert!(out.len() >= 4);
    assert!(matches!(&out[0], DetectorOutput::Content(s) if s.contains("Let me check")));
    assert!(matches!(&out[1], DetectorOutput::ToolCallStart { name, .. } if name == "get_weather"));
    assert!(matches!(&out[2], DetectorOutput::ToolCallDelta { .. }));
    assert!(matches!(&out[3], DetectorOutput::ToolCallEnd { .. }));
    assert!(det.has_tool_calls());
}

#[test]
fn streaming_detector_mistral_split_tokens() {
    // Simulate token-by-token streaming of a Mistral tool call.
    let mut det = StreamingToolDetector::new();
    let chunks = [
        "[TOOL_",
        "CALLS]",
        "get_weather",
        "[ARGS]",
        "{\"city\":",
        "\"Paris",
        "\"}",
    ];
    let mut outs: Vec<DetectorOutput> = Vec::new();
    for c in chunks {
        outs.extend(det.process(c));
    }
    assert!(
        det.has_tool_calls(),
        "Expected tool call, got {} events",
        outs.len()
    );
    let has_start = outs
        .iter()
        .any(|o| matches!(o, DetectorOutput::ToolCallStart { name, .. } if name == "get_weather"));
    assert!(has_start, "Missing ToolCallStart");
    let has_delta = outs
        .iter()
        .any(|o| matches!(o, DetectorOutput::ToolCallDelta { args, .. } if args.contains("Paris")));
    assert!(has_delta, "Missing ToolCallDelta with Paris");
}

#[test]
fn mistral_native_parser_format_tool_calls() {
    let parser = MistralNativeParser;
    let calls = vec![IncomingToolCall {
        id: Some("call_1".into()),
        function: IncomingFunction {
            name: "get_weather".into(),
            arguments: r#"{"location":"Paris"}"#.into(),
        },
    }];
    let formatted = parser.format_tool_calls(&calls);
    assert!(formatted.contains("[TOOL_CALLS]"));
    assert!(formatted.contains("get_weather"));
    assert!(formatted.contains("[ARGS]"));
    assert!(formatted.contains("Paris"));
}

#[test]
fn mistral_native_parser_name_and_dispatch() {
    let p = ToolCallFormat::Mistral.into_parser();
    assert_eq!(p.name(), "mistral");
    assert!("mistral".parse::<ToolCallFormat>().is_ok());
}

#[test]
fn find_balanced_json_end_simple() {
    assert_eq!(find_balanced_json_end(r#"{"a":1}"#), Some(7));
    assert_eq!(find_balanced_json_end(r#"{"a":{"b":2}}"#), Some(13));
    assert_eq!(find_balanced_json_end(r#"{"a":1}extra"#), Some(7));
    assert_eq!(find_balanced_json_end(r#"{"a":1"#), None);
    assert_eq!(find_balanced_json_end("not json"), None);
}

#[test]
fn find_balanced_json_end_string_braces() {
    // Braces inside strings must not affect depth counting.
    assert_eq!(find_balanced_json_end(r#"{"msg":"hi } there"}"#), Some(20));
    // Escaped quotes must not terminate the string.
    assert_eq!(find_balanced_json_end(r#"{"msg":"say \"hi\""}"#), Some(20));
}

#[test]
fn parse_mistral_truncated_args_recovered() {
    // max_tokens cut the response mid-object — the largest balanced
    // prefix should still be parsed. Here the second call's args are
    // truncated, so only the first call should succeed.
    let input = "[TOOL_CALLS]search[ARGS]{\"q\":\"rust\"}[TOOL_CALLS]broken[ARGS]{\"incomp";
    let (_, calls) = parse_tool_calls(input);
    assert!(!calls.is_empty());
    assert_eq!(calls[0].function.name, "search");
}

#[test]
fn qwen3_tool_call_close_inside_param_value_not_terminating() {
    // A parameter value contains the literal `</tool_call>` substring.
    // The bare `find` would terminate the call body early; the new
    // depth-aware scan must wait for the close after `</parameter>`.
    let input = "<tool_call>\n<function=write>\n<parameter=content>before</tool_call> after</parameter>\n</function>\n</tool_call>";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "write");
    assert!(
        calls[0].function.arguments.contains("</tool_call>"),
        "value should preserve embedded close substring: {}",
        calls[0].function.arguments,
    );
}

#[test]
fn qwen3_missing_param_close_doesnt_swallow_next_param() {
    // If a parameter is missing its `</parameter>` close, recovery
    // should stop at the next `<parameter=` so the subsequent param
    // is not absorbed into the prior value (vllm #38158 regression).
    let input = "<tool_call>\n<function=foo>\n<parameter=a>first<parameter=b>second</parameter>\n</function>\n</tool_call>";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    let args: serde_json::Value =
        serde_json::from_str(&calls[0].function.arguments).expect("args parse");
    assert_eq!(args["a"].as_str().unwrap_or(""), "first");
    assert_eq!(args["b"].as_str().unwrap_or(""), "second");
}

// ── Salvage pass: <parameter=NAME> mis-typed as <function=NAME> ──

#[test]
fn salvage_param_as_function_recovers_write_call() {
    // Repro of Qwen3.5-35B-A3B-FP8 session ses_2401e91f1ffeFiB1kzYpnc2pFX —
    // model emitted the write tool with `<parameter=write>` instead of
    // `<function=write>`, no `<tool_call>` wrapper. Salvage pass must
    // reconstruct it.
    let input = "<parameter=write>\n\
            <parameter=content>\n\
            [package]\nname = \"calc\"\nversion = \"0.1.0\"\n\
            </parameter>\n\
            <parameter=filePath>\n\
            /tmp/calc-test26/Cargo.toml\n\
            </parameter>\n\
            </function>";
    let pipeline = ToolCallPipeline::bare_function_default();
    let (_content, calls) = pipeline.run(input);
    assert_eq!(calls.len(), 1, "salvage should recover exactly one call");
    assert_eq!(calls[0].function.name, "write");
    let args: serde_json::Value =
        serde_json::from_str(&calls[0].function.arguments).expect("args parse");
    assert!(
        args["content"].as_str().unwrap_or("").contains("[package]"),
        "content arg should carry the Cargo.toml body: {}",
        calls[0].function.arguments,
    );
    assert_eq!(
        args["filePath"].as_str().unwrap_or(""),
        "/tmp/calc-test26/Cargo.toml"
    );
}

#[test]
fn salvage_does_not_fire_when_legitimate_function_opener_present() {
    // A well-formed `<function=X>` block must be handled by the primary
    // pass; the salvage must not double-parse / misfire.
    let input = "prelude\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>";
    let pipeline = ToolCallPipeline::bare_function_default();
    let (content, calls) = pipeline.run(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "bash");
    assert_eq!(content.as_deref(), Some("prelude"));
}

#[test]
fn salvage_requires_closing_function_tag() {
    // Malformed output WITHOUT a `</function>` close is ambiguous — it
    // might be genuine prose that happens to use the `<parameter=...>`
    // syntax. Salvage must NOT fire.
    let input = "Some prose with <parameter=key> but no close tag.";
    let pipeline = ToolCallPipeline::bare_function_default();
    let (_content, calls) = pipeline.run(input);
    assert!(
        calls.is_empty(),
        "salvage must not fire without </function>"
    );
}

#[test]
fn salvage_rejects_non_identifier_name() {
    // First `<parameter=...>` has a non-identifier name → reject the
    // salvage entirely (don't treat punctuation / URL-like values as a
    // function name).
    let input = "<parameter=some/thing>\n\
            <parameter=k>v</parameter>\n\
            </function>";
    let pipeline = ToolCallPipeline::bare_function_default();
    let (_content, calls) = pipeline.run(input);
    assert!(
        calls.is_empty(),
        "non-identifier first param must not salvage"
    );
}

// ── qwen3_coder system-prompt anti-narration hardening ──

#[test]
fn qwen3_coder_prompt_has_antinarration_hardening() {
    // Regression pin for the anti-narration / declarative-logic
    // structure of the qwen3_coder parser's system prompt.
    // F34 (2026-04-26): replaced the prior 50-line WRONG-pattern
    // wall with a single positive-shot example + declarative
    // sentence (per Wei et al. on contrastive in-context
    // learning — negative examples imitation-trap pattern
    // models).
    let parser = Qwen3CoderParser;
    let prompt = parser.system_prompt(&[], &ToolChoice::Mode("auto".into()));
    // <IMMEDIATE_TOOL_USE> block is still present.
    assert!(
        prompt.contains("<IMMEDIATE_TOOL_USE>"),
        "IMMEDIATE_TOOL_USE block missing"
    );
    // Declarative sentence about parameter values being the only
    // place tool inputs should appear (replaces the old WRONG/
    // RIGHT walls).
    assert!(
        prompt.contains("ONLY place file content, commands"),
        "declarative parameter-content sentence missing"
    );
    // Anti-simulation rule preserved.
    assert!(
        prompt.contains("NEVER simulate a tool response"),
        "anti-simulation rule missing"
    );
    // Bare-tag rule preserved.
    assert!(
        prompt.contains("NEVER emit bare tags"),
        "bare-tag rule missing"
    );
    // Existing required-param rule preserved.
    assert!(
        prompt.contains("EVERY required parameter MUST have a non-empty value"),
        "required-param rule missing"
    );
}

#[test]
fn qwen3_coder_f33_bash_description_has_retry_rule() {
    // F33 (2026-04-26): the Bash tool's `description` field
    // gets a one-line retry rule appended so the model sees it
    // on every Bash call (tool descriptions are part of the
    // schema render, attended every emission).
    let parser = Qwen3CoderParser;
    let bash_tool = ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "Bash".into(),
            description: Some("Execute a shell command.".into()),
            parameters: None,
        },
    };
    let prompt = parser.system_prompt(
        std::slice::from_ref(&bash_tool),
        &ToolChoice::Mode("auto".into()),
    );
    assert!(
        prompt.contains("[atlas-f33]"),
        "F33 marker missing from Bash description"
    );
    assert!(
        prompt.contains("do NOT retry the same command"),
        "F33 retry-rule body missing from Bash description"
    );
    // Other tools should not be touched.
    let write_tool = ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "Write".into(),
            description: Some("Write a file.".into()),
            parameters: None,
        },
    };
    let prompt_write_only = parser.system_prompt(
        std::slice::from_ref(&write_tool),
        &ToolChoice::Mode("auto".into()),
    );
    assert!(
        !prompt_write_only.contains("[atlas-f33]"),
        "F33 marker should only attach to Bash, not Write"
    );
}

// ── Per-parser LeakMarkers ──

#[test]
fn leak_markers_default_is_empty() {
    // A parser without an override returns `LeakMarkers::EMPTY` — the
    // sanitizer short-circuits to pass-through. Cover every parser
    // that opts out.
    let defaults: &[Box<dyn ToolCallParser>] = &[
        Box::new(HermesParser),
        Box::new(Gemma4Parser),
        Box::new(MistralNativeParser),
        Box::new(BareJsonParser),
    ];
    for parser in defaults {
        let m = parser.leak_markers();
        assert!(
            m.orphan_open.is_empty(),
            "{}: orphan_open should be empty by default",
            parser.name()
        );
        assert!(
            m.close.is_empty(),
            "{}: close should be empty by default",
            parser.name()
        );
    }
}

#[test]
fn leak_markers_qwen3_coder_has_parameter_open() {
    let m = Qwen3CoderParser.leak_markers();
    // `<parameter=` catches the common orphan-param leak; `<tool_response>`
    // catches hallucinated simulated tool exchanges (observed in
    // opencode session ses_23f6593f4ffe43lhflzkIlWGez).
    assert!(m.orphan_open.contains(&"<parameter="));
    assert!(m.orphan_open.contains(&"<tool_response>"));
    assert!(m.close.contains(&"</parameter>"));
    assert!(m.close.contains(&"</function>"));
    assert!(m.close.contains(&"</tool_call>"));
    assert!(m.close.contains(&"</tool_response>"));
}

#[test]
fn leak_markers_minimax_has_invoke_open() {
    let m = MinimaxXmlParser.leak_markers();
    // Both inner-element starts should be declared so a truncated
    // `<invoke name="write">` leaks don't bypass the scanner just
    // because no `<parameter name=` was emitted yet.
    assert!(m.orphan_open.contains(&"<invoke name=\""));
    assert!(m.orphan_open.contains(&"<parameter name=\""));
    // Close set must include the outer namespaced tag AND the
    // detector-rewritten bare `</tool_call>`.
    assert!(m.close.contains(&"</invoke>"));
    assert!(m.close.contains(&"</parameter>"));
    assert!(m.close.contains(&"</minimax:tool_call>"));
    assert!(m.close.contains(&"</tool_call>"));
}
