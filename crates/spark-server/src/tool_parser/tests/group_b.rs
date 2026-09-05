// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::super::*;

#[test]
fn parse_qwen3_coder_multiple_params() {
    let input = "<tool_call>\n\
            <function=search>\n\
            <parameter=query>\nrust programming\n</parameter>\n\
            <parameter=limit>\n10\n</parameter>\n\
            </function>\n\
            </tool_call>";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["query"], "rust programming");
    assert_eq!(args["limit"], "10"); // All param values are strings
}

#[test]
fn parse_qwen3_coder_two_consecutive_functions_no_field_bleed() {
    // Reference failure: opencode-session.md 2026-04-25.
    // Model emitted two consecutive <function=...> blocks; the
    // parser merged params from the second into the first's args
    // dict, producing a single Write call with command/timeout
    // (bash fields) AND filePath/content (write fields).
    //
    // Post-fix: the parameter loop hard-stops at `</function>`,
    // so each function block yields its own tool call with only
    // its own parameters. This goes through the bare-function
    // pass (no `<tool_call>` wrapper) which iterates over each
    // `<function=...>` independently.
    let input = "<function=write>\n\
            <parameter=filePath>\n/tmp/x.gitignore\n</parameter>\n\
            <parameter=content>\ntarget/\n</parameter>\n\
            </function>\n\
            <function=bash>\n\
            <parameter=command>\nls -la\n</parameter>\n\
            <parameter=description>\nlist files\n</parameter>\n\
            <parameter=timeout>\n180000\n</parameter>\n\
            </function>";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(
        calls.len(),
        2,
        "must emit 2 distinct tool calls — got {calls:#?}"
    );
    assert_eq!(calls[0].function.name, "write");
    assert_eq!(calls[1].function.name, "bash");

    let write_args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(write_args["filePath"], "/tmp/x.gitignore");
    assert_eq!(write_args["content"], "target/");
    assert!(
        write_args.get("command").is_none(),
        "write must not have bash field `command`: {write_args:?}"
    );
    assert!(
        write_args.get("timeout").is_none(),
        "write must not have bash field `timeout`: {write_args:?}"
    );
    assert!(
        write_args.get("description").is_none(),
        "write must not have bash field `description`: {write_args:?}"
    );

    let bash_args: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
    assert_eq!(bash_args["command"], "ls -la");
    assert_eq!(bash_args["timeout"], "180000");
    assert!(
        bash_args.get("filePath").is_none(),
        "bash must not have write field `filePath`: {bash_args:?}"
    );
    assert!(
        bash_args.get("content").is_none(),
        "bash must not have write field `content`: {bash_args:?}"
    );
}

#[test]
fn parse_qwen3_coder_call_direct_two_funcs_returns_first_only_no_bleed() {
    // Direct test of `parse_qwen3_coder_call`: when fed text that
    // contains two consecutive `<function=...>` blocks, it must
    // return ONLY the first call's parameters — no fields from
    // the second block leak into the first's args. This is the
    // exact contract the BareFunctionAttrPass relies on (it
    // advances past `</function>` between calls).
    let input = "<function=write>\n\
            <parameter=filePath>\n/tmp/y.txt\n</parameter>\n\
            <parameter=content>\nhello\n</parameter>\n\
            </function>\n\
            <function=bash>\n\
            <parameter=command>\nrm -rf /\n</parameter>\n\
            </function>";
    let tc = parse_qwen3_coder_call(input, 0).expect("must parse first function");
    assert_eq!(tc.function.name, "write");
    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
    assert_eq!(args["filePath"], "/tmp/y.txt");
    assert_eq!(args["content"], "hello");
    assert!(
        args.get("command").is_none(),
        "param loop must NOT cross `</function>` boundary: {args:?}"
    );
}

#[test]
fn parse_qwen3_coder_unclosed_param_then_next_function_does_not_bleed() {
    // Recovery case: first function's last `<parameter=>` lacks a
    // proper `</parameter>` close, so the value extends to
    // `</function>`. The fix's `advanced_to_func_close` flag must
    // break the loop at that point, NOT fall through to the next
    // function's parameters.
    let input = "<function=write>\n\
            <parameter=filePath>\n/tmp/y.txt\n</parameter>\n\
            <parameter=content>\nbody-without-close\n\
            </function>\n\
            <function=bash>\n\
            <parameter=command>\necho hi\n</parameter>\n\
            </function>";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 2);
    let write_args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    // The value should contain the body trimmed; missing-close
    // recovery is fine, the key constraint is no `command` leak.
    assert!(
        write_args["content"]
            .as_str()
            .unwrap()
            .contains("body-without-close")
    );
    assert!(
        write_args.get("command").is_none(),
        "no bleed across `</function>` boundary even with unclosed param"
    );
}

#[test]
fn parse_qwen3_coder_with_content() {
    let input = "Let me check the weather.\n\
            <tool_call>\n\
            <function=get_weather>\n\
            <parameter=location>\nTokyo\n</parameter>\n\
            </function>\n\
            </tool_call>";
    let (c, calls) = parse_tool_calls(input);
    assert_eq!(c.unwrap(), "Let me check the weather.");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
}

#[test]
fn streaming_detector_qwen3_coder() {
    let mut det = StreamingToolDetector::new();
    let out = det.process(
        "Hi <tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
    );
    assert!(out.len() >= 2);
    assert!(matches!(&out[0], DetectorOutput::Content(s) if s.contains("Hi")));
    assert!(matches!(&out[1], DetectorOutput::ToolCall(tc, 0) if tc.function.name == "f"));
    assert!(det.has_tool_calls());
}

// ── Format/trait dispatch ──

#[test]
fn tool_call_format_from_str() {
    assert!("hermes".parse::<ToolCallFormat>().is_ok());
    assert!("qwen3_coder".parse::<ToolCallFormat>().is_ok());
    assert!("unknown".parse::<ToolCallFormat>().is_err());
}

#[test]
fn into_parser_returns_correct_name() {
    let h = ToolCallFormat::Hermes.into_parser();
    assert_eq!(h.name(), "hermes");
    let q = ToolCallFormat::Qwen3Coder.into_parser();
    assert_eq!(q.name(), "qwen3_coder");
}

#[test]
fn hermes_parser_system_prompt_contains_json() {
    let parser = HermesParser;
    let tools = vec![ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "test".into(),
            description: None,
            parameters: None,
        },
    }];
    let prompt = parser.system_prompt(
        &tools,
        &ToolChoice::Mode("auto".into()),
        &crate::tool_parser::PromptLevers::OFF,
    );
    assert!(prompt.contains("\"name\":\"test\""));
    assert!(prompt.contains("<tools>"));
}

#[test]
fn tscg_is_decided_per_model_not_per_process() {
    // The regression the `OnceLock<bool>` allowed: the first model to load
    // fixed TSCG for every later one, so a hot-swap rendered the new model's
    // schemas under the old model's setting. Levers travel with the call, so
    // both answers are reachable in one process.
    let parser = HermesParser;
    let tools = vec![ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "test".into(),
            description: Some("A test function".into()),
            parameters: None,
        },
    }];
    let tc = ToolChoice::Mode("auto".into());
    let off = parser.system_prompt(&tools, &tc, &crate::tool_parser::PromptLevers::OFF);
    let on = parser.system_prompt(&tools, &tc, &crate::tool_parser::PromptLevers::new(true));
    assert!(
        off.contains("\"name\":\"test\""),
        "TSCG off keeps the JSON body"
    );
    assert!(
        !on.contains("\"name\":\"test\""),
        "TSCG on replaces it with signatures"
    );
    assert!(on.contains("test"), "the tool is still described");
}

#[test]
fn qwen3_coder_parser_system_prompt_contains_xml() {
    let parser = Qwen3CoderParser;
    let tools = vec![ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "test".into(),
            description: Some("A test function".into()),
            parameters: None,
        },
    }];
    let prompt = parser.system_prompt(
        &tools,
        &ToolChoice::Mode("auto".into()),
        &crate::tool_parser::PromptLevers::OFF,
    );
    assert!(prompt.contains("\"name\":\"test\""));
    assert!(prompt.contains("A test function"));
    assert!(prompt.contains("<function=example_function_name>"));
}

#[test]
fn format_tool_response_default() {
    let parser = HermesParser;
    let resp = parser.format_tool_response("{\"temp\": 20}");
    assert_eq!(resp, "<tool_response>\n{\"temp\": 20}\n</tool_response>");
}

// ── Tag-style fallback (bare <function> without <tool_call> wrapper) ──

#[test]
fn parse_bare_tag_style_simple() {
    let input = "I'll help you.\n\
            <function>get_weather</function>\
            <parameters>\
            <name>location</name>\
            <value>Paris</value>\
            </parameters>\
            </function>";
    let (c, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Paris");
    assert_eq!(c.unwrap(), "I'll help you.");
}

#[test]
fn parse_bare_tag_style_nested_params() {
    let input = "<function>search</function>\
            <parameters>\
            <parameter><name>query</name><value>rust</value></parameter>\
            <parameter><name>limit</name><value>10</value></parameter>\
            </parameters>";
    let (c, calls) = parse_tool_calls(input);
    assert!(c.is_none());
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["query"], "rust");
    assert_eq!(args["limit"], 10);
}

#[test]
fn parse_tag_style_inside_tool_call_wrapper() {
    let input = "<tool_call>\n\
            <function>get_weather</function>\
            <parameters><name>location</name><value>Tokyo</value></parameters>\
            </tool_call>";
    let (c, calls) = parse_tool_calls(input);
    assert!(c.is_none());
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Tokyo");
}

#[test]
fn parse_bare_attribute_style_no_wrapper() {
    let input = "Let me look that up.\n\
            <function=get_weather>\n\
            <parameter=location>\nBerlin\n</parameter>\n\
            </function>";
    let (c, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Berlin");
    assert_eq!(c.unwrap(), "Let me look that up.");
}

#[test]
fn parse_mike_screenshot_format() {
    // Exact format from mike2022014545's screenshot
    let input = "I see you're working with a complex codebase.\n\
            <function>question</function>\
            <parameters>\
            <name>questions</name>\
            <type>array</type>\
            <required>true</required>\
            <items>\
            <type>string</type>\
            <value>What would you like me to help you with?</value>\
            </items>\
            </parameters>\
            </function>";
    let (c, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "question");
    assert_eq!(c.unwrap(), "I see you're working with a complex codebase.");
}

#[test]
fn streaming_bare_function_flush() {
    // Regression: a bare tag-style call split across stream chunks so the
    // NAME closes (`</function>`) before the `<parameters>` block does.
    // `bare_function_end` used to treat the name-closing `</function>` as
    // end-of-call whenever `</parameters>` hadn't streamed in yet, emitting
    // a zero-argument call from `process()` alone and leaking the orphaned
    // `<parameters>...` text as raw content on the next call (it has no
    // `<function` prefix, so nothing recognizes it as part of the tool
    // call). Fixed by holding the block incomplete once `<parameters>` has
    // opened, until its close (or end-of-stream via `flush()`) arrives.
    let mut det = StreamingToolDetector::new();
    let out1 = det.process(
        "Hello <function>test</function><parameters><name>x</name><value>1</value>",
    );
    let premature_tool_calls = out1
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCall(..)))
        .count();
    assert_eq!(
        premature_tool_calls, 0,
        "must not complete the call before </parameters> arrives (params would be dropped)"
    );
    // `</parameters>` never arrives (e.g. max_tokens cutoff) — end of
    // stream. Flush must recover the call, with its argument, from the
    // buffered partial block — this is the actual flush() code path this
    // test's name refers to.
    let out2 = det.flush();
    let all: Vec<_> = out1.into_iter().chain(out2).collect();
    let has_content = all
        .iter()
        .any(|o| matches!(o, DetectorOutput::Content(s) if s.contains("Hello")));
    let tool_call = all.iter().find_map(|o| match o {
        DetectorOutput::ToolCall(tc, _) => Some(tc),
        _ => None,
    });
    assert!(has_content, "Should have content before function tag");
    let tc = tool_call.expect("Should detect bare function tag on flush");
    assert_eq!(tc.function.name, "test");
    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
    assert_eq!(
        args["x"], 1,
        "flush must recover the parameter value, not drop it: {args:?}"
    );
}

// ── Duplicated name sanitization (Bash=Bash bug) ──

#[test]
fn parse_qwen3_coder_duplicated_name_equals() {
    // Model artifact: <function=Bash=Bash> should parse as "Bash"
    let input = "<tool_call>\n\
            <function=Bash=Bash>\n\
            <parameter=command>\nwhich cargo\n</parameter>\n\
            </function>\n\
            </tool_call>";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "Bash");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["command"], "which cargo");
}

#[test]
fn parse_bare_duplicated_name_equals() {
    // Same bug without <tool_call> wrapper
    let input = "<function=Write=Write>\n\
            <parameter=file_path>\n/tmp/test.txt\n</parameter>\n\
            <parameter=content>\nhello\n</parameter>\n\
            </function>";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "Write");
}

#[test]
fn parse_qwen3_coder_name_with_trailing_garbage() {
    // Model artifact: <function=Bash=command> (confused param with name)
    let input = "<tool_call>\n\
            <function=Bash=command>\n\
            <parameter=command>\nls\n</parameter>\n\
            </function>\n\
            </tool_call>";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "Bash");
}

#[test]
fn parse_qwen3_coder_json_args_grammar_mode() {
    // Grammar-constrained output: JSON between <function=NAME> and </function>
    let input = "<tool_call>\
            <function=Bash>{\"command\":\"which cargo\"}</function>\
            </tool_call>";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "Bash");
    // JSON arguments should be extracted even without <parameter> tags
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["command"], "which cargo");
}

#[test]
fn parse_qwen3_coder_namespaced_function_name() {
    let input = "<tool_call>\n\
            <function=browser:search>\n\
            <parameter=query>\nrust namespace parser\n</parameter>\n\
            </function>\n\
            </tool_call>";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
}

#[test]
fn parse_hermes_namespaced_function_name() {
    let input = r#"<tool_call>{"name":"browser:search","arguments":{"query":"rust"}}</tool_call>"#;
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["query"], "rust");
}

#[test]
fn parse_mistral_namespaced_function_name() {
    let input = r#"[TOOL_CALLS]browser:search[ARGS]{"query":"rust"}"#;
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
}

#[test]
fn parse_bare_mistral_namespaced_function_name() {
    let input = r#"browser:search{"query":"rust"}"#;
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
}

#[test]
fn parse_gemma4_namespaced_function_name() {
    let input = "<|tool_call>call:browser:search{query:<|\"|>rust<|\"|>}<tool_call|>";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
}

#[test]
fn parse_minimax_namespaced_function_name() {
    let input = r#"<tool_call><invoke name="browser:search"><parameter name="query">rust</parameter></invoke></tool_call>"#;
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
}

// ── Mistral native format ──
