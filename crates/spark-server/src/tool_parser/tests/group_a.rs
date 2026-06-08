// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::super::*;

use super::*;

// ── Hermes format ──

#[test]
fn parse_minimax_xml_drops_write_with_empty_filepath() {
    // F80b: parser-side defense. The model self-truncates a long
    // `content` then JSON-completes `<parameter name="filePath">
    // </parameter>` (empty). Without this, parse_minimax_xml_call
    // would return Some(ToolCall) with filePath="" — F78 catches
    // it downstream but only after the call has already been
    // emitted into the stream. Returning None here aborts cleanly.
    let body = "<invoke name=\"write\">\n\
            <parameter name=\"content\">some code</parameter>\n\
            <parameter name=\"filePath\"></parameter>\n\
            </invoke>";
    let res = parse_minimax_xml_call(body, 0);
    assert!(res.is_none(), "expected drop, got {res:?}");
}

#[test]
fn parse_minimax_xml_drops_write_with_whitespace_filepath() {
    let body = "<invoke name=\"write\">\n\
            <parameter name=\"content\">x</parameter>\n\
            <parameter name=\"filePath\">   </parameter>\n\
            </invoke>";
    let res = parse_minimax_xml_call(body, 0);
    assert!(
        res.is_none(),
        "expected whitespace-only path drop, got {res:?}"
    );
}

#[test]
fn parse_minimax_xml_keeps_bash_with_empty_path_field() {
    // bash isn't a write/edit tool — F80b does NOT apply to it.
    // (`path` field appearing on bash would be unusual but should
    // pass through.)
    let body = "<invoke name=\"bash\">\n\
            <parameter name=\"command\">ls</parameter>\n\
            </invoke>";
    let res = parse_minimax_xml_call(body, 0);
    assert!(res.is_some(), "bash should pass even without path");
}

#[test]
fn parse_minimax_xml_keeps_write_with_valid_filepath() {
    let body = "<invoke name=\"write\">\n\
            <parameter name=\"content\">hi</parameter>\n\
            <parameter name=\"filePath\">/tmp/x.rs</parameter>\n\
            </invoke>";
    let res = parse_minimax_xml_call(body, 0);
    assert!(res.is_some());
    let args: serde_json::Value = serde_json::from_str(&res.unwrap().function.arguments).unwrap();
    assert_eq!(args["filePath"], "/tmp/x.rs");
}

#[test]
fn validate_rejects_write_with_empty_filepath() {
    // F78: opencode loop reproduction — model emitted
    // `{"content":"...","filePath":""}` and the Write tool failed
    // with EISDIR forever. Validation must reject this.
    let tool = ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "write".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "filePath": {"type": "string"}
                },
                "required": ["content", "filePath"]
            })),
        },
    };
    let call = ToolCall {
        id: "x".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "write".to_string(),
            arguments: r#"{"content":"some code","filePath":""}"#.to_string(),
        },
    };
    let res = validate_single_tool_call(&call, &[tool]);
    assert!(res.is_err(), "expected reject, got {res:?}");
    assert!(
        res.as_ref().unwrap_err().contains("non-empty"),
        "error should mention non-empty: {}",
        res.unwrap_err()
    );
}

#[test]
fn validate_allows_read_with_empty_path() {
    // Theia getWorkspaceFileList passes path="" — must keep
    // working. Only WRITE_FAMILY rejects empty paths.
    let tool = ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "read".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            })),
        },
    };
    let call = ToolCall {
        id: "x".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "read".to_string(),
            arguments: r#"{"path":""}"#.to_string(),
        },
    };
    assert!(validate_single_tool_call(&call, &[tool]).is_ok());
}

#[test]
fn parse_hermes_single_call() {
    let (c, calls) =
        parse_tool_calls("<tool_call>\n{\"name\":\"f\",\"arguments\":{\"x\":1}}\n</tool_call>");
    assert!(c.is_none());
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "f");
    assert_eq!(calls[0].function.arguments, "{\"x\":1}");
}

#[test]
fn parse_hermes_with_content_and_multiple() {
    let (c, calls) = parse_tool_calls(
        "Hello\n<tool_call>\n{\"name\":\"a\",\"arguments\":{}}\n</tool_call>\n\
             <tool_call>\n{\"name\":\"b\",\"arguments\":{}}\n</tool_call>",
    );
    assert_eq!(c.unwrap(), "Hello");
    assert_eq!(calls.len(), 2);
    // IDs are globally-unique 16-hex-digit counter values; don't
    // assert on specific numbers (test order affects counter).
    assert!(calls[0].id.starts_with("call_"));
    assert!(calls[1].id.starts_with("call_"));
    assert_ne!(calls[0].id, calls[1].id);
    assert_eq!(calls[0].function.name, "a");
    assert_eq!(calls[1].function.name, "b");
}

#[test]
fn parse_no_calls() {
    let (c, calls) = parse_tool_calls("just text");
    assert_eq!(c.unwrap(), "just text");
    assert!(calls.is_empty());
}

#[test]
fn streaming_detector_hermes() {
    let mut det = StreamingToolDetector::new();
    let out = det.process("Hi <tool_call>\n{\"name\":\"f\",\"arguments\":{}}\n</tool_call>");
    assert!(out.len() >= 2);
    assert!(matches!(&out[0], DetectorOutput::Content(s) if s.contains("Hi")));
    assert!(matches!(&out[1], DetectorOutput::ToolCall(tc, 0) if tc.function.name == "f"));
    assert!(det.has_tool_calls());
}

/// F73 (2026-04-29): the streaming detector must recognise all three
/// envelope forms produced by MiniMax M2.7 (canonical via the
/// 200052 special-token, BPE-broken via the `:_` straddler, and
/// the `<tool_call>` shape that downstream code normalises to).
/// Before F73 the detector only saw `<tool_call>`/`<|tool_call>`,
/// so the model's `<minimax:_call>...<invoke ...></invoke>...</minimax:tool_call>`
/// passed straight through as content and `tool_calls` ended up
/// `None` — the bug observed live in opencode session
/// `ses_225891319ffeu33G5iHCMGwvgV`.
#[test]
fn streaming_detector_minimax_envelope_canonical() {
    let mut det = StreamingToolDetector::new();
    let body = "<minimax:tool_call>\n\
            <invoke name=\"bash\">\n\
            <parameter name=\"command\">uname -r</parameter>\n\
            </invoke>\n\
            </minimax:tool_call>";
    let out = det.process(body);
    let names: Vec<String> = out
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::ToolCall(tc, _) => Some(tc.function.name.clone()),
            DetectorOutput::ToolCallStart { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert!(
        det.has_tool_calls(),
        "detector must report a tool_call, got names={names:?}"
    );
    assert!(
        names.iter().any(|n| n == "bash"),
        "expected bash tool, got names={names:?}"
    );
}

#[test]
fn streaming_detector_minimax_envelope_bpe_broken() {
    // The exact failure shape from opencode-session.md
    // `ses_225891319ffeu33G5iHCMGwvgV`: open is the BPE-broken
    // `<minimax:_call>`, close is the canonical
    // `</minimax:tool_call>`. Detector still extracts a valid
    // tool call.
    let mut det = StreamingToolDetector::new();
    let body = "<minimax:_call>\n\
            <invoke name=\"bash\">\n\
            <parameter name=\"command\">mkdir -p /tmp/calc-test74/src /tmp/calc-test74/tests</parameter>\n\
            <parameter name=\"description\">Create project directories</parameter>\n\
            </invoke>\n\
            </minimax:tool_call>";
    let out = det.process(body);
    let tc = out.iter().find_map(|o| match o {
        DetectorOutput::ToolCall(tc, _) => Some(tc.clone()),
        _ => None,
    });
    assert!(
        det.has_tool_calls(),
        "broken-envelope detector must still extract a tool_call (tc found: {})",
        tc.is_some()
    );
    let tc = tc.expect("ToolCall output expected");
    assert_eq!(tc.function.name, "bash");
    let args: serde_json::Value =
        serde_json::from_str(&tc.function.arguments).expect("args must be JSON");
    assert!(
        args["command"]
            .as_str()
            .unwrap()
            .contains("/tmp/calc-test74"),
        "command arg lost: {args}"
    );
}

/// F75 chunk-split safety: the BPE-broken `<minimax:_call>` open
/// tag arrives split across stream chunks. Before adding it to
/// `safe_emit_len`'s prefix list, the detector emitted the
/// `<minimax:` trailing prefix as content (because none of the
/// listed tags started with `<minimax:`), then on the next chunk
/// the rest landed in an empty buffer and the open tag was lost.
/// Result: the full envelope leaked to `content` and
/// `has_tool_calls=false` — exactly opencode-session.md
/// `ses_224cc79f4ffeUtq7NFV9YMTVMH`.
#[test]
fn streaming_detector_minimax_bpe_broken_split_chunks() {
    let mut det = StreamingToolDetector::new();
    // Simulate per-token chunked arrival of the broken envelope.
    // The `:_` BPE token splits the open tag at byte 9.
    let chunks = [
        "<minimax",
        ":_call>",
        "\n<invoke name=\"bash\">\n",
        "<parameter name=\"command\">",
        "mkdir -p /tmp/calc-test74/src",
        "</parameter>\n</invoke>\n",
        "</minimax:tool_call>",
    ];
    let mut all_outputs = Vec::new();
    for c in chunks {
        all_outputs.extend(det.process(c));
    }
    let tcs: Vec<_> = all_outputs
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::ToolCall(tc, _) => Some(tc.clone()),
            _ => None,
        })
        .collect();
    assert!(
        det.has_tool_calls(),
        "envelope missed under chunked arrival"
    );
    assert_eq!(tcs.len(), 1, "expected 1 ToolCall");
    assert_eq!(tcs[0].function.name, "bash");
    // No content output should leak the envelope opener.
    for o in &all_outputs {
        if let DetectorOutput::Content(s) = o {
            assert!(
                !s.contains("<minimax:"),
                "envelope text leaked to content: {s:?}"
            );
        }
    }
}

/// F75 (2026-04-29): the actual opencode-session.md
/// `ses_224cc79f4ffeUtq7NFV9YMTVMH` failure shape — TWO `<invoke>`
/// blocks inside the BPE-broken envelope. Before F75 the detector
/// extracted only one and dropped the second; live response had
/// `has_tool_calls=false` because parse_one_call short-circuits to
/// the first `<invoke>` and the rest fall through as content.
#[test]
fn streaming_detector_minimax_envelope_bpe_broken_two_invokes() {
    let mut det = StreamingToolDetector::new();
    let body = "<minimax:_call>\n\
            <invoke name=\"bash\">\n\
            <parameter name=\"command\">mkdir -p /tmp/calc-test74/src</parameter>\n\
            <parameter name=\"description\">Create project directory structure</parameter>\n\
            </invoke>\n\
            <invoke name=\"bash\">\n\
            <parameter name=\"command\">mkdir -p /tmp/calc-test74/tests</parameter>\n\
            <parameter name=\"description\">Create tests directory</parameter>\n\
            </invoke>\n\
            </minimax:tool_call>";
    let out = det.process(body);
    let tcs: Vec<_> = out
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::ToolCall(tc, _) => Some(tc.clone()),
            _ => None,
        })
        .collect();
    assert!(det.has_tool_calls(), "no tool_calls extracted at all");
    assert_eq!(
        tcs.len(),
        2,
        "expected 2 ToolCall outputs, got {}",
        tcs.len()
    );
    for (i, tc) in tcs.iter().enumerate() {
        assert_eq!(tc.function.name, "bash", "call {i} wrong name");
        let args: serde_json::Value =
            serde_json::from_str(&tc.function.arguments).expect("args must be JSON");
        let cmd = args["command"].as_str().unwrap_or("");
        if i == 0 {
            assert!(
                cmd.contains("src"),
                "first call should be src dir, got {cmd}"
            );
        } else {
            assert!(
                cmd.contains("tests"),
                "second call should be tests dir, got {cmd}"
            );
        }
    }
}

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
    // value (mirroring path A) → validator passes (only
    // WRITE_FAMILY rejects empty paths; `exec` is not in that
    // list). The chat_stream::tool_handlers fix calls this same
    // chain inside handle_tool_call_delta so streaming behaviour
    // matches.
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
    assert!(
        validate_single_tool_call(&calls[0], std::slice::from_ref(&tool)).is_ok(),
        "validator passes once required key is present (non-WRITE-family)"
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
