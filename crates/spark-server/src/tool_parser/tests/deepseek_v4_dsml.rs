// SPDX-License-Identifier: AGPL-3.0-only

use super::super::*;

const CALL: &str = r#"answer

<｜DSML｜tool_calls>
<｜DSML｜invoke name="search">
<｜DSML｜parameter name="query" string="true">DeepSeek V4</｜DSML｜parameter>
<｜DSML｜parameter name="limit" string="false">3</｜DSML｜parameter>
<｜DSML｜parameter name="fresh" string="false">true</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"#;

#[test]
fn blocking_dsml_parser_preserves_typed_arguments() {
    let (content, calls) = parse_tool_calls(CALL);
    assert_eq!(content.as_deref(), Some("answer"));
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args,
        serde_json::json!({"query": "DeepSeek V4", "limit": 3, "fresh": true})
    );
}

#[test]
fn streaming_dsml_parser_handles_split_special_tokens() {
    let mut detector = StreamingToolDetector::new();
    let mut outputs = Vec::new();
    let chars: Vec<char> = CALL.chars().collect();
    for chunk in chars.chunks(7) {
        outputs.extend(detector.process(&chunk.iter().collect::<String>()));
    }
    outputs.extend(detector.flush());

    let mut content = String::new();
    let mut calls = Vec::new();
    for output in outputs {
        match output {
            DetectorOutput::Content(text) => content.push_str(&text),
            DetectorOutput::ToolCall(call, index) => calls.push((index, call)),
            _ => {}
        }
    }
    assert_eq!(content.trim(), "answer");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, 0);
    assert_eq!(calls[0].1.function.name, "search");
}

#[test]
fn deepseek_v4_parser_is_registered() {
    let parser = "deepseek_v4"
        .parse::<ToolCallFormat>()
        .expect("registered DSML parser")
        .into_parser();
    assert_eq!(parser.name(), "deepseek_v4");
    assert!(
        parser
            .system_prompt(&[], &ToolChoice::Mode("auto".into()), &PromptLevers::OFF)
            .is_empty()
    );
}

/// Drive `CALL` truncated at `cut` through the streaming detector in 7-char
/// chunks plus a final `flush`, and collapse the outputs the same way the
/// existing streaming test does.
fn stream_truncated(cut: usize) -> (String, Vec<ToolCall>) {
    let mut detector = StreamingToolDetector::new();
    let mut outputs = Vec::new();
    let chars: Vec<char> = CALL[..cut].chars().collect();
    for chunk in chars.chunks(7) {
        outputs.extend(detector.process(&chunk.iter().collect::<String>()));
    }
    outputs.extend(detector.flush());
    let mut content = String::new();
    let mut calls = Vec::new();
    for output in outputs {
        match output {
            DetectorOutput::Content(text) => content.push_str(&text),
            DetectorOutput::ToolCall(call, _) => calls.push(call),
            _ => {}
        }
    }
    (content, calls)
}

/// Batch4 leftover (streaming-vs-blocking DSML parity): an UNTERMINATED
/// envelope whose invokes are complete — the close tag lost to truncation
/// (`finish_reason="length"`) or EOS drift — must yield the same tool call
/// from both paths. The blocking parser used to return ZERO calls and leak
/// the raw `<｜DSML｜…` markup into content while the streaming flush
/// salvaged the invoke.
#[test]
fn unterminated_envelope_blocking_matches_streaming_salvage() {
    let cut = CALL.len() - DSML_CLOSE.len();
    assert!(
        CALL[cut..].starts_with(DSML_CLOSE),
        "cut must drop the close tag"
    );

    let (blocking_content, blocking_calls) = parse_tool_calls(&CALL[..cut]);
    let (streaming_content, streaming_calls) = stream_truncated(cut);

    assert_eq!(blocking_content.as_deref(), Some("answer"));
    assert_eq!(streaming_content.trim(), "answer");
    assert_eq!(blocking_calls.len(), 1);
    assert_eq!(streaming_calls.len(), 1);
    assert_eq!(blocking_calls[0].function.name, "search");
    assert_eq!(
        blocking_calls[0].function.name,
        streaming_calls[0].function.name
    );
    let expected = serde_json::json!({"query": "DeepSeek V4", "limit": 3, "fresh": true});
    let blocking_args: serde_json::Value =
        serde_json::from_str(&blocking_calls[0].function.arguments).unwrap();
    let streaming_args: serde_json::Value =
        serde_json::from_str(&streaming_calls[0].function.arguments).unwrap();
    assert_eq!(blocking_args, expected);
    assert_eq!(streaming_args, expected);
}

/// Batch4 leftover (streaming-vs-blocking DSML parity): an unterminated
/// envelope with NO complete invoke (truncated mid-parameter) parses as a
/// tool call in neither path, and neither path may destroy the bytes. The
/// streaming flush used to `mem::take` the buffer and then emit nothing,
/// silently dropping the whole envelope; blocking returns it as content.
#[test]
fn unterminated_envelope_with_no_complete_invoke_is_content_in_both_paths() {
    let cut = CALL.find("DeepSeek V4").expect("param value present") + "DeepSeek V".len();
    let envelope_start = CALL.find(DSML_OPEN).expect("envelope present");
    let raw_envelope = &CALL[envelope_start..cut];

    let (blocking_content, blocking_calls) = parse_tool_calls(&CALL[..cut]);
    let (streaming_content, streaming_calls) = stream_truncated(cut);

    assert!(blocking_calls.is_empty());
    assert!(streaming_calls.is_empty());
    let blocking_content = blocking_content.expect("blocking keeps the bytes");
    assert!(
        blocking_content.contains(raw_envelope),
        "blocking content lost the envelope: {blocking_content:?}"
    );
    assert!(
        streaming_content.contains(raw_envelope),
        "streaming content lost the envelope: {streaming_content:?}"
    );
}
