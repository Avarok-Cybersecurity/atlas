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
