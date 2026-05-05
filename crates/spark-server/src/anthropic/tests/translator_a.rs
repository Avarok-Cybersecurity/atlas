// SPDX-License-Identifier: AGPL-3.0-only

use super::super::*;

#[test]
fn translator_text_only_request_roundtrips() {
    let req = MessagesRequest {
        model: "claude".into(),
        max_tokens: 100,
        system: Some(SystemContent::Text("You are helpful.".into())),
        messages: vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Text("Hi".into()),
        }],
        temperature: Some(0.7),
        top_k: None,
        top_p: None,
        tools: None,
        tool_choice: None,
        stop_sequences: vec![],
        stream: false,
        thinking: None,
    };
    let chat = anthropic_to_chat_request_json(&req);
    assert_eq!(chat["model"], "claude");
    assert_eq!(chat["max_tokens"], 100);
    // f32 → f64 round-trip → not bit-exact; check within tolerance.
    let temp = chat["temperature"].as_f64().unwrap();
    assert!((temp - 0.7).abs() < 1e-5, "temperature ≈ 0.7, got {temp}");
    let msgs = chat["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "You are helpful.");
    assert_eq!(msgs[1]["role"], "user");
    assert_eq!(msgs[1]["content"], "Hi");
    // Penalty fields must NOT be set so the OpenAI-side preset wins.
    assert!(chat.get("repetition_penalty").is_none());
    assert!(chat.get("presence_penalty").is_none());
    assert!(chat.get("dry_multiplier").is_none());
}

#[test]
fn translator_tool_use_assistant_msg_collapses_to_tool_calls() {
    let req = MessagesRequest {
        model: "claude".into(),
        max_tokens: 100,
        system: None,
        messages: vec![AnthropicMessage {
            role: "assistant".into(),
            content: AnthropicContent::Blocks(vec![
                ContentBlock::Text {
                    text: "I'll write the file.".into(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_1".into(),
                    name: "write".into(),
                    input: serde_json::json!({"path": "/tmp/x.txt", "content": "hi"}),
                },
            ]),
        }],
        temperature: None,
        top_k: None,
        top_p: None,
        tools: None,
        tool_choice: None,
        stop_sequences: vec![],
        stream: false,
        thinking: None,
    };
    let chat = anthropic_to_chat_request_json(&req);
    let msgs = chat["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "assistant");
    assert_eq!(msgs[0]["content"], "I'll write the file.");
    let tcs = msgs[0]["tool_calls"].as_array().unwrap();
    assert_eq!(tcs.len(), 1);
    assert_eq!(tcs[0]["id"], "toolu_1");
    assert_eq!(tcs[0]["function"]["name"], "write");
    // arguments must be a STRING (OpenAI shape), not an object.
    let args = tcs[0]["function"]["arguments"].as_str().unwrap();
    assert!(args.contains("/tmp/x.txt"));
}

#[test]
fn translator_tool_result_blocks_split_into_role_tool() {
    // User message with text + tool_result must split into a user
    // text message + a separate role="tool" message carrying
    // tool_call_id and the result text.
    let req = MessagesRequest {
        model: "claude".into(),
        max_tokens: 100,
        system: None,
        messages: vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Blocks(vec![
                ContentBlock::Text {
                    text: "follow up".into(),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".into(),
                    content: Some(ToolResultContent::Text("Wrote.".into())),
                    is_error: None,
                },
            ]),
        }],
        temperature: None,
        top_k: None,
        top_p: None,
        tools: None,
        tool_choice: None,
        stop_sequences: vec![],
        stream: false,
        thinking: None,
    };
    let chat = anthropic_to_chat_request_json(&req);
    let msgs = chat["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["content"], "follow up");
    assert_eq!(msgs[1]["role"], "tool");
    assert_eq!(msgs[1]["tool_call_id"], "toolu_1");
    assert_eq!(msgs[1]["content"], "Wrote.");
}

#[test]
fn translator_tool_result_is_error_prepends_marker() {
    // F6 (2026-04-26): when Anthropic's `is_error: true` is set,
    // the OpenAI tool message content must be prefixed with
    // `[tool error]\n` so the model has a structural failure
    // signal. Repro from dump fix26 seq 27.
    let req = MessagesRequest {
        model: "claude".into(),
        max_tokens: 100,
        system: None,
        messages: vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "toolu_1".into(),
                content: Some(ToolResultContent::Text(
                    "Exit code 127\n/bin/bash: line 1: cargo: command not found".into(),
                )),
                is_error: Some(true),
            }]),
        }],
        temperature: None,
        top_k: None,
        top_p: None,
        tools: None,
        tool_choice: None,
        stop_sequences: vec![],
        stream: false,
        thinking: None,
    };
    let chat = anthropic_to_chat_request_json(&req);
    let msgs = chat["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "tool");
    let content = msgs[0]["content"].as_str().unwrap();
    assert!(
        content.starts_with("[tool error]\n"),
        "expected `[tool error]\\n` prefix, got: {content:?}"
    );
    assert!(
        content.contains("cargo: command not found"),
        "original error text must be preserved after the prefix"
    );
}

#[test]
fn translator_tool_result_no_is_error_passes_text_unchanged() {
    // Successful tool results must NOT receive the prefix —
    // is_error absent or false leaves the content untouched.
    let req = MessagesRequest {
        model: "claude".into(),
        max_tokens: 100,
        system: None,
        messages: vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Blocks(vec![
                ContentBlock::ToolResult {
                    tool_use_id: "toolu_a".into(),
                    content: Some(ToolResultContent::Text("Wrote 42 lines.".into())),
                    is_error: None,
                },
                ContentBlock::ToolResult {
                    tool_use_id: "toolu_b".into(),
                    content: Some(ToolResultContent::Text("Done.".into())),
                    is_error: Some(false),
                },
            ]),
        }],
        temperature: None,
        top_k: None,
        top_p: None,
        tools: None,
        tool_choice: None,
        stop_sequences: vec![],
        stream: false,
        thinking: None,
    };
    let chat = anthropic_to_chat_request_json(&req);
    let msgs = chat["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["content"], "Wrote 42 lines.");
    assert_eq!(msgs[1]["content"], "Done.");
    for m in msgs {
        assert!(
            !m["content"].as_str().unwrap().starts_with("[tool error]"),
            "non-error tool_result must not receive the marker"
        );
    }
}

#[test]
fn translator_tool_result_is_error_default_serde() {
    // Verify serde_json deserialisation: explicit `is_error: true`
    // round-trips, and an absent field deserialises to None
    // (which behaves as the success path).
    let with_err: ContentBlock = serde_json::from_str(
        r#"{"type":"tool_result","tool_use_id":"x","content":"oops","is_error":true}"#,
    )
    .unwrap();
    match with_err {
        ContentBlock::ToolResult { is_error, .. } => {
            assert_eq!(is_error, Some(true));
        }
        _ => panic!("wrong variant"),
    }
    let no_field: ContentBlock = serde_json::from_str(
        r#"{"type":"tool_result","tool_use_id":"x","content":"ok"}"#,
    )
    .unwrap();
    match no_field {
        ContentBlock::ToolResult { is_error, .. } => {
            assert_eq!(is_error, None);
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn translator_tool_choice_any_maps_to_required() {
    let req = MessagesRequest {
        model: "claude".into(),
        max_tokens: 100,
        system: None,
        messages: vec![],
        temperature: None,
        top_k: None,
        top_p: None,
        tools: Some(vec![AnthropicTool {
            name: "x".into(),
            description: None,
            input_schema: serde_json::json!({"type":"object"}),
        }]),
        tool_choice: Some(AnthropicToolChoice {
            choice_type: "any".into(),
            name: None,
        }),
        stop_sequences: vec![],
        stream: false,
        thinking: None,
    };
    let chat = anthropic_to_chat_request_json(&req);
    assert_eq!(chat["tool_choice"], "required");
    let tools = chat["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["function"]["name"], "x");
}

#[test]
fn translator_strips_x_anthropic_system_billing_blocks() {
    let req = MessagesRequest {
        model: "claude".into(),
        max_tokens: 100,
        system: Some(SystemContent::Blocks(vec![
            SystemBlock {
                block_type: "text".into(),
                text: Some("x-anthropic-cch=abc123".into()),
            },
            SystemBlock {
                block_type: "text".into(),
                text: Some("Real instruction.".into()),
            },
        ])),
        messages: vec![],
        temperature: None,
        top_k: None,
        top_p: None,
        tools: None,
        tool_choice: None,
        stop_sequences: vec![],
        stream: false,
        thinking: None,
    };
    let chat = anthropic_to_chat_request_json(&req);
    let sys_content = chat["messages"][0]["content"].as_str().unwrap();
    assert!(!sys_content.contains("x-anthropic-"));
    assert!(sys_content.contains("Real instruction."));
}

// ── Response translator tests (chat_to_anthropic_response) ──

#[test]
fn response_translator_text_only() {
    let chat = serde_json::json!({
        "id": "chatcmpl-abc",
        "model": "qwen-served",
        "choices": [{
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    });
    let r = chat_to_anthropic_response(&chat, "claude-echo".into());
    assert_eq!(r.id, "msg_abc");
    assert_eq!(r.model, "claude-echo");
    assert_eq!(r.usage.input_tokens, 5);
    assert_eq!(r.usage.output_tokens, 1);
    assert_eq!(r.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(r.content.len(), 1);
    match &r.content[0] {
        ResponseBlock::Text { text } => assert_eq!(text, "hello"),
        other => panic!("unexpected block: {:?}", other),
    }
}

