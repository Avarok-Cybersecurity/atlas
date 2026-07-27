// SPDX-License-Identifier: AGPL-3.0-only
//
// Anthropic request adapter: the wire request lowers DIRECTLY into the
// chat IR (no OpenAI-wire-JSON hop), carrying images, reasoning, tool
// calls, and the tool_result error flag as typed structure (issue
// #165 + IR migration).

use super::super::types::{MessagesRequest, MessagesResponse};
use crate::ir::message::ImageSource;
use crate::ir::{ChatRequest, ContentPart, ImageData, Role, ThinkingDirective};

/// Lower an Anthropic request body into the IR envelope (the same path
/// `handlers::messages` takes).
fn lower(req_json: serde_json::Value) -> ChatRequest {
    let req: MessagesRequest = serde_json::from_value(req_json).expect("MessagesRequest");
    req.into()
}

fn text_of(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|p| match p {
            ContentPart::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect()
}

fn images_of(parts: &[ContentPart]) -> Vec<&ImageData> {
    parts
        .iter()
        .filter_map(|p| match p {
            ContentPart::Image(ImageSource { data }) => Some(data),
            _ => None,
        })
        .collect()
}

#[test]
fn user_image_block_is_carried() {
    let ir = lower(serde_json::json!({
        "model": "m", "max_tokens": 16,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}}
        ]}]
    }));
    let user = ir
        .messages
        .iter()
        .find(|m| m.role == Role::User)
        .expect("user msg");
    assert_eq!(
        images_of(&user.content),
        vec![&ImageData::Base64("data:image/png;base64,AAA".into())]
    );
    assert_eq!(text_of(&user.content), "what is this?");
    // Canonical part order: images first, then the joined text.
    assert!(matches!(user.content[0], ContentPart::Image(_)));
}

#[test]
fn tool_result_image_and_error_flag_are_carried() {
    let ir = lower(serde_json::json!({
        "model": "m", "max_tokens": 16,
        "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "is_error": true, "content": [
                {"type": "text", "text": "Exit code 127"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "BBB"}}
            ]}
        ]}]
    }));
    let tool = ir
        .messages
        .iter()
        .find(|m| m.role == Role::Tool)
        .expect("tool msg");
    assert_eq!(tool.tool_call_id.as_deref(), Some("t1"));
    assert_eq!(
        images_of(&tool.content),
        vec![&ImageData::Base64("data:image/jpeg;base64,BBB".into())]
    );
    // The error flag is structural now; the `[tool error]\n` marker is
    // rendered by msg_entry, not baked into the text.
    assert!(tool.tool_error);
    assert_eq!(text_of(&tool.content), "Exit code 127");
}

#[test]
fn thinking_block_becomes_first_class_reasoning() {
    let ir = lower(serde_json::json!({
        "model": "m", "max_tokens": 16,
        "messages": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "let me reason"},
            {"type": "text", "text": "the answer is 4"}
        ]}]
    }));
    let asst = ir
        .messages
        .iter()
        .find(|m| m.role == Role::Assistant)
        .expect("assistant msg");
    assert_eq!(
        asst.reasoning.as_ref().map(|r| r.text.as_str()),
        Some("let me reason")
    );
    assert_eq!(text_of(&asst.content), "the answer is 4");
}

#[test]
fn url_image_source_becomes_url_variant() {
    let ir = lower(serde_json::json!({
        "model": "m", "max_tokens": 16,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}
        ]}]
    }));
    let user = ir
        .messages
        .iter()
        .find(|m| m.role == Role::User)
        .expect("user msg");
    // Typed as Url — the pipeline rejects it with an explicit 400
    // instead of feeding a URL to the base64 decoder.
    assert_eq!(
        images_of(&user.content),
        vec![&ImageData::Url("https://example.com/a.png".into())]
    );
}

#[test]
fn tool_use_and_envelope_fields_are_carried() {
    let ir = lower(serde_json::json!({
        "model": "claude-ish", "max_tokens": 99, "stream": true,
        "temperature": 0.3, "top_k": 5, "top_p": 0.9,
        "stop_sequences": ["STOP"],
        "system": [{"type": "text", "text": "sys"}, {"type": "text", "text": "x-anthropic-billing"}],
        "tools": [{"name": "get_weather", "description": "w", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"},
        "thinking": {"type": "enabled", "budget_tokens": 512},
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "SF"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "sunny"}
            ]}
        ]
    }));
    assert_eq!(ir.model, "claude-ish");
    assert_eq!(ir.max_tokens, 99);
    assert!(ir.stream);
    assert_eq!(ir.sampling.temperature, Some(0.3));
    assert_eq!(ir.sampling.top_k, Some(5));
    assert_eq!(ir.sampling.top_p, Some(0.9));
    assert_eq!(ir.stop, vec!["STOP".to_string()]);
    assert_eq!(ir.n, 1);
    assert_eq!(ir.thinking, ThinkingDirective::On { budget: Some(512) });

    // System: billing block filtered, text kept.
    assert_eq!(ir.messages[0].role, Role::System);
    assert_eq!(text_of(&ir.messages[0].content), "sys");

    // tool_use → structured tool call with parsed arguments.
    let asst = &ir.messages[1];
    assert_eq!(asst.role, Role::Assistant);
    assert_eq!(asst.tool_calls.len(), 1);
    assert_eq!(asst.tool_calls[0].id, "c1");
    assert_eq!(asst.tool_calls[0].name, "get_weather");
    assert_eq!(
        asst.tool_calls[0].arguments,
        serde_json::json!({"city": "SF"})
    );

    // tool_result → Tool message linked by id, no error flag.
    let tool = &ir.messages[2];
    assert_eq!(tool.role, Role::Tool);
    assert_eq!(tool.tool_call_id.as_deref(), Some("c1"));
    assert!(!tool.tool_error);
    assert_eq!(text_of(&tool.content), "sunny");

    // Tools + tool_choice converted to the neutral definitions.
    assert_eq!(ir.tools.len(), 1);
    assert_eq!(ir.tools[0].function.name, "get_weather");
    match ir.tool_choice {
        Some(crate::tool_parser::ToolChoice::Mode(ref m)) => assert_eq!(m, "required"),
        ref other => panic!("expected Mode(required), got {other:?}"),
    }
}

#[test]
fn thinking_disabled_and_unspecified_directives() {
    let ir = lower(serde_json::json!({
        "model": "m", "max_tokens": 16,
        "thinking": {"type": "disabled", "budget_tokens": 100},
        "messages": [{"role": "user", "content": "hi"}]
    }));
    assert_eq!(ir.thinking, ThinkingDirective::Off);

    let ir = lower(serde_json::json!({
        "model": "m", "max_tokens": 16,
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": "hi"}]
    }));
    assert_eq!(ir.thinking, ThinkingDirective::On { budget: None });

    let ir = lower(serde_json::json!({
        "model": "m", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}]
    }));
    assert_eq!(ir.thinking, ThinkingDirective::Unspecified);
}

#[test]
fn rendered_prompt_json_matches_retired_json_hop_output() {
    // GOLDEN captured from the retired anthropic_to_chat_request_json
    // path (pre-deletion): the same fixture, run through the old
    // wire→OpenAI-JSON→IncomingMessage→IR lowering, produced exactly
    // this build_json_messages output. The direct adapter must render
    // identical prompt bytes (kv-cache prefix stability across the
    // migration).
    let ir = lower(serde_json::json!({
        "model": "m", "max_tokens": 32,
        "system": "Be helpful.",
        "messages": [
            {"role": "user", "content": "check the weather"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "need the tool"},
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "SF"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "c1", "is_error": true, "content": "network down"}
            ]},
            {"role": "user", "content": "try again"}
        ]
    }));
    let out = crate::api::chat::test_build_msg_entries(&ir.messages, true).expect("builds");
    let json = crate::api::chat::test_build_json_messages(&out);
    let expected = serde_json::json!([
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "check the weather"},
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "SF"}}}
            ],
            "reasoning_content": "need the tool"
        },
        {"role": "tool", "content": "[tool error]\nnetwork down"},
        {"role": "user", "content": "try again"}
    ]);
    assert_eq!(serde_json::Value::Array(json), expected);
}

#[test]
fn wire_surfaces_lower_to_identical_ir_core() {
    // The same conversation expressed in the Anthropic and OpenAI chat
    // wire dialects must lower to the SAME IR messages/thinking/stop —
    // the whole point of the narrow waist. (The Responses surface
    // lowers through the chat wire, so chat parity covers it.)
    let anth = lower(serde_json::json!({
        "model": "m", "max_tokens": 64,
        "system": "Be helpful.",
        "temperature": 0.5, "top_p": 0.9, "top_k": 4,
        "stop_sequences": ["END"],
        "thinking": {"type": "enabled", "budget_tokens": 512},
        "tools": [{"name": "get_weather", "description": "w",
                    "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
                {"type": "text", "text": "what is this?"}
            ]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "SF"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "sunny"}
            ]},
        ]
    }));

    let chat: crate::openai::ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "m", "max_tokens": 64,
        "temperature": 0.5, "top_p": 0.9, "top_k": 4,
        "stop": ["END"],
        "thinking": {"type": "enabled", "budget_tokens": 512},
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "w",
                    "parameters": {"type": "object"}}}],
        "messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                {"type": "text", "text": "what is this?"}
            ]},
            {"role": "assistant", "content": "Checking.", "reasoning_content": "hmm",
             "tool_calls": [{"id": "c1", "type": "function",
                              "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ]
    }))
    .expect("chat wire parses");
    let chat_ir = ChatRequest::from(chat);

    assert_eq!(anth.messages, chat_ir.messages);
    assert_eq!(anth.thinking, chat_ir.thinking);
    assert_eq!(anth.stop, chat_ir.stop);
    assert_eq!(anth.max_tokens, chat_ir.max_tokens);
    assert_eq!(anth.sampling.temperature, chat_ir.sampling.temperature);
    assert_eq!(anth.sampling.top_p, chat_ir.sampling.top_p);
    assert_eq!(anth.sampling.top_k, chat_ir.sampling.top_k);
    assert_eq!(anth.tools.len(), chat_ir.tools.len());
    assert_eq!(anth.tools[0].function.name, chat_ir.tools[0].function.name);
    assert_eq!(
        anth.tools[0].function.parameters,
        chat_ir.tools[0].function.parameters
    );
}

// ── response direction (IR → MessagesResponse) ──

fn ir_response(choice: crate::ir::Choice) -> crate::ir::ChatResponse {
    crate::ir::ChatResponse {
        id: "abc".into(),
        model: "m".into(),
        created: 1,
        choices: vec![choice],
        usage: crate::ir::Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            cached_prompt_tokens: 3,
            reasoning_tokens: 2,
            time_to_first_token_ms: 0.0,
            response_tokens_per_second: 0.0,
        },
    }
}

fn choice() -> crate::ir::Choice {
    crate::ir::Choice {
        index: 0,
        content: Some("hello".into()),
        reasoning: None,
        tool_calls: Vec::new(),
        refusal: None,
        finish_reason: crate::ir::FinishReason::Stop,
        matched_stop: None,
        logprobs: None,
    }
}

#[test]
fn response_maps_reasoning_text_tools_and_usage() {
    let mut c = choice();
    c.reasoning = Some("thinking".into());
    c.tool_calls = vec![crate::ir::message::ToolCall {
        id: "c1".into(),
        name: "f".into(),
        arguments: serde_json::json!({"x": 1}),
    }];
    c.finish_reason = crate::ir::FinishReason::ToolCalls;
    let json = serde_json::to_value(MessagesResponse::from(ir_response(c))).unwrap();
    assert_eq!(json["id"], "msg_abc");
    assert_eq!(json["model"], "m");
    assert_eq!(json["stop_reason"], "tool_use");
    assert_eq!(json["usage"]["input_tokens"], 10);
    assert_eq!(json["usage"]["output_tokens"], 5);
    // B5: prefix-cache hits surface as Anthropic cache accounting.
    assert_eq!(json["usage"]["cache_read_input_tokens"], 3);
    assert_eq!(json["content"][0]["type"], "thinking");
    assert_eq!(json["content"][0]["thinking"], "thinking");
    assert_eq!(json["content"][1]["type"], "text");
    assert_eq!(json["content"][1]["text"], "hello");
    assert_eq!(json["content"][2]["type"], "tool_use");
    assert_eq!(json["content"][2]["name"], "f");
    assert_eq!(json["content"][2]["input"], serde_json::json!({"x": 1}));
}

#[test]
fn response_stop_reason_mapping() {
    // content_filter → refusal (Anthropic's dedicated stop reason).
    let mut c = choice();
    c.finish_reason = crate::ir::FinishReason::ContentFilter;
    let json = serde_json::to_value(MessagesResponse::from(ir_response(c))).unwrap();
    assert_eq!(json["stop_reason"], "refusal");

    // length → max_tokens.
    let mut c = choice();
    c.finish_reason = crate::ir::FinishReason::Length;
    let json = serde_json::to_value(MessagesResponse::from(ir_response(c))).unwrap();
    assert_eq!(json["stop_reason"], "max_tokens");

    // A client stop sequence gets stop_sequence + the matched echo.
    let mut c = choice();
    c.matched_stop = Some("END".into());
    let json = serde_json::to_value(MessagesResponse::from(ir_response(c))).unwrap();
    assert_eq!(json["stop_reason"], "stop_sequence");
    assert_eq!(json["stop_sequence"], "END");
}

#[test]
fn response_empty_content_omits_text_block() {
    let mut c = choice();
    c.content = Some(String::new());
    let json = serde_json::to_value(MessagesResponse::from(ir_response(c))).unwrap();
    assert_eq!(json["content"].as_array().unwrap().len(), 0);
    assert_eq!(json["stop_reason"], "end_turn");
}
