// SPDX-License-Identifier: AGPL-3.0-only

//! Chat-completions wire-format contracts: `token_ids` opt-in and the
//! reasoning single-field rule.

use crate::openai::*;

// ── return_token_ids wire format ────────────────────────────────────

#[test]
fn token_ids_absent_by_default_keeps_wire_byte_identical() {
    // PCND: a client that did not opt in must see no `token_ids` key.
    let chunk = ChatCompletionChunk::content_chunk("m", "id", "hi".into());
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(!json.contains("token_ids"), "default wire changed: {json}");
    // Empty `with_token_ids` is a no-op (still absent).
    let chunk =
        ChatCompletionChunk::content_chunk("m", "id", "hi".into()).with_token_ids(Vec::new());
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(!json.contains("token_ids"));
}

#[test]
fn with_token_ids_stamps_first_choice() {
    let chunk =
        ChatCompletionChunk::content_chunk("m", "id", "hi".into()).with_token_ids(vec![10, 20, 30]);
    assert_eq!(chunk.choices[0].token_ids, vec![10, 20, 30]);
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(json.contains("\"token_ids\":[10,20,30]"), "{json}");
    // No choices (usage-only chunk) → no panic, no-op.
    let usage = Usage {
        prompt_tokens: 1,
        completion_tokens: 1,
        total_tokens: 2,
        prompt_tokens_details: None,
        completion_tokens_details: None,
        time_to_first_token_ms: 0.0,
        response_tokens_per_second: 0.0,
        expert_activation: None,
    };
    let chunk = ChatCompletionChunk::usage_only_chunk("m", "id", usage).with_token_ids(vec![1, 2]);
    assert!(chunk.choices.is_empty());
}

// ── reasoning wire format: exactly one field ────────────────────────
// A response carrying BOTH `reasoning_content` and a `reasoning` mirror is
// rejected by strict OpenAI-compatible clients (they assert exactly one).
// Atlas emits only `reasoning_content` — these lock that contract in.

#[test]
fn reasoning_delta_emits_only_reasoning_content() {
    let chunk = ChatCompletionChunk::reasoning_chunk("m", "id", "thinking".into());
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(
        json.contains("\"reasoning_content\":\"thinking\""),
        "reasoning_content missing: {json}"
    );
    assert!(
        !json.contains("\"reasoning\":"),
        "mirror `reasoning` field leaked into stream delta: {json}"
    );
}

#[test]
fn blocking_message_emits_only_reasoning_content() {
    let msg = ChatMessage {
        role: "assistant".into(),
        reasoning_content: Some("thinking".into()),
        content: Some("hi".into()),
        tool_calls: None,
        annotations: None,
        refusal: None,
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(
        json.contains("\"reasoning_content\":\"thinking\""),
        "reasoning_content missing: {json}"
    );
    assert!(
        !json.contains("\"reasoning\":"),
        "mirror `reasoning` field leaked into message: {json}"
    );
}

// ── response constructors carry reasoning_content ───────────────────
// Both convenience constructors must thread the reasoning trace through
// to the wire message — hardwired `None` silently dropped thinking turns.

fn test_usage() -> Usage {
    Usage {
        prompt_tokens: 1,
        completion_tokens: 1,
        total_tokens: 2,
        prompt_tokens_details: None,
        completion_tokens_details: None,
        time_to_first_token_ms: 0.0,
        response_tokens_per_second: 0.0,
        expert_activation: None,
    }
}

#[test]
fn new_response_carries_reasoning_content() {
    let resp = ChatCompletionResponse::new(
        "m",
        "hi".into(),
        Some("thinking".into()),
        test_usage(),
        "stop",
    );
    assert_eq!(
        resp.choices[0].message.reasoning_content.as_deref(),
        Some("thinking")
    );
    let json = serde_json::to_string(&resp).unwrap();
    assert!(
        json.contains("\"reasoning_content\":\"thinking\""),
        "reasoning_content missing: {json}"
    );
    // Absent reasoning stays off the wire entirely.
    let resp = ChatCompletionResponse::new("m", "hi".into(), None, test_usage(), "stop");
    let json = serde_json::to_string(&resp).unwrap();
    assert!(!json.contains("reasoning_content"), "{json}");
}

#[test]
fn tool_call_response_carries_reasoning_content() {
    let resp = ChatCompletionResponse::with_tool_calls(
        "m",
        None,
        Some("need the tool".into()),
        vec![crate::tool_parser::ToolCall {
            id: "call_1".into(),
            call_type: "function".into(),
            function: crate::tool_parser::FunctionCall {
                name: "get_weather".into(),
                arguments: "{}".into(),
            },
        }],
        test_usage(),
    );
    assert_eq!(
        resp.choices[0].message.reasoning_content.as_deref(),
        Some("need the tool")
    );
    assert_eq!(resp.choices[0].finish_reason, "tool_calls");
    let json = serde_json::to_string(&resp).unwrap();
    assert!(
        json.contains("\"reasoning_content\":\"need the tool\""),
        "reasoning_content missing: {json}"
    );
}
