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

// ── `stop_reason` extension field (#927 / #1000 / #1002) ────────────

/// Blocking twin of the streaming assertions in
/// `encode_stream::tests`: the non-streaming `chat.completion` choice
/// reports `finish_reason: "length"` AND names the guard that cut it.
///
/// Round-13 cell V (`--prefill-varlen-batch`) is the receipt — 6 of 16
/// responses truncated at 49 tokens by the content-loop watchdog, all
/// of them reporting bare `"length"`. `finish_reason` is deliberately
/// unchanged here: the detail is an extension FIELD (unknown fields are
/// ignored by every SDK; unknown enum VALUES hard-fail typed clients),
/// which is the seam vLLM uses for its own `stop_reason`.
#[test]
fn blocking_choice_carries_stop_reason_beside_length() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            reasoning_content: None,
            annotations: None,
            refusal: None,
            content: Some("the the the".to_string()),
            tool_calls: None,
        },
        finish_reason: "length".to_string(),
        logprobs: None,
        stop_reason: Some("content_loop_watchdog"),
    };
    let json = serde_json::to_value(&choice).expect("serializable");
    assert_eq!(json["finish_reason"], "length");
    assert_eq!(json["stop_reason"], "content_loop_watchdog");
}

/// NEGATIVE: a natural EOS stop must not gain a key. `to_value` rather
/// than a substring check, so `"stop_reason":null` cannot sneak past —
/// absent and null are different bytes to a client that branches on
/// `"stop_reason" in choice`.
#[test]
fn blocking_choice_omits_stop_reason_on_natural_stop() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            reasoning_content: None,
            annotations: None,
            refusal: None,
            content: Some("done.".to_string()),
            tool_calls: None,
        },
        finish_reason: "stop".to_string(),
        logprobs: None,
        stop_reason: None,
    };
    let json = serde_json::to_value(&choice).expect("serializable");
    assert_eq!(json["finish_reason"], "stop");
    assert!(
        json.get("stop_reason").is_none(),
        "key must be ABSENT, not null: {json}"
    );
    assert!(
        !serde_json::to_string(&choice)
            .unwrap()
            .contains("stop_reason"),
        "wire moved for a normal stop"
    );
}
