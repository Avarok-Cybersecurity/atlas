// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn sse(payload: &str) -> Value {
    serde_json::from_str(payload).unwrap()
}

#[test]
fn chunked_body_split_mid_line_still_yields_one_intact_line() {
    // The failure this decoder exists to prevent: a chunk boundary in the
    // middle of a `data:` line. Naive line-splitting emits two broken halves,
    // both fail to parse as JSON, and the token vanishes from the count.
    let mut r = Reader::default();
    let head = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n";
    assert!(r.push(head).unwrap().is_empty());

    let first = "data: {\"choices\":[{\"de";
    let second = "lta\":{\"content\":\"hi\"}}]}\n";
    let mut wire = Vec::new();
    wire.extend_from_slice(format!("{:x}\r\n{first}\r\n", first.len()).as_bytes());
    let lines = r.push(&wire).unwrap();
    assert!(lines.is_empty(), "no complete line yet, got {lines:?}");

    let mut wire2 = Vec::new();
    wire2.extend_from_slice(format!("{:x}\r\n{second}\r\n", second.len()).as_bytes());
    let lines = r.push(&wire2).unwrap();
    assert_eq!(lines.len(), 1);
    assert!(lines[0].ends_with("}]}"), "{:?}", lines[0]);
    assert!(serde_json::from_str::<Value>(lines[0].strip_prefix("data: ").unwrap()).is_ok());
}

#[test]
fn identity_body_is_read_straight_through() {
    let mut r = Reader::default();
    let lines = r
        .push(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: a\ndata: b\n")
        .unwrap();
    assert_eq!(lines, vec!["data: a", "data: b"]);
}

#[test]
fn a_non_200_status_is_an_error_not_an_empty_stream() {
    let mut r = Reader::default();
    let err = r
        .push(b"HTTP/1.1 404 Not Found\r\n\r\n")
        .unwrap_err()
        .to_string();
    assert!(err.contains("404"), "{err}");
}

#[test]
fn content_deltas_accumulate_text_and_token_count() {
    let mut out = ChatOutcome::default();
    assert!(apply_chunk(
        &sse(r#"{"choices":[{"delta":{"content":"He"}}]}"#),
        &mut out
    ));
    assert!(apply_chunk(
        &sse(r#"{"choices":[{"delta":{"content":"llo"}}]}"#),
        &mut out
    ));
    // A role-only chunk carries no token and must not inflate the count.
    assert!(!apply_chunk(
        &sse(r#"{"choices":[{"delta":{"role":"assistant"}}]}"#),
        &mut out
    ));
    assert_eq!(out.text, "Hello");
    assert_eq!(out.completion_tokens, 2);
}

#[test]
fn tool_call_deltas_assemble_by_index() {
    let mut out = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"id":"c1","function":{"name":"get_","arguments":"{\"a\""}}]}}]}"#),
        &mut out,
    );
    apply_chunk(
        &sse(r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"name":"weather","arguments":":1}"}}]}}]}"#),
        &mut out,
    );
    assert_eq!(out.tool_calls.len(), 1);
    assert_eq!(out.tool_calls[0].id, "c1");
    assert_eq!(out.tool_calls[0].name, "get_weather");
    assert_eq!(out.tool_calls[0].arguments, r#"{"a":1}"#);
}

#[test]
fn server_usage_overrides_the_streamed_delta_count() {
    let mut out = ChatOutcome::default();
    apply_chunk(&sse(r#"{"choices":[{"delta":{"content":"x"}}]}"#), &mut out);
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":37,"prompt_tokens":12,
                "prompt_tokens_details":{"cached_tokens":8}},"choices":[]}"#),
        &mut out,
    );
    assert_eq!(out.completion_tokens, 37);
    assert_eq!(out.prompt_tokens, 12);
    assert_eq!(out.cached_prompt_tokens, 8);
}

#[test]
fn finish_reason_is_captured() {
    let mut out = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#),
        &mut out,
    );
    assert_eq!(out.finish_reason.as_deref(), Some("stop"));
}
