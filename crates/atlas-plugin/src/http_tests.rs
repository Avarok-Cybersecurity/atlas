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
    // The guarantee is that a non-200 never reads as a successful empty
    // stream. *When* it is raised changed deliberately: the reader now waits
    // for the body, because the body is where the server explains itself. With
    // no Content-Length and no body, as here, that wait ends at EOF.
    let mut r = Reader::default();
    let lines = r.push(b"HTTP/1.1 404 Not Found\r\n\r\n").unwrap();
    assert!(lines.is_empty(), "a failed response yields no data lines");
    let err = r.finish().unwrap_err().to_string();
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
fn a_reasoning_delta_is_a_token_and_starts_the_clock() {
    // Regression: a thinking model streams `reasoning_content` first, and
    // counting only `content` made TTFT measure the time to the END of the
    // reasoning block. On a short prompt the reply is reasoning ALMOST
    // ENTIRELY -- 59 of 64 tokens on the observed run -- so the benchmark
    // reported "no token was emitted" and measured nothing at all.
    let mut out = ChatOutcome::default();
    assert!(
        apply_chunk(
            &sse(r#"{"choices":[{"delta":{"reasoning_content":"Let me think"}}]}"#),
            &mut out
        ),
        "a reasoning delta must count as carried, or it cannot start the TTFT clock"
    );
    assert!(apply_chunk(
        &sse(r#"{"choices":[{"delta":{"content":"4"}}]}"#),
        &mut out
    ));
    // Reasoning stays OUT of `text`: every scorer downstream parses `text`
    // for the answer, so folding thinking into it would score the model's
    // chain of thought as its reply.
    assert_eq!(out.text, "4", "reasoning must not leak into the answer");
    assert_eq!(out.reasoning, "Let me think");
    assert_eq!(
        out.completion_tokens, 2,
        "both are decoded tokens -- the server's usage.completion_tokens \
         includes reasoning_tokens, and the streamed count must agree"
    );
}

#[test]
fn an_empty_reasoning_delta_carries_nothing() {
    let mut out = ChatOutcome::default();
    assert!(!apply_chunk(
        &sse(r#"{"choices":[{"delta":{"reasoning_content":""}}]}"#),
        &mut out
    ));
    assert_eq!(out.completion_tokens, 0);
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

// ---- non-200 responses: the body carries the explanation ----

/// A 503 with the server's own JSON error body, split at `at` bytes to
/// reproduce headers and body arriving in separate reads.
fn error_response(body: &str) -> Vec<u8> {
    format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

const NO_MODEL_BODY: &str = r#"{"error":{"message":"no model is loaded — Open the Library (press 4), choose a model and a recipe, and start it; then retry this request.","type":"model_not_loaded"}}"#;

#[test]
fn an_error_body_reaches_the_caller_instead_of_just_the_status_line() {
    // The observed failure: benchmarking a modelless server reported only
    // `endpoint returned "HTTP/1.1 503 Service Unavailable"`, discarding the
    // one part of the response that said what to do about it.
    let mut r = Reader::default();
    let err = r.push(&error_response(NO_MODEL_BODY)).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("503"), "keeps the status: {msg}");
    assert!(msg.contains("Library"), "and carries the hint: {msg}");
}

#[test]
fn an_error_body_arriving_after_its_headers_is_still_reported() {
    // Headers can land in a read of their own — the original code bailed on
    // that first read, so the body was guaranteed unread whenever this split
    // occurred, which is precisely when the body was largest.
    let whole = error_response(NO_MODEL_BODY);
    let split = find(&whole, b"\r\n\r\n").unwrap() + 4;

    let mut r = Reader::default();
    assert!(
        r.push(&whole[..split]).is_ok(),
        "headers alone are not yet a verdict — the body is still coming"
    );
    let msg = format!("{}", r.push(&whole[split..]).unwrap_err());
    assert!(msg.contains("Library"), "hint survives the split: {msg}");
}

#[test]
fn a_chunked_error_body_is_decoded_before_it_is_parsed() {
    // This is how the server actually sends errors — `transfer-encoding:
    // chunked`, no Content-Length. Collecting the body raw yields
    // `14A\r\n{...}\r\n0\r\n\r\n`, which is not JSON, so the hint was dropped
    // and the failure was indistinguishable from "there was no body". Caught
    // only by running it against a real modelless server; every
    // Content-Length fixture passed while this was broken.
    let body = NO_MODEL_BODY;
    let raw = format!(
        "HTTP/1.1 503 Service Unavailable\r\ncontent-type: application/json\r\n\
         transfer-encoding: chunked\r\n\r\n{:X}\r\n{body}\r\n0\r\n\r\n",
        body.len()
    );
    let mut r = Reader::default();
    let msg = format!("{}", r.push(raw.as_bytes()).unwrap_err());
    assert!(msg.contains("503"), "{msg}");
    assert!(msg.contains("Library"), "chunked hint must survive: {msg}");
}

#[test]
fn a_chunked_error_split_across_reads_is_still_decoded() {
    let body = NO_MODEL_BODY;
    let raw = format!(
        "HTTP/1.1 503 Service Unavailable\r\ntransfer-encoding: chunked\r\n\r\n\
         {:X}\r\n{body}\r\n0\r\n\r\n",
        body.len()
    );
    let bytes = raw.as_bytes();
    // Split inside the chunk data — the case naive framing gets wrong.
    let cut = bytes.len() - 40;
    let mut r = Reader::default();
    assert!(r.push(&bytes[..cut]).is_ok(), "partial chunk: keep waiting");
    let msg = format!("{}", r.push(&bytes[cut..]).unwrap_err());
    assert!(msg.contains("Library"), "{msg}");
}

#[test]
fn an_error_without_content_length_is_reported_at_eof() {
    // No length and no chunking means "the body ends when I close". Waiting for
    // a length that will never arrive would hang the run instead of failing it.
    let mut r = Reader::default();
    let raw = format!("HTTP/1.1 500 Internal Server Error\r\n\r\n{NO_MODEL_BODY}");
    assert!(r.push(raw.as_bytes()).is_ok(), "still open, still waiting");
    let msg = format!("{}", r.finish().unwrap_err());
    assert!(msg.contains("500"), "{msg}");
    assert!(msg.contains("Library"), "{msg}");
}

#[test]
fn an_unparseable_error_body_still_reports_the_status() {
    // A proxy's HTML, a truncated body, a plain-text gateway error: the status
    // line is all there is, and it must not be lost trying to parse the rest.
    let mut r = Reader::default();
    let raw = "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 9\r\n\r\n<html></".to_string();
    let _ = r.push(raw.as_bytes());
    let msg = format!("{}", r.finish().unwrap_err());
    assert!(msg.contains("502"), "{msg}");
}

#[test]
fn a_200_response_is_completely_unaffected_by_the_error_path() {
    // The success path runs on every request; it must not have acquired a new
    // branch that can misfire.
    let mut r = Reader::default();
    let raw = "HTTP/1.1 200 OK\r\n\r\ndata: {\"a\":1}\n";
    let lines = r.push(raw.as_bytes()).unwrap();
    assert_eq!(lines, vec!["data: {\"a\":1}".to_string()]);
    assert!(r.finish().is_ok(), "no pending error on a good response");
}

#[test]
fn a_huge_error_body_is_capped_rather_than_buffered_without_limit() {
    let mut r = Reader::default();
    let big = "x".repeat(MAX_ERROR_BODY + 4096);
    let raw = format!("HTTP/1.1 503 Service Unavailable\r\n\r\n{big}");
    // Bails at the cap rather than growing until the sender stops.
    assert!(r.push(raw.as_bytes()).is_err());
    assert!(r.body.len() <= MAX_ERROR_BODY + 4096);
}

#[test]
fn message_from_body_ignores_bodies_that_are_not_openai_shaped() {
    assert_eq!(message_from_body(""), None);
    assert_eq!(message_from_body("{"), None);
    assert_eq!(message_from_body(r#"{"error":"str"}"#), None);
    assert_eq!(message_from_body(r#"{"error":{"message":"  "}}"#), None);
    assert_eq!(
        message_from_body(r#"{"error":{"message":"boom"}}"#).as_deref(),
        Some("boom")
    );
}

#[test]
fn a_whole_chunked_error_response_yields_its_message_in_one_shot() {
    // The TUI chat pane has the entire response, not a stream. It must get the
    // same answer as the streaming reader — including de-chunking, which is
    // the step a hand-written second parser would have missed.
    let body = NO_MODEL_BODY;
    let raw = format!(
        "HTTP/1.1 503 Service Unavailable\r\ntransfer-encoding: chunked\r\n\r\n\
         {:X}\r\n{body}\r\n0\r\n\r\n",
        body.len()
    );
    let msg = error_message_from_response(raw.as_bytes()).expect("a message is there");
    assert!(msg.contains("Library"), "{msg}");
}

#[test]
fn a_successful_response_has_no_error_message_to_extract() {
    let raw = "HTTP/1.1 200 OK\r\n\r\ndata: {\"a\":1}\n";
    assert_eq!(error_message_from_response(raw.as_bytes()), None);
}

#[test]
fn an_error_response_with_an_unreadable_body_yields_nothing_rather_than_junk() {
    let raw = "HTTP/1.1 502 Bad Gateway\r\n\r\n<html>nope</html>";
    assert_eq!(error_message_from_response(raw.as_bytes()), None);
}
