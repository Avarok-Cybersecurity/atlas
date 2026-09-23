// SPDX-License-Identifier: AGPL-3.0-only

//! A streamed request that FAILED must reach the caller as an error, never as
//! a successful (0-token) completion.
//!
//! The G25 serve answered ~1000 requests after a poisoned verify with HTTP 200
//! and no tokens, and the concurrency harness reported 0 errors: the client
//! skipped the in-band `{"error":…}` frame as an unknown chunk and treated EOF
//! as the end of a completed stream. A CHILD of `http_tests` so the loopback
//! mock (`endpoint_answering`) stays single-sourced.

use super::*;

const SSE_HEAD: &str =
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";

async fn stream(body: &str) -> Result<ChatOutcome> {
    let target = endpoint_answering(format!("{SSE_HEAD}{body}")).await;
    chat_stream(&target, &serde_json::json!({}), Duration::from_secs(5)).await
}

fn delta(text: &str) -> String {
    format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{text}\"}}}}]}}\n\n")
}

/// Chat's OpenAI envelope, exactly as spark-server's `handle_error` writes it,
/// followed by the `[DONE]` its encoder always appends.
#[tokio::test]
async fn a_chat_error_frame_is_a_request_error() {
    let err = stream(&format!(
        "{}data: {{\"error\":{{\"message\":\"prefill_chunk failed: status 901\",\
         \"type\":\"server_error\",\"code\":500}}}}\n\ndata: [DONE]\n\n",
        delta("a")
    ))
    .await
    .expect_err("an in-band error frame must not read as a completion");
    assert!(
        format!("{err:#}").contains("prefill_chunk failed: status 901"),
        "the server's reason must reach the caller: {err:#}"
    );
}

/// Legacy `/v1/completions`' published frame, `{"error":"<msg>"}`, before any
/// token — the exact 0-token shape the G25 harness counted as a success.
#[tokio::test]
async fn a_legacy_completions_error_frame_is_a_request_error() {
    let err = stream("data: {\"error\":\"Scheduler queue closed\"}\n\ndata: [DONE]\n\n")
        .await
        .expect_err("the string-shaped frame is an error too");
    assert!(
        format!("{err:#}").contains("Scheduler queue closed"),
        "{err:#}"
    );
}

/// Tokens, then EOF: no finish_reason, no `[DONE]`. A cut-off stream.
#[tokio::test]
async fn a_truncated_stream_is_a_request_error() {
    let err = stream(&format!("{}{}", delta("a"), delta("b")))
        .await
        .expect_err("a stream cut off mid-response must not read as a completion");
    assert!(format!("{err:#}").contains("truncated"), "{err:#}");
    // And the empty 200: headers, then EOF.
    stream("")
        .await
        .expect_err("an empty 200 stream is not a 0-token success");
}

/// NEGATIVE CONTROLS: a completed stream still reads as one, whether the
/// server ends it with `[DONE]` or only with its finish chunk.
#[tokio::test]
async fn a_completed_stream_is_still_a_success() {
    let out = stream(&format!("{}data: [DONE]\n\n", delta("a")))
        .await
        .expect("[DONE] ends a stream");
    assert_eq!(out.text, "a");
    let out = stream(&format!(
        "{}data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n",
        delta("b")
    ))
    .await
    .expect("a finish_reason chunk followed by EOF is a completed stream");
    assert_eq!(out.finish_reason.as_deref(), Some("stop"));
}

#[test]
fn stream_error_reads_both_published_shapes_and_nothing_else() {
    let v = |s: &str| serde_json::from_str::<Value>(s).unwrap();
    assert_eq!(
        stream_error(&v(r#"{"error":{"message":"m","type":"server_error"}}"#)).as_deref(),
        Some("m")
    );
    assert_eq!(stream_error(&v(r#"{"error":"m"}"#)).as_deref(), Some("m"));
    assert_eq!(stream_error(&v(r#"{"error":null,"choices":[]}"#)), None);
    assert_eq!(
        stream_error(&v(r#"{"choices":[{"delta":{"content":"error"}}]}"#)),
        None
    );
}
