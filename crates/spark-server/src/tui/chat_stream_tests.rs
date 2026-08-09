// SPDX-License-Identifier: AGPL-3.0-only

//! The loopback SSE client, driven against a real socket.
//!
//! Everything here goes over TCP on purpose. The framing bugs this client can
//! have — a frame split across two reads, a chunk carrying both halves of a
//! reply, an error body that has not arrived when the status line has — only
//! exist because the bytes arrive in pieces, and a test that hands the parser
//! one complete buffer cannot see any of them.

use std::sync::mpsc::channel;

use super::fake::{dead_port, refuse, serve, sse};
use super::*;

/// Run one request to completion and collect everything it emitted.
async fn run(port: u16) -> Vec<ChatDelta> {
    let (tx, rx) = channel();
    stream_chat(
        port,
        vec![("user".into(), "hi".into())],
        ThinkingRequest::Auto,
        tx,
    )
    .await;
    rx.try_iter().collect()
}

/// A stable one-line rendering of a delta, so assertions read as a script.
fn tags(ds: &[ChatDelta]) -> Vec<String> {
    ds.iter()
        .map(|d| match d {
            ChatDelta::Token(t) => format!("tok:{t}"),
            ChatDelta::Reasoning(t) => format!("rsn:{t}"),
            ChatDelta::Done {
                tokens,
                reasoning_tokens,
                ..
            } => format!("done:{tokens}/{reasoning_tokens}"),
            ChatDelta::Error(e) => format!("err:{e}"),
        })
        .collect()
}

fn frame(json: &str) -> Vec<u8> {
    format!("data: {json}\n\n").into_bytes()
}

#[tokio::test]
async fn a_refused_connection_is_reported_rather_than_leaving_the_pane_blank() {
    let ds = run(dead_port()).await;
    assert_eq!(ds.len(), 1, "one terminal delta, nothing else");
    assert!(
        matches!(&ds[0], ChatDelta::Error(e) if e.starts_with("connect:")),
        "{:?}",
        tags(&ds)
    );
}

#[tokio::test]
async fn a_non_200_carries_the_servers_own_explanation_not_just_the_status() {
    // Showing the status line alone turned every failure into "503 Service
    // Unavailable" while the server had already said what to do about it.
    let f = serve(|s| {
        refuse(
            s,
            "503 Service Unavailable",
            "application/json",
            r#"{"error":{"message":"no model is loaded; run `spark serve`"}}"#,
        )
    });
    let ds = run(f.port).await;
    let ChatDelta::Error(e) = &ds[0] else {
        panic!("{:?}", tags(&ds))
    };
    assert!(e.contains("503"), "{e}");
    assert!(e.contains("no model is loaded"), "{e}");
}

#[tokio::test]
async fn a_non_200_whose_body_is_not_json_falls_back_to_the_status_line() {
    // A proxy's HTML page must not be pasted into the transcript.
    let f = serve(|s| refuse(s, "502 Bad Gateway", "text/html", "<html>nginx</html>"));
    let ds = run(f.port).await;
    assert_eq!(tags(&ds), vec!["err:HTTP/1.1 502 Bad Gateway"]);
}

#[tokio::test]
async fn a_non_200_whose_body_arrives_after_the_headers_is_still_read() {
    // Headers can land in a read of their own; the client must go back for the
    // body instead of giving up on the first buffer it sees.
    let f = serve(|s| {
        use std::io::Write as _;
        let body = r#"{"error":{"message":"the model is still loading"}}"#;
        let _ = s.write_all(
            format!(
                "HTTP/1.1 503 Service Unavailable\r\nContent-Length: {}\r\n\r\n",
                body.len()
            )
            .as_bytes(),
        );
        let _ = s.flush();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let _ = s.write_all(body.as_bytes());
        let _ = s.flush();
    });
    let ds = run(f.port).await;
    let ChatDelta::Error(e) = &ds[0] else {
        panic!("{:?}", tags(&ds))
    };
    assert!(e.contains("still loading"), "{e}");
}

#[tokio::test]
async fn a_frame_split_mid_character_across_reads_arrives_whole() {
    // The reply is decoded per COMPLETE line; a read that ends inside a
    // multi-byte character must not be lossy-decoded into a replacement char.
    let json = r#"{"choices":[{"delta":{"content":"héllo — ✓"}}]}"#;
    let whole = frame(json);
    // One byte past a UTF-8 lead byte, so the first read ends mid-character.
    let cut = whole.iter().position(|b| *b == 0xE2).expect("multi-byte") + 1;
    let (a, b) = whole.split_at(cut);
    let (a, b) = (a.to_vec(), b.to_vec());
    let f = serve(move |s| sse(s, &[&a, &b, b"data: [DONE]\n\n"]));
    let ds = run(f.port).await;
    assert_eq!(tags(&ds), vec!["tok:héllo — ✓", "done:1/0"]);
}

#[tokio::test]
async fn a_chunk_carrying_both_halves_forwards_the_reasoning_first() {
    // Rare but legal. The reasoning half is the earlier event, and emitting it
    // second would seal the thinking clock before its own text landed.
    let f = serve(|s| {
        sse(
            s,
            &[
                &frame(r#"{"choices":[{"delta":{"reasoning_content":"hm","content":"Paris."}}]}"#),
                b"data: [DONE]\n\n",
            ],
        )
    });
    let ds = run(f.port).await;
    assert_eq!(tags(&ds), vec!["rsn:hm", "tok:Paris.", "done:1/1"]);
}

#[tokio::test]
async fn keepalives_comments_and_unparseable_frames_do_not_break_the_stream() {
    let f = serve(|s| {
        sse(
            s,
            &[
                b": keepalive\n\n",
                b"event: ping\n",
                b"data: {not json\n\n",
                b"datamissing-space\n\n",
                &frame(r#"{"choices":[{"delta":{"content":"ok"}}]}"#),
                b"data: [DONE]\n\n",
            ],
        )
    });
    let ds = run(f.port).await;
    assert_eq!(tags(&ds), vec!["tok:ok", "done:1/0"]);
}

#[tokio::test]
async fn deltas_with_nothing_readable_in_them_are_not_counted_as_tokens() {
    // An empty string, a null, a non-string, and a tool-call-only delta all
    // reach the pane; counting any of them would inflate the token footer and
    // stamp a TTFT on a delta the user cannot read.
    let f = serve(|s| {
        sse(
            s,
            &[
                &frame(r#"{"choices":[{"delta":{"content":""}}]}"#),
                &frame(r#"{"choices":[{"delta":{"content":null}}]}"#),
                &frame(r#"{"choices":[{"delta":{"content":7}}]}"#),
                &frame(r#"{"choices":[{"delta":{"role":"assistant"}}]}"#),
                &frame(r#"{"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}"#),
                &frame(r#"{"choices":[]}"#),
                b"data: [DONE]\n\n",
            ],
        )
    });
    let ds = run(f.port).await;
    assert_eq!(tags(&ds), vec!["done:0/0"]);
    let ChatDelta::Done {
        ttft_ms, tok_per_s, ..
    } = &ds[0]
    else {
        panic!("Done")
    };
    assert!(ttft_ms.is_none(), "nothing readable ever arrived");
    assert!(tok_per_s.is_none(), "no division by an empty reply");
}

#[tokio::test]
async fn a_stream_that_ends_without_done_still_reports_what_it_measured() {
    // The observed zero-answer replies are exactly this shape; reporting
    // nothing made them look like a bug in the pane rather than in the reply.
    let f = serve(|s| {
        sse(
            s,
            &[
                &frame(r#"{"choices":[{"delta":{"reasoning_content":"a"}}]}"#),
                &frame(r#"{"choices":[{"delta":{"reasoning_content":"b"}}]}"#),
            ],
        )
    });
    let ds = run(f.port).await;
    assert_eq!(tags(&ds), vec!["rsn:a", "rsn:b", "done:0/2"]);
    let ChatDelta::Done {
        ttft_ms,
        answer_ttft_ms,
        think_ms,
        tok_per_s,
        ..
    } = &ds[2]
    else {
        panic!("Done")
    };
    assert!(ttft_ms.is_some(), "the first reasoning delta IS the TTFT");
    assert!(answer_ttft_ms.is_none(), "nothing readable ever landed");
    assert!(think_ms.is_some(), "the span falls back to the last delta");
    assert!(
        tok_per_s.is_some(),
        "reasoning tokens are still decode work"
    );
}

#[tokio::test]
async fn a_very_long_single_frame_is_delivered_as_one_token() {
    // 64 KiB in one delta crosses several 8 KiB socket reads; a parser that
    // treated a read boundary as a line boundary would split it.
    let long = "x".repeat(64 * 1024);
    let body = frame(&serde_json::json!({"choices":[{"delta":{"content":long}}]}).to_string());
    let f = serve(move |s| sse(s, &[&body, b"data: [DONE]\n\n"]));
    let ds = run(f.port).await;
    let ChatDelta::Token(t) = &ds[0] else {
        panic!("{:?}", ds.len())
    };
    assert_eq!(t.len(), 64 * 1024);
    assert_eq!(tags(&ds[1..]), vec!["done:1/0"]);
}

#[tokio::test]
async fn ansi_escapes_in_the_model_output_are_passed_through_verbatim() {
    // Sanitizing is the renderer's job, not the wire's: silently rewriting the
    // text here would make the transcript disagree with what the model said.
    let raw = "\u{1b}[31mred\u{1b}[0m\u{7}";
    let body = frame(&serde_json::json!({"choices":[{"delta":{"content":raw}}]}).to_string());
    let f = serve(move |s| sse(s, &[&body, b"data: [DONE]\n\n"]));
    let ds = run(f.port).await;
    let ChatDelta::Token(t) = &ds[0] else {
        panic!("token")
    };
    assert_eq!(t, raw);
}

#[tokio::test]
async fn the_request_is_a_streaming_completion_carrying_the_history_in_order() {
    let f = serve(|s| sse(s, &[b"data: [DONE]\n\n"]));
    let (tx, _rx) = channel::<ChatDelta>();
    stream_chat(
        f.port,
        vec![
            ("user".into(), "hi".into()),
            ("assistant".into(), "hello".into()),
            ("user".into(), "and now?".into()),
        ],
        ThinkingRequest::Off,
        tx,
    )
    .await;
    let req = f.request.recv().expect("the server saw a request");
    assert!(
        req.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"),
        "{req}"
    );
    assert!(req.contains("Accept: text/event-stream"), "{req}");
    let body: serde_json::Value =
        serde_json::from_str(req.split("\r\n\r\n").nth(1).expect("a body")).expect("valid JSON");
    assert_eq!(body["stream"], serde_json::json!(true));
    assert_eq!(
        body["messages"],
        serde_json::json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "and now?"},
        ]),
        "multi-turn history, in order, roles intact"
    );
    assert_eq!(
        body["chat_template_kwargs"],
        serde_json::json!({"enable_thinking": false})
    );
}

#[tokio::test]
async fn a_stream_whose_reader_hung_up_stops_instead_of_filling_a_dead_channel() {
    // The TUI can exit mid-reply. Without the send-error check the task would
    // sit in this loop until the server ran out of things to say.
    let f = serve(|s| {
        use std::io::Write as _;
        let body = frame(r#"{"choices":[{"delta":{"content":"tick"}}]}"#);
        let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
        for _ in 0..100_000 {
            if s.write_all(&body).is_err() {
                return;
            }
        }
    });
    let (tx, rx) = channel::<ChatDelta>();
    drop(rx);
    let done = tokio::time::timeout(
        std::time::Duration::from_secs(20),
        stream_chat(
            f.port,
            vec![("user".into(), "hi".into())],
            ThinkingRequest::On,
            tx,
        ),
    )
    .await;
    assert!(done.is_ok(), "the task gave up once nobody was listening");
}

#[test]
fn nonempty_accepts_only_a_non_empty_string() {
    assert_eq!(nonempty(&serde_json::json!("hi")), Some("hi".into()));
    assert_eq!(nonempty(&serde_json::json!("")), None);
    assert_eq!(nonempty(&serde_json::json!(null)), None);
    assert_eq!(nonempty(&serde_json::json!(3)), None);
    assert_eq!(nonempty(&serde_json::json!({"a": "b"})), None);
}

#[test]
fn find_subslice_reports_the_first_match_and_nothing_when_absent() {
    assert_eq!(find_subslice(b"ab\r\n\r\ncd", b"\r\n\r\n"), Some(2));
    assert_eq!(find_subslice(b"\r\n\r\n", b"\r\n\r\n"), Some(0));
    assert_eq!(find_subslice(b"ab\r\ncd", b"\r\n\r\n"), None);
    // A needle longer than the haystack must not panic on `windows`.
    assert_eq!(find_subslice(b"ab", b"\r\n\r\n"), None);
}

#[test]
fn an_empty_conversation_still_produces_a_well_formed_body() {
    let body: serde_json::Value =
        serde_json::from_str(&request_body(&[], ThinkingRequest::On)).expect("valid JSON");
    assert_eq!(body["messages"], serde_json::json!([]));
    assert_eq!(body["model"], serde_json::json!("atlas-tui"));
}
