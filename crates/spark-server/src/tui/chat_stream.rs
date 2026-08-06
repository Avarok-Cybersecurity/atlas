// SPDX-License-Identifier: AGPL-3.0-only

//! The Chat pane's loopback SSE client.
//!
//! Split out of `chat.rs` at the per-file cap; unchanged in shape. Requests
//! traverse the normal HTTP path — indistinguishable from an external client,
//! zero scheduler coupling — and answers cross to the render thread over the
//! std mpsc that `ChatState::pump` drains each tick.

use std::sync::mpsc::Sender;
use std::time::Instant;

use super::chat::ChatDelta;
use super::chat_thinking::ThinkingRequest;

/// Cap on how much of a non-200 body to read before giving up on it.
///
/// An error body is a small JSON object; anything larger is a proxy's HTML or a
/// server that will not stop talking, and neither is worth unbounded memory on
/// the path where something has already gone wrong.
const MAX_ERROR_BODY: usize = 64 * 1024;

/// Clocks for one reply.
///
/// `first_any` is the honest TTFT: the first delta of ANY kind. Stamping it on
/// the first `content` delta alone reported 12-20 SECONDS on a model that had
/// answered in 410 ms, because 200-odd `reasoning_content` deltas streamed
/// first and none of them counted. `first_answer` is the genuinely different
/// second number — when something the user reads finally appeared.
#[derive(Default)]
struct Clocks {
    first_any: Option<Instant>,
    first_answer: Option<Instant>,
    first_reasoning: Option<Instant>,
    last_reasoning: Option<Instant>,
    tokens: usize,
    reasoning_tokens: usize,
}

impl Clocks {
    fn done(&self, started: Instant) -> ChatDelta {
        let ms = |t: Instant| (t - started).as_secs_f64() * 1000.0;
        // Reasoning that never terminates before the answer starts is the
        // normal case; reasoning that never terminates AT ALL (the observed
        // zero-answer replies) falls back to the last delta seen, so the
        // summary still reports a real span instead of nothing.
        let think_ms = self.first_reasoning.and_then(|start| {
            self.first_answer
                .or(self.last_reasoning)
                .map(|end| (end - start).as_secs_f64() * 1000.0)
        });
        let total = self.tokens + self.reasoning_tokens;
        // Decode rate over EVERY token generated, reasoning included: they are
        // the same decode work, and dividing the whole wall by the answer
        // tokens alone would advertise a stall that never happened.
        let tok_per_s = self.first_any.filter(|_| total > 0).map(|t| {
            let gen_secs = t.elapsed().as_secs_f64().max(1e-3);
            total as f64 / gen_secs
        });
        ChatDelta::Done {
            ttft_ms: self.first_any.map(ms),
            answer_ttft_ms: self.first_answer.map(ms),
            think_ms,
            tok_per_s,
            tokens: self.tokens,
            reasoning_tokens: self.reasoning_tokens,
        }
    }
}

/// Build the request body.
///
/// `Auto` omits `chat_template_kwargs` entirely rather than sending a guess at
/// the model's default — see [`ThinkingRequest`].
fn request_body(messages: &[(String, String)], thinking: ThinkingRequest) -> String {
    let mut body = serde_json::json!({
        "model": "atlas-tui",
        "stream": true,
        "messages": messages
            .iter()
            .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
            .collect::<Vec<_>>(),
    });
    if let Some(enable) = thinking.enable_thinking() {
        body["chat_template_kwargs"] = serde_json::json!({ "enable_thinking": enable });
    }
    body.to_string()
}

/// POST the chat request and forward SSE deltas. Plain HTTP/1.1 over a
/// loopback TcpStream — no TLS, no client stack beyond tokio's.
pub(super) async fn stream_chat(
    port: u16,
    messages: Vec<(String, String)>,
    thinking: ThinkingRequest,
    tx: Sender<ChatDelta>,
) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let started = Instant::now();
    let mut c = Clocks::default();
    let body = request_body(&messages, thinking);

    let mut stream = match tokio::net::TcpStream::connect(("127.0.0.1", port)).await {
        Ok(s) => s,
        Err(e) => {
            let _ = tx.send(ChatDelta::Error(format!("connect: {e}")));
            return;
        }
    };
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\
         Content-Type: application/json\r\nAccept: text/event-stream\r\n\
         Connection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    if let Err(e) = stream.write_all(req.as_bytes()).await {
        let _ = tx.send(ChatDelta::Error(format!("write: {e}")));
        return;
    }

    // Read the whole response incrementally; parse SSE `data:` lines from the
    // (possibly chunked) body. Chunked framing is tolerated by line-splitting:
    // SSE data lines never contain bare hex-length lines' shape ambiguity in
    // practice because each chunk boundary falls between lines for this
    // server (axum writes whole events).
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let mut header_done = false;
    let mut consumed = 0usize;
    loop {
        let n = match stream.read(&mut tmp).await {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                let _ = tx.send(ChatDelta::Error(format!("read: {e}")));
                return;
            }
        };
        buf.extend_from_slice(&tmp[..n]);
        if !header_done {
            if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..pos]).to_string();
                if !head.starts_with("HTTP/1.1 200") && !head.starts_with("HTTP/1.0 200") {
                    let status = head.lines().next().unwrap_or("?").to_string();
                    // Drain the rest of the response before giving up on it.
                    // Showing only the status line turned every failure into
                    // "HTTP/1.1 503 Service Unavailable" — the server had
                    // already explained itself in the body, and the pane threw
                    // the explanation away. The body is not necessarily in
                    // `buf` yet: headers can arrive in a read of their own.
                    while buf.len() < MAX_ERROR_BODY {
                        match stream.read(&mut tmp).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        }
                    }
                    // The WHOLE response, headers included: the error body is
                    // chunked, and de-chunking is the shared reader's job.
                    let msg = atlas_plugin::http::error_message_from_response(&buf)
                        .map(|m| format!("{status} — {m}"))
                        .unwrap_or(status);
                    let _ = tx.send(ChatDelta::Error(msg));
                    return;
                }
                consumed = pos + 4;
                header_done = true;
            } else {
                continue;
            }
        }
        // Process complete lines.
        while let Some(nl) = buf[consumed..].iter().position(|b| *b == b'\n') {
            let line_end = consumed + nl;
            let line = String::from_utf8_lossy(&buf[consumed..line_end])
                .trim()
                .to_string();
            consumed = line_end + 1;
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                let _ = tx.send(c.done(started));
                return;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };
            let delta = &v["choices"][0]["delta"];
            // Reasoning FIRST: a chunk carrying both is rare but legal, and
            // the reasoning half of it is the earlier event.
            if let Some(text) = nonempty(&delta["reasoning_content"]) {
                let now = Instant::now();
                c.first_any.get_or_insert(now);
                c.first_reasoning.get_or_insert(now);
                c.last_reasoning = Some(now);
                c.reasoning_tokens += 1;
                if tx.send(ChatDelta::Reasoning(text)).is_err() {
                    return; // TUI gone
                }
            }
            if let Some(text) = nonempty(&delta["content"]) {
                let now = Instant::now();
                c.first_any.get_or_insert(now);
                c.first_answer.get_or_insert(now);
                c.tokens += 1;
                if tx.send(ChatDelta::Token(text)).is_err() {
                    return; // TUI gone
                }
            }
        }
    }
    // The stream ended without `[DONE]`. Report what was measured anyway —
    // the zero-answer replies observed with `response_format` + thinking on
    // are exactly this shape, and reporting nothing made them look like a bug
    // in the pane rather than in the reply.
    let _ = tx.send(c.done(started));
}

fn nonempty(v: &serde_json::Value) -> Option<String> {
    v.as_str().filter(|s| !s.is_empty()).map(str::to_string)
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// The loopback HTTP server the client tests run against. `pub(super)` so the
/// reducer's tests in `chat_more_tests` share this one definition rather than
/// growing a second, kinder fake of their own.
#[cfg(test)]
#[path = "chat_fake_server.rs"]
pub(super) mod fake;

#[cfg(test)]
#[path = "chat_stream_tests.rs"]
mod stream_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn kwargs(req: ThinkingRequest) -> Option<serde_json::Value> {
        let body: serde_json::Value =
            serde_json::from_str(&request_body(&[("user".into(), "hi".into())], req))
                .expect("valid JSON");
        body.get("chat_template_kwargs").cloned()
    }

    #[test]
    fn auto_sends_no_thinking_key_at_all() {
        assert_eq!(kwargs(ThinkingRequest::Auto), None);
    }

    #[test]
    fn off_and_on_send_the_key_that_is_actually_honored() {
        // ★ `enable_thinking` nested under `chat_template_kwargs`. A bare
        // `{"thinking": false}` is accepted and then ignored, which is
        // indistinguishable from a working toggle until you read the reply.
        assert_eq!(
            kwargs(ThinkingRequest::Off),
            Some(serde_json::json!({"enable_thinking": false}))
        );
        assert_eq!(
            kwargs(ThinkingRequest::On),
            Some(serde_json::json!({"enable_thinking": true}))
        );
    }

    #[test]
    fn the_ttft_clock_stops_on_the_first_delta_of_any_kind() {
        // The measured bug: 410 ms of real TTFT reported as 12-20 s because
        // only `content` deltas were counted.
        let started = Instant::now();
        let mut c = Clocks::default();
        let think = Instant::now();
        c.first_any = Some(think);
        c.first_reasoning = Some(think);
        c.reasoning_tokens = 200;
        std::thread::sleep(std::time::Duration::from_millis(15));
        let answer = Instant::now();
        c.first_answer = Some(answer);
        c.tokens = 10;
        let ChatDelta::Done {
            ttft_ms,
            answer_ttft_ms,
            think_ms,
            ..
        } = c.done(started)
        else {
            panic!("Done")
        };
        let (ttft, ans) = (ttft_ms.expect("ttft"), answer_ttft_ms.expect("answer"));
        assert!(ans > ttft, "the answer landed after the first token");
        assert!(think_ms.expect("thought") >= 10.0, "and it thought first");
    }

    #[test]
    fn a_reply_that_only_ever_thought_still_reports_a_span() {
        // `response_format` + thinking on returned all reasoning and no answer
        // on 2 of 4 requests. The footer must still be a real measurement.
        let started = Instant::now();
        let mut c = Clocks::default();
        let t = Instant::now();
        c.first_any = Some(t);
        c.first_reasoning = Some(t);
        c.last_reasoning = Some(t + std::time::Duration::from_millis(500));
        c.reasoning_tokens = 247;
        let ChatDelta::Done {
            think_ms,
            tokens,
            reasoning_tokens,
            tok_per_s,
            ..
        } = c.done(started)
        else {
            panic!("Done")
        };
        assert_eq!(tokens, 0, "no answer arrived");
        assert_eq!(reasoning_tokens, 247);
        assert!((think_ms.expect("thought") - 500.0).abs() < 1.0);
        assert!(
            tok_per_s.is_some(),
            "reasoning tokens are still decode work"
        );
    }

    #[test]
    fn a_reply_with_no_tokens_at_all_reports_no_rate() {
        let c = Clocks::default();
        let ChatDelta::Done {
            ttft_ms, tok_per_s, ..
        } = c.done(Instant::now())
        else {
            panic!("Done")
        };
        assert!(ttft_ms.is_none());
        assert!(tok_per_s.is_none(), "no division by an empty reply");
    }
}
