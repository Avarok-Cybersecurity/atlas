// SPDX-License-Identifier: AGPL-3.0-only

//! The warm-up sanity check, ported from `run_tier.sh:159-182`.
//!
//! The harness header states the reason plainly: a direct API probe asserts the
//! model answers `4` for `2+2` and "HALTS on failure — saves the operator from
//! waiting 25 min on a catastrophic regression."
//!
//! `/v1/models` returning 200 only proves a server is listening. It says
//! nothing about whether the checkpoint still decodes, which is exactly the
//! failure that costs a whole tier.

use std::time::Duration;

use anyhow::{Result, bail};
use serde_json::json;

use crate::http;
use crate::plugin::PluginHandle;

/// Verbatim from the harness's `warmup_endpoint` body.
pub const SANITY_PROMPT: &str = "What is 2+2? Respond with just the number.";

/// `run_tier.sh:168-170` merges `content` **and** `reasoning_content` before
/// grepping — "some configs route to reasoning", so grading the content field
/// alone would fail a thinking model that answered correctly.
pub fn answered(text: &str, reasoning: &str) -> bool {
    [text, reasoning].iter().any(|channel| {
        channel.match_indices('4').any(|(index, _)| {
            let before = channel[..index].chars().next_back();
            let after = channel[index + 1..].chars().next();
            let identifier = |c: char| c.is_ascii_alphanumeric() || c == '_';
            !before.is_some_and(identifier) && !after.is_some_and(identifier)
        })
    })
}

/// Ask the endpoint 2+2 and fail the run if it cannot say 4.
pub async fn sanity_check(handle: &PluginHandle, timeout: Duration) -> Result<()> {
    let target = handle.target();
    let body = json!({
        "model": target.model,
        "messages": [{"role": "user", "content": SANITY_PROMPT}],
        // 80 tokens, as the harness allows. Thinking is disabled for THIS probe
        // only (and only here) because 80 tokens is not a thinking budget — the
        // Gate A trajectories themselves keep thinking on, per the module docs.
        // `chat_template_kwargs.enable_thinking` is the key that is honoured;
        // a bare `thinking` field is silently ignored.
        "chat_template_kwargs": {"enable_thinking": false},
        "max_tokens": 80,
        "temperature": 0.0,
        "stream": true,
    });
    let outcome = http::chat_stream(target, &body, timeout).await?;
    if outcome.text.trim().is_empty() && outcome.reasoning.trim().is_empty() {
        bail!(
            "warm-up: {} returned no parseable response",
            target.base_url
        );
    }
    if !answered(&outcome.text, &outcome.reasoning) {
        bail!(
            "warm-up: {} did not answer '4' to 2+2 — catastrophic regression, halting: {:?}",
            target.base_url,
            crate::benchmarks::one_line(format!("{} {}", outcome.text, outcome.reasoning))
        );
    }
    Ok(())
}

// **A repeat-until-settled warm-up was tried here on 2026-08-07, and REJECTED
// on the measurement.** It is written down because the reasoning that leads to
// it is sound and someone will reach for it again.
//
// The observation it was built on holds. Repeating ONE tool-attached request
// against a fresh 35B FP8 serve on the Gate A recipe:
//
// | regime | `--speculative` | identical replies |
// |---|---|---|
// | first 6 requests of a fresh serve | on  | 1/6 (3 distinct) |
// | first 6 requests of a fresh serve | off | 4/6 (2 distinct) |
// | next 6 on the same process        | on  | 6/6 |
// | next 6 on the same process        | off | 6/6 |
//
// So a cold endpoint is not repeatable and a warmed one is — and a tier starts
// a fresh serve and measures iterations 0, 1, 2 straight into the cold regime.
// Sending discardable probes until two replies matched (settling took 3) looks
// like the obvious fix.
//
// It made the gate WORSE, and not marginally: with the warm-up, N=10 scored
// **3/10 webserver_ok · 1/10 followed_directions**, with 8 turns of 90 ending
// in `finish_reason: length` — the model degenerating into a repetition loop
// mid-run. The same binary with the warm-up removed and nothing else changed
// scored **3/3 webserver_ok** with no degeneration at all. Probing leaves
// prefix-cache and SSM-snapshot state behind that the real requests then
// partially match, and a partially matched SSM snapshot is not a cheaper
// prefix — it is the wrong recurrent state.
//
// Two rules follow, both paid for: do not send this endpoint traffic the
// measurement does not need, and A/B any determinism fix against the score
// before shipping it.

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;
    use crate::artifacts::ArtifactStore;
    use crate::plugin::TargetEndpoint;

    async fn reasoning_server() -> (u16, tokio::sync::oneshot::Receiver<serde_json::Value>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (sent, request) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let mut buf = [0u8; 4096];
            let (header_end, length) = loop {
                let n = socket.read(&mut buf).await.unwrap();
                assert!(n > 0, "client closed before sending its request");
                bytes.extend_from_slice(&buf[..n]);
                let Some(header_end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&bytes[..header_end]);
                let length = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length: ")
                            .and_then(|value| value.parse::<usize>().ok())
                    })
                    .expect("request must declare content length");
                if bytes.len() >= header_end + 4 + length {
                    break (header_end, length);
                }
            };
            let body = serde_json::from_slice(&bytes[header_end + 4..header_end + 4 + length])
                .expect("request body must be JSON");
            sent.send(body).unwrap();

            let body = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"2+2 is 4\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
            socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        (port, request)
    }

    #[test]
    fn the_prompt_is_the_harness_prompt() {
        let harness = include_str!("../../../../../bench/fp8_dgx2_drift/harness/run_tier.sh");
        let request = harness
            .lines()
            .find_map(|line| {
                line.trim_start()
                    .strip_prefix("-d '")
                    .and_then(|body| body.strip_suffix("' 2>&1)"))
            })
            .expect("run_tier.sh must carry the warm-up request as JSON");
        let body: serde_json::Value =
            serde_json::from_str(request).expect("the harness warm-up request must be valid JSON");
        assert_eq!(body["messages"][0]["content"], SANITY_PROMPT);
    }

    #[test]
    fn either_channel_may_carry_the_answer() {
        assert!(answered("4", ""));
        assert!(answered("", "2+2 is 4"));
        assert!(answered("The answer is 4.", ""));
        // A reply that never says 4 is the catastrophic-regression signal.
        assert!(!answered("", ""));
        assert!(!answered("five", "let me think"));
        assert!(!answered("42", ""), "42 is not the answer to 2+2");
        assert!(
            !answered("status 404", ""),
            "an error code is not an answer"
        );
        assert!(
            !answered("4ever", ""),
            "a digit inside a word is not an answer"
        );
    }

    #[test]
    fn the_harness_uses_the_same_answer_boundary() {
        let harness = include_str!("../../../../../bench/fp8_dgx2_drift/harness/run_tier.sh");
        assert!(
            harness.contains("grep -Eq '(^|[^[:alnum:]_])4([^[:alnum:]_]|$)'"),
            "the shell and Rust preflights must reject 42 and 404 alike"
        );
    }

    #[tokio::test]
    async fn sanity_check_sends_the_declared_request_and_reads_reasoning() {
        let (port, request) = reasoning_server().await;
        let (events, receiver) = std::sync::mpsc::channel();
        let handle = PluginHandle::new(
            1,
            TargetEndpoint::local(port, "test-model"),
            ArtifactStore::with_root(std::env::temp_dir().join("atlas-preflight-test")),
            events,
            Arc::new(AtomicBool::new(false)),
        );

        sanity_check(&handle, Duration::from_secs(2))
            .await
            .expect("a standalone 4 in reasoning must pass");
        drop(receiver);
        let body = request.await.expect("server must observe the request");
        assert_eq!(body["model"], "test-model");
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": SANITY_PROMPT}])
        );
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"enable_thinking": false})
        );
        assert_eq!(body["max_tokens"], 80);
        assert_eq!(body["temperature"], 0.0);
        assert_eq!(body["stream"], true);
    }
}
