// SPDX-License-Identifier: AGPL-3.0-only

//! A minimal OpenAI-compatible SSE server, for driving benchmarks end to end.
//!
//! It answers `/v1/models` and streams `/v1/chat/completions` in **chunked**
//! transfer-encoding with a deliberate mid-line chunk split, because that is the
//! framing case the client's decoder exists to survive and the one a naive
//! line-splitter drops tokens on.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

pub struct MockEndpoint {
    pub port: u16,
    /// Chat completions served so far.
    pub requests: Arc<AtomicUsize>,
}

/// Start the mock on an ephemeral port. Each reply streams `tokens` content
/// deltas, `ttft_ms` after the request, then `[DONE]`.
pub async fn start(tokens: usize, ttft: Duration, gap: Duration) -> MockEndpoint {
    start_saying(None, tokens, ttft, gap).await
}

/// As [`start`], but every completion answers with `reply` instead of filler.
///
/// One implementation, so the chunk-splitting the decoder is tested against is
/// identical in both cases.
pub async fn start_saying(
    reply: Option<String>,
    tokens: usize,
    ttft: Duration,
    gap: Duration,
) -> MockEndpoint {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let requests = Arc::new(AtomicUsize::new(0));
    let counter = requests.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let counter = counter.clone();
            let reply = reply.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 64 * 1024];
                let mut request = Vec::new();
                // Read until the headers end; the body follows Content-Length
                // but the mock does not need it.
                loop {
                    let Ok(n) = socket.read(&mut buf).await else {
                        return;
                    };
                    if n == 0 {
                        return;
                    }
                    request.extend_from_slice(&buf[..n]);
                    if request.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&request).to_string();
                if head.starts_with("GET /v1/models") {
                    // CHUNKED, like Atlas — not Content-Length. A reader that
                    // parses from the first `{` to the end of the buffer trips
                    // over the trailing `0\r\n\r\n`, which is exactly how the
                    // model check came to be silently useless on a real server.
                    let body = r#"{"object":"list","data":[{"id":"mock"}]}"#;
                    let _ = socket
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n                              Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                        )
                        .await;
                    let _ = write_chunk(&mut socket, body).await;
                    let _ = socket.write_all(b"0\r\n\r\n").await;
                    return;
                }
                counter.fetch_add(1, Ordering::Relaxed);
                let _ = socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                          Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                    )
                    .await;
                tokio::time::sleep(ttft).await;
                // A canned reply is one delta; filler is `tokens` of them.
                let deltas: Vec<String> = match &reply {
                    Some(text) => vec![text.clone()],
                    None => (0..tokens).map(|i| format!("t{i} ")).collect(),
                };
                let tokens = deltas.len();
                for (i, delta) in deltas.iter().enumerate() {
                    let escaped = delta.replace('\\', "\\\\").replace('"', "\\\"");
                    let payload = format!(
                        "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{escaped}\"}}}}]}}\n"
                    );
                    if i == 0 {
                        // Split the FIRST event across two chunks, mid-line.
                        let (a, b) = payload.split_at(payload.len() / 2);
                        if write_chunk(&mut socket, a).await.is_err() {
                            return;
                        }
                        if write_chunk(&mut socket, b).await.is_err() {
                            return;
                        }
                    } else if write_chunk(&mut socket, &payload).await.is_err() {
                        return;
                    }
                    tokio::time::sleep(gap).await;
                }
                let usage = format!(
                    "data: {{\"usage\":{{\"completion_tokens\":{tokens},\"prompt_tokens\":42,\
                     \"prompt_tokens_details\":{{\"cached_tokens\":40}}}},\"choices\":[]}}\n"
                );
                let _ = write_chunk(&mut socket, &usage).await;
                let _ = write_chunk(&mut socket, "data: [DONE]\n").await;
                let _ = socket.write_all(b"0\r\n\r\n").await;
                let _ = socket.shutdown().await;
            });
        }
    });
    MockEndpoint { port, requests }
}

async fn write_chunk(socket: &mut tokio::net::TcpStream, text: &str) -> std::io::Result<()> {
    socket
        .write_all(format!("{:x}\r\n{text}\r\n", text.len()).as_bytes())
        .await
}
