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
                    let body = br#"{"object":"list","data":[{"id":"mock"}]}"#;
                    let _ = socket
                        .write_all(
                            format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                                body.len()
                            )
                            .as_bytes(),
                        )
                        .await;
                    let _ = socket.write_all(body).await;
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
                for i in 0..tokens {
                    let payload = format!(
                        "data: {{\"choices\":[{{\"delta\":{{\"content\":\"t{i} \"}}}}]}}\n"
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
