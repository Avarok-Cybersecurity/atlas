// SPDX-License-Identifier: AGPL-3.0-only

//! Chat tab backend: a loopback SSE client against this server's own
//! `/v1/chat/completions`. Requests traverse the normal HTTP path —
//! indistinguishable from an external client, zero scheduler coupling.
//!
//! The request runs on the tokio runtime (Handle captured at TUI start); the
//! stream's deltas cross to the TUI thread over a std mpsc that the event
//! loop drains each tick.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq)]
pub enum Role {
    User,
    Model,
}

pub struct ChatMessage {
    pub role: Role,
    pub text: String,
    /// Reply footer stats (model messages, once complete).
    pub ttft_ms: Option<f64>,
    pub tok_per_s: Option<f64>,
    pub tokens: usize,
}

pub enum ChatDelta {
    Token(String),
    Done {
        ttft_ms: Option<f64>,
        tok_per_s: Option<f64>,
        tokens: usize,
    },
    Error(String),
}

#[derive(Default)]
pub struct ChatState {
    pub transcript: Vec<ChatMessage>,
    pub input: String,
    pub streaming: bool,
    /// Transcript viewport, in WRAPPED rows above the bottom. `None` follows the
    /// streaming tip; `Some(n)` holds station n rows up. Same contract as the Main
    /// log pane's `log_scroll`, so both panes answer to the same keys.
    pub scroll: Option<usize>,
    rx: Option<Receiver<ChatDelta>>,
    cancel: Option<tokio::sync::oneshot::Sender<()>>,
    runtime: Option<tokio::runtime::Handle>,
}

impl ChatState {
    pub fn set_runtime(&mut self, handle: tokio::runtime::Handle) {
        self.runtime = Some(handle);
    }

    /// Scroll the transcript by `rows` (positive = back toward older turns).
    /// Landing at or past the bottom restores follow, so a stream that is running
    /// keeps painting its tip without a second keypress.
    pub fn scroll_by(&mut self, rows: i32) {
        let cur = self.scroll.unwrap_or(0) as i32;
        let next = cur + rows;
        self.scroll = if next <= 0 { None } else { Some(next as usize) };
    }

    /// Snap back to the live tip.
    pub fn follow(&mut self) {
        self.scroll = None;
    }

    /// Send the current input as a user message and start streaming a reply.
    pub fn send(&mut self, port: u16) {
        let prompt = self.input.trim().to_string();
        if prompt.is_empty() || self.streaming {
            return;
        }
        self.input.clear();
        // Sending is an explicit "show me the new reply" — resume follow.
        self.follow();
        self.transcript.push(ChatMessage {
            role: Role::User,
            text: prompt.clone(),
            ttft_ms: None,
            tok_per_s: None,
            tokens: 0,
        });
        self.transcript.push(ChatMessage {
            role: Role::Model,
            text: String::new(),
            ttft_ms: None,
            tok_per_s: None,
            tokens: 0,
        });
        let Some(rt) = self.runtime.clone() else {
            if let Some(last) = self.transcript.last_mut() {
                last.text = "(chat unavailable: no runtime handle)".into();
            }
            return;
        };
        let (tx, rx) = channel();
        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
        self.rx = Some(rx);
        self.cancel = Some(cancel_tx);
        self.streaming = true;
        // History (excluding the empty model placeholder) for multi-turn.
        let messages: Vec<(String, String)> = self
            .transcript
            .iter()
            .filter(|m| !(m.role == Role::Model && m.text.is_empty()))
            .map(|m| {
                (
                    match m.role {
                        Role::User => "user".to_string(),
                        Role::Model => "assistant".to_string(),
                    },
                    m.text.clone(),
                )
            })
            .collect();
        rt.spawn(async move {
            tokio::select! {
                _ = stream_chat(port, messages, tx.clone()) => {}
                _ = cancel_rx => {
                    let _ = tx.send(ChatDelta::Done { ttft_ms: None, tok_per_s: None, tokens: 0 });
                }
            }
        });
    }

    pub fn cancel(&mut self) {
        if let Some(c) = self.cancel.take() {
            let _ = c.send(());
        }
    }

    /// Drain pending deltas into the transcript (event-loop tick).
    pub fn pump(&mut self) {
        let Some(rx) = &self.rx else { return };
        let deltas: Vec<ChatDelta> = rx.try_iter().collect();
        for d in deltas {
            match d {
                ChatDelta::Token(t) => {
                    if let Some(m) = self.transcript.last_mut() {
                        m.text.push_str(&t);
                        m.tokens += 1;
                    }
                }
                ChatDelta::Done {
                    ttft_ms,
                    tok_per_s,
                    tokens,
                } => {
                    if let Some(m) = self.transcript.last_mut() {
                        m.ttft_ms = ttft_ms;
                        m.tok_per_s = tok_per_s;
                        if tokens > 0 {
                            m.tokens = tokens;
                        }
                    }
                    self.streaming = false;
                    self.rx = None;
                    self.cancel = None;
                    return;
                }
                ChatDelta::Error(e) => {
                    if let Some(m) = self.transcript.last_mut() {
                        m.text = format!("(error: {e})");
                    }
                    self.streaming = false;
                    self.rx = None;
                    self.cancel = None;
                    return;
                }
            }
        }
    }
}

/// POST the chat request and forward SSE deltas. Plain HTTP/1.1 over a
/// loopback TcpStream — no TLS, no client stack beyond tokio's.
async fn stream_chat(port: u16, messages: Vec<(String, String)>, tx: Sender<ChatDelta>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let started = Instant::now();
    let mut first_token: Option<Instant> = None;
    let mut tokens = 0usize;

    let body = serde_json::json!({
        "model": "atlas-tui",
        "stream": true,
        "messages": messages
            .iter()
            .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
            .collect::<Vec<_>>(),
    })
    .to_string();

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
                    let _ = tx.send(ChatDelta::Error(status));
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
                let ttft = first_token.map(|t| (t - started).as_secs_f64() * 1000.0);
                let tps = first_token.map(|t| {
                    let gen_secs = t.elapsed().as_secs_f64().max(1e-3);
                    tokens as f64 / gen_secs
                });
                let _ = tx.send(ChatDelta::Done {
                    ttft_ms: ttft,
                    tok_per_s: tps,
                    tokens,
                });
                return;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data)
                && let Some(delta) = v["choices"][0]["delta"]["content"]
                    .as_str()
                    .filter(|s| !s.is_empty())
            {
                if first_token.is_none() {
                    first_token = Some(Instant::now());
                }
                tokens += 1;
                if tx.send(ChatDelta::Token(delta.to_string())).is_err() {
                    return; // TUI gone
                }
            }
        }
    }
    let _ = tx.send(ChatDelta::Done {
        ttft_ms: None,
        tok_per_s: None,
        tokens,
    });
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}
