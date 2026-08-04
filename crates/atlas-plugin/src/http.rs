// SPDX-License-Identifier: AGPL-3.0-only

//! A minimal OpenAI-compatible client: one streaming chat call, measured.
//!
//! Deliberately raw HTTP/1.1 over `tokio::net::TcpStream` — the targets are
//! loopback or a LAN box, never TLS, and a benchmark must not measure a client
//! stack it did not intend to. This is the same shape as `tui::chat`, with one
//! difference that matters here: **chunked transfer-encoding is decoded
//! properly**. Line-splitting the raw socket works until a chunk boundary lands
//! mid-`data:` line, at which point the JSON fails to parse and the token is
//! silently dropped — invisible in a chat pane, but this client's token counts
//! are the measurement.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::plugin::TargetEndpoint;

/// A tool call assembled from streamed deltas.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Raw JSON string as emitted by the model (not parsed — BFCL scores the
    /// arguments itself and a re-serialization would change them).
    pub arguments: String,
}

/// Everything one request produced, plus its timings.
#[derive(Clone, Debug, Default)]
pub struct ChatOutcome {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    /// Client-measured: request start → first content/tool delta.
    pub ttft_ms: Option<f64>,
    /// Client-measured decode inter-token latency.
    pub tpot_ms: Option<f64>,
    /// Client-measured: request start → last byte.
    pub e2e_ms: f64,
    /// Streamed delta count, overridden by the server's `usage` when present.
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub cached_prompt_tokens: usize,
}

/// POST `/v1/chat/completions` with `"stream": true` and measure it.
pub async fn chat_stream(
    target: &TargetEndpoint,
    body: &Value,
    timeout: Duration,
) -> Result<ChatOutcome> {
    tokio::time::timeout(timeout, chat_stream_inner(target, body))
        .await
        .map_err(|_| anyhow!("request exceeded {:.0}s", timeout.as_secs_f64()))?
}

async fn chat_stream_inner(target: &TargetEndpoint, body: &Value) -> Result<ChatOutcome> {
    let (host, port) = target.host_port()?;
    let payload = serde_json::to_string(body)?;
    let mut sock = TcpStream::connect((host.as_str(), port))
        .await
        .with_context(|| format!("connecting to {}", target.base_url))?;
    // Nagle would coalesce our single request write with nothing, but it also
    // delays the server's first small SSE frame on some stacks — and that frame
    // IS the TTFT measurement.
    let _ = sock.set_nodelay(true);
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {host}:{port}\r\n\
         Content-Type: application/json\r\nAccept: text/event-stream\r\n\
         Connection: close\r\nContent-Length: {}\r\n\r\n{payload}",
        payload.len()
    );

    let started = Instant::now();
    sock.write_all(request.as_bytes()).await.context("write")?;

    let mut reader = Reader::default();
    let mut out = ChatOutcome::default();
    let mut first_delta: Option<Instant> = None;
    let mut last_delta: Option<Instant> = None;
    let mut buf = [0u8; 16 * 1024];
    'read: loop {
        let n = sock.read(&mut buf).await.context("read")?;
        if n == 0 {
            // EOF. If an error response was still being collected — no
            // Content-Length, server closed to signal the end — report it now
            // with whatever body arrived, rather than returning an empty
            // success.
            reader.finish()?;
            break;
        }
        for line in reader.push(&buf[..n])? {
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                break 'read;
            }
            let Ok(chunk) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            if apply_chunk(&chunk, &mut out) {
                let now = Instant::now();
                first_delta.get_or_insert(now);
                last_delta = Some(now);
            }
        }
    }
    out.e2e_ms = started.elapsed().as_secs_f64() * 1000.0;
    out.ttft_ms = first_delta.map(|t| (t - started).as_secs_f64() * 1000.0);
    // TPOT is the DECODE rate: it excludes prefill, so it is measured from the
    // first delta, not from the request start, and needs at least two tokens.
    if let (Some(f), Some(l)) = (first_delta, last_delta)
        && out.completion_tokens >= 2
        && l > f
    {
        out.tpot_ms =
            Some((l - f).as_secs_f64() * 1000.0 / (out.completion_tokens.saturating_sub(1)) as f64);
    }
    Ok(out)
}

/// Fold one SSE chunk into the outcome. Returns true when it carried a token.
fn apply_chunk(chunk: &Value, out: &mut ChatOutcome) -> bool {
    if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
        if let Some(c) = usage.get("completion_tokens").and_then(Value::as_u64) {
            out.completion_tokens = c as usize;
        }
        if let Some(p) = usage.get("prompt_tokens").and_then(Value::as_u64) {
            out.prompt_tokens = p as usize;
        }
        if let Some(c) = usage
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(Value::as_u64)
        {
            out.cached_prompt_tokens = c as usize;
        }
    }
    let Some(choice) = chunk.get("choices").and_then(|c| c.get(0)) else {
        return false;
    };
    if let Some(r) = choice.get("finish_reason").and_then(Value::as_str) {
        out.finish_reason = Some(r.to_string());
    }
    let Some(delta) = choice.get("delta") else {
        return false;
    };
    let mut carried = false;
    if let Some(content) = delta.get("content").and_then(Value::as_str)
        && !content.is_empty()
    {
        out.text.push_str(content);
        out.completion_tokens += 1;
        carried = true;
    }
    if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
        for call in calls {
            let idx = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            if out.tool_calls.len() <= idx {
                out.tool_calls.resize(idx + 1, ToolCall::default());
            }
            let slot = &mut out.tool_calls[idx];
            if let Some(id) = call.get("id").and_then(Value::as_str) {
                slot.id.push_str(id);
            }
            if let Some(f) = call.get("function") {
                if let Some(name) = f.get("name").and_then(Value::as_str) {
                    slot.name.push_str(name);
                }
                if let Some(args) = f.get("arguments").and_then(Value::as_str) {
                    slot.arguments.push_str(args);
                }
            }
            carried = true;
        }
    }
    carried
}

/// The model ids `GET /v1/models` reports.
///
/// [`probe`] checks only the status line; this parses the body, which is the
/// difference between "something is listening" and "it is serving what you
/// asked for". Atlas answers a completion regardless of the `model` field, so
/// a wrong name is otherwise invisible until the numbers look strange.
pub async fn list_models(target: &TargetEndpoint, timeout: Duration) -> Result<Vec<String>> {
    let body = get_models(target, timeout).await?;
    let start = body
        .find('{')
        .context("no JSON in the /v1/models response")?;
    // Parse the FIRST value and ignore whatever follows. Atlas replies with
    // `Transfer-Encoding: chunked`, so the body carries hex length prefixes and
    // a terminating `0\r\n\r\n`; plain `from_str` fails on those trailing
    // bytes, which is exactly how this check came to be silently useless
    // against a real server while passing against a Content-Length mock.
    let doc: serde_json::Value = serde_json::Deserializer::from_str(&body[start..])
        .into_iter()
        .next()
        .context("/v1/models returned an empty body")?
        .context("/v1/models did not return JSON")?;
    Ok(doc
        .get("data")
        .and_then(|d| d.as_array())
        .map(|rows| {
            rows.iter()
                .filter_map(|r| r.get("id").and_then(|i| i.as_str()))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default())
}

/// `GET /v1/models` — used as a reachability probe before a sweep starts, so a
/// wrong port fails in a second instead of producing a suspiciously fast run.
pub async fn probe(target: &TargetEndpoint, timeout: Duration) -> Result<()> {
    let (host, port) = target.host_port()?;
    let fut = async {
        let mut sock = TcpStream::connect((host.as_str(), port)).await?;
        let req =
            format!("GET /v1/models HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
        sock.write_all(req.as_bytes()).await?;
        let mut head = Vec::new();
        let mut buf = [0u8; 1024];
        while head.len() < 512 {
            let n = sock.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            head.extend_from_slice(&buf[..n]);
        }
        anyhow::Ok(String::from_utf8_lossy(&head).into_owned())
    };
    let head = tokio::time::timeout(timeout, fut)
        .await
        .map_err(|_| anyhow!("{} did not answer within {:?}", target.base_url, timeout))?
        .with_context(|| format!("probing {}", target.base_url))?;
    let status = head.lines().next().unwrap_or_default();
    if !status.contains(" 200") {
        bail!("{} /v1/models returned {status:?}", target.base_url);
    }
    Ok(())
}

/// The whole `/v1/models` response, headers included.
async fn get_models(target: &TargetEndpoint, timeout: Duration) -> Result<String> {
    let (host, port) = target.host_port()?;
    let fut = async {
        let mut sock = TcpStream::connect((host.as_str(), port)).await?;
        let req =
            format!("GET /v1/models HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
        sock.write_all(req.as_bytes()).await?;
        let mut body = Vec::new();
        let mut buf = [0u8; 4096];
        // Read to EOF: `Connection: close` means the server ends the body, and
        // a fixed cap would truncate a long model list into invalid JSON.
        loop {
            let n = sock.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&buf[..n]);
            if body.len() > 1 << 20 {
                break;
            }
        }
        anyhow::Ok(String::from_utf8_lossy(&body).into_owned())
    };
    tokio::time::timeout(timeout, fut)
        .await
        .map_err(|_| anyhow!("{} did not answer within {:?}", target.base_url, timeout))?
        .with_context(|| format!("reading models from {}", target.base_url))
}

/// Pull the human-readable message out of an OpenAI-shaped error body.
///
/// Returns `None` for anything that is not a JSON object carrying
/// `error.message` as a string — empty, truncated, HTML from a proxy, a
/// plain-text 502 — so callers can fall back to the status line.
///
/// This lives in `atlas-plugin` rather than in the server because
/// `spark-server` depends on this crate and not the reverse: putting it here is
/// what lets the benchmark reader and the server's TUI share one definition
/// instead of two that can drift.
pub fn message_from_body(body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let msg = v.get("error")?.get("message")?.as_str()?.trim();
    (!msg.is_empty()).then(|| msg.to_string())
}

/// The human-readable message from a complete non-200 HTTP response.
///
/// Takes the whole response — status line, headers and body — and returns
/// `None` unless it is a non-200 carrying a parseable OpenAI-shaped error.
///
/// For callers that have the entire response in hand rather than a stream, and
/// deliberately routed through this module's own incremental reader rather than
/// given a second parser (no intra-doc link: that reader is private, and a
/// public item linking to a private one is a rustdoc error under this crate's
/// `deny(warnings)`):
/// error bodies are framed like any other, this server sends them
/// `transfer-encoding: chunked`, and a second de-chunker written by hand is
/// exactly how one caller ends up understanding `14A\r\n{...}` and the other
/// not. There is one decoder, and both use it.
pub fn error_message_from_response(raw: &[u8]) -> Option<String> {
    let mut r = Reader::default();
    // Both may bail — that IS the error being reported — and the decoded body
    // is what we came for either way.
    let _ = r.push(raw);
    let _ = r.finish();
    r.error_status.as_ref()?;
    message_from_body(String::from_utf8_lossy(&r.body).trim())
}

/// Cap on how much of a non-200 body to buffer before reporting it.
const MAX_ERROR_BODY: usize = 64 * 1024;

fn is_chunked(head: &str) -> bool {
    head.lines()
        .any(|l| l.to_ascii_lowercase().starts_with("transfer-encoding:") && l.contains("chunked"))
}

/// `Content-Length` from a header block, if declared and parseable.
fn content_length(head: &str) -> Option<usize> {
    head.lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split_once(':'))
        .and_then(|(_, v)| v.trim().parse().ok())
}

/// Incremental HTTP response reader: status/header parse, chunked decode, then
/// complete lines out. Owns the only place framing can go wrong.
#[derive(Default)]
struct Reader {
    raw: Vec<u8>,
    header_end: Option<usize>,
    chunked: bool,
    body: Vec<u8>,
    consumed: usize,
    /// Set once a non-200 status line is seen. The reader then switches to
    /// collecting the body rather than bailing on the spot — see `push`.
    error_status: Option<String>,
    error_len: Option<usize>,
    /// The terminal zero-length chunk has been consumed.
    chunks_done: bool,
}

impl Reader {
    /// Feed socket bytes, get back the complete body lines they completed.
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>> {
        self.raw.extend_from_slice(bytes);
        // Already known to be an error: keep collecting the explanation.
        if self.error_status.is_some() {
            self.collect_error()?;
            return Ok(Vec::new());
        }
        if self.header_end.is_none() {
            let Some(pos) = find(&self.raw, b"\r\n\r\n") else {
                return Ok(Vec::new());
            };
            let head = String::from_utf8_lossy(&self.raw[..pos]).into_owned();
            let status = head.lines().next().unwrap_or_default().to_string();
            if !status.contains(" 200") {
                // Do NOT bail here. The body carries the server's own
                // explanation — including the actionable hint — and at this
                // point it may not have arrived yet: headers can land in a read
                // of their own. Bailing on the status line alone is why every
                // benchmark failure against a modelless server read
                // `endpoint returned "HTTP/1.1 503 Service Unavailable"` and
                // nothing about how to fix it.
                self.error_status = Some(status);
                self.error_len = content_length(&head);
                // Errors are framed like anything else — this server sends
                // them chunked — so the framing must be decoded before the
                // body is JSON. Reading it raw yields `14A\r\n{...}\r\n0` and
                // a parse failure that looks exactly like "no body".
                self.chunked = is_chunked(&head);
                self.header_end = Some(pos + 4);
                self.consumed = pos + 4;
                self.collect_error()?;
                return Ok(Vec::new());
            }
            self.chunked = is_chunked(&head);
            self.header_end = Some(pos + 4);
            self.consumed = pos + 4;
        }
        if self.chunked {
            self.decode_chunks()?;
        } else {
            self.body.extend_from_slice(&self.raw[self.consumed..]);
            self.consumed = self.raw.len();
        }
        Ok(self.take_lines())
    }

    /// Accumulate an error body, decoding its framing, and bail once complete.
    ///
    /// "Complete" is the terminal chunk when chunked, the declared
    /// `Content-Length` when there is one, and the cap otherwise. A body with
    /// none of those is reported at EOF by [`Reader::finish`], so a server that
    /// never says how much it is sending and never closes cannot wedge the run.
    fn collect_error(&mut self) -> Result<()> {
        if self.chunked {
            // A malformed chunk header in an error body is not worth failing
            // twice over: report the status we already have.
            if self.decode_chunks().is_err() {
                return self.fail();
            }
        } else {
            self.body.extend_from_slice(&self.raw[self.consumed..]);
            self.consumed = self.raw.len();
        }
        let have = self.body.len();
        let done =
            self.chunks_done || self.error_len.is_some_and(|n| have >= n) || have >= MAX_ERROR_BODY;
        if done { self.fail() } else { Ok(()) }
    }

    /// Report a pending error, whatever arrived. Called at EOF.
    fn finish(&self) -> Result<()> {
        if self.error_status.is_some() {
            self.fail()?;
        }
        Ok(())
    }

    fn fail(&self) -> Result<()> {
        let status = self.error_status.clone().unwrap_or_default();
        let text = String::from_utf8_lossy(&self.body);
        match message_from_body(text.trim()) {
            Some(m) => bail!("endpoint returned {status:?}: {m}"),
            None => bail!("endpoint returned {status:?}"),
        }
    }

    /// Pull every whole chunk currently buffered into `body`. A partial chunk
    /// stays in `raw` until the rest arrives — which is exactly the case naive
    /// line-splitting gets wrong.
    fn decode_chunks(&mut self) -> Result<()> {
        loop {
            let rest = &self.raw[self.consumed..];
            let Some(nl) = find(rest, b"\r\n") else {
                return Ok(());
            };
            let header = std::str::from_utf8(&rest[..nl]).unwrap_or("");
            // A chunk-extension (`;name=value`) is legal after the size.
            let size_hex = header.split(';').next().unwrap_or("").trim();
            let size = usize::from_str_radix(size_hex, 16)
                .map_err(|_| anyhow!("malformed chunk size {size_hex:?}"))?;
            let start = nl + 2;
            let end = start + size;
            // +2 for the CRLF that terminates the chunk data.
            if rest.len() < end + 2 {
                return Ok(());
            }
            self.body.extend_from_slice(&rest[start..end]);
            self.consumed += end + 2;
            if size == 0 {
                self.chunks_done = true;
                return Ok(());
            }
        }
    }

    fn take_lines(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        let mut start = 0;
        while let Some(nl) = find(&self.body[start..], b"\n") {
            let end = start + nl;
            out.push(
                String::from_utf8_lossy(&self.body[start..end])
                    .trim()
                    .to_string(),
            );
            start = end + 1;
        }
        self.body.drain(..start);
        out
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
#[path = "http_tests.rs"]
mod tests;
