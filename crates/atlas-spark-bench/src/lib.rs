// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]

//! Atlas Spark HTTP benchmark client.
//!
//! Shared utilities for Criterion benchmarks and correctness tests.
//! Expects a running Atlas Spark server (default: `http://localhost:8888`).

pub mod gpu;

use std::io::BufRead;
use std::sync::Barrier;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

// ── Configuration ──

pub fn server_url() -> String {
    std::env::var("ATLAS_BENCH_URL").unwrap_or_else(|_| "http://localhost:8888".into())
}

pub fn require_server() -> String {
    let url = server_url();
    match ureq::get(&format!("{url}/health")).call() {
        Ok(resp) if resp.status() == 200 => url,
        Ok(resp) => panic!("Server at {url} returned status {}", resp.status()),
        Err(e) => panic!("Server not reachable at {url}: {e}. Start Atlas Spark first."),
    }
}

// ── Request / Response types ──

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<Message<'a>>,
    max_tokens: usize,
    temperature: f32,
    stream: bool,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ResponseChoice>,
    usage: ResponseUsage,
}

#[derive(Deserialize)]
struct ResponseChoice {
    message: ResponseMessage,
    finish_reason: String,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct ResponseUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[derive(Deserialize)]
struct ChunkPayload {
    choices: Vec<ChunkChoice>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    delta: ChunkDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChunkDelta {
    content: Option<String>,
}

// ── Result types ──

pub struct BlockingResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
    pub elapsed: Duration,
}

pub struct StreamResult {
    pub text: String,
    pub token_count: usize,
    pub ttft: Duration,
    pub decode_duration: Duration,
    pub total_duration: Duration,
    pub decode_tok_s: f64,
    pub finish_reason: String,
}

// ── Blocking request ──

pub fn send_blocking(url: &str, prompt: &str, max_tokens: usize) -> Result<BlockingResult> {
    let body = ChatRequest {
        model: "qwen3",
        messages: vec![Message {
            role: "user",
            content: prompt,
        }],
        max_tokens,
        temperature: 0.0,
        stream: false,
    };

    let t_start = Instant::now();
    let resp = ureq::post(&format!("{url}/v1/chat/completions"))
        .send_json(&body)
        .context("POST request failed")?;

    let (_, mut body) = resp.into_parts();
    let parsed: ChatResponse = body.read_json().context("Failed to parse response JSON")?;
    let elapsed = t_start.elapsed();

    let choice = parsed
        .choices
        .into_iter()
        .next()
        .context("No choices in response")?;

    Ok(BlockingResult {
        text: choice.message.content,
        prompt_tokens: parsed.usage.prompt_tokens,
        completion_tokens: parsed.usage.completion_tokens,
        finish_reason: choice.finish_reason,
        elapsed,
    })
}

// ── Streaming request with timing ──

pub fn send_streaming(url: &str, prompt: &str, max_tokens: usize) -> Result<StreamResult> {
    let body = ChatRequest {
        model: "qwen3",
        messages: vec![Message {
            role: "user",
            content: prompt,
        }],
        max_tokens,
        temperature: 0.0,
        stream: true,
    };

    let t_start = Instant::now();
    let resp = ureq::post(&format!("{url}/v1/chat/completions"))
        .send_json(&body)
        .context("POST streaming request failed")?;

    let (_, body) = resp.into_parts();
    let reader = std::io::BufReader::new(body.into_reader());
    let mut token_count: usize = 0;
    let mut t_first: Option<Instant> = None;
    let mut t_last: Option<Instant> = None;
    let mut finish_reason = String::new();
    let mut text_parts = Vec::new();

    for line_result in reader.lines() {
        let line: String = line_result.context("Failed to read SSE line")?;
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload == "[DONE]" {
            break;
        }
        let chunk: ChunkPayload = match serde_json::from_str(payload) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let Some(choice) = chunk.choices.first() else {
            continue;
        };
        if let Some(ref fr) = choice.finish_reason {
            finish_reason = fr.clone();
        }
        if let Some(ref content) = choice.delta.content {
            let now = Instant::now();
            if t_first.is_none() {
                t_first = Some(now);
            }
            t_last = Some(now);
            token_count += 1;
            text_parts.push(content.clone());
        }
    }

    let total_duration = t_start.elapsed();
    let ttft = t_first.map(|tf| tf - t_start).unwrap_or(total_duration);
    let decode_duration = match (t_first, t_last) {
        (Some(tf), Some(tl)) if tl > tf => tl - tf,
        _ => Duration::ZERO,
    };
    let decode_tok_s = if !decode_duration.is_zero() && token_count >= 2 {
        (token_count - 1) as f64 / decode_duration.as_secs_f64()
    } else {
        0.0
    };

    Ok(StreamResult {
        text: text_parts.concat(),
        token_count,
        ttft,
        decode_duration,
        total_duration,
        decode_tok_s,
        finish_reason,
    })
}

// ── Concurrent streaming ──

pub fn send_concurrent_streaming(
    url: &str,
    prompt: &str,
    max_tokens: usize,
    concurrency: usize,
) -> Vec<Result<StreamResult>> {
    let barrier = Barrier::new(concurrency);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..concurrency)
            .map(|_| {
                s.spawn(|| {
                    barrier.wait();
                    send_streaming(url, prompt, max_tokens)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

// ── Prompt generators ──

pub fn short_prompt() -> &'static str {
    "What is the capital of France?"
}

pub fn medium_prompt() -> String {
    "Explain quantum computing. ".repeat(4) + "Be concise."
}

pub fn long_prompt() -> String {
    "Explain quantum computing. ".repeat(16) + "Be concise."
}

pub fn very_long_prompt() -> String {
    "Explain quantum computing. ".repeat(32) + "Be concise."
}
