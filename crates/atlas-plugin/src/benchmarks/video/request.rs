// SPDX-License-Identifier: AGPL-3.0-only

//! Building the requests.

use base64::Engine;
use serde_json::{Value, json};

/// The question every ordered-color leg asks.
///
/// Phrased to make the answer a fact rather than a judgement: name the
/// colors, in order, nothing else. "Describe this video" would invite prose
/// that is pleasant to read and impossible to score — and, worse, prose a
/// server that had stopped splicing embeddings entirely can still produce.
pub const ORDER_PROMPT: &str = "This video is a sequence of solid background colors. List the colors in the order they \
     appear, separated by commas. Answer with only the color names.";

/// A clip as the API wants it.
pub fn data_uri(mime: &str, bytes: &[u8]) -> String {
    let mut s = format!("data:{mime};base64,");
    base64::engine::general_purpose::STANDARD.encode_string(bytes, &mut s);
    s
}

/// One request carrying a single video.
///
/// Temperature 0 and thinking off, matching the image benchmark and for the
/// same reasons: every assertion is about what the model SAW, so sampling
/// variance is noise, and a reasoning block can eat the whole token budget and
/// return empty content — which reads as a video failure and is not one.
pub fn video_body(model: &str, mime: &str, bytes: &[u8], prompt: &str, max_tokens: usize) -> Value {
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": data_uri(mime, bytes)}},
            {"type": "text", "text": prompt},
        ]}],
    })
}

/// One request carrying an image AND a video, in that order.
///
/// The order is the point: it is the contract between collection order,
/// template marker order and pad expansion. A desync there yields a pad run of
/// the wrong length filled with the wrong embeddings, and nothing errors.
pub fn mixed_body(
    model: &str,
    png: &[u8],
    mime: &str,
    video: &[u8],
    prompt: &str,
    max_tokens: usize,
) -> Value {
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri("image/png", png)}},
            {"type": "video_url", "video_url": {"url": data_uri(mime, video)}},
            {"type": "text", "text": prompt},
        ]}],
    })
}

/// The control: the same question, with nothing attached.
pub fn text_only_body(model: &str, prompt: &str, max_tokens: usize) -> Value {
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": prompt}],
    })
}

/// Does this error read as "the server cannot decode that container"?
///
/// A server without ffmpeg is a DEPLOYMENT choice, not a defect, so those legs
/// are skipped rather than failed — the same call the image benchmark makes
/// for an image beyond the encoder's capacity. Matching on the server's own
/// wording is deliberate: those strings are the operator-facing contract and
/// are asserted in `video_decode_ffmpeg`'s tests, so a reworded error breaks a
/// test rather than silently turning skips into failures.
pub fn is_decoder_unavailable(err: &str) -> bool {
    let e = err.to_lowercase();
    let quoted_binary_unavailable = e
        .split_once(" could not be run:")
        .is_some_and(|(binary, _)| binary.starts_with('"') && binary.ends_with('"'));
    e.contains("subprocess decoding is disabled")
        || (e.contains("could not run ") && e.contains("is ffmpeg installed"))
        || quoted_binary_unavailable
}

#[cfg(test)]
#[path = "request_tests.rs"]
mod request_tests;
