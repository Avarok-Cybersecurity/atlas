// SPDX-License-Identifier: AGPL-3.0-only

//! Building a multimodal chat request, and reading the geometry back out.

use anyhow::{Context, Result};
use base64::Engine;
use serde_json::{Value, json};

/// A fixture as the API wants it: a base64 `data:` URI.
///
/// Atlas deliberately rejects `http(s)` image URLs and tells the caller to
/// send a data URI instead, so this is the only shape that works today. When
/// the opt-in remote-fetch flag lands, this stays the benchmark's shape
/// regardless — a benchmark that depended on the server making outbound
/// requests would be measuring the network.
pub fn data_uri(png: &[u8]) -> String {
    let mut s = String::from("data:image/png;base64,");
    base64::engine::general_purpose::STANDARD.encode_string(png, &mut s);
    s
}

/// One chat request carrying `images` followed by `prompt`.
///
/// Temperature 0 throughout: every assertion here is about what the model
/// SAW, so sampling variance is pure noise. Thinking is off for the same
/// reason — a reasoning block would spend the token budget without changing
/// whether the image was encoded correctly, and on a thinking-first
/// checkpoint it can consume the whole of `max_tokens` and return empty
/// content, which reads as a vision failure and is not one.
pub fn body(model: &str, images: &[&[u8]], prompt: &str, max_tokens: usize) -> Value {
    let mut content: Vec<Value> = images
        .iter()
        .map(|png| json!({"type": "image_url", "image_url": {"url": data_uri(png)}}))
        .collect();
    content.push(json!({"type": "text", "text": prompt}));
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": content}],
    })
}

/// Vision tokens in a reply, inferred by subtracting the chat template's own
/// cost from `prompt_tokens`.
///
/// The server reports `prompt_tokens` for the whole rendered prompt, so the
/// template overhead has to come out before the number means anything. It is
/// measured once per run from a calibration request whose vision-token count
/// is known, rather than hard-coded: the overhead is a property of the
/// checkpoint's chat template and moves when the template does.
pub fn vision_tokens(prompt_tokens: usize, overhead: usize) -> Result<usize> {
    prompt_tokens.checked_sub(overhead).with_context(|| {
        format!(
            "prompt_tokens {prompt_tokens} is below the measured template overhead \
             {overhead} — the calibration request and this one did not render the same \
             template, so the subtraction is meaningless"
        )
    })
}

#[cfg(test)]
#[path = "request_tests.rs"]
mod request_tests;
