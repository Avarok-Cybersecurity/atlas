// SPDX-License-Identifier: AGPL-3.0-only

//! Tokenizer wrapper using HuggingFace tokenizers + minijinja chat template.
//!
//! Loads the model's official Jinja template from `tokenizer_config.json` and
//! renders it with minijinja for byte-exact alignment with the model's training
//! format. No fallback — if there's no Jinja template, the model is misconfigured.

use anyhow::Result;
use tokenizers::Tokenizer;

/// F76 (2026-04-29): pre-parse `tool_calls[*].function.arguments` from
/// OpenAI's wire format (JSON-encoded string) into the JSON value the
/// model's chat template expects. MiniMax M2.7's template iterates
/// `tool_call.function.arguments.items()` which crashes on a string.
/// We rebuild the message list with parsed arguments where present,
/// leaving every other field untouched. Returns a fresh Vec rather
/// than mutating the caller's slice.
fn normalize_tool_call_arguments(messages: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut total_parsed = 0usize;
    let mut total_seen = 0usize;
    let out: Vec<_> = messages
        .iter()
        .map(|msg| {
            let mut msg = msg.clone();
            let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) else {
                return msg;
            };
            for tc in tool_calls.iter_mut() {
                let Some(function) = tc.get_mut("function") else {
                    continue;
                };
                let Some(args) = function.get_mut("arguments") else {
                    continue;
                };
                total_seen += 1;
                let parsed_owned = if let Some(s) = args.as_str() {
                    serde_json::from_str::<serde_json::Value>(s).ok()
                } else {
                    None
                };
                if let Some(parsed) = parsed_owned {
                    *args = parsed;
                    total_parsed += 1;
                }
                // If parse fails or args wasn't a string, leave as-is —
                // template may handle via tojson, or surface the
                // original error for the operator.
            }
            msg
        })
        .collect();
    if total_seen > 0 {
        tracing::debug!(
            "F76 normalize: {}/{} tool_call arguments parsed string→dict",
            total_parsed,
            total_seen,
        );
    }
    out
}

/// Wraps a HuggingFace tokenizer with Jinja chat template support.
mod chat_impl;
mod jinja_helpers;
mod message_preprocess;

pub(crate) use message_preprocess::{autoclose_assistant_think, resolve_think_control};

pub struct ChatTokenizer {
    tokenizer: Tokenizer,
    eos_token_id: u32,
    supports_thinking: bool,
    /// Compiled Jinja chat template (from tokenizer_config.json).
    #[allow(dead_code)]
    chat_template: String,
    /// Precompiled minijinja environment (avoids re-creating + re-compiling each call).
    jinja_env: minijinja::Environment<'static>,
    /// OpenAI-variant template: gates historical `<think>` wrappers on enable_thinking.
    /// Falls back to jinja_env if no openai/ variant exists.
    openai_jinja_env: Option<minijinja::Environment<'static>>,
}

/// Wrapper around tokenizers::DecodeStream that hides the generic parameters.
/// O(1) per step vs O(n) for full re-decode.
pub struct StreamingDecoder<'a> {
    inner: tokenizers::DecodeStream<
        'a,
        tokenizers::models::ModelWrapper,
        tokenizers::normalizers::NormalizerWrapper,
        tokenizers::pre_tokenizers::PreTokenizerWrapper,
        tokenizers::processors::PostProcessorWrapper,
        tokenizers::decoders::DecoderWrapper,
    >,
}

impl StreamingDecoder<'_> {
    /// Feed one token. Returns Some(text) when valid UTF-8 is ready.
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        self.inner
            .step(id)
            .map_err(|e| anyhow::anyhow!("Streaming decode error: {e}"))
    }
}

#[cfg(test)]
mod tests;
