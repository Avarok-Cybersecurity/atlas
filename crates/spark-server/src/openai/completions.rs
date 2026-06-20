// SPDX-License-Identifier: AGPL-3.0-only

use serde::{Deserialize, Serialize};

use super::*;

// ── Legacy /v1/completions types (OpenAI standard) ──

/// Completion request (non-chat, raw prompt).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct CompletionRequest {
    pub model: String,
    /// OpenAI-compatible `prompt`: a string, an array of strings, an array
    /// of integer token IDs, or an array of token-ID arrays (batch). The
    /// token-ID forms bypass tokenization and feed the scheduler verbatim
    /// (see `PromptInput` and the handler in `api/completions.rs`).
    pub prompt: PromptInput,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    /// Top-k: keep only the k highest-probability tokens before sampling.
    pub top_k: Option<u32>,
    /// Top-p (nucleus): keep smallest set of tokens whose cumulative probability >= p.
    pub top_p: Option<f32>,
    /// Top-n-sigma: filter tokens in logit space before temperature scaling.
    /// 0.0 = disabled.
    pub top_n_sigma: Option<f32>,
    /// Min-p: keep tokens with prob >= min_p * max_prob (post-softmax).
    /// 0.0 = disabled.
    pub min_p: Option<f32>,
    /// Repetition penalty: penalize tokens that have already been generated.
    /// 1.0 = disabled.
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Per-token logit bias.
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    #[serde(default)]
    pub stream: bool,
    /// Stop sequences (same as chat completions).
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Vec<String>,
    /// Seed for deterministic sampling (same as chat completions).
    pub seed: Option<u64>,
    /// Per-request override for the vLLM-anchored token-loop detector
    /// (see `RepetitionDetectionParams` in `chat_request.rs`). None =
    /// use server default.
    #[serde(default)]
    pub repetition_detection: Option<RepetitionDetectionParams>,
}

/// OpenAI-compatible `prompt` field. Mirrors the four shapes the
/// `/v1/completions` spec permits:
///   - `"hello"`              → `Text`
///   - `[128000, 9906, ...]`  → `TokenIds`     (bypasses tokenization)
///   - `[[128000], [9906]]`   → `TokenIdBatch` (bypasses tokenization)
///   - `["hello", "world"]`   → `TextArray`    (joined, then tokenized)
///
/// ── serde-untagged ordering rationale ──
/// `#[serde(untagged)]` tries variants top-to-bottom and accepts the first
/// that deserializes. The four JSON shapes are mutually exclusive *by value
/// type*, so there is no real collision:
///   - A JSON string only matches `Text`.
///   - An array of integers matches `TokenIds` but never `TextArray`
///     (integers are not strings) nor `TokenIdBatch` (integers are not
///     sub-arrays).
///   - An array of strings matches only `TextArray`.
///   - An array of arrays matches only `TokenIdBatch`.
/// The single genuinely ambiguous input is the empty array `[]`, which
/// satisfies every array variant; it resolves to the first array variant
/// listed (`TokenIds([])`), i.e. an empty token sequence — semantically
/// identical to an empty prompt, so ordering is harmless. Integer variants
/// are listed before `TextArray` so that an all-integer array is never
/// coerced; out-of-`u32`-range or negative numbers fail every variant and
/// surface as a clean "did not match any variant" 400 (fail-fast).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    /// Plain text prompt — tokenized by the server.
    Text(String),
    /// Pre-tokenized prompt: integer token IDs fed to the scheduler
    /// verbatim (no tokenization, no BOS prepended).
    TokenIds(Vec<u32>),
    /// Batch of pre-tokenized prompts. The legacy `/v1/completions`
    /// handler is single-prompt, so the batch is flattened (concatenated)
    /// into one token sequence, matching the existing string-array
    /// behavior of joining multiple string prompts.
    TokenIdBatch(Vec<Vec<u32>>),
    /// Array of text prompts — joined (matching prior behavior) then
    /// tokenized.
    TextArray(Vec<String>),
}

/// Completion response.
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

impl CompletionResponse {
    pub fn new(model: &str, text: String, usage: Usage, finish_reason: &str) -> Self {
        Self {
            id: format!("cmpl-{}", uuid_v4()),
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            choices: vec![CompletionChoice {
                index: 0,
                text,
                finish_reason: finish_reason.to_string(),
            }],
            usage,
        }
    }
}

/// SSE streaming chunk for completions.
#[derive(Debug, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunkChoice {
    pub index: usize,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

impl CompletionChunk {
    /// Content text chunk.
    pub fn text_chunk(model: &str, id: &str, text: String) -> Self {
        Self {
            id: id.to_string(),
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            choices: vec![CompletionChunkChoice {
                index: 0,
                text,
                finish_reason: None,
            }],
            usage: None,
        }
    }

    /// Final chunk with finish_reason and usage.
    pub fn done_chunk(model: &str, id: &str, finish_reason: &str, usage: Usage) -> Self {
        Self {
            id: id.to_string(),
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            choices: vec![CompletionChunkChoice {
                index: 0,
                text: String::new(),
                finish_reason: Some(finish_reason.to_string()),
            }],
            usage: Some(usage),
        }
    }
}

// ── Tokenize endpoint types ──

/// Request body for POST /tokenize.
#[derive(Debug, Deserialize)]
pub struct TokenizeRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    /// Raw text to tokenize (mutually exclusive with `messages`).
    pub prompt: Option<String>,
    /// Chat messages to tokenize via the chat template (mutually exclusive with `prompt`).
    pub messages: Option<Vec<IncomingMessage>>,
}

/// Response body for POST /tokenize.
#[derive(Debug, Serialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
    pub count: usize,
}
