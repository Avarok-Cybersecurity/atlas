// SPDX-License-Identifier: AGPL-3.0-only

#[allow(unused_imports)]
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use super::*;

// ── Legacy /v1/completions types (OpenAI standard) ──

/// Completion request (non-chat, raw prompt).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct CompletionRequest {
    pub model: String,
    #[serde(deserialize_with = "deserialize_prompt")]
    pub prompt: String,
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
}

/// Accept `prompt` as a string or array of strings (joined).
fn deserialize_prompt<'de, D>(d: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum RawPrompt {
        Str(String),
        Arr(Vec<String>),
    }
    match RawPrompt::deserialize(d)? {
        RawPrompt::Str(s) => Ok(s),
        RawPrompt::Arr(v) => Ok(v.join("")),
    }
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
