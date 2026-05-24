// SPDX-License-Identifier: AGPL-3.0-only

use serde::{Deserialize, Serialize};

use super::*;

/// Chat completion response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    /// Echo of the request's `service_tier` (OpenAI-compatible).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Echo of the request's `metadata` (OpenAI-compatible).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Token usage and performance timing.
///
/// Standard OpenAI fields (`prompt_tokens`, `completion_tokens`, `total_tokens`)
/// plus timing extensions that OpenWebUI and other frontends display in tooltips.
/// Field naming follows llama.cpp / Ollama conventions for broad compatibility.
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// Prefix-cache + audio token breakdown of the prompt (OpenAI-compatible).
    /// Populated when Atlas's prefix cache served any portion of the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Reasoning + audio + prediction breakdown of the completion
    /// (OpenAI-compatible). `reasoning_tokens` counts the tokens emitted
    /// inside `<think>...</think>` (or the equivalent for each model type).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// Time to first token in milliseconds (prefill duration).
    #[serde(rename = "time_to_first_token_ms")]
    pub time_to_first_token_ms: f64,
    /// Decode throughput in tokens per second.
    #[serde(rename = "response_token/s")]
    pub response_tokens_per_second: f64,
}

/// Prompt-token breakdown (OpenAI-compatible `prompt_tokens_details`).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PromptTokensDetails {
    /// Tokens served by the prefix cache (no prefill compute cost).
    pub cached_tokens: usize,
    /// Audio-input tokens. Always 0 on Atlas until audio modality lands.
    pub audio_tokens: usize,
}

/// Completion-token breakdown (OpenAI-compatible `completion_tokens_details`).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompletionTokensDetails {
    /// Tokens generated inside a thinking/reasoning block
    /// (`<think>...</think>`, `[THINK]...[/THINK]`, etc.). Counted in
    /// `completion_tokens` as well — this is the portion attributable to
    /// chain-of-thought.
    pub reasoning_tokens: usize,
    /// Audio-output tokens. Always 0 on Atlas until audio modality lands.
    pub audio_tokens: usize,
    /// Predicted-output (`prediction`) tokens that matched generation.
    /// Always 0 on Atlas — we don't implement predicted outputs yet.
    pub accepted_prediction_tokens: usize,
    /// Predicted-output tokens that were rejected. Always 0 on Atlas.
    pub rejected_prediction_tokens: usize,
}

/// Top log-probability for a single alternative token.
#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

/// Log-probability information for a single generated token.
#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprobInfo {
    pub token: String,
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogprob>,
}

/// Per-choice logprobs container (OpenAI-compatible).
#[derive(Debug, Clone, Serialize)]
pub struct ChoiceLogprobs {
    pub content: Vec<TokenLogprobInfo>,
}

/// Model list response.
#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

impl ChatCompletionResponse {
    pub fn new(model: &str, content: String, usage: Usage, finish_reason: &str) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid_v4()),
            object: "chat.completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    reasoning_content: None,
                    reasoning: None,
                    annotations: extract_url_annotations(&content),
                    refusal: None,
                    content: Some(content),
                    tool_calls: None,
                },
                finish_reason: finish_reason.to_string(),
                logprobs: None,
            }],
            usage,
            service_tier: None,
            metadata: None,
        }
    }

    pub fn with_tool_calls(
        model: &str,
        content: Option<String>,
        tool_calls: Vec<crate::tool_parser::ToolCall>,
        usage: Usage,
    ) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid_v4()),
            object: "chat.completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    reasoning_content: None,
                    reasoning: None,
                    annotations: content.as_deref().and_then(extract_url_annotations),
                    refusal: None,
                    content,
                    tool_calls: Some(tool_calls),
                },
                finish_reason: "tool_calls".to_string(),
                logprobs: None,
            }],
            usage,
            service_tier: None,
            metadata: None,
        }
    }
}
