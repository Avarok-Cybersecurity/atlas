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
    /// Which MoE experts this request's prompt routed to. Present only when
    /// the request set `report_expert_metadata` and the serve was started
    /// with `--expert-telemetry`; absent (not empty) otherwise, so a
    /// consumer can tell "not instrumented" from "no experts used".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_activation: Option<ExpertActivation>,
}

/// Per-request MoE expert routing (Atlas extension on `usage`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertActivation {
    /// Which part of the request these numbers cover — `"prefill"` today.
    /// Named on the wire so prompt routing is never read as whole-request
    /// routing.
    pub scope: String,
    pub top_k: u32,
    pub num_experts: u32,
    /// Token positions folded in. `Σcounts == tokens_routed * top_k` when
    /// every routed slot carried weight.
    pub tokens_routed: u64,
    /// Token positions that ran but are not in the counts. Non-zero means
    /// this report covers a prefix of the request.
    pub unattributed_rows: u64,
    /// Of `tokens_routed`, how many came from decode rather than the prompt.
    #[serde(skip_serializing_if = "is_zero")]
    pub decode_tokens_routed: u64,
    /// Decode positions that ran without being attributed (MTP verify rows).
    #[serde(skip_serializing_if = "is_zero")]
    pub decode_unattributed_rows: u64,
    pub layers: Vec<ExpertLayerActivation>,
}

/// Omit a zero counter rather than emit it: a field that is always present
/// and always 0 trains readers to ignore it.
fn is_zero(v: &u64) -> bool {
    *v == 0
}

/// One MoE layer's routing, as parallel ascending-by-expert-id arrays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertLayerActivation {
    /// Absolute model layer index — the numbering MODEL.toml
    /// `[expert_categories] layers."<L>"` uses.
    pub layer: usize,
    pub experts: Vec<u32>,
    /// Routed token-slots that chose each expert.
    pub counts: Vec<u32>,
    /// Summed post-renormalization routing weight per expert.
    pub mass: Vec<f32>,
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
    /// Predicted tokens that matched generation. Atlas has no client-supplied
    /// `prediction` feature; this reports the SPECULATIVE-DECODE draft tokens
    /// the MTP verify step accepted for this request — the same "predicted
    /// tokens that matched generation" meaning, with the server as the
    /// predictor. 0 when speculation is off or nothing was accepted.
    pub accepted_prediction_tokens: usize,
    /// Predicted-output tokens that were rejected. Always 0 on Atlas —
    /// rejected MTP drafts are not client-billable and are not reported here.
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
    /// Context window the server will actually accept, in tokens.
    ///
    /// Not an OpenAI field — a vLLM extension that clients (LiteLLM, aider,
    /// Continue, OpenWebUI) probe to size requests without a round trip that
    /// fails at the scheduler. It is DERIVED from `AppState::max_seq_len`, the
    /// same value the admission path enforces, so the advertised ceiling and
    /// the enforced one cannot drift apart.
    ///
    /// `None` (omitted from the wire) when no model is loaded: fabricating a 0
    /// would read as "zero context" rather than "unknown".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,
}

impl ModelInfo {
    /// The ONE place an advertised entry is built.
    ///
    /// Both `/v1/models` list sites and the retrieve handler go through here so
    /// the advertised ceiling is DERIVED from the value admission enforces
    /// (`AppState::max_seq_len`) rather than restated. A second construction
    /// site is how the wire and the scheduler drift apart.
    pub fn advertise(id: String, max_seq_len: usize) -> Self {
        Self {
            id,
            object: "model".to_string(),
            created: crate::ids::unix_timestamp(),
            owned_by: "atlas-spark".to_string(),
            max_model_len: Some(max_seq_len),
        }
    }
}

impl ChatCompletionResponse {
    pub fn new(
        model: &str,
        content: String,
        reasoning_content: Option<String>,
        usage: Usage,
        finish_reason: &str,
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
                    reasoning_content,
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
        reasoning_content: Option<String>,
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
                    reasoning_content,
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
