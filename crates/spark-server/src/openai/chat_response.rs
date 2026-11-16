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
    /// Decode window in milliseconds: first token → final (terminal)
    /// chunk, on the server's clock. ADDITIVE (2026-09-20): lets a client
    /// compute Inter-Token Latency, `decode_time_ms / (completion_tokens
    /// − 1)`, without depending on SSE arrival times. Undefined below two
    /// completion tokens — the client decides that, this is the raw window.
    #[serde(rename = "decode_time_ms")]
    pub decode_time_ms: f64,
    /// Total server-side request time in milliseconds: scheduler receipt →
    /// final chunk (`time_to_first_token_ms + decode_time_ms`; see
    /// `ir::Usage::total_time_ms` for what the origin excludes). ADDITIVE.
    #[serde(rename = "total_time_ms")]
    pub total_time_ms: f64,
    /// Atlas fingerprints and confidence summary (`atlas_token_sha`,
    /// `atlas_output_sha256`, `atlas_confidence`). Flattened into the usage
    /// object and absent unless the request asked for them.
    #[serde(flatten)]
    pub atlas: AtlasUsageWire,
}

/// The wire form of [`crate::api::evidence::UsageExtras`]: every key is
/// omitted when unset, so a response to a request that asked for nothing is
/// byte-identical to one from before these keys existed.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AtlasUsageWire {
    /// SHA-256 of the generated token ids (little-endian `u32`), lowercase hex.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub atlas_token_sha: Option<String>,
    /// SHA-256 over the content and reasoning digests, lowercase hex.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub atlas_output_sha256: Option<String>,
    /// Confidence over the first K generated tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub atlas_confidence: Option<AtlasConfidenceWire>,
}

/// `usage.atlas_confidence`.
#[derive(Debug, Clone, Serialize)]
pub struct AtlasConfidenceWire {
    /// Mean probability of the model's top-1 token.
    pub mean_top1: f32,
    /// Smallest top-1 minus top-2 probability.
    pub min_margin: f32,
    /// Fraction of tokens whose top-1 probability was below 0.5.
    pub low_frac: f32,
    /// Tokens the summary covers (at most K).
    pub tokens: u32,
}

impl From<&crate::api::evidence::UsageExtras> for AtlasUsageWire {
    fn from(x: &crate::api::evidence::UsageExtras) -> Self {
        use crate::api::evidence::to_hex;
        Self {
            atlas_token_sha: x.token_sha.as_ref().map(to_hex),
            atlas_output_sha256: x.output_sha.as_ref().map(to_hex),
            atlas_confidence: x.confidence.map(|c| AtlasConfidenceWire {
                mean_top1: c.mean_top1,
                min_margin: c.min_margin,
                low_frac: c.low_frac,
                tokens: c.tokens,
            }),
        }
    }
}

impl From<&crate::ir::Usage> for Usage {
    /// `ir::Usage` → OpenAI wire usage — THE mapping. `total_tokens` is
    /// prompt + completion, the audio counters are pinned to 0, the
    /// accept count carries through, and the timing block is the IR's
    /// three raw components plus the derived total. The blocking encoder,
    /// the streaming encoder and the `--dump` capture all read this one
    /// function; before it each held its own copy of the field list, and a
    /// key added to one was silently absent from the others.
    fn from(u: &crate::ir::Usage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.prompt_tokens + u.completion_tokens,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: u.cached_prompt_tokens,
                audio_tokens: 0,
            }),
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: u.reasoning_tokens,
                audio_tokens: 0,
                accepted_prediction_tokens: u.accepted_prediction_tokens,
                rejected_prediction_tokens: 0,
            }),
            time_to_first_token_ms: u.time_to_first_token_ms,
            response_tokens_per_second: u.response_tokens_per_second,
            decode_time_ms: u.decode_time_ms,
            total_time_ms: u.total_time_ms(),
            atlas: AtlasUsageWire::from(&u.atlas),
        }
    }
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
            owned_by: "avarok-spark".to_string(),
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
            system_fingerprint: Some("fp_avarok".to_string()),
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
            system_fingerprint: Some("fp_avarok".to_string()),
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

#[cfg(test)]
mod atlas_usage_tests {
    use super::*;
    use crate::api::evidence::{UsageExtras, token_sha};
    use crate::scheduler::conf_summary::ConfSummary;

    const ATLAS_KEYS: [&str; 3] = ["atlas_token_sha", "atlas_output_sha256", "atlas_confidence"];

    #[test]
    fn a_response_to_a_plain_request_gains_no_key() {
        let v = serde_json::to_value(Usage::from(&crate::ir::Usage::default())).unwrap();
        let obj = v.as_object().unwrap();
        for k in ATLAS_KEYS {
            assert!(
                !obj.contains_key(k),
                "{k} must be absent unless requested: {v}"
            );
        }
        // The shipped keys are still there.
        for k in [
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "decode_time_ms",
        ] {
            assert!(obj.contains_key(k), "{k}: {v}");
        }
    }

    #[test]
    fn requested_fields_are_flattened_into_usage() {
        let ir = crate::ir::Usage {
            atlas: UsageExtras {
                token_sha: Some(token_sha(&[1, 2, 3])),
                output_sha: Some(token_sha(&[])),
                confidence: Some(ConfSummary {
                    tokens: 32,
                    mean_top1: 0.75,
                    min_margin: 0.125,
                    low_frac: 0.25,
                }),
            },
            ..Default::default()
        };
        let v = serde_json::to_value(Usage::from(&ir)).unwrap();
        assert_eq!(
            v["atlas_token_sha"],
            "4636993d3e1da4e9d6b8f87b79e8f7c6d018580d52661950eabc3845c5897a4d"
        );
        assert_eq!(
            v["atlas_output_sha256"],
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(v["atlas_confidence"]["mean_top1"], 0.75);
        assert_eq!(v["atlas_confidence"]["min_margin"], 0.125);
        assert_eq!(v["atlas_confidence"]["low_frac"], 0.25);
        assert_eq!(v["atlas_confidence"]["tokens"], 32);
    }

    #[test]
    fn each_field_is_independent() {
        let ir = crate::ir::Usage {
            atlas: UsageExtras {
                token_sha: Some(token_sha(&[])),
                ..Default::default()
            },
            ..Default::default()
        };
        let v = serde_json::to_value(Usage::from(&ir)).unwrap();
        assert!(v.get("atlas_token_sha").is_some());
        assert!(v.get("atlas_output_sha256").is_none());
        assert!(v.get("atlas_confidence").is_none());
    }
}
