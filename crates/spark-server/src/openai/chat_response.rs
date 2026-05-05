// SPDX-License-Identifier: AGPL-3.0-only

#[allow(unused_imports)]
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
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

// ── SSE streaming types (OpenAI format) ──

/// SSE streaming chunk.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
    pub logprobs: Option<ChoiceLogprobs>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Reasoning trace chunk (from <think>...</think>). Streamed during
    /// the thinking phase when enable_thinking=true. Cline and Roo Code
    /// both check for this field via `"reasoning_content" in delta`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Forward-compatible alias: mirrors `reasoning_content` for clients that
    /// use the shorter `reasoning` field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Omitted when absent — Cline, Roo Code, and most OpenAI-compatible clients
    /// expect content to be missing (not `null`) in role and done chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::tool_parser::ChunkToolCall>>,
    /// Refusal signal. Atlas emits this on a single terminal delta chunk
    /// (just before the `done` chunk) when the accumulated streamed
    /// content matches a known refusal pattern. OpenAI's streaming
    /// refusal model is fragment-by-fragment; Atlas only classifies
    /// post-hoc so the signal lands as one chunk. Safety-aware clients
    /// that branch on `delta.refusal` will still see a non-null value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl ChatCompletionChunk {
    /// First chunk: sends the assistant role.
    pub fn role_chunk(model: &str, id: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".to_string()),
                    reasoning_content: None,
                    reasoning: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
        }
    }

    /// Reasoning content delta chunk (from <think>...</think> when enable_thinking=true).
    /// Cline and Roo Code check for `delta.reasoning_content` in streaming.
    pub fn reasoning_chunk(model: &str, id: &str, text: String) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: Some(text.clone()),
                    reasoning: Some(text),
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
        }
    }

    /// Content delta chunk.
    pub fn content_chunk(model: &str, id: &str, text: String) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    reasoning: None,
                    content: Some(text),
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
        }
    }

    /// Tool call start chunk — emits role, id, type, name with empty arguments.
    /// Per OpenAI streaming spec, the first tool_call delta carries
    /// `role: "assistant"` and `content: null` alongside the metadata.
    pub fn tool_call_start_chunk(
        model: &str,
        id: &str,
        tc: &crate::tool_parser::ToolCall,
        tc_index: usize,
    ) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".to_string()),
                    reasoning_content: None,
                    reasoning: None,
                    content: None,
                    tool_calls: Some(vec![crate::tool_parser::ChunkToolCall {
                        index: tc_index,
                        id: Some(tc.id.clone()),
                        call_type: Some(tc.call_type.clone()),
                        function: crate::tool_parser::ChunkFunction {
                            name: Some(tc.function.name.clone()),
                            arguments: String::new(),
                        },
                    }]),
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
        }
    }

    /// Tool call argument fragment chunk — emits a partial arguments string.
    /// Per OpenAI streaming spec, subsequent deltas carry incremental argument
    /// fragments. Callers should split the full arguments into small pieces
    /// (~20 chars) and call this for each fragment.
    pub fn tool_call_args_fragment(model: &str, id: &str, tc_index: usize, fragment: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    reasoning: None,
                    content: None,
                    tool_calls: Some(vec![crate::tool_parser::ChunkToolCall {
                        index: tc_index,
                        id: None,
                        call_type: None,
                        function: crate::tool_parser::ChunkFunction {
                            name: None,
                            arguments: fragment.to_string(),
                        },
                    }]),
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
        }
    }

    /// Final chunk with finish_reason, empty delta, and usage.
    pub fn done_chunk(model: &str, id: &str, finish_reason: &str, usage: Usage) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    reasoning: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: Some(finish_reason.to_string()),
                logprobs: None,
            }],
            usage: Some(usage),
        }
    }

    /// When `stream_options.include_usage=true`, OpenAI emits a separate
    /// chunk with `choices:[]` and populated `usage` BEFORE the final
    /// `finish_reason`-carrying chunk. This helper emits that usage-only
    /// chunk.
    pub fn usage_only_chunk(model: &str, id: &str, usage: Usage) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: Vec::new(),
            usage: Some(usage),
        }
    }

    /// Delta chunk carrying only `refusal: "<sentence>"`. Atlas emits
    /// this once, just before the terminal `done`/`usage_only` chunk,
    /// when `refusal::detect` classifies the accumulated streamed
    /// content as a refusal. OpenAI's streaming refusal model sends
    /// multiple `delta.refusal` fragments; we send a single post-hoc
    /// signal because Atlas classifies after the stream is complete.
    pub fn refusal_chunk(model: &str, id: &str, refusal: String) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    reasoning: None,
                    content: None,
                    tool_calls: None,
                    refusal: Some(refusal),
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
        }
    }

    /// Final chunk carrying only `finish_reason`, with `usage:null`. Used
    /// when `include_usage=true` (the usage sits in `usage_only_chunk`
    /// emitted just before this one).
    pub fn final_chunk_no_usage(model: &str, id: &str, finish_reason: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    reasoning: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: Some(finish_reason.to_string()),
                logprobs: None,
            }],
            usage: None,
        }
    }
}

// ── Legacy /v1/completions types (OpenAI standard) ──
