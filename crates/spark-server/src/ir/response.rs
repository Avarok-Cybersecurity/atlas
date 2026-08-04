// SPDX-License-Identifier: AGPL-3.0-only
//
// Canonical chat IR (response direction). The blocking pipeline
// produces exactly one of these per request; each API surface encodes
// it into its own wire format (OpenAI chat JSON, Anthropic
// MessagesResponse, Responses API JSON). No surface re-parses another
// surface's serialized body.

use super::message::ToolCall;

/// A complete (non-streaming) chat response.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    /// Bare response id (a uuid) — surfaces apply their wire prefixes
    /// (`chatcmpl-`, `msg_`, `resp_`).
    pub id: String,
    /// Served model name (encoders read this — no side-channel param).
    pub model: String,
    /// Unix seconds at response build time.
    pub created: u64,
    /// One entry per requested choice. `n > 1` is only reachable from
    /// the OpenAI surface; other adapters pin `n = 1` and their
    /// encoders read the first choice.
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

/// One generated choice.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub index: usize,
    /// Assistant text (`None` mirrors the wire's `content: null`, e.g.
    /// after a refusal strip).
    pub content: Option<String>,
    /// Reasoning/thinking trace, when the model produced one.
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    /// Refusal message (safety classifier), when set.
    pub refusal: Option<String>,
    pub finish_reason: FinishReason,
    /// The client stop sequence that terminated generation, when one
    /// did. Feeds Anthropic's `stop_sequence` field.
    pub matched_stop: Option<String>,
    /// Per-token logprobs (opt-in). Only the OpenAI surface encodes
    /// these today.
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Neutral logprob report: sampled token + alternatives.
#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceLogprobs {
    pub content: Vec<TokenLogprob>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f32,
    /// `(token, logprob)` alternatives, highest first.
    pub top: Vec<(String, f32)>,
}

/// Token accounting, including the detail counters the wire formats
/// surface (prefix-cache hits, reasoning tokens) and Atlas's
/// performance extensions.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Prompt tokens served from the prefix cache
    /// (OpenAI `prompt_tokens_details.cached_tokens`, Anthropic
    /// `cache_read_input_tokens`).
    pub cached_prompt_tokens: usize,
    /// Completion tokens spent inside the thinking block
    /// (OpenAI `completion_tokens_details.reasoning_tokens`).
    pub reasoning_tokens: usize,
    /// Atlas perf extensions; encoders may ignore.
    pub time_to_first_token_ms: f64,
    pub response_tokens_per_second: f64,
}

/// Why generation stopped. `Other` preserves unknown engine reasons
/// losslessly (PCND: no silent default).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Other(String),
}

impl From<&str> for FinishReason {
    /// Map the engine's finish-reason string (the scheduler's internal
    /// vocabulary, which happens to match OpenAI's wire strings).
    fn from(s: &str) -> Self {
        match s {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolCalls,
            "content_filter" => FinishReason::ContentFilter,
            other => FinishReason::Other(other.to_string()),
        }
    }
}

impl FinishReason {
    /// The canonical wire string (OpenAI-compatible surfaces emit it
    /// verbatim; other surfaces map per their own vocabulary).
    pub fn as_wire(&self) -> &str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::Other(s) => s,
        }
    }
}
