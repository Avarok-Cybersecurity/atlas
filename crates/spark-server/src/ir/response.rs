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
    /// The server-side degeneration guard that cut this choice, when
    /// one did (`"content_loop_watchdog"`, `"fuzzy_repetition"`, …);
    /// `None` for every ordinary stop. Companion to — never a
    /// replacement for — `finish_reason`: a guard cut keeps reporting
    /// `"length"` on the OpenAI wire (see [`FINISH_REASON_TIMEOUT`]'s
    /// note on why no further non-standard enum VALUE may be minted),
    /// and this carries the detail as an extension FIELD instead.
    ///
    /// #927 / #1000 / #1002, round-13 cell V
    /// (`--prefill-varlen-batch`): 6 of 16 responses were cut at 49
    /// tokens by the content-loop watchdog while reporting
    /// `finish_reason: "length"`; the client could not distinguish a
    /// quality cut from a budget stop.
    ///
    /// Both surfaces produce it. Streaming carries
    /// `ActiveSeq::guard_stop` on the Done frame
    /// (`StreamDelta::Finish::stop_reason`); blocking carries the same
    /// value on `api::InferenceResponse::guard_stop`, set by
    /// `scheduler::lifecycle::finish_sequence` and read by
    /// `api::chat_blocking_choice`. The blocking surface is the one the
    /// round-13 probe actually ran (`stream=false`), so it is not
    /// optional.
    pub stop_reason: Option<&'static str>,
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
    /// Speculative-decode draft tokens the verify step ACCEPTED (MTP path)
    /// (OpenAI `completion_tokens_details.accepted_prediction_tokens` — the
    /// field's meaning is "predicted tokens that matched generation", which
    /// Atlas's self-drafted MTP predictions are). 0 when speculation is off.
    pub accepted_prediction_tokens: usize,
    /// Atlas perf extensions; encoders may ignore.
    pub time_to_first_token_ms: f64,
    pub response_tokens_per_second: f64,
}

/// Wire string for a response cut short by the server-side request
/// deadline (`--request-timeout`, or the per-request `timeout` field).
///
/// Deliberately NOT one of OpenAI's four spec reasons: a deadline
/// truncation must be distinguishable from a legitimate `max_tokens`
/// stop ("length") and from a natural end ("stop"), or the client
/// silently loses output with no way to tell. It is carried as
/// `FinishReason::Other` and round-trips verbatim through `as_wire`.
///
/// KNOWN TRADEOFF (2026-08-09): a non-standard `finish_reason` is a
/// client-compatibility hazard — strictly typed clients hard-fail on
/// unknown variants (Rust `async-openai` fails deserialization outright,
/// which is what forced TGI to drop its `eos_token` value; pydantic-ai
/// raised on OpenRouter's non-standard "error"). "timeout" is kept as a
/// deliberate, shipped exception because silent truncation is worse; do
/// NOT add further non-standard values — server-side guard cuts map to
/// "length" and carry their detail in the `stop_reason` extension FIELD
/// (see `scheduler::lifecycle::guard_stop_wire_reason` for the mapping
/// and [`Choice::stop_reason`] for the field). Unknown FIELDS are
/// ignored by every SDK; unknown enum VALUES are what hard-fail typed
/// clients — which is why the detail rides a new key rather than a
/// fifth `finish_reason` string. vLLM makes the same split with its own
/// `stop_reason` extension.
pub const FINISH_REASON_TIMEOUT: &str = "timeout";

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
