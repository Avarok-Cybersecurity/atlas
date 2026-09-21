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
    /// Speculative-decode draft tokens the verify step ACCEPTED (MTP path)
    /// (OpenAI `completion_tokens_details.accepted_prediction_tokens` — the
    /// field's meaning is "predicted tokens that matched generation", which
    /// Atlas's self-drafted MTP predictions are). 0 when speculation is off.
    pub accepted_prediction_tokens: usize,
    /// Atlas perf extensions; encoders may ignore.
    ///
    /// All three timing components are read off the SCHEDULER'S clock
    /// (`scheduler::types::ActiveSeq::request_start` / `decode_start`),
    /// which by design starts when the scheduler picks the request up —
    /// HTTP parsing, template rendering, tokenisation and queue wait
    /// precede it (see `scheduler::prefill_a_step`). Sharing one origin
    /// is what makes them subtractable: a client's Inter-Token Latency
    /// per the AIPerf definition, `(request_latency − TTFT) /
    /// (output_tokens − 1)`, is exactly `decode_time_ms /
    /// (completion_tokens − 1)` here, with no queue time leaking into
    /// the numerator.
    pub time_to_first_token_ms: f64,
    /// The decode window: first emitted token → the scheduler's terminal
    /// (done) frame, which is the final chunk of the response on the
    /// server side. The raw numerator of the server-clock ITL.
    pub decode_time_ms: f64,
    /// `(completion_tokens − 1) / decode_time` — see
    /// [`Usage::decode_rate_tok_s`]. Kept beside its primitives because
    /// it is a shipped wire key (`response_token/s`); new consumers
    /// should read `decode_time_ms` and `completion_tokens` instead.
    pub response_tokens_per_second: f64,
}

impl Usage {
    /// Total server-side request duration: scheduler receipt → terminal
    /// frame. DERIVED — the two components are the stored primitives —
    /// so it is defined once here and never assembled by an encoder.
    ///
    /// Ends at the terminal frame, not at the socket: the usage block
    /// must exist before the final chunk can be serialised, so the
    /// encoder's own serialisation and flush (sub-millisecond) are
    /// necessarily outside it.
    pub fn total_time_ms(&self) -> f64 {
        self.time_to_first_token_ms + self.decode_time_ms
    }

    /// The `response_token/s` rate, `(completion_tokens − 1) / decode_s`:
    /// the first token is produced by prefill, so only the remaining
    /// `n − 1` are decode work. The ONE definition — the blocking and
    /// streaming chat paths and both completions paths call this rather
    /// than restating the formula.
    ///
    /// `0.0` when undefined (no decode window, or fewer than two
    /// tokens): a SHIPPED contract for this key, which vLLM-comparison
    /// tooling reads. Consumers that need the undefined case to be
    /// distinguishable read the raw components, where it is.
    pub fn decode_rate_tok_s(completion_tokens: usize, decode_time_ms: f64) -> f64 {
        if decode_time_ms > 0.0 && completion_tokens > 0 {
            completion_tokens.saturating_sub(1) as f64 / (decode_time_ms / 1000.0)
        } else {
            0.0
        }
    }
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
/// "stop" and carry their detail in the `guard_stop` side-channel (see
/// `scheduler::lifecycle::guard_stop_wire_reason`).
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
