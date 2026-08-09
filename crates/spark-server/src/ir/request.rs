// SPDX-License-Identifier: AGPL-3.0-only
//
// Canonical chat IR (request direction): the provider-agnostic request
// envelope the whole internal pipeline consumes. Each API surface
// (OpenAI chat, OpenAI Responses, Anthropic Messages) resolves its wire
// format into `ChatRequest` at the edge; no internal module reads a
// wire type.
//
// Deliberately absent: echo-only wire fields (service_tier, store,
// metadata, stream_options.include_usage) — those never influence
// generation, so each surface handler keeps its parsed wire request
// and echoes them at encode time (`api/chat/echo.rs`). Accepted-and-
// ignored compat fields (audio, prediction, modalities, user, …) stay
// wire-only for the same reason (PCND: the IR carries exactly what
// internals read).

/// Provider-agnostic chat request envelope.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// Client-requested model name (logging + response echo).
    pub model: String,
    pub messages: Vec<super::Message>,
    /// Tool definitions; empty = none. (`tool_parser::ToolDefinition`
    /// is already provider-neutral — every surface parses into it.)
    pub tools: Vec<crate::tool_parser::ToolDefinition>,
    pub tool_choice: Option<crate::tool_parser::ToolChoice>,
    pub sampling: SamplingParams,
    pub max_tokens: usize,
    pub min_tokens: usize,
    /// Stop sequences (generation halts when one is produced).
    pub stop: Vec<String>,
    pub stream: bool,
    /// Number of choices (>1 only reachable from the OpenAI surface;
    /// other adapters pin 1).
    pub n: usize,
    /// Output shape constraint. `None` = unconstrained text (the wire
    /// `{"type":"text"}` maps to `None` at the edge).
    pub response_format: Option<ResponseFormat>,
    /// Client thinking intent (edge-resolved). Server/model defaults
    /// fold in later (`api/chat/thinking.rs`).
    pub thinking: ThinkingDirective,
    /// Per-request token-loop detector override.
    pub repetition_detection: Option<crate::api::inference_types::RepetitionDetectionParams>,
    /// M2 per-request LoRA routing: optional resident adapter NAME for
    /// this request (independent of `model`). `None` = installed active
    /// adapter; resolved to a pool slot at the handler edge.
    pub adapter: Option<String>,
    /// NLLB (encoder-decoder): per-request source language token NAME
    /// (e.g. `eng_Latn`); resolved to a token id via the server
    /// tokenizer at dispatch. `None` = deployment default.
    pub src_lang: Option<String>,
    /// NLLB: per-request target language token NAME. See [`Self::src_lang`].
    pub tgt_lang: Option<String>,
    /// NLLB beam search width. `None` = single-hypothesis decode.
    pub num_beams: Option<u32>,
    /// NLLB beam search length penalty (only read when `num_beams > 1`).
    pub length_penalty: Option<f32>,
    /// NLLB beam search early stopping (only read when `num_beams > 1`).
    pub early_stopping: Option<bool>,
    /// Per-token logit bias, already parsed from the wire's string-key
    /// map at the edge (non-numeric keys dropped, matching history).
    pub logit_bias: Vec<(u32, f32)>,
    /// Logprob alternatives per token, already resolved from the wire's
    /// `logprobs` + `top_logprobs` pair at the edge. `None` = disabled.
    pub top_logprobs: Option<u8>,
    /// Deterministic-sampling seed.
    pub seed: Option<u64>,
    /// Request timeout in seconds; `None` = server default.
    pub timeout_secs: Option<f32>,
    /// Emit sampled token IDs on stream chunks (vLLM extension).
    pub return_token_ids: bool,
}

/// Client sampling parameters. `None` = client silent → the server's
/// generation_config / MODEL.toml preset default applies downstream
/// (`api/chat/sampling_setup.rs`).
#[derive(Debug, Clone, Copy, Default)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub top_n_sigma: Option<f32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
}

/// Output shape constraint (neutral form of OpenAI's `response_format`).
#[derive(Debug, Clone)]
pub enum ResponseFormat {
    /// Output must be valid JSON (any shape).
    JsonObject,
    /// Output must match the given JSON schema.
    JsonSchema {
        /// Schema name (used for logging).
        name: String,
        schema: serde_json::Value,
        /// Enforce strict adherence.
        strict: bool,
    },
}

/// Client thinking intent, resolved at the API edge from each wire's
/// channels (Anthropic `thinking`, OpenAI `reasoning.effort`, vLLM
/// `chat_template_kwargs` / `thinking_token_budget`, Atlas legacy
/// `enable_thinking`). Internal code reads only this — never the wire
/// fields.
///
/// Model-default folding (MODEL.toml `[behavior].thinking_default`)
/// happens in `api/chat/thinking.rs` when the directive is
/// [`Unspecified`](ThinkingDirective::Unspecified); the directive itself
/// never encodes a server or model default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingDirective {
    /// No client signal. The server-level default directive (from
    /// `--default-chat-template-kwargs`) applies if set, else the model
    /// default from MODEL.toml.
    #[default]
    Unspecified,
    /// Client explicitly disabled thinking.
    Off,
    /// Client explicitly enabled thinking. `budget: None` means no
    /// explicit budget — defer to the per-model `max_thinking_budget`
    /// cap rather than a conservative hardcoded default (a hard low cap
    /// force-injects `</think>` mid-reasoning and wrecks tool
    /// selection).
    On { budget: Option<u32> },
}

impl ThinkingDirective {
    /// True when the client stated any explicit thinking intent —
    /// enabled OR disabled. Server-side policies (e.g. MODEL.toml
    /// `thinking_in_tools=false`) only apply when this is false.
    pub fn is_explicit(&self) -> bool {
        !matches!(self, ThinkingDirective::Unspecified)
    }
}
