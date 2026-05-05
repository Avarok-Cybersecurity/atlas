// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

/// Grammar specification for constrained decoding.
///
/// Either tool-call grammar (Hermes/Qwen3-Coder format) or response-format
/// grammar (generic JSON or JSON-schema).
#[derive(Clone)]
pub enum GrammarSpec {
    /// Tool-call constrained decoding: tools, parser instance, use_triggers flag.
    ///
    /// F69 (2026-04-29): carries the `Arc<dyn ToolCallParser>` itself
    /// rather than a stringly-typed `parser_name`. The scheduler
    /// dispatches `parser.compile_tool_grammar(engine, tools, use_triggers)`
    /// — the parser is the single source of truth for both response
    /// scanning AND grammar generation. This eliminates the prior
    /// `match parser_name.as_str()` in
    /// `scheduler.rs::compile_grammar_state` that could drift from the
    /// parser identity stored in `AppState.tool_call_parser`.
    ToolCall {
        tools: Vec<tool_parser::ToolDefinition>,
        parser: std::sync::Arc<dyn tool_parser::ToolCallParser>,
        use_triggers: bool,
    },
    /// Response format: any valid JSON (json_object).
    JsonObject,
    /// Response format: JSON matching a specific schema.
    JsonSchema { schema: String },
}

/// Request submitted to the scheduler.
pub enum InferenceRequest {
    /// Blocking: waits for full response.
    Blocking {
        prompt_tokens: Vec<u32>,
        /// Session hash for SSM snapshot isolation (hash of first 64 prompt tokens).
        session_hash: u64,
        /// Preprocessed image data: (pixels `[P,1536]` f32, grid_h, grid_w) per image.
        image_pixels: Vec<(Vec<f32>, usize, usize)>,
        max_tokens: usize,
        /// Minimum tokens before allowing EOS/stop (0 = no minimum).
        min_tokens: usize,
        temperature: f32,
        /// Top-k: keep only the k highest-probability tokens (0 = disabled).
        top_k: u32,
        /// Top-p (nucleus): cumulative probability threshold (1.0 = disabled).
        top_p: f32,
        /// Top-n-sigma: filter tokens by logit z-score (0.0 = disabled).
        top_n_sigma: f32,
        /// Min-p: keep tokens with prob >= min_p * max_prob (0.0 = disabled).
        min_p: f32,
        /// Repetition penalty (1.0 = disabled).
        repetition_penalty: f32,
        /// Presence penalty — OpenAI-style, additive (0.0 = disabled).
        presence_penalty: f32,
        /// Frequency penalty — OpenAI-style, additive (0.0 = disabled).
        frequency_penalty: f32,
        /// DRY (Don't-Repeat-Yourself) n-gram sampler. Populated from
        /// `sampling_presets.tools.dry_multiplier` etc. when the model's
        /// MODEL.toml opts in (e.g. qwen3.5-35b-a3b/MODEL.toml sets
        /// `dry_multiplier = 0.8` for the tools preset). 0.0 = disabled.
        dry_multiplier: f32,
        dry_base: f32,
        dry_allowed_length: u32,
        /// LZ penalty (A.1, arXiv:2504.20131). Per-extension n-gram
        /// penalty over the recent 256-token window. Populated from
        /// MODEL.toml `[sampling.*].lz_penalty`. 0.0 = disabled.
        lz_penalty: f32,
        /// Per-token logit bias: (token_id, bias_value) pairs.
        logit_bias: Vec<(u32, f32)>,
        /// Per-request stop token IDs (from OpenAI `stop` parameter).
        stop_tokens: Vec<u32>,
        /// Whether thinking mode is enabled for this request.
        enable_thinking: bool,
        /// Max thinking tokens before forcing `</think>`. None = unlimited.
        thinking_budget: Option<u32>,
        /// Whether a tool call is required (tool_choice="required").
        require_tool_call: bool,
        /// Suppress `<tool_call>` token when tool call loop detected (≥3 identical).
        suppress_tool_call: bool,
        /// F60 (2026-04-27): disable MTP speculative decoding for this
        /// sequence. Set when the request has tools active and the
        /// `ATLAS_DISABLE_MTP_FOR_TOOLS` env-gate is on (default true).
        /// Hybrid GDN+attention models (Qwen3.5-35B-A3B) exhibit
        /// documented SSM state corruption under MTP rejection on
        /// agentic workloads (89% reject rate observed). vLLM #36872,
        /// #38106 + STree (arXiv:2505.14969) recommend disabling
        /// speculative decode for tool-use traffic. The MTP rollback
        /// machinery is correct on inspection, but bypassing it
        /// entirely on tool-use turns eliminates the entire failure
        /// class while preserving MTP for non-tool chat workloads.
        disable_mtp: bool,
        /// Grammar specification for constrained decoding (tools or response_format).
        grammar_spec: Option<GrammarSpec>,
        /// Seed for deterministic sampling. None = non-deterministic.
        seed: Option<u64>,
        /// Number of top logprobs to return per token. None = disabled.
        top_logprobs: Option<u8>,
        /// Request timeout as absolute deadline. None = no timeout.
        timeout_at: Option<std::time::Instant>,
        response_tx: tokio::sync::oneshot::Sender<anyhow::Result<InferenceResponse>>,
    },
    /// Streaming: sends tokens as they're generated.
    Streaming {
        prompt_tokens: Vec<u32>,
        /// Session hash for SSM snapshot isolation (hash of first 64 prompt tokens).
        session_hash: u64,
        /// Preprocessed image data: (pixels `[P,1536]` f32, grid_h, grid_w) per image.
        image_pixels: Vec<(Vec<f32>, usize, usize)>,
        max_tokens: usize,
        /// Minimum tokens before allowing EOS/stop (0 = no minimum).
        min_tokens: usize,
        temperature: f32,
        /// Top-k: keep only the k highest-probability tokens (0 = disabled).
        top_k: u32,
        /// Top-p (nucleus): cumulative probability threshold (1.0 = disabled).
        top_p: f32,
        /// Top-n-sigma: filter tokens by logit z-score (0.0 = disabled).
        top_n_sigma: f32,
        /// Min-p: keep tokens with prob >= min_p * max_prob (0.0 = disabled).
        min_p: f32,
        /// Repetition penalty (1.0 = disabled).
        repetition_penalty: f32,
        /// Presence penalty — OpenAI-style, additive (0.0 = disabled).
        presence_penalty: f32,
        /// Frequency penalty — OpenAI-style, additive (0.0 = disabled).
        frequency_penalty: f32,
        /// DRY (Don't-Repeat-Yourself) n-gram sampler (see Blocking
        /// variant for rationale).
        dry_multiplier: f32,
        dry_base: f32,
        dry_allowed_length: u32,
        /// LZ penalty (A.1, see Blocking variant).
        lz_penalty: f32,
        /// Per-token logit bias: (token_id, bias_value) pairs.
        logit_bias: Vec<(u32, f32)>,
        /// Per-request stop token IDs (from OpenAI `stop` parameter).
        stop_tokens: Vec<u32>,
        /// Whether thinking mode is enabled for this request.
        enable_thinking: bool,
        /// Max thinking tokens before forcing `</think>`. None = unlimited.
        thinking_budget: Option<u32>,
        /// Whether a tool call is required (tool_choice="required").
        require_tool_call: bool,
        /// Suppress `<tool_call>` token when tool call loop detected (≥3 identical).
        suppress_tool_call: bool,
        /// F60 (2026-04-27): disable MTP speculative decoding for this
        /// sequence. Set when the request has tools active and the
        /// `ATLAS_DISABLE_MTP_FOR_TOOLS` env-gate is on (default true).
        /// Hybrid GDN+attention models (Qwen3.5-35B-A3B) exhibit
        /// documented SSM state corruption under MTP rejection on
        /// agentic workloads (89% reject rate observed). vLLM #36872,
        /// #38106 + STree (arXiv:2505.14969) recommend disabling
        /// speculative decode for tool-use traffic. The MTP rollback
        /// machinery is correct on inspection, but bypassing it
        /// entirely on tool-use turns eliminates the entire failure
        /// class while preserving MTP for non-tool chat workloads.
        disable_mtp: bool,
        /// Grammar specification for constrained decoding (tools or response_format).
        grammar_spec: Option<GrammarSpec>,
        /// Seed for deterministic sampling. None = non-deterministic.
        seed: Option<u64>,
        /// Number of top logprobs to return per token. None = disabled.
        top_logprobs: Option<u8>,
        /// Request timeout as absolute deadline. None = no timeout.
        timeout_at: Option<std::time::Instant>,
        token_tx: tokio::sync::mpsc::Sender<StreamEvent>,
    },
}

/// Per-token logprobs data extracted from the logits buffer.
#[derive(Clone)]
pub struct TokenLogprobs {
    /// The sampled token ID.
    pub token_id: u32,
    /// Log-probability of the sampled token.
    pub logprob: f32,
    /// Top-K alternative tokens with their logprobs, sorted descending.
    pub top: Vec<(u32, f32)>,
}

/// Response from the scheduler (blocking path).
pub struct InferenceResponse {
    pub output_tokens: Vec<u32>,
    pub finish_reason: String,
    pub time_to_first_token_ms: f64,
    pub decode_time_ms: f64,
    /// Per-token logprobs (populated when top_logprobs is requested).
    pub logprobs: Vec<TokenLogprobs>,
    /// Number of generated tokens that were inside the thinking/reasoning
    /// block (counted AS PART OF `output_tokens.len()`). Reported to clients
    /// as `usage.completion_tokens_details.reasoning_tokens`.
    pub reasoning_tokens: u32,
    /// Number of prompt tokens served by the prefix cache (no prefill compute
    /// cost). Reported as `usage.prompt_tokens_details.cached_tokens`.
    pub cached_prompt_tokens: u32,
}

/// Events sent during streaming generation.
pub enum StreamEvent {
    Token(u32),
    /// Token with logprobs data (when top_logprobs is requested).
    TokenWithLogprobs(u32, TokenLogprobs),
    Done {
        finish_reason: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        time_to_first_token_ms: f64,
        decode_time_ms: f64,
        /// Tokens inside the thinking/reasoning block (for usage details).
        reasoning_tokens: u32,
        /// Prefix-cached prompt tokens (for usage details).
        cached_prompt_tokens: u32,
    },
    Error(String),
}
