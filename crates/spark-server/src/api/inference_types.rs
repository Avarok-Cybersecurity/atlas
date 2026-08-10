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

/// Per-request override for the vLLM-anchored token-loop detector.
///
/// Mirrors vLLM's `RepetitionDetectionParams`
/// (`vllm/sampling_params.py:111-144`):
/// - `min_pattern_size` → smallest pattern length (in tokens) to consider
/// - `max_pattern_size` → largest pattern length to consider
/// - `min_count` → number of end-anchored back-to-back repeats that
///   constitute a "loop"
///
/// When attached to a request, these override the boot-global
/// thresholds (`CONTENT_LOOP_PERIOD_MIN` / `_MAX` /
/// `CONTENT_LOOP_MIN_REPEATS` and the thinking-loop equivalents) for
/// that single sequence's content-loop and thinking-loop detectors.
///
/// Defined here (not in a wire module) because the scheduler and
/// chat-stream internals consume it; the API surfaces deserialize into
/// it at their edges.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RepetitionDetectionParams {
    pub min_pattern_size: u32,
    pub max_pattern_size: u32,
    pub min_count: u32,
}

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
        /// Prompt token slice. `Arc`-wrapped so the scheduler request,
        /// the streaming context, and the Tier 5c retry path can share
        /// the read-only data without cloning ~40 KB for a typical
        /// long-context opencode prompt.
        prompt_tokens: std::sync::Arc<Vec<u32>>,
        /// Session hash for SSM snapshot isolation (hash of first 64 prompt tokens).
        session_hash: u64,
        /// M2 per-request LoRA routing: adapter pool SLOT this request applies
        /// (`-1` = defer to installed active; byte-identical to today). Set from
        /// the resolved `adapter` field; carried onto `SequenceState.adapter_slot`.
        adapter_slot: i32,
        /// Per-request source-language token id (0 = deployment default);
        /// carried onto `SequenceState.src_lang_id`.
        src_lang_id: u32,
        /// Per-request target-language token id (0 = deployment default);
        /// carried onto `SequenceState.tgt_lang_id`.
        tgt_lang_id: u32,
        /// NLLB beam search: beams per request (1 = greedy); carried onto
        /// `SequenceState.num_beams`.
        num_beams: u32,
        /// NLLB beam search: length penalty; carried onto
        /// `SequenceState.length_penalty`.
        length_penalty: f32,
        /// NLLB beam search: early stopping; carried onto
        /// `SequenceState.early_stopping`.
        early_stopping: bool,
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
        /// Per-request override for the vLLM-anchored token-loop detector.
        /// `None` = use the boot-global watchdog parameters.
        repetition_detection: Option<RepetitionDetectionParams>,
        /// Whether a tool call is required (tool_choice="required").
        require_tool_call: bool,
        /// #192: whether the request declared tools (chat `tools` non-empty
        /// with a parser available). Unlike `grammar_spec` it stays true even
        /// when constrained decoding is unavailable or disabled
        /// (MODEL.toml `disable_tool_grammar`, parser opt-out, compile
        /// failure), so the scheduler can gate multi-tool-call continuation
        /// on "tools defined" rather than "grammar attached": with tools
        /// present, `</tool_call>` is NOT a hard stop — generation continues
        /// so the model can emit parallel calls, ending at natural EOS.
        tools_present: bool,
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
        /// Legacy /v1/completions `logprobs`: Some(k) = collect prompt-token
        /// logprobs (top-k alternatives) during prefill. Forces full
        /// recompute (prefix cache bypassed) so every position is scored.
        prompt_logprobs: Option<u8>,
        /// Legacy /v1/completions `echo`: prepend the prompt to the
        /// response text (handler-side; carried here for symmetry/logging).
        echo: bool,
        /// Request timeout as absolute deadline. None = no timeout.
        timeout_at: Option<std::time::Instant>,
        response_tx: tokio::sync::oneshot::Sender<anyhow::Result<InferenceResponse>>,
    },
    /// Streaming: sends tokens as they're generated.
    Streaming {
        /// Prompt token slice. `Arc`-wrapped so the scheduler request,
        /// the streaming context, and the Tier 5c retry path can share
        /// the read-only data without cloning ~40 KB for a typical
        /// long-context opencode prompt.
        prompt_tokens: std::sync::Arc<Vec<u32>>,
        /// Session hash for SSM snapshot isolation (hash of first 64 prompt tokens).
        session_hash: u64,
        /// M2 per-request LoRA routing: adapter pool SLOT (`-1` = defer to
        /// installed active). See the Blocking variant.
        adapter_slot: i32,
        /// Per-request source-language token id (0 = deployment default).
        /// See the Blocking variant.
        src_lang_id: u32,
        /// Per-request target-language token id (0 = deployment default).
        /// See the Blocking variant.
        tgt_lang_id: u32,
        /// NLLB beam search: beams per request (1 = greedy). See Blocking.
        num_beams: u32,
        /// NLLB beam search: length penalty. See Blocking.
        length_penalty: f32,
        /// NLLB beam search: early stopping. See Blocking.
        early_stopping: bool,
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
        /// Per-request override for the vLLM-anchored token-loop detector.
        /// `None` = use the boot-global watchdog parameters.
        repetition_detection: Option<RepetitionDetectionParams>,
        /// Whether a tool call is required (tool_choice="required").
        require_tool_call: bool,
        /// #192: whether the request declared tools — see the Blocking
        /// variant for the multi-tool-call continuation contract.
        tools_present: bool,
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
        /// Legacy /v1/completions `logprobs`: Some(k) = collect prompt-token
        /// logprobs during prefill (see Blocking variant).
        prompt_logprobs: Option<u8>,
        /// Legacy /v1/completions `echo` (see Blocking variant).
        echo: bool,
        /// Request timeout as absolute deadline. None = no timeout.
        timeout_at: Option<std::time::Instant>,
        token_tx: tokio::sync::mpsc::Sender<StreamEvent>,
        /// Cooperative cancellation flag, shared with the streaming
        /// pipeline. Set true by chat_stream guards (tool-call loop
        /// cap, watchdog, etc.) to ask the scheduler to terminate
        /// the sequence at the next decode boundary. The scheduler
        /// reads it in `emit_step::emit_token`; flipping it true is
        /// equivalent to receiving an EOS — `a.finished = true`, the
        /// usual finalize path runs, and `handle_done` emits the
        /// proper final-chunk (`finish_reason="length"` via the
        /// `tool_loop_capped` override) plus `[DONE]`. Without this,
        /// `stop_string_triggered` only suppresses *output*; the
        /// scheduler keeps generating until natural EOS / max_tokens,
        /// which on a degenerate-loop response can hang.
        cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
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
    /// Prompt-token logprobs (legacy /v1/completions `logprobs` + `echo`):
    /// one entry per prompt position i in [0, prompt_len-1) scoring token
    /// i+1. Empty unless requested. The handler prepends the null entry
    /// for the first prompt token (no preceding context).
    pub prompt_logprobs: Vec<TokenLogprobs>,
}

/// Events sent during streaming generation.
pub enum StreamEvent {
    Token(u32),
    /// Token with logprobs data (when top_logprobs is requested).
    TokenWithLogprobs(u32, TokenLogprobs),
    /// Prompt-token logprobs, emitted once right after prefill when
    /// `prompt_logprobs` was requested (legacy completions echo path).
    PromptLogprobs(Vec<TokenLogprobs>),
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
        /// Server-side guard that force-finished the sequence (e.g.
        /// "fuzzy_repetition"), if any — dump/observability only, never
        /// part of the OpenAI wire format.
        guard_stop: Option<&'static str>,
    },
    Error(String),
}
