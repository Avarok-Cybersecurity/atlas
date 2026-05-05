// SPDX-License-Identifier: AGPL-3.0-only

//! Shared scheduler-internal type definitions, factored out of `mod.rs`
//! for the ≤500 LoC cap (refactor wave-4e). Visibility is `pub(super)`
//! throughout — these types are scheduler-internal, but every sibling
//! file (decode_step, mtp_step, lifecycle, etc.) accesses them via
//! `super::*`.

#![allow(dead_code)]

use std::time::Instant;

use anyhow::Result;
use spark_model::traits::SequenceState;

use crate::api::{InferenceRequest, InferenceResponse, StreamEvent};
use crate::grammar::GrammarState;

/// Shared queue between receiver thread and scheduler.
pub(super) struct PendingQueue {
    pub requests: Vec<InferenceRequest>,
    pub closed: bool,
}

/// How to deliver results for an active sequence.
pub(super) enum ResponseSink {
    Blocking(Option<tokio::sync::oneshot::Sender<Result<InferenceResponse>>>),
    Streaming(tokio::sync::mpsc::Sender<StreamEvent>),
}

/// An in-progress chunked prefill (prompt being processed in chunks).
pub(super) struct PrefillInProgress {
    pub prompt_tokens: Vec<u32>,
    pub session_hash: u64,
    pub seq: SequenceState,
    pub chunk_offset: usize,
    pub max_tokens: usize,
    pub min_tokens: usize,
    pub eos_tokens: Vec<u32>,
    pub sink: ResponseSink,
    pub request_start: Instant,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub top_n_sigma: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub repetition_penalty_window: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub lz_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    pub dry_sequence_breakers: Vec<u32>,
    pub logit_bias: Vec<(u32, f32)>,
    pub enable_thinking: bool,
    pub thinking_budget: Option<u32>,
    /// Per-server spontaneous-thinking budget (from MODEL.toml
    /// `[behavior].max_thinking_budget`). When the model emits a
    /// `<think>` token without the request having explicitly enabled
    /// thinking, this caps how many thinking tokens it can produce
    /// before `</think>` is force-emitted. Replaces a previous
    /// hard-coded 512-token fallback.
    pub spontaneous_think_budget: u32,
    pub require_tool_call: bool,
    pub suppress_tool_call: bool,
    /// F60 (2026-04-27): MTP-disable flag (propagated to ActiveSeq).
    pub disable_mtp: bool,
    pub grammar_state: Option<GrammarState>,
    pub seed: Option<u64>,
    pub top_logprobs: Option<u8>,
    pub timeout_at: Option<Instant>,
}

/// An in-flight sequence participating in batched decode.
pub(super) struct ActiveSeq {
    pub seq: SequenceState,
    pub session_hash: u64,
    pub last_token: u32,
    pub output_tokens: Vec<u32>,
    pub remaining: usize,
    pub min_tokens: usize,
    pub eos_tokens: Vec<u32>,
    pub finished: bool,
    pub sink: ResponseSink,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub top_n_sigma: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub repetition_penalty_window: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub lz_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    pub dry_sequence_breakers: Vec<u32>,
    pub logit_bias: Vec<(u32, f32)>,
    /// Tracks whether the model is inside `<think>...</think>` reasoning.
    pub inside_thinking: bool,
    /// Whether the request opted into thinking mode (`enable_thinking=true`).
    /// When false but the model spontaneously emits `<think>`, the thinking-content
    /// tokens MUST NOT be streamed to the client.
    pub enable_thinking: bool,
    /// Max thinking tokens before forcing `</think>`. None = unlimited.
    pub thinking_budget: Option<u32>,
    /// Per-server spontaneous-thinking budget (from MODEL.toml
    /// `[behavior].max_thinking_budget`).
    pub spontaneous_think_budget: u32,
    /// Number of thinking tokens generated so far (counted while inside_thinking).
    pub thinking_tokens: u32,
    /// When true, the next decode step must produce the `</think>` token.
    pub force_end_thinking: bool,
    /// Consecutive tokens where top-1 softmax prob >= 0.95 (for confidence early stop).
    pub consecutive_confident: u32,
    /// Token ID for `</think>` (needed for budget enforcement in emit_token).
    pub think_end_token: Option<u32>,
    /// Token ID for `<think>` (needed for spontaneous thinking detection in emit_token).
    pub think_start_token: Option<u32>,
    /// True after the first `</think>` token is generated.
    pub think_ended: bool,
    /// One-shot signal: set when `</think>` was the most recently emitted token.
    pub think_just_ended: bool,
    /// Consecutive `</think>` tokens skipped outside thinking. Safety limit: 50.
    pub think_skip_count: u32,
    /// Token ID for `</tool_call>` — acts as a stop token for one-call-per-response.
    pub tool_call_end_token: Option<u32>,
    /// When true AND grammar_state is None, EOS tokens are suppressed until
    /// `<tool_call>` is generated (legacy fallback).
    pub require_tool_call: bool,
    /// Token ID for `<tool_call>` (legacy fallback when grammar is unavailable).
    pub tool_call_start_token: Option<u32>,
    /// True after `<tool_call>` generated in output (not inside thinking).
    pub tool_call_opened: bool,
    /// True between emission of `<tool_call>`/`<function=…>` (open) and
    /// `</tool_call>`/`</function>` (close).
    pub inside_tool_body: bool,
    /// When true, `<tool_call>` token logit is set to -inf during decode.
    pub suppress_tool_call: bool,
    /// F60 (2026-04-27): when true, MTP speculative decoding is bypassed.
    pub disable_mtp: bool,
    /// True after the first non-thinking content token has been generated.
    pub content_started: bool,
    /// Number of content tokens emitted post-`</think>`.
    pub content_tokens: u32,
    /// Free-text tokens emitted since the last `<tool_call>` opened.
    pub prose_tokens_since_last_tool: u32,
    /// F10 (2026-04-26): how many times the thinking-loop watchdog has fired.
    pub think_watchdog_fires: u32,
    /// F26 (2026-04-26): consecutive sample steps with collapsed entropy.
    pub entropy_collapse_streak: u32,
    /// F27 (2026-04-26): ring buffer of recent logit-distribution fingerprints.
    pub f27_fingerprint_ring: std::collections::VecDeque<u64>,
    pub f27_attractor_streak: u32,
    pub f27_last_emitted_token: u32,
    /// Grammar state for constrained decoding (tool_choice="required").
    pub grammar_state: Option<GrammarState>,
    /// MTP draft tokens awaiting verification.
    pub pending_drafts: Vec<u32>,
    /// Timestamp of the last token emission (for TBT deadline tracking).
    pub last_token_time: Instant,
    /// Timestamp when the request entered prefill (for TTFT).
    pub request_start: Instant,
    /// Decode start time (set after prefill completes, for decode throughput).
    pub decode_start: Instant,
    /// Seed for deterministic sampling.
    pub seed: Option<u64>,
    /// Number of top logprobs to return per token. None = disabled.
    pub top_logprobs: Option<u8>,
    /// Accumulated logprobs data for blocking responses.
    pub logprobs_data: Vec<crate::api::TokenLogprobs>,
    /// Request timeout deadline. None = no timeout.
    pub timeout_at: Option<Instant>,
    /// Adaptive sampling state.
    pub adaptive: crate::adaptive_sampler::AdaptiveSamplingState,
    /// Number of prompt tokens served by the prefix cache (no prefill cost).
    pub cached_prompt_tokens: u32,
}

/// A sequence that has been swapped out to disk (KV + SSM state saved to file).
pub(super) struct SwappedSeq {
    pub tokens: Vec<u32>,
    pub session_hash: u64,
    pub seq_len: usize,
    pub num_blocks: usize,
    pub last_token: u32,
    pub output_tokens: Vec<u32>,
    pub remaining: usize,
    pub min_tokens: usize,
    pub eos_tokens: Vec<u32>,
    pub sink: ResponseSink,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub top_n_sigma: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub repetition_penalty_window: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub lz_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    pub dry_sequence_breakers: Vec<u32>,
    pub logit_bias: Vec<(u32, f32)>,
    pub inside_thinking: bool,
    pub enable_thinking: bool,
    pub thinking_budget: Option<u32>,
    pub spontaneous_think_budget: u32,
    pub thinking_tokens: u32,
    pub force_end_thinking: bool,
    pub consecutive_confident: u32,
    pub think_end_token: Option<u32>,
    pub think_start_token: Option<u32>,
    pub think_ended: bool,
    pub think_just_ended: bool,
    pub think_skip_count: u32,
    pub require_tool_call: bool,
    pub suppress_tool_call: bool,
    /// F60 (2026-04-27): MTP-disable flag preserved across snapshot/restore.
    pub disable_mtp: bool,
    pub content_started: bool,
    pub content_tokens: u32,
    pub prose_tokens_since_last_tool: u32,
    pub think_watchdog_fires: u32,
    pub entropy_collapse_streak: u32,
    pub f27_fingerprint_ring: std::collections::VecDeque<u64>,
    pub f27_attractor_streak: u32,
    pub f27_last_emitted_token: u32,
    pub tool_call_start_token: Option<u32>,
    pub tool_call_opened: bool,
    pub tool_call_end_token: Option<u32>,
    pub last_token_time: Instant,
    pub request_start: Instant,
    pub decode_start: Instant,
    pub seed: Option<u64>,
    pub top_logprobs: Option<u8>,
    pub logprobs_data: Vec<crate::api::TokenLogprobs>,
    /// Number of prompt tokens served by the prefix cache (no prefill cost).
    pub cached_prompt_tokens: u32,
    pub timeout_at: Option<Instant>,
    pub swap_id: u64,
}
