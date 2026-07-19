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
use crate::openai::RepetitionDetectionParams;

/// Shared queue between receiver thread and scheduler.
pub(super) struct PendingQueue {
    pub requests: Vec<InferenceRequest>,
    pub closed: bool,
}

/// Per-request slice of a co-dispatched batched-ViT encode. When >=2 image
/// requests are admitted in one tick, the scheduler encodes all their images
/// in ONE `forward_batched` call (block GEMM weights read once over Σpatches)
/// and hands each request the offsets it owns in the shared packed `buf_out`.
/// `Default` (all zero) means "not co-dispatched" → the request self-encodes,
/// reading from row 0 / grid 0 exactly as the legacy single-request path.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct VisionSlice {
    /// First `buf_out` row (post-merge patch) this request owns.
    pub patch_row_offset: usize,
    /// First `vision_image_grids` index this request owns.
    pub grid_index_offset: usize,
    /// Number of images this request contributed to the batch.
    pub num_images: usize,
    /// Total post-merge rows this request owns (Σ merged_p over its images).
    pub patch_row_count: usize,
}

/// How to deliver results for an active sequence.
pub(super) enum ResponseSink {
    Blocking(Option<tokio::sync::oneshot::Sender<Result<InferenceResponse>>>),
    Streaming(tokio::sync::mpsc::Sender<StreamEvent>),
}

/// An in-progress chunked prefill (prompt being processed in chunks).
pub(super) struct PrefillInProgress {
    /// Arc-wrapped so the original request, the per-prefill scheduler
    /// state, and any retry path (Tier 5c) can share the read-only
    /// token slice without copying ~40 KB on every long prompt.
    pub prompt_tokens: std::sync::Arc<Vec<u32>>,
    pub session_hash: u64,
    pub seq: SequenceState,
    pub chunk_offset: usize,
    pub max_tokens: usize,
    pub min_tokens: usize,
    pub eos_tokens: Vec<u32>,
    pub sink: ResponseSink,
    /// Cooperative cancellation flag — see ActiveSeq for the contract.
    /// Propagated to ActiveSeq when this PrefillInProgress promotes.
    pub cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
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
    /// Per-request override for the vLLM-anchored token-loop detector.
    /// Propagated to `ActiveSeq` on promotion. `None` = use the
    /// boot-global watchdog parameters.
    pub repetition_detection: Option<RepetitionDetectionParams>,
    /// Per-server spontaneous-thinking budget (from MODEL.toml
    /// `[behavior].max_thinking_budget`). When the model emits a
    /// `<think>` token without the request having explicitly enabled
    /// thinking, this caps how many thinking tokens it can produce
    /// before `</think>` is force-emitted. Replaces a previous
    /// hard-coded 512-token fallback.
    pub spontaneous_think_budget: u32,
    pub require_tool_call: bool,
    /// #192: request declared tools (propagated to ActiveSeq on promotion).
    pub tools_present: bool,
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
    /// Cooperative cancellation flag from the streaming pipeline.
    /// `Some` for streaming requests with the flag wired through;
    /// `None` for blocking requests. `emit_step::emit_token` reads
    /// it on every token and finalises the sequence when set —
    /// equivalent to receiving an EOS. Set by `chat_stream` guards
    /// (tool-call loop cap, loop-watchdog) so the scheduler stops
    /// generating instead of just having its output suppressed.
    pub cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
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
    /// Per-request override for the vLLM-anchored token-loop detector
    /// (content-loop + thinking-loop). `None` = use the boot-global
    /// watchdog parameters. Mirrors vLLM's `RepetitionDetectionParams`.
    pub repetition_detection: Option<RepetitionDetectionParams>,
    /// Per-server spontaneous-thinking budget (from MODEL.toml
    /// `[behavior].max_thinking_budget`).
    pub spontaneous_think_budget: u32,
    /// Number of thinking tokens generated so far (counted while inside_thinking).
    pub thinking_tokens: u32,
    /// When true, the next decode step must produce the `</think>` token.
    pub force_end_thinking: bool,
    /// Decode-step counter incremented while `force_end_thinking` is
    /// armed but the injection is deferred (waiting for a sentence-
    /// boundary token or fence close). Reset to 0 on the false→true
    /// arm transition and on the true→false reset (`</think>` emitted
    /// or model exited thinking). Bounded by
    /// [`crate::scheduler::confidence::MAX_SENTENCE_DEFER_TOKENS`] —
    /// past that the caller computes `hard_override = true` and
    /// `should_inject_think_end` fires unconditionally.
    pub sentence_defer_count: u32,
    /// Consecutive tokens where top-1 softmax prob >= 0.95 (for confidence early stop).
    pub consecutive_confident: u32,
    /// True while the model is inside an unclosed ``` code fence within
    /// the current thinking block. Toggled on each sampled code-fence
    /// token. The F2 confidence early-stop is suppressed while this is
    /// set: code is near-deterministic (high top-1 prob) but that is
    /// NOT a "done reasoning" signal — braking here truncates the model
    /// mid-statement. Per-seq state, persisted across decode steps and
    /// snapshots.
    pub in_code_fence: bool,
    /// Token ID for `</think>` (needed for budget enforcement in emit_token).
    pub think_end_token: Option<u32>,
    /// Token ID for `<think>` (needed for spontaneous thinking detection in emit_token).
    pub think_start_token: Option<u32>,
    /// True after the first `</think>` token is generated.
    pub think_ended: bool,
    /// One-shot signal: set when `</think>` was the most recently emitted token.
    pub think_just_ended: bool,
    /// Tokens emitted since `</think>` (0 while thinking; resets if the model
    /// re-enters a think block). Consumed by the DFlash spec-resume guard
    /// (ATLAS_DFLASH_RESUME_GUARD) to keep the answer's opening tokens on
    /// serial decode, where the T=0 verify-vs-decode low-margin flips
    /// concentrate (measured 2026-07-07).
    pub post_think_emitted: u32,
    /// Adaptive speculation (ATLAS_DFLASH_ADAPTIVE=1): rolling accept window
    /// + suspend/re-probe state. Transient — reset on swap/restore (a
    /// resumed sequence re-measures). See `adaptive_spec` module docs.
    pub spec_adapt: crate::scheduler::adaptive_spec::AdaptState,
    /// Consecutive `</think>` tokens skipped outside thinking. Safety limit: 50.
    pub think_skip_count: u32,
    /// Token ID for `</tool_call>`. Hard-stops only NON-tool requests
    /// (spurious tool-call in plain chat); tool-armed requests continue past
    /// it so multiple/parallel calls can follow (#192).
    pub tool_call_end_token: Option<u32>,
    /// When true AND grammar_state is None, EOS tokens are suppressed until
    /// `<tool_call>` is generated (legacy fallback).
    pub require_tool_call: bool,
    /// F4 (2026-06-02): sticky "this is a tool request" flag, set once at
    /// prefill when a grammar is attached OR the legacy tool-call path is
    /// active. Unlike `grammar_state.is_some()` it survives a graceful
    /// grammar disengage (`emit_step` drops `grammar_state` to `None` to
    /// salvage a turn), so the inter-tool prose-budget watchdog does not
    /// go inert when the grammar disengages mid-response. Default false ⇒
    /// no-op for non-tool requests (plain chat is never prose-capped).
    pub tool_request: bool,
    /// #192: the request declared tools (from the API layer's `tools_active`).
    /// Unlike `tool_request` this does NOT arm the prose-budget / post-think
    /// watchdogs; it only gates multi-tool-call continuation: when true, a
    /// `</tool_call>` outside a grammar does not finish the sequence —
    /// generation continues (vLLM parity) so the model can emit parallel
    /// calls, ending at natural EOS. When false (plain chat), a spurious
    /// `</tool_call>` keeps its historical hard stop.
    pub tools_present: bool,
    /// Token ID for `<tool_call>` (legacy fallback when grammar is unavailable).
    pub tool_call_start_token: Option<u32>,
    /// True after `<tool_call>` generated in output (not inside thinking).
    pub tool_call_opened: bool,
    /// True between emission of `<tool_call>`/`<function=…>` (open) and
    /// `</tool_call>`/`</function>` (close).
    pub inside_tool_body: bool,
    /// Fix A (2026-06-05): true once a complete `</tool_call>` has been emitted;
    /// gates the EOS-escape (helpers::tool_eos_escape_enabled).
    pub tool_call_completed: bool,
    /// Number of `<tool_call>` openers emitted AFTER the first one completed
    /// (i.e. while `tool_call_completed == true`), outside thinking. On a
    /// `tool_choice="auto"` grammar turn the grammar never reaches a terminal
    /// state (`stop_after_first=false`), so the EOS-escape is the only way to
    /// stop — but re-entering a tool body (`inside_tool_body=true`) defeats the
    /// escape's `!inside_tool_body` guard. A degenerating FP8/long-context model
    /// loops emitting full `<tool_call>…</tool_call>` blocks as content; each
    /// closes cleanly so the envelope-streak guard never fires, and EOS is
    /// suppressed to the max_tokens cap (8k-tok runaway, ~260s wasted). This
    /// counter detects that repetition: once it exceeds
    /// `MAX_POST_COMPLETION_TOOL_OPENS` the escape is force-armed so the model's
    /// natural EOS can end the turn. Legit multi-call turns are unaffected — we
    /// only stop SUPPRESSING EOS, never force-finish, so a model still mid-call
    /// keeps generating until it actually samples EOS. Reset never needed (the
    /// turn ends once it trips).
    pub post_completion_tool_opens: u32,
    /// Consecutive tokens emitted while `inside_tool_body=true`. When
    /// this exceeds `MAX_TOOL_BODY_TOKENS` (emit_step.rs), the response
    /// is force-ended: the model has emitted a `<tool_call>` opener but
    /// never reached a matching close — observed live 2026-05-24 on
    /// NVFP4 Qwen3.6 (opencode-nvfp4.jsonl seq=15: 8221 tokens, all
    /// suppressed by sanitizer as unclosed tool-call envelope, hit
    /// max_tokens=8192). 1024 tokens is enough headroom for legitimate
    /// long tool-call bodies (large `content` field on a `write` call)
    /// while bounding worst-case wasted decode at ~15s @ 65 tok/s.
    /// Resets to 0 on tool_call_end emission.
    pub tool_body_streak_tokens: u32,
    /// Tier-1 (Epoch 1) sampler byte counter: True between the model
    /// emitting `<parameter=KEY>` and the matching `</parameter>` close.
    /// While true AND `param_body_chars_emitted == 0`, decode_logits_seq.rs
    /// masks token id 510 (`</`, first token of `</parameter>`) with bias
    /// -8.0 so the model is forced to emit at least one non-close token
    /// before the close-tag's first byte can be sampled. Defends against
    /// xgrammar's failure to enforce `minLength: 1` on json_schema body
    /// (3 grammar attempts so far — regex `\S` sandwich, regex `+`
    /// quantifier, json_schema style qwen_xml with minLength:1 — none
    /// enforce due to upstream xgrammar ε-edge bugs documented in
    /// `bench/fp8_dgx2_drift/research_synthesis.md`).
    pub inside_parameter_body: bool,
    /// Tier-1 byte counter — number of tokens emitted INSIDE
    /// `<parameter=KEY>…</parameter>` body so far. Reset to 0 on opener.
    /// Increments by 1 per token while inside; used as the mask-gate.
    pub param_body_chars_emitted: u32,
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
    /// Phase-C: how many times a degeneration watchdog has rolled this
    /// sequence back to a boundary and re-steered. Capped at
    /// [`atlas_kernels::ROLLBACK_RESTEER_CAP`]; once the cap is hit the
    /// watchdog reverts to a hard stop. See
    /// [`super::rollback::rollback_to_boundary`].
    pub rollback_count: u32,
    /// Phase-C: decode-time SSM-snapshot ring for hybrid (attention +
    /// Mamba/SSM) models. Records a bounded set of SSM `h_state` +
    /// `conv_state` snapshots taken at boundary tokens so a watchdog
    /// rollback can restore the recurrent state — not just the KV
    /// cache — to the chosen boundary. Disabled (`capacity == 0`,
    /// every op a no-op) for pure-attention models. See
    /// [`super::ssm_decode_ring::SsmDecodeRing`].
    pub ssm_rollback_ring: super::ssm_decode_ring::SsmDecodeRing,
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

impl ActiveSeq {
    /// Consume one token of generation budget (SSOT for the two decode
    /// paths: MTP `emit_step::emit_token` and non-MTP
    /// `decode_logits_content::handle_content_token`).
    ///
    /// `remaining` reaching 0 must finish the sequence at the next
    /// `remaining == 0` check. A second decrement at 0 means a budget
    /// desync (e.g. a token processed after the length stop should have
    /// fired): in release builds the old bare `-= 1` wrapped to
    /// usize::MAX, unbounding generation entirely (issue #94's
    /// never-terminating ngram loop; debug builds panicked with
    /// 'attempt to subtract with overflow'). Saturate, log, and finish
    /// instead — never wrap, never hide the desync.
    pub fn consume_generation_budget(&mut self) {
        if !consume_budget(&mut self.remaining) {
            tracing::warn!(
                output_tokens = self.output_tokens.len(),
                "generation budget decremented at 0 (token processed after \
                 length stop should have fired) — finishing sequence instead \
                 of wrapping"
            );
            self.finished = true;
        }
    }
}

/// Decrement `remaining` by one. Returns false (without touching
/// `remaining`) when it is already 0 — the caller must finish the
/// sequence rather than wrap.
pub(super) fn consume_budget(remaining: &mut usize) -> bool {
    if *remaining == 0 {
        return false;
    }
    *remaining -= 1;
    true
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
    /// Per-request override for the vLLM-anchored token-loop detector,
    /// preserved across snapshot/restore.
    pub repetition_detection: Option<RepetitionDetectionParams>,
    pub spontaneous_think_budget: u32,
    pub thinking_tokens: u32,
    pub force_end_thinking: bool,
    pub sentence_defer_count: u32,
    pub consecutive_confident: u32,
    pub in_code_fence: bool,
    pub think_end_token: Option<u32>,
    pub think_start_token: Option<u32>,
    pub think_ended: bool,
    pub think_just_ended: bool,
    pub post_think_emitted: u32,
    pub think_skip_count: u32,
    pub require_tool_call: bool,
    /// F4 (2026-06-02): sticky tool-request flag, preserved across
    /// snapshot/restore (the grammar state itself is not serializable, so
    /// this is the only signal that a resumed sequence was tool-active).
    pub tool_request: bool,
    /// #192: request declared tools, preserved across snapshot/restore so a
    /// resumed multi-call turn keeps continuing past `</tool_call>`.
    pub tools_present: bool,
    pub suppress_tool_call: bool,
    /// F60 (2026-04-27): MTP-disable flag preserved across snapshot/restore.
    pub disable_mtp: bool,
    pub content_started: bool,
    pub content_tokens: u32,
    pub prose_tokens_since_last_tool: u32,
    pub think_watchdog_fires: u32,
    /// Phase-C: watchdog rollback counter, preserved across snapshot/restore.
    pub rollback_count: u32,
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

#[cfg(test)]
mod budget_tests {
    use super::consume_budget;

    #[test]
    fn consume_budget_decrements_to_zero() {
        let mut r = 2usize;
        assert!(consume_budget(&mut r));
        assert_eq!(r, 1);
        assert!(consume_budget(&mut r));
        assert_eq!(r, 0);
    }

    #[test]
    fn consume_budget_at_zero_signals_finish_and_never_wraps() {
        // Regression: a bare `remaining -= 1` at 0 wrapped to usize::MAX in
        // release builds, unbounding generation (issue #94 ngram runaway),
        // and panicked in debug builds.
        let mut r = 0usize;
        assert!(!consume_budget(&mut r));
        assert_eq!(r, 0);
    }
}
