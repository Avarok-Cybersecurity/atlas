// SPDX-License-Identifier: AGPL-3.0-only

//! Shared `#[cfg(test)]` fixtures for the scheduler tests.
//!
//! The scheduler's existing unit tests all test PURE cores (see
//! `rollback_tests.rs`, `emit_step.rs::cc6_envelope_streak_tests`)
//! precisely because no `ActiveSeq` fixture existed. That is fine for a
//! pure predicate, but it cannot catch a guard whose bug is in the
//! ORDER of statements inside `emit_token` — which is exactly the
//! `think_skip_count` reset. This module supplies one real `ActiveSeq`
//! so behavioural tests can drive the real entry point.
//!
//! SSOT: one fixture for every scheduler test that needs an `ActiveSeq`;
//! new call sites extend this, they do not copy it.

use super::types::{ActiveSeq, ResponseSink};
use super::{DEFAULT_LZ_PENALTY, SsmDecodeRing};
use crate::api::InferenceResponse;
use anyhow::Result;
use spark_model::traits::SequenceState;
use std::time::Instant;

pub(super) const EOS: &[u32] = &[151645];
const TOOL_END: Option<u32> = Some(151658);

pub(super) type RespRx = tokio::sync::oneshot::Receiver<Result<InferenceResponse>>;

/// A real `ActiveSeq` with a blocking oneshot sink. `min_tokens` is
/// deliberately set to a value DIFFERENT from `remaining` (7) so a
/// call-site mutation that passes the wrong field flips a test red.
pub(super) fn test_seq(
    output_tokens: Vec<u32>,
    remaining: usize,
    guard_stop: Option<&'static str>,
    seq_len: usize,
) -> (ActiveSeq, RespRx) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let now = Instant::now();
    let mut seq = SequenceState::host_only(0);
    seq.seq_len = seq_len;
    let a = ActiveSeq {
        seq,
        session_hash: 0,
        last_token: output_tokens.last().copied().unwrap_or(0),
        output_tokens,
        remaining,
        min_tokens: 7,
        eos_tokens: EOS.to_vec(),
        finished: true,
        guard_stop,
        param_close_pending: 0,
        sink: ResponseSink::Blocking(Some(tx)),
        cancel_flag: None,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        repetition_penalty_window: 256,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: DEFAULT_LZ_PENALTY,
        dry_multiplier: 0.0,
        dry_base: 0.0,
        dry_allowed_length: 0,
        dry_sequence_breakers: Vec::new(),
        logit_bias: Vec::new(),
        inside_thinking: false,
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: None,
        spontaneous_think_budget: 0,
        thinking_tokens: 0,
        force_end_thinking: false,
        think_force_closed: false,
        sentence_defer_count: 0,
        consecutive_confident: 0,
        in_code_fence: false,
        think_end_token: None,
        think_start_token: None,
        think_ended: false,
        think_just_ended: false,
        post_think_emitted: 0,
        spec_adapt: Default::default(),
        think_skip_count: 0,
        tool_call_end_token: TOOL_END,
        require_tool_call: false,
        tool_request: false,
        tools_present: false,
        tool_call_start_token: None,
        tool_call_opened: false,
        inside_tool_body: false,
        tool_call_completed: false,
        post_completion_tool_opens: 0,
        tool_body_streak_tokens: 0,
        inside_parameter_body: false,
        param_body_chars_emitted: 0,
        suppress_tool_call: false,
        disable_mtp: false,
        mtp_acct: Default::default(),
        content_started: false,
        content_tokens: 0,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: 0,
        rollback_count: 0,
        ssm_rollback_ring: SsmDecodeRing::new(0),
        grammar_state: None,
        pending_drafts: Vec::new(),
        pending_draft_conf: Vec::new(),
        last_token_time: now,
        request_start: now,
        decode_start: now,
        seed: None,
        top_logprobs: None,
        logprobs_data: Vec::new(),
        timeout_at: None,
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(0.0),
        cached_prompt_tokens: 0,
        preempt_immune_until_tokens: 0,
    };
    (a, rx)
}

/// A `PrefillInProgress` with a blocking oneshot sink — the shape the
/// scheduler is holding when a shutdown lands mid-prefill.
pub(super) fn test_prefill(prompt: Vec<u32>) -> (super::types::PrefillInProgress, RespRx) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let p = super::types::PrefillInProgress {
        prompt_tokens: std::sync::Arc::new(prompt),
        session_hash: 0,
        seq: SequenceState::host_only(0),
        chunk_offset: 0,
        max_tokens: 16,
        min_tokens: 0,
        eos_tokens: EOS.to_vec(),
        sink: ResponseSink::Blocking(Some(tx)),
        cancel_flag: None,
        request_start: Instant::now(),
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        repetition_penalty_window: 256,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: DEFAULT_LZ_PENALTY,
        dry_multiplier: 0.0,
        dry_base: 0.0,
        dry_allowed_length: 0,
        dry_sequence_breakers: Vec::new(),
        logit_bias: Vec::new(),
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: None,
        spontaneous_think_budget: 0,
        require_tool_call: false,
        tools_present: false,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_state: None,
        seed: None,
        top_logprobs: None,
        timeout_at: None,
    };
    (p, rx)
}

/// A real `PrefillInProgress` at `chunk_offset == 0`, identified by
/// `session_hash` so an ordering test can name the request it is tracking.
///
/// `eos_tokens` is empty and `temperature` is 0.0 on purpose: that is the
/// `sample_token` greedy fast path (`argmax_on_device`), which a stub `Model`
/// can answer without a device logits buffer. `max_tokens` is 8 (> 1) so
/// promotion pushes onto `active` instead of finishing immediately.
pub(super) fn test_prefill_ident(
    id: u64,
    prompt_len: usize,
) -> (super::types::PrefillInProgress, RespRx) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let p = super::types::PrefillInProgress {
        prompt_tokens: std::sync::Arc::new(vec![1u32; prompt_len]),
        session_hash: id,
        seq: SequenceState::host_only(id as usize),
        chunk_offset: 0,
        max_tokens: 8,
        min_tokens: 0,
        eos_tokens: Vec::new(),
        sink: ResponseSink::Blocking(Some(tx)),
        cancel_flag: None,
        request_start: Instant::now(),
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        repetition_penalty_window: 256,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: DEFAULT_LZ_PENALTY,
        dry_multiplier: 0.0,
        dry_base: 0.0,
        dry_allowed_length: 0,
        dry_sequence_breakers: Vec::new(),
        logit_bias: Vec::new(),
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: None,
        spontaneous_think_budget: 0,
        require_tool_call: false,
        tools_present: false,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_state: None,
        seed: None,
        top_logprobs: None,
        timeout_at: None,
    };
    (p, rx)
}
