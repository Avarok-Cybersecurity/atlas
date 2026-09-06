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
use spark_model::traits::Model;
use spark_model::traits::SequenceState;
use spark_runtime::gpu::DevicePtr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
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

// ── Preemption fixtures ──────────────────────────────────────────────────
// Moved here from `preempt_tests.rs`, which this stack pushed from 481 to 510
// lines against a 500 cap. Copying the stub into a sibling file would have
// satisfied the cap and broken SSOT; this module's own header already declares
// it the home for shared scheduler fixtures, and `shutdown_drain_tests` will
// want the same stub. Byte-exact move.

/// Scripted stub: `decode_batch` fails with the KV-exhausted error for the
/// first `fail_decodes` calls, then succeeds. Records every free/cache/
/// prefill so the tests can assert the preemption side effects.
#[derive(Default)]
pub(super) struct PreemptStubModel {
    pub(super) fail_decodes: AtomicUsize,
    /// When set, `decode_batch` always fails with this message instead.
    pub(super) hard_error: Option<&'static str>,
    /// When set, `compact_sequence` always fails with this message.
    pub(super) fail_compact: Option<&'static str>,
    pub(super) compact_calls: AtomicUsize,
    pub(super) decode_calls: AtomicUsize,
    pub(super) freed_slots: Mutex<Vec<usize>>,
    pub(super) cached_seqs: AtomicUsize,
    pub(super) prefilled: Mutex<Vec<Vec<u32>>>,
    pub(super) vision_pad: Option<u32>,
    pub(super) free_blocks: AtomicUsize,
    pub(super) total_blocks: usize,
    pub(super) reclaimable: AtomicUsize,
}

impl PreemptStubModel {
    pub(super) fn failing(n: usize) -> Self {
        Self {
            fail_decodes: AtomicUsize::new(n),
            ..Default::default()
        }
    }

    /// A stub whose `compact_sequence` always fails — the swap-out tests'
    /// entry point (`swap_out_tests.rs`). Lives here so the stub's fields
    /// stay private to their own module.
    pub(super) fn failing_compact(msg: &'static str) -> Self {
        Self {
            fail_compact: Some(msg),
            ..Default::default()
        }
    }
}

impl Model for PreemptStubModel {
    fn prefill(&self, t: &[u32], s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        self.prefilled.lock().unwrap().push(t.to_vec());
        // Mirror the real contract: prefill populates tokens/seq_len/prompt_len.
        s.tokens.extend_from_slice(t);
        s.seq_len = s.tokens.len();
        s.prompt_len = t.len();
        Ok(DevicePtr::NULL)
    }
    fn decode(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!("unused in preempt tests")
    }
    fn prefill_chunk(
        &self,
        _t: &[u32],
        _s: &mut SequenceState,
        _cs: usize,
        _cl: usize,
        _last: bool,
        _st: u64,
    ) -> Result<DevicePtr> {
        anyhow::bail!("unused in preempt tests")
    }
    fn decode_batch(
        &self,
        _t: &[u32],
        _s: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<DevicePtr> {
        self.decode_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(msg) = self.hard_error {
            anyhow::bail!("{msg}");
        }
        if self
            .fail_decodes
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok()
        {
            anyhow::bail!("KV cache exhausted: no free blocks");
        }
        Ok(DevicePtr::NULL)
    }
    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        anyhow::bail!("unused in preempt tests")
    }
    fn generate_speculative(
        &self,
        _p: &[u32],
        _params: &spark_runtime::sampler::SamplingParams,
        _n: usize,
    ) -> Result<spark_model::engine::GenerateResult> {
        anyhow::bail!("unused in preempt tests")
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        anyhow::bail!("unused in preempt tests")
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        anyhow::bail!("unused in preempt tests")
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        anyhow::bail!("unused in preempt tests")
    }
    fn run_mtp_propose(
        &self,
        _t: u32,
        _p: usize,
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<Option<u32>> {
        anyhow::bail!("unused in preempt tests")
    }
    fn run_mtp_propose_multi(
        &self,
        _t: u32,
        _p: usize,
        _n: usize,
        _s: &mut SequenceState,
        _st: u64,
        _bm: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("unused in preempt tests")
    }
    fn trim_proposer_state(&self, _s: &mut SequenceState, _n: usize, _st: u64) -> Result<()> {
        Ok(())
    }
    fn vocab_size(&self) -> usize {
        0
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState::host_only(0))
    }
    fn copy_logits_to_host(&self, _p: DevicePtr, _d: &mut [u8]) -> Result<()> {
        Ok(())
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn argmax_on_device(&self, _p: DevicePtr, _st: u64) -> Result<u32> {
        anyhow::bail!("unused in preempt tests")
    }
    fn argmax_batch(&self, _p: DevicePtr, _n: usize, _st: u64) -> Result<Vec<u32>> {
        anyhow::bail!("unused in preempt tests")
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn checkpoint_ssm_states(&self, _s: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn rollback_ssm_states(&self, _s: &mut SequenceState, _n: usize) -> Result<()> {
        Ok(())
    }
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!("unused in preempt tests")
    }
    fn cache_sequence(&self, _s: &SequenceState) {
        self.cached_seqs.fetch_add(1, Ordering::SeqCst);
    }
    fn free_sequence(&self, s: &mut SequenceState) -> Result<()> {
        self.freed_slots.lock().unwrap().push(s.slot_idx);
        Ok(())
    }
    fn compact_sequence(&self, _s: &mut SequenceState, _slot: usize) -> Result<()> {
        self.compact_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(msg) = self.fail_compact {
            anyhow::bail!("{msg}");
        }
        Ok(())
    }
    /// The trait default bails ("swap not supported"); the swap-out tests
    /// need the SUCCESS path to reach `spill_out_sequence`, so record a
    /// byte and return Ok. Preempt tests all pass `spill: None` and never
    /// reach this.
    fn save_sequence_state(
        &self,
        _seq: &SequenceState,
        writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        writer.write_all(&[0u8])?;
        Ok(())
    }
    fn detach_slot_for_reuse(&self, _s: &mut SequenceState) {}
    fn save_hidden_for_mtp(&self, _i: usize, _st: u64) -> Result<()> {
        Ok(())
    }
    fn tokens_contain_vision_pad(&self, tokens: &[u32]) -> bool {
        self.vision_pad
            .map(|pad| tokens.contains(&pad))
            .unwrap_or(false)
    }
    fn num_free_blocks(&self) -> usize {
        self.free_blocks.load(Ordering::SeqCst)
    }
    fn num_total_blocks(&self) -> usize {
        self.total_blocks
    }
    fn reclaim_prefix_blocks(&self, num_blocks: usize) -> usize {
        let take = num_blocks.min(self.reclaimable.load(Ordering::SeqCst));
        self.reclaimable.fetch_sub(take, Ordering::SeqCst);
        self.free_blocks.fetch_add(take, Ordering::SeqCst);
        take
    }
}

/// An unfinished decode-active sequence at `slot` with `n_out` generated
/// tokens and a known prompt in `seq.tokens`.
pub(super) fn active_seq(slot: usize, n_out: usize) -> (ActiveSeq, super::test_support::RespRx) {
    let out: Vec<u32> = (100..100 + n_out as u32).collect();
    let (mut a, rx) = test_seq(out, 50, None, 4 + n_out);
    a.finished = false;
    a.seq.slot_idx = slot;
    // prompt [1,2,3,4] + all PROCESSED outputs (everything but last_token).
    a.seq.tokens = vec![1, 2, 3, 4];
    let n = a.output_tokens.len();
    a.seq
        .tokens
        .extend_from_slice(&a.output_tokens[..n.saturating_sub(1)]);
    (a, rx)
}

pub(super) fn streaming_seq(
    slot: usize,
    n_out: usize,
) -> (
    ActiveSeq,
    tokio::sync::mpsc::Receiver<crate::api::StreamEvent>,
) {
    let (a, _rx) = active_seq(slot, n_out);
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    let mut a = a;
    a.sink = ResponseSink::Streaming(tx);
    (a, rx)
}

// ── decode_batch_with_preemption ─────────────────────────────────────────
