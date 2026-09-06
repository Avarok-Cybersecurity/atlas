// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for decode-time KV preemption with resume (`preempt.rs`).
//!
//! The bug class under test is client-facing: the pre-resume code called
//! `send_error` on the victim, emitting a mid-stream SSE error frame and
//! ending the stream with no finish chunk — an HTTP-200 "success" with
//! silently truncated content. These tests drive the REAL retry loop
//! (`decode_batch_with_preemption`) against a scripted `Model` stub and
//! assert on the transport seam the API layer consumes (the per-request
//! `StreamEvent` channel): a preempted victim's channel must stay OPEN and
//! EMPTY — no `StreamEvent::Error`, no premature `Done`.

use super::preempt::{
    PREEMPT_IMMUNITY_TOKENS, choose_decode_victim, decode_batch_with_preemption, preempt_requeue,
    resume_preempted_seq, resume_preempted_seqs,
};
use super::test_support::test_seq;
use super::types::{ActiveSeq, ResponseSink};
use anyhow::Result;
use spark_model::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Scripted stub: `decode_batch` fails with the KV-exhausted error for the
/// first `fail_decodes` calls, then succeeds. Records every free/cache/
/// prefill so the tests can assert the preemption side effects.
#[derive(Default)]
pub(super) struct PreemptStubModel {
    fail_decodes: AtomicUsize,
    /// When set, `decode_batch` always fails with this message instead.
    hard_error: Option<&'static str>,
    /// When set, `compact_sequence` always fails with this message.
    fail_compact: Option<&'static str>,
    pub(super) compact_calls: AtomicUsize,
    decode_calls: AtomicUsize,
    pub(super) freed_slots: Mutex<Vec<usize>>,
    cached_seqs: AtomicUsize,
    prefilled: Mutex<Vec<Vec<u32>>>,
    vision_pad: Option<u32>,
    free_blocks: AtomicUsize,
    total_blocks: usize,
    reclaimable: AtomicUsize,
}

impl PreemptStubModel {
    fn failing(n: usize) -> Self {
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

#[test]
fn kv_exhaustion_requeues_least_progress_victim_and_sends_nothing() {
    let model = PreemptStubModel::failing(1);
    let (a0, _rx0) = active_seq(0, 5);
    let (victim, mut victim_rx) = streaming_seq(1, 2); // least progress
    let (a2, _rx2) = active_seq(2, 9);
    let mut active = vec![a0, victim, a2];
    let mut swapped = Vec::new();
    let mut preempted = Vec::new();

    let logits =
        decode_batch_with_preemption(&model, &mut active, None, &mut swapped, &mut preempted);

    // The batch survived: one retry after one preemption.
    assert!(logits.is_some());
    assert_eq!(model.decode_calls.load(Ordering::SeqCst), 2);
    assert_eq!(active.len(), 2);
    // Least-progress victim (slot 1, 2 tokens) was requeued, not killed.
    assert_eq!(preempted.len(), 1);
    assert!(swapped.is_empty());
    assert_eq!(preempted[0].a.output_tokens.len(), 2);
    // Its GPU state was freed, its KV offered to the prefix cache first.
    assert_eq!(*model.freed_slots.lock().unwrap(), vec![1]);
    assert_eq!(model.cached_seqs.load(Ordering::SeqCst), 1);
    // STREAM CONTRACT: the victim's channel is OPEN and EMPTY — no
    // mid-stream Error frame, no Done. (The old code sent
    // StreamEvent::Error here: an HTTP-200 with silent truncation.)
    assert!(matches!(
        victim_rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
    // History retained for the re-prefill: prompt + processed outputs.
    assert_eq!(preempted[0].tokens, vec![1, 2, 3, 4, 100]);
    assert_eq!(preempted[0].a.last_token, 101);
}

#[test]
fn non_kv_error_still_fails_the_whole_batch() {
    let model = PreemptStubModel {
        hard_error: Some("CUDA error 700: illegal memory access"),
        ..Default::default()
    };
    let (a0, mut rx0) = active_seq(0, 3);
    let (a1, mut rx1) = active_seq(1, 4);
    let mut active = vec![a0, a1];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    let logits =
        decode_batch_with_preemption(&model, &mut active, None, &mut swapped, &mut preempted);
    assert!(logits.is_none());
    assert!(active.is_empty() && preempted.is_empty() && swapped.is_empty());
    // Non-recoverable errors still reach the clients.
    assert!(rx0.try_recv().expect("response sent").is_err());
    assert!(rx1.try_recv().expect("response sent").is_err());
}

#[test]
fn single_sequence_exhaustion_is_not_preemptible() {
    // With one sequence, preemption cannot free anything the survivor
    // needs — the existing terminal path is kept.
    let model = PreemptStubModel {
        hard_error: Some("KV cache exhausted: no free blocks"),
        ..Default::default()
    };
    let (a0, mut rx0) = active_seq(0, 3);
    let mut active = vec![a0];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    let logits =
        decode_batch_with_preemption(&model, &mut active, None, &mut swapped, &mut preempted);
    assert!(logits.is_none());
    assert!(preempted.is_empty());
    assert!(rx0.try_recv().expect("response sent").is_err());
}

// ── choose_decode_victim ─────────────────────────────────────────────────

#[test]
fn victim_policy_least_progress_wins() {
    let model = PreemptStubModel::default();
    let (a0, _r0) = active_seq(0, 7);
    let (a1, _r1) = active_seq(1, 3);
    let (a2, _r2) = active_seq(2, 12);
    let active = vec![a0, a1, a2];
    assert_eq!(choose_decode_victim(&model, &active, false), Some(1));
}

#[test]
fn victim_policy_starvation_guard_skips_resumed_until_progress() {
    let model = PreemptStubModel::default();
    let (mut a0, _r0) = active_seq(0, 3);
    // Just resumed: 3 generated, immune until 3 + PREEMPT_IMMUNITY_TOKENS.
    a0.preempt_immune_until_tokens = a0.output_tokens.len() + PREEMPT_IMMUNITY_TOKENS;
    let (a1, _r1) = active_seq(1, 9);
    let active = vec![a0, a1];
    // The immune least-progress seq is skipped; the other is chosen.
    assert_eq!(choose_decode_victim(&model, &active, false), Some(1));

    // Once it has generated PREEMPT_IMMUNITY_TOKENS more, immunity lapses.
    let (mut a0, _r0) = active_seq(0, 3 + PREEMPT_IMMUNITY_TOKENS);
    a0.preempt_immune_until_tokens = 3 + PREEMPT_IMMUNITY_TOKENS;
    let (a1, _r1) = active_seq(1, 200);
    let active = vec![a0, a1];
    assert_eq!(choose_decode_victim(&model, &active, false), Some(0));
}

#[test]
fn victim_policy_all_immune_still_yields_a_victim() {
    // Immunity must never convert a recoverable exhaustion into a
    // batch-wide error: with every candidate immune, one is chosen anyway.
    let model = PreemptStubModel::default();
    let (mut a0, _r0) = active_seq(0, 4);
    a0.preempt_immune_until_tokens = usize::MAX;
    let (mut a1, _r1) = active_seq(1, 2);
    a1.preempt_immune_until_tokens = usize::MAX;
    let active = vec![a0, a1];
    assert_eq!(choose_decode_victim(&model, &active, false), Some(1));
}

#[test]
fn victim_policy_vision_requeue_excluded_but_spill_allowed() {
    const PAD: u32 = 999;
    let model = PreemptStubModel {
        vision_pad: Some(PAD),
        ..Default::default()
    };
    let (mut a0, _r0) = active_seq(0, 2);
    a0.seq.tokens.insert(2, PAD); // image KV: not re-prefillable from tokens
    let (a1, _r1) = active_seq(1, 8);
    let active = vec![a0, a1];
    // Requeue (no spill): the vision seq is ineligible despite least progress.
    assert_eq!(choose_decode_victim(&model, &active, false), Some(1));
    // Spill saves KV verbatim: the vision seq is eligible again.
    assert_eq!(choose_decode_victim(&model, &active, true), Some(0));
}

// ── requeue → resume round trip ──────────────────────────────────────────

#[test]
fn resume_reprefills_exact_history_and_preserves_stream_state() {
    let model = PreemptStubModel::default();
    let (a, _rx) = active_seq(3, 6);
    let last_token = a.last_token;
    let out_before = a.output_tokens.clone();
    let remaining_before = a.remaining;
    let history = a.seq.tokens.clone();

    let p = preempt_requeue(&model, a);
    assert_eq!(p.tokens, history);
    assert_eq!(*model.freed_slots.lock().unwrap(), vec![3]);

    let resumed = resume_preempted_seq(&model, p).expect("resume succeeds");
    // The re-prefill processed EXACTLY the retained history — the pending
    // last_token is decoded next, not re-prefilled and never re-emitted.
    assert_eq!(*model.prefilled.lock().unwrap(), vec![history.clone()]);
    assert_eq!(resumed.seq.tokens, history);
    assert_eq!(resumed.seq.seq_len, history.len());
    assert_eq!(resumed.last_token, last_token);
    assert_eq!(resumed.output_tokens, out_before);
    assert_eq!(resumed.remaining, remaining_before);
    assert!(!resumed.finished);
    // Starvation guard armed.
    assert_eq!(
        resumed.preempt_immune_until_tokens,
        out_before.len() + PREEMPT_IMMUNITY_TOKENS
    );
}

#[test]
fn resume_loop_gates_on_blocks_and_reclaims_from_prefix_cache() {
    let model = PreemptStubModel {
        total_blocks: 100,
        free_blocks: AtomicUsize::new(0),
        reclaimable: AtomicUsize::new(50),
        ..Default::default()
    };
    let (a, _rx) = active_seq(0, 4);
    let p = {
        let mut history_seq = a;
        history_seq.seq.tokens = (0..32).collect(); // 32-token history
        preempt_requeue(&model, history_seq)
    };
    let mut preempted = vec![p];
    let mut active = Vec::new();
    // block_size 16 → needs 32/16+1 = 3 blocks (+1 headroom = 4): free 0,
    // but 50 reclaimable → the loop must ASK the prefix cache and resume.
    resume_preempted_seqs(&model, &mut active, &mut preempted, 8, 16);
    assert_eq!(active.len(), 1);
    assert!(preempted.is_empty());

    // With nothing free AND nothing reclaimable, it stays parked (no error).
    let model2 = PreemptStubModel {
        total_blocks: 100,
        ..Default::default()
    };
    let (a2, mut rx2) = active_seq(0, 4);
    let p2 = preempt_requeue(&model2, a2);
    let mut preempted2 = vec![p2];
    let mut active2 = Vec::new();
    resume_preempted_seqs(&model2, &mut active2, &mut preempted2, 8, 16);
    assert!(active2.is_empty());
    assert_eq!(preempted2.len(), 1);
    assert!(matches!(
        rx2.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    ));
}

#[test]
fn resume_loop_errors_out_a_sequence_that_can_never_fit() {
    let model = PreemptStubModel {
        total_blocks: 2, // pool smaller than the history
        ..Default::default()
    };
    let (a, mut rx) = active_seq(0, 4);
    let p = {
        let mut s = a;
        s.seq.tokens = (0..64).collect();
        preempt_requeue(&model, s)
    };
    let mut preempted = vec![p];
    let mut active = Vec::new();
    resume_preempted_seqs(&model, &mut active, &mut preempted, 8, 16);
    assert!(preempted.is_empty() && active.is_empty());
    // The client is told, not left hanging forever.
    assert!(rx.try_recv().expect("error delivered").is_err());
}
