// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill-queue ORDERING tests for the promote/continue pair.
//!
//! `prefilling` is a FIFO: `phase_start_prefills` appends arrivals with
//! `push`, and `continue_in_progress_prefills` advances `prefilling[0]` —
//! the queue HEAD — one chunk per tick. The only other mutation of that
//! vector is the removal in `phase_promote_prefills`, so that removal is
//! what has to keep the FIFO a FIFO.
//!
//! It did not. `swap_remove(0)` backfills the head with the vector's LAST
//! element, i.e. the NEWEST arrival, so `[A,B,C]` becomes `[C,B]` when A
//! completes: the queue is served LIFO at the head and `B` only ever gets
//! a chunk if the queue happens to drain to length 2 at the instant a
//! prefill completes. Under sustained arrivals that never happens and `B`
//! waits forever — nothing downstream re-sorts `prefilling`, and the
//! request-deadline sweep in `mod_helpers` only walks `active`, so a
//! starved prefill is not even timed out.
//!
//! Reachability: the single-stream path runs whenever both Q12 batched
//! branches are ineligible. The case exercised here is the common one —
//! a `--speculative` serve with one decode active (`single_active_with_spec`),
//! which disqualifies `can_batch_mixed`, while a non-empty `active`
//! disqualifies `can_batch_prefill_only`.
//!
//! These tests drive the REAL `continue_in_progress_prefills` (admission →
//! chunk → promote) over many ticks against a scripted `Model` stub; the
//! ordering assertions are on the production queue itself, not on a model
//! of it.

use anyhow::Result;
use spark_model::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;

use super::phase_continue_prefills::continue_in_progress_prefills;
use super::phase_promote_prefills::promote_completed_prefills;
use super::sched_ctx::SchedCtx;
use super::test_support::{test_prefill_ident, test_seq};
use super::types::{ActiveSeq, PrefillInProgress};
use crate::scheduling_policy::FifoPolicy;

/// The sampled first token. Not an EOS (the fixture's `eos_tokens` is
/// empty), so promotion pushes onto `active` rather than finishing.
const FIRST: u32 = 7;

/// One prompt = one chunk = one tick of prefill work.
const CHUNK: usize = 4;

/// Minimal `Model`: every prefill chunk succeeds and the greedy sampler
/// (temperature 0.0, no suppressed ids) answers from `argmax_on_device`.
/// Everything else is unreachable on the single-stream prefill path.
#[derive(Default)]
struct PrefillStubModel;

impl Model for PrefillStubModel {
    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        _is_last: bool,
        _stream: u64,
    ) -> Result<DevicePtr> {
        // Mirror the real contract: the chunk's tokens land in the sequence.
        seq.tokens
            .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
        seq.seq_len = seq.tokens.len();
        Ok(DevicePtr::NULL)
    }
    fn argmax_on_device(&self, _logits_ptr: DevicePtr, _stream: u64) -> Result<u32> {
        Ok(FIRST)
    }
    fn vocab_size(&self) -> usize {
        32
    }
    fn free_sequence(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn cache_sequence(&self, _seq: &SequenceState) {}
    fn detach_slot_for_reuse(&self, _seq: &mut SequenceState) {}
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState::host_only(0))
    }
    fn copy_logits_to_host(&self, _l: DevicePtr, _dst: &mut [u8]) -> Result<()> {
        unreachable!("greedy fast path never reads logits back")
    }
    fn prefill(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("chunked prefill only")
    }
    fn decode(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("decode is driven by mod.rs, not by this harness")
    }
    fn decode_batch(
        &self,
        _t: &[u32],
        _s: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<DevicePtr> {
        unreachable!("decode is driven by mod.rs, not by this harness")
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        unreachable!("no speculation in this harness")
    }
    fn argmax_batch(&self, _l: DevicePtr, _n: usize, _st: u64) -> Result<Vec<u32>> {
        unreachable!("batched decode is not driven here")
    }
    fn checkpoint_ssm_states(&self, _s: &mut SequenceState) -> Result<()> {
        unreachable!("no speculation in this harness")
    }
    fn rollback_ssm_states(&self, _s: &mut SequenceState, _n: usize) -> Result<()> {
        unreachable!("no speculation in this harness")
    }
    fn compact_sequence(&self, _s: &mut SequenceState, _new_slot: usize) -> Result<()> {
        unreachable!("no compaction in this harness")
    }
    fn save_hidden_for_mtp(&self, _token_idx: usize, _st: u64) -> Result<()> {
        unreachable!("no speculation in this harness")
    }
    fn run_mtp_propose(
        &self,
        _t: u32,
        _p: usize,
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<Option<u32>> {
        unreachable!("no speculation in this harness")
    }
    fn run_mtp_propose_multi(
        &self,
        _t: u32,
        _p: usize,
        _n: usize,
        _s: &mut SequenceState,
        _st: u64,
        _mask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        unreachable!("no speculation in this harness")
    }
    fn trim_proposer_state(&self, _s: &mut SequenceState, _n: usize, _st: u64) -> Result<()> {
        unreachable!("no speculation in this harness")
    }
    fn generate_speculative(
        &self,
        _p: &[u32],
        _params: &spark_runtime::sampler::SamplingParams,
        _n: usize,
    ) -> Result<spark_model::engine::GenerateResult> {
        unreachable!("no speculation in this harness")
    }
}

/// What one run of the tick loop observed.
struct Run {
    /// `(session_hash, tick)` for every prefill promoted into `active`.
    promoted: Vec<(u64, usize)>,
    /// Initially-queued ids that never got a single prefill chunk.
    stuck: Vec<u64>,
}

/// Drive the real admit → continue → promote loop for `ticks` ticks.
///
/// `seed_depth` requests (ids `1..=seed_depth`) are already queued when the
/// loop starts, and one NEW request arrives every tick — the sustained-arrival
/// condition under which head reordering becomes unbounded starvation. Arrival
/// is applied before the continue phase, the order `mod.rs` uses
/// (`start_new_requests` then `continue_in_progress_prefills`).
fn drive(seed_depth: usize, ticks: usize) -> Run {
    let model = PrefillStubModel;
    let policy = FifoPolicy;
    let sched = SchedCtx::for_test();

    // Response receivers, kept alive so no sink observes a closed channel.
    let mut keep = Vec::new();
    let mut active: Vec<ActiveSeq> = Vec::new();
    let mut prefilling: Vec<PrefillInProgress> = Vec::new();

    // One long-lived decode occupant — a `--speculative` serve at C=1. That
    // is `single_active_with_spec`, which disqualifies the Q12 mixed batch,
    // and a non-empty `active` disqualifies the Q12 prefill batch, so every
    // tick takes the single-stream path under test. `session_hash` 0 is
    // reserved for it; prefill ids start at 1.
    let (occupant, rx) = test_seq(vec![1], usize::MAX, None, 8);
    keep.push(rx);
    active.push(occupant);

    let seeded: Vec<u64> = (1..=seed_depth as u64).collect();
    for &id in &seeded {
        let (p, rx) = test_prefill_ident(id, CHUNK);
        keep.push(rx);
        prefilling.push(p);
    }

    let mut next_id = seed_depth as u64 + 1;
    let mut promoted: Vec<(u64, usize)> = Vec::new();

    for tick in 0..ticks {
        let (p, rx) = test_prefill_ident(next_id, CHUNK);
        keep.push(rx);
        prefilling.push(p);
        next_id += 1;

        continue_in_progress_prefills(
            &model,
            &policy,
            &mut active,
            &mut prefilling,
            CHUNK, // max_prefill_tokens
            CHUNK, // max_batch_tokens
            false, // always_mixed
            0,     // prefill_stream
            0,     // prefill_event
            true,  // use_mtp
            false, // use_self_speculative
            false, // use_ngram_speculative
            None,
            None,
            None,
            None,
            None,
            false,
            &sched,
        );

        // The rest of `mod.rs` decodes and retires. Here every promoted
        // prefill retires at once, leaving the single decode occupant at
        // index 0 (promotion appends).
        for a in active.drain(1..) {
            promoted.push((a.session_hash, tick));
        }
    }

    let stuck = seeded
        .into_iter()
        .filter(|id| !promoted.iter().any(|(p, _)| p == id))
        .collect();
    Run { promoted, stuck }
}

/// The starvation test. Every request already in the queue must get its
/// prefill even though a newer one arrives on every single tick.
///
/// Pre-fix (`swap_remove`) this is red from depth 2 upward: the head is
/// backfilled with the newest arrival, so ids 2..=depth never advance in
/// 500 ticks. Depth 1 passes either way — the harness discriminates on the
/// ordering, it does not fail everything.
#[test]
fn every_queued_prefill_advances_under_sustained_arrivals() {
    for depth in 1..=8 {
        let run = drive(depth, 500);
        assert!(
            run.stuck.is_empty(),
            "queue depth {depth}: request(s) {:?} never got a prefill chunk in 500 ticks",
            run.stuck,
        );
    }
}

/// The wait is BOUNDED by the queue depth, not merely finite: with one
/// chunk of work per request the request that is `k`-th in line is served
/// on tick `k`.
#[test]
fn queued_prefills_are_served_in_arrival_order() {
    for depth in 1..=8 {
        let run = drive(depth, 500);
        for (rank, id) in (1..=depth as u64).enumerate() {
            let tick = run
                .promoted
                .iter()
                .find(|(p, _)| *p == id)
                .map(|(_, t)| *t)
                .unwrap_or_else(|| panic!("depth {depth}: request {id} never promoted"));
            assert_eq!(
                tick, rank,
                "depth {depth}: request {id} was {rank} places from the head but ran on tick {tick}",
            );
        }
    }
}

/// Control: the loop really runs. Every tick admits one request and
/// completes one prefill, so a 500-tick run promotes 500 of them. This
/// stays GREEN before the fix — the pre-fix failure above is about WHICH
/// requests ran, not about a harness that never got off the ground.
#[test]
fn the_tick_loop_makes_progress_every_tick() {
    let run = drive(8, 500);
    assert_eq!(
        run.promoted.len(),
        500,
        "expected one promotion per tick, got {}",
        run.promoted.len(),
    );
}

/// Unit control on the removal itself: promoting the head must leave the
/// rest of the queue in arrival order.
#[test]
fn promoting_the_head_preserves_the_order_of_the_remainder() {
    let model = PrefillStubModel;
    let mut keep = Vec::new();
    let mut prefilling: Vec<PrefillInProgress> = (1..=4)
        .map(|id| {
            let (p, rx) = test_prefill_ident(id, CHUNK);
            keep.push(rx);
            p
        })
        .collect();
    let mut active: Vec<ActiveSeq> = Vec::new();

    promote_completed_prefills(
        &model,
        &mut prefilling,
        vec![(0, Some(FIRST))],
        &mut active,
        None,
        None,
        None,
        None,
        4096,
    );

    let order: Vec<u64> = prefilling.iter().map(|p| p.session_hash).collect();
    assert_eq!(
        order,
        vec![2, 3, 4],
        "head removal must not reorder the queue"
    );
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].session_hash, 1);
}

/// The batched paths hand `promote_completed_prefills` SEVERAL completed
/// indices at once. Removing them must still leave the survivors in
/// arrival order — the reverse-index walk is an implementation detail of
/// keeping the indices valid, not licence to reorder.
#[test]
fn promoting_several_at_once_preserves_the_order_of_the_remainder() {
    let model = PrefillStubModel;
    let mut keep = Vec::new();
    let mut prefilling: Vec<PrefillInProgress> = (1..=5)
        .map(|id| {
            let (p, rx) = test_prefill_ident(id, CHUNK);
            keep.push(rx);
            p
        })
        .collect();
    let mut active: Vec<ActiveSeq> = Vec::new();

    // Indices 0 and 2 complete (ids 1 and 3), in the caller's arrival order.
    promote_completed_prefills(
        &model,
        &mut prefilling,
        vec![(0, Some(FIRST)), (2, Some(FIRST))],
        &mut active,
        None,
        None,
        None,
        None,
        4096,
    );

    let order: Vec<u64> = prefilling.iter().map(|p| p.session_hash).collect();
    assert_eq!(
        order,
        vec![2, 4, 5],
        "multi-removal must not reorder the queue"
    );
    let promoted: Vec<u64> = active.iter().map(|a| a.session_hash).collect();
    assert_eq!(
        promoted,
        vec![3, 1],
        "promotion still walks the indices in reverse"
    );
}
