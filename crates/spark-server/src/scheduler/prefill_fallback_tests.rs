// SPDX-License-Identifier: AGPL-3.0-only

//! A declined batched-prefill wave must not cost anyone their response.
//!
//! THE BUG (#927, H100 round 11 cell E). With `ATLAS_PREFILL_VARLEN=1` at
//! C=16 the server logged
//!
//! ```text
//! ERROR spark::scheduler::phase_continue_prefills::run_batched_prefill:
//!   Batched prefill error (wave of 6 streams, 6 prefilling): prefill_inner:
//!   batched mode requires seq_len_start > 0 (paged path); got seq_len_start=0.
//!   Caller must fall back to per-stream for this chunk.
//! ```
//!
//! and returned **sixteen HTTP 200 responses with zero tokens, no
//! `finish_reason` and no usage frame**, on 3/3 ladder reps (`err=0` at the
//! client). A recheck on the same process gave one clean rep and one that
//! dropped 3 of 16 — intermittent silent truncation, the failure shape that is
//! indistinguishable from a correct empty answer.
//!
//! Two independent defects produced it, and both are covered here:
//!   1. `run_batched_prefill_step` failed every stream in the wave instead of
//!      running the per-stream path the error message itself asked for, and
//!   2. `promote_completed_prefills` freed a failed prefill's sequence while
//!      DROPPING its `ResponseSink`, so the client's channel simply closed.
//!
//! These tests drive the real `continue_in_progress_prefills` → promote loop
//! against a stub whose batched answer is scripted, and assert on the sinks.

use super::phase_continue_prefills::continue_in_progress_prefills;
use super::sched_ctx::SchedCtx;
use super::test_support::{RespRx, test_prefill_ident};
use super::test_support_prefill::{BatchedBehaviour, FIRST, PrefillStubModel};
use super::types::{ActiveSeq, PrefillInProgress};
use crate::scheduling_policy::FifoPolicy;
use std::sync::atomic::Ordering;

/// One prompt = one chunk = one tick of prefill work.
const CHUNK: usize = 4;
/// The receipt's burst width.
const N: usize = 16;

struct Harness {
    active: Vec<ActiveSeq>,
    prefilling: Vec<PrefillInProgress>,
    rx: Vec<RespRx>,
}

fn harness(n: usize) -> Harness {
    let mut prefilling = Vec::new();
    let mut rx = Vec::new();
    for id in 1..=n as u64 {
        let (p, r) = test_prefill_ident(id, CHUNK);
        prefilling.push(p);
        rx.push(r);
    }
    Harness {
        active: Vec::new(),
        prefilling,
        rx,
    }
}

/// Run one scheduler tick. `active` starts empty, so the batched-prefill-only
/// branch (`can_batch_prefill_only`) is the one under test.
fn tick(model: &PrefillStubModel, h: &mut Harness) {
    let policy = FifoPolicy;
    let sched = SchedCtx::for_test();
    continue_in_progress_prefills(
        model,
        &policy,
        &mut h.active,
        &mut h.prefilling,
        CHUNK, // max_prefill_tokens
        CHUNK, // max_batch_tokens
        false, // always_mixed
        0,     // prefill_stream
        0,     // prefill_event
        false, // use_mtp
        false, // use_self_speculative
        false, // use_ngram_speculative
        None,
        None,
        None,
        None,
        None,
        false, // adaptive_sampling
        &sched,
    );
}

#[test]
fn a_declined_wave_is_re_run_per_stream_and_every_request_completes() {
    let model = PrefillStubModel::with_batched(BatchedBehaviour::Decline);
    let mut h = harness(N);

    tick(&model, &mut h);

    assert_eq!(
        model.batched_calls.load(Ordering::Relaxed),
        1,
        "the wave must be attempted batched exactly once",
    );
    assert_eq!(
        model.single_calls.load(Ordering::Relaxed),
        N,
        "a decline must re-run EVERY member of the wave on the single-stream \
         path — this is the count that was 0 when the bug shipped",
    );
    assert!(
        h.prefilling.is_empty(),
        "every stream finished its only chunk",
    );
    assert_eq!(
        h.active.len(),
        N,
        "all {N} requests must promote into decode; the receipt's failure was \
         {N} requests promoted into nothing",
    );
    for a in &h.active {
        assert_eq!(
            a.last_token, FIRST,
            "each promoted request has its first token"
        );
    }
    // Nothing was completed, so nothing has answered its sink yet — the
    // requests are alive, which is the point.
    for r in &mut h.rx {
        assert!(
            matches!(
                r.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            ),
            "a promoted request's sink must still be open",
        );
    }
}

#[test]
fn a_hard_batched_failure_still_answers_every_request() {
    // An ADMITTED batch that fails mid-forward owns KV state, so re-running it
    // per-stream would double-allocate. The requests must still be told.
    let model = PrefillStubModel::with_batched(BatchedBehaviour::HardError);
    let mut h = harness(N);

    tick(&model, &mut h);

    assert_eq!(
        model.single_calls.load(Ordering::Relaxed),
        0,
        "a hard failure must NOT be retried per-stream",
    );
    assert!(h.prefilling.is_empty(), "the failed streams are retired");
    assert!(h.active.is_empty(), "none of them promote");
    for (i, r) in h.rx.iter_mut().enumerate() {
        match r.try_recv() {
            Ok(Err(_)) => {}
            Ok(Ok(_)) => panic!("request {i} reported success after a failed prefill"),
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => panic!(
                "request {i}'s sink was DROPPED — that is the silent HTTP-200 \
                 with no body and no finish_reason (#927 cell E)",
            ),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                panic!("request {i} was never answered")
            }
        }
    }
}

#[test]
fn the_control_the_batched_path_completes_everyone_when_it_works() {
    // Discriminator: the two tests above must fail for the reason they name,
    // not because the harness never gets off the ground.
    let model = PrefillStubModel::with_batched(BatchedBehaviour::PerStream);
    let mut h = harness(N);

    tick(&model, &mut h);

    assert_eq!(model.batched_calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        model.single_calls.load(Ordering::Relaxed),
        0,
        "no fallback when the batch succeeds",
    );
    assert_eq!(h.active.len(), N);
}
