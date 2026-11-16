// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: tests of the asynchronous KV restore queue (lane L27). The queue core runs over
//! fake items and scripted ticket states (no GPU, no tier): a parked request never blocks the
//! tick, is released exactly once, in park order, and is cancelled cleanly; every ticket is
//! settled exactly once; the exits, refusals and the no-second-park rule. Then the real
//! `InferenceRequest` adapter, the pending-queue release, the row-off fast path and `/metrics`.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use super::park::{Exit, Report};
use super::*;
use crate::api::{InferenceRequest, InferenceResponse};
use crate::scheduler::types::PendingQueue;

#[derive(Clone, Debug)]
struct Fake {
    id: usize,
    gone: Rc<Cell<bool>>,
    deadline: Option<Instant>,
}

impl Item for Fake {
    fn identity(&self) -> usize {
        self.id
    }
    fn client_gone(&self) -> bool {
        self.gone.get()
    }
    fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
}

fn fake(id: usize) -> Fake {
    Fake {
        id,
        gone: Rc::new(Cell::new(false)),
        deadline: None,
    }
}

/// Scripted ticket states (unknown tickets are pending), with a log of polls and settles.
#[derive(Default)]
struct Script {
    states: RefCell<HashMap<u32, Poll>>,
    polls: Cell<usize>,
    settled: RefCell<Vec<u32>>,
}

impl Script {
    fn set(&self, t: u32, p: Poll) {
        self.states.borrow_mut().insert(t, p);
    }
    fn state(&self, t: &u32) -> Poll {
        self.polls.set(self.polls.get() + 1);
        *self.states.borrow().get(t).unwrap_or(&Poll::Pending)
    }
    fn settled(&self) -> Vec<u32> {
        let mut v = self.settled.borrow().clone();
        v.sort_unstable();
        v
    }
}

/// One poll of `q` at `now` against the script.
fn tick(q: &mut Core<Fake, u32>, s: &Script, now: Instant, flush: bool) -> Report<Fake> {
    q.poll(
        now,
        flush,
        |t| s.state(t),
        |t| s.settled.borrow_mut().push(t),
    )
}

fn ids(r: &Report<Fake>) -> Vec<usize> {
    r.readmit.iter().map(|f| f.id).collect()
}

fn exits(r: &Report<Fake>) -> Vec<Exit> {
    r.exits.iter().map(|(e, _)| *e).collect()
}

#[test]
fn a_parked_request_never_blocks_the_tick_and_is_released_exactly_once() {
    let s = Script::default();
    let mut q: Core<Fake, u32> = Core::new();
    let t0 = Instant::now();
    q.park(fake(1), 1, true, t0).unwrap();
    q.park(fake(2), 2, true, t0).unwrap();
    // Both copies in flight: the poll returns at once with nothing released (decode goes on).
    let r = tick(&mut q, &s, t0, false);
    assert!(r.readmit.is_empty() && r.exits.is_empty());
    assert_eq!((q.parked(), s.polls.get()), (2, 2));
    s.set(1, Poll::Ready);
    let r = tick(&mut q, &s, t0 + Duration::from_millis(5), false);
    assert_eq!((ids(&r), exits(&r)), (vec![1], vec![Exit::Ready]));
    assert_eq!(r.exits[0].1, Duration::from_millis(5));
    assert_eq!(s.settled(), vec![1], "a finished ticket is settled at once");
    for _ in 0..3 {
        assert!(
            tick(&mut q, &s, t0, false).readmit.is_empty(),
            "released once, never again"
        );
    }
    s.set(2, Poll::Ready);
    assert_eq!(ids(&tick(&mut q, &s, t0, false)), vec![2]);
    assert!(q.is_idle());
    assert_eq!(s.settled(), vec![1, 2], "every ticket settled exactly once");
}

#[test]
fn release_follows_park_order_and_a_slow_copy_does_not_hold_back_later_ones() {
    let s = Script::default();
    let mut q: Core<Fake, u32> = Core::new();
    let t0 = Instant::now();
    for i in 1..=4 {
        q.park(fake(i), i as u32, true, t0).unwrap();
    }
    s.set(4, Poll::Ready);
    s.set(2, Poll::Ready);
    let r = tick(&mut q, &s, t0, false);
    assert_eq!(ids(&r), vec![2, 4]);
    assert_eq!(q.parked(), 2, "1 and 3 still copying");
}

#[test]
fn a_disconnected_client_is_cancelled_and_its_ticket_settled_when_the_copy_ends() {
    let s = Script::default();
    let mut q: Core<Fake, u32> = Core::new();
    let t0 = Instant::now();
    let a = fake(1);
    q.park(a.clone(), 1, true, t0).unwrap();
    a.gone.set(true);
    let r = tick(&mut q, &s, t0, false);
    assert!(
        r.readmit.is_empty(),
        "a cancelled request is dropped, not admitted"
    );
    assert_eq!(exits(&r), vec![Exit::Cancelled]);
    assert_eq!((q.parked(), q.draining()), (0, 1));
    assert!(
        s.settled().is_empty(),
        "never settled while its copy runs (that would block)"
    );
    tick(&mut q, &s, t0, false);
    assert_eq!(q.draining(), 1);
    s.set(1, Poll::Ready);
    tick(&mut q, &s, t0, false);
    assert!(q.is_idle());
    assert_eq!(s.settled(), vec![1]);
    // Cancelled after the copy completed: settled in the same poll.
    let b = fake(2);
    q.park(b.clone(), 2, true, t0).unwrap();
    s.set(2, Poll::Ready);
    b.gone.set(true);
    let r = tick(&mut q, &s, t0, false);
    assert_eq!(exits(&r), vec![Exit::Cancelled]);
    assert!(q.is_idle());
    assert_eq!(s.settled(), vec![1, 2]);
}

#[test]
fn failed_restores_timeouts_and_deadlines_are_admitted_for_recompute() {
    let s = Script::default();
    let mut q: Core<Fake, u32> = Core::new();
    let t0 = Instant::now();
    q.park(fake(1), 1, true, t0).unwrap();
    let mut late = fake(2);
    late.deadline = Some(t0 + Duration::from_secs(1));
    q.park(late, 2, true, t0).unwrap();
    q.park(fake(3), 3, true, t0).unwrap();
    s.set(1, Poll::Failed);
    let r = tick(&mut q, &s, t0 + Duration::from_secs(2), false);
    assert_eq!(ids(&r), vec![1, 2]);
    assert_eq!(exits(&r), vec![Exit::Failed, Exit::Deadline]);
    assert_eq!(q.draining(), 1, "the deadline exit's copy is still running");
    assert_eq!(s.settled(), vec![1]);
    let r = tick(&mut q, &s, t0 + MAX_PARK, false);
    assert_eq!((ids(&r), exits(&r)), (vec![3], vec![Exit::Timeout]));
    assert_eq!((q.parked(), q.draining()), (0, 2));
    s.set(2, Poll::Ready);
    s.set(3, Poll::Failed);
    tick(&mut q, &s, t0 + MAX_PARK, false);
    assert!(q.is_idle());
    assert_eq!(s.settled(), vec![1, 2, 3]);
}

#[test]
fn a_flush_releases_everything_at_once() {
    let s = Script::default();
    let mut q: Core<Fake, u32> = Core::new();
    let t0 = Instant::now();
    for i in 1..=3 {
        q.park(fake(i), i as u32, true, t0).unwrap();
    }
    s.set(2, Poll::Ready);
    let r = tick(&mut q, &s, t0, true);
    assert_eq!(ids(&r), vec![1, 2, 3]);
    assert_eq!(exits(&r), vec![Exit::Flushed, Exit::Ready, Exit::Flushed]);
    assert_eq!((q.parked(), q.draining()), (0, 2));
}

#[test]
fn refusals_return_the_item_and_its_ticket() {
    let mut q: Core<Fake, u32> = Core::new();
    let t0 = Instant::now();
    let (item, ticket, why) = q.park(fake(1), 1, false, t0).unwrap_err();
    assert_eq!((item.id, ticket, why), (1, 1, Refusal::Disabled));
    assert!(q.is_idle(), "a refused ticket stays with the caller");
    for i in 0..MAX_PARKED {
        q.park(fake(100 + i), 100 + i as u32, true, t0).unwrap();
    }
    assert_eq!(q.refusal(&fake(9), true), Some(Refusal::Full));
    let (_, ticket, why) = q.park(fake(9), 9, true, t0).unwrap_err();
    assert_eq!((ticket, why), (9, Refusal::Full));
    assert_eq!(q.parked(), MAX_PARKED);
}

#[test]
fn a_released_request_is_never_parked_a_second_time() {
    let s = Script::default();
    let mut q: Core<Fake, u32> = Core::new();
    let t0 = Instant::now();
    q.park(fake(7), 1, true, t0).unwrap();
    s.set(1, Poll::Failed);
    assert_eq!(ids(&tick(&mut q, &s, t0, false)), vec![7]);
    // Re-admitted, the admission site tries again: refused once (synchronous handling).
    assert_eq!(q.refusal(&fake(7), true), Some(Refusal::Readmitted));
    let (_, _, why) = q.park(fake(7), 2, true, t0).unwrap_err();
    assert_eq!(why, Refusal::Readmitted);
    // The mark is consumed: a later request at the same address parks normally.
    q.park(fake(7), 3, true, t0).unwrap();
}

#[test]
fn the_readmit_mark_expires_and_a_stuck_drain_is_settled_after_max_drain() {
    let s = Script::default();
    let mut q: Core<Fake, u32> = Core::new();
    let t0 = Instant::now();
    q.park(fake(5), 1, true, t0).unwrap();
    s.set(1, Poll::Ready);
    tick(&mut q, &s, t0, false);
    for _ in 0..=READMIT_MEMORY_TICKS {
        tick(&mut q, &s, t0, false);
    }
    assert_eq!(
        q.refusal(&fake(5), true),
        None,
        "forgotten after READMIT_MEMORY_TICKS"
    );
    // A request flushed while its copy never ends.
    q.park(fake(6), 99, true, t0).unwrap();
    tick(&mut q, &s, t0, true);
    let r = tick(&mut q, &s, t0 + MAX_DRAIN - Duration::from_secs(1), false);
    assert_eq!((q.draining(), r.expired), (1, 0));
    let r = tick(&mut q, &s, t0 + MAX_DRAIN, false);
    assert_eq!((q.draining(), r.expired), (0, 1));
    assert_eq!(s.settled(), vec![1, 99]);
}

// ── The real request type ──

type Rx = tokio::sync::oneshot::Receiver<anyhow::Result<InferenceResponse>>;

fn request(prompt: Vec<u32>, timeout_at: Option<Instant>) -> (InferenceRequest, Rx) {
    let (response_tx, rx) = tokio::sync::oneshot::channel();
    let req = InferenceRequest::Blocking {
        prompt_tokens: Arc::new(prompt),
        session_hash: 0,
        adapter_slot: -1,
        src_lang_id: 0,
        tgt_lang_id: 0,
        num_beams: 1,
        length_penalty: 1.0,
        early_stopping: false,
        image_pixels: vec![],
        max_tokens: 2,
        min_tokens: 0,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 0.0,
        dry_allowed_length: 0,
        lz_penalty: 0.0,
        logit_bias: vec![],
        stop_tokens: vec![],
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: None,
        require_tool_call: false,
        tools_present: false,
        suppress_tool_call: false,
        disable_mtp: true,
        grammar_spec: None,
        seed: Some(1),
        top_logprobs: None,
        prompt_logprobs: None,
        echo: false,
        timeout_at,
        response_tx,
    };
    (req, rx)
}

#[test]
fn the_request_adapter_sees_identity_disconnect_and_deadline() {
    let deadline = Instant::now() + Duration::from_secs(3);
    let (a, rx_a) = request(vec![1, 2, 3], Some(deadline));
    let (b, _rx_b) = request(vec![1, 2, 3], None);
    assert_eq!(a.identity(), a.identity());
    assert_ne!(
        a.identity(),
        b.identity(),
        "equal prompts, distinct requests"
    );
    assert_eq!((a.deadline(), b.deadline()), (Some(deadline), None));
    assert!(!a.client_gone());
    drop(rx_a);
    assert!(a.client_gone(), "the client hung up");
    let s = Script::default();
    let mut q: Core<InferenceRequest, u32> = Core::new();
    assert!(q.park(a, 1, true, Instant::now()).is_ok());
    let r = q.poll(Instant::now(), false, |t| s.state(t), |_| {});
    assert!(r.readmit.is_empty());
    assert_eq!(r.exits[0].0, Exit::Cancelled);
}

fn pending_with(reqs: Vec<InferenceRequest>) -> Arc<(Mutex<PendingQueue>, Condvar)> {
    Arc::new((
        Mutex::new(PendingQueue {
            requests: reqs,
            closed: false,
            rotations: Vec::new(),
        }),
        Condvar::new(),
    ))
}

fn prompts(p: &Arc<(Mutex<PendingQueue>, Condvar)>) -> Vec<u32> {
    p.0.lock()
        .requests
        .iter()
        .map(|r| r.prompt_tokens_arc()[0])
        .collect()
}

#[test]
fn released_requests_go_to_the_front_in_park_order() {
    let (newer, _r0) = request(vec![30], None);
    let pending = pending_with(vec![newer]);
    let (a, _r1) = request(vec![10], None);
    let (b, _r2) = request(vec![20], None);
    assert!(
        !release(&pending, vec![a, b], true, 1),
        "released something: no idle wait"
    );
    assert_eq!(prompts(&pending), vec![10, 20, 30]);
}

#[test]
fn only_an_otherwise_idle_tick_waits_and_only_briefly() {
    let pending = pending_with(Vec::new());
    let t = Instant::now();
    assert!(
        release(&pending, Vec::new(), true, 1),
        "idle with a parked restore: bounded wait"
    );
    assert!(t.elapsed() < Duration::from_secs(1), "the wait is bounded");
    assert!(
        !release(&pending, Vec::new(), false, 1),
        "decode running: never waits"
    );
    assert!(
        !release(&pending, Vec::new(), true, 0),
        "nothing parked: never waits"
    );
    let (r, _rx) = request(vec![1], None);
    let busy = pending_with(vec![r]);
    assert!(
        !release(&busy, Vec::new(), true, 1),
        "a request is waiting: admit it now"
    );
    pending.0.lock().closed = true;
    assert!(!release(&pending, Vec::new(), true, 1), "shutdown: no wait");
}

#[test]
fn with_nothing_parked_the_tick_returns_at_once_and_touches_nothing() {
    let (r, _rx) = request(vec![4], None);
    let pending = pending_with(vec![r]);
    let model = crate::scheduler::lifecycle_tests::StubModel;
    assert!(!super::tick(&model, &pending, true));
    assert!(!super::tick(&model, &pending, false));
    assert_eq!(prompts(&pending), vec![4]);
}

#[test]
fn the_metrics_text_names_every_series() {
    let mut s = String::new();
    render_prometheus(&mut s);
    for name in [
        "# TYPE atlas_kv_restore_parked gauge",
        "# TYPE atlas_kv_restore_parked_seconds_total counter",
        "atlas_kv_restore_parked_total ",
        "atlas_kv_restore_draining ",
        "atlas_kv_restore_exits_total{outcome=\"cancelled\"}",
        "atlas_kv_restore_park_refused_total{reason=\"readmitted\"}",
    ] {
        assert!(s.contains(name), "{name} missing from:\n{s}");
    }
}
