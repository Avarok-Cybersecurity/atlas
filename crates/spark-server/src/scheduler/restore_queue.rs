// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/integration/vllm/lmcache_mp_connector.py:1360-1420 (commit 3aefae1, Apache-2.0: an external KV load runs asynchronously between scheduler steps); no code copied.

//! Responsibility: the asynchronous KV restore queue (lane L27, row `kv_restore_async`, contract
//! 3.7.5): the scheduler-facing entry points. A new request whose prefix the host KV tier is
//! restoring waits here, outside the batch, while its host-to-device copy runs on the tier's copy
//! stream; every other sequence keeps its decode ticks. At the top of each tick [`tick`] polls the
//! parked restores (never blocking) and puts the requests whose copy finished back at the FRONT
//! of the pending queue, so they go through the normal admission path (policy, batch cap,
//! admission watermark) and find their prefix as an HBM hit.
//!
//! Exits from the queue (`restore_queue/core.rs`), each counted on `/metrics`:
//! * **ready**: the copy completed (the glue re-inserted the blocks into the prefix cache);
//! * **failed**: the restore failed or its poll errored: the request is admitted and recomputes;
//! * **timeout** / **deadline**: parked longer than [`MAX_PARK`], or past the request's own
//!   deadline: admitted now (fairness: a slow copy never starves a request);
//! * **cancelled**: the client disconnected while parked: the request is dropped and its ticket
//!   is polled to completion in the background, so the restored blocks land in the prefix cache
//!   (evictable like any cached block) and no lease or block is leaked;
//! * **flushed**: shutdown, or a queued LoRA rotation (which applies only at quiescence, and a
//!   restore keyed under one adapter must not be admitted under another): every parked request
//!   is admitted at once.
//!
//! Every ticket ends in `ModelPortExt::kv_tier_settle` (lane L26b) once its poll is no longer
//! `Pending`; a request that leaves while its copy still runs leaves the ticket in the drain.
//!
//! A request that left the queue is never parked a second time (LMCache's bypass rule):
//! [`try_park`] refuses it once and the caller handles that restore synchronously.
//!
//! Parking holds no batch slot, so admission accounting is unchanged. A sequence that already
//! holds a slot ([`ActiveSeq`]) is never parked: outside `active` its slot would not count toward
//! the batch cap, so [`try_park`] returns it for synchronous handling.
//!
//! The queue is per scheduler thread (thread-local: one scheduler loop per model), so the free
//! functions of contract 3.7.5 need no handle. With the row off nothing is ever parked and
//! [`tick`] returns at once.

#[path = "restore_queue/metrics.rs"]
mod metrics;
#[path = "restore_queue/core.rs"]
mod park;

#[cfg(test)]
#[path = "restore_queue_tests.rs"]
mod tests;

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use spark_model::traits::Model;
use spark_runtime::kv_host_tier::{RestoreState, RestoreTicket};

pub(crate) use self::metrics::render_prometheus;
use self::park::{Core, Item, Poll, Refusal};
use super::types::{ActiveSeq, PendingQueue};
use crate::api::InferenceRequest;

/// Most requests parked at once per scheduler; more are restored synchronously.
pub(crate) const MAX_PARKED: usize = 64;
/// Longest park before the request is admitted anyway (a 32K-token 4B prefix copies in well
/// under a second at 8+ GB/s; ten seconds means the copy is stuck).
pub(crate) const MAX_PARK: Duration = Duration::from_secs(10);
/// Longest a drained ticket is polled before it is settled anyway (settling waits for the copy;
/// a copy this late means the device is stuck).
pub(crate) const MAX_DRAIN: Duration = Duration::from_secs(60);
/// Bounded wait of an otherwise idle tick while restores are parked.
pub(crate) const IDLE_POLL: Duration = Duration::from_millis(2);
/// Ticks a readmitted request is remembered (the no-second-park rule).
pub(crate) const READMIT_MEMORY_TICKS: u64 = 1024;

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Publish the resolved `kv_restore_async` row (`port_apply`). Off by default.
pub(crate) fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
    if on {
        tracing::info!(
            "kv_restore_async: on (host-tier restores park the request; decode goes on)"
        );
    }
}

pub(crate) fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

impl Item for InferenceRequest {
    fn identity(&self) -> usize {
        match self {
            InferenceRequest::Blocking { prompt_tokens, .. }
            | InferenceRequest::Streaming { prompt_tokens, .. } => {
                Arc::as_ptr(prompt_tokens) as usize
            }
        }
    }

    fn client_gone(&self) -> bool {
        match self {
            InferenceRequest::Blocking { response_tx, .. } => response_tx.is_closed(),
            InferenceRequest::Streaming { token_tx, .. } => token_tx.is_closed(),
        }
    }

    fn deadline(&self) -> Option<Instant> {
        self.timeout_at()
    }
}

/// The scheduler thread's queue and its share of the gauges.
struct Queue {
    core: Core<InferenceRequest, RestoreTicket>,
    shown: metrics::Shown,
}

impl Queue {
    fn sync_gauges(&mut self) {
        let (p, d) = (self.core.parked(), self.core.draining());
        self.shown.sync(p, d);
    }
}

thread_local! {
    static QUEUE: RefCell<Queue> = RefCell::new(Queue {
        core: Core::new(),
        shown: metrics::Shown::default(),
    });
}

/// Something [`try_park`] accepts: a new request (parked) or an admitted sequence (refused).
pub(crate) trait Parkable: Sized {
    fn park_with(self, ticket: RestoreTicket) -> Result<(), (Self, RestoreTicket)>;
}

impl Parkable for InferenceRequest {
    fn park_with(self, ticket: RestoreTicket) -> Result<(), (Self, RestoreTicket)> {
        QUEUE.with(|q| {
            let mut q = q.borrow_mut();
            let res = match q.core.park(self, ticket, enabled(), Instant::now()) {
                Ok(()) => {
                    metrics::parked_one();
                    Ok(())
                }
                Err((req, ticket, why)) => {
                    metrics::refused(why);
                    if why == Refusal::Full {
                        tracing::debug!(
                            "kv_restore_async: {MAX_PARKED} restores parked; this one is synchronous"
                        );
                    }
                    Err((req, ticket))
                }
            };
            q.sync_gauges();
            res
        })
    }
}

impl Parkable for ActiveSeq {
    fn park_with(self, ticket: RestoreTicket) -> Result<(), (Self, RestoreTicket)> {
        metrics::refused(Refusal::HoldsSlot);
        Err((self, ticket))
    }
}

/// Contract 3.7.5: park `seq` until `ticket`'s restore completes; the queue then owns the ticket
/// and settles it. `Err` returns the item for synchronous handling (row off, already parked
/// once, queue full, or a slot-holding [`ActiveSeq`]), and with it the ticket (an addition to
/// the contract's signature, as lane L26b's admission site uses it): the caller finishes that
/// restore itself with `ModelPortExt::kv_tier_settle`. [`would_park`] asks without side effect.
/// `allow(dead_code)` keeps this lane buildable on its own commit, before L26b's call exists.
#[allow(dead_code)]
pub(crate) fn try_park<P: Parkable>(
    seq: P,
    ticket: RestoreTicket,
) -> Result<(), (P, RestoreTicket)> {
    seq.park_with(ticket)
}

/// Whether [`try_park`] would park `req` now (no side effect), for an admission site that wants
/// to skip starting an asynchronous restore that would be refused.
#[allow(dead_code)]
pub(crate) fn would_park(req: &InferenceRequest) -> bool {
    QUEUE.with(|q| q.borrow().core.refusal(req, enabled()).is_none())
}

fn poll_ticket(model: &dyn Model, t: &RestoreTicket) -> Poll {
    let Some(ext) = model.port_ext() else {
        return Poll::Failed;
    };
    match ext.kv_restore_poll(t) {
        Ok(RestoreState::Pending) => Poll::Pending,
        Ok(RestoreState::Ready) => Poll::Ready,
        Ok(RestoreState::Failed) => Poll::Failed,
        Err(e) => {
            tracing::warn!(
                "kv_restore_async: restore poll failed; the prefix is recomputed: {e:#}"
            );
            Poll::Failed
        }
    }
}

/// Hand a finished ticket back (lane L26b's `kv_tier_settle`). A model without the extension
/// never issued one; dropping it is all there is to do.
fn settle_ticket(model: &dyn Model, t: RestoreTicket) {
    if let Some(ext) = model.port_ext() {
        ext.kv_tier_settle(t);
    }
}

/// Top-of-tick hook (`scheduler/mod.rs`). Polls the parked restores and puts released requests at
/// the front of the pending queue. `idle`: nothing active, prefilling, swapped or preempted.
/// Returns true only when the tick has nothing else to do but restores are parked: it then waited
/// up to [`IDLE_POLL`] (waking early on a new request) and the caller starts the next tick
/// instead of blocking on the request condvar, which nothing would signal.
pub(super) fn tick(
    model: &dyn Model,
    pending: &Arc<(Mutex<PendingQueue>, Condvar)>,
    idle: bool,
) -> bool {
    let polled = QUEUE.with(|q| {
        let mut q = q.borrow_mut();
        if q.core.is_idle() {
            return None;
        }
        let flush = {
            let g = pending.0.lock();
            g.closed || !g.rotations.is_empty()
        };
        let report = q.core.poll(
            Instant::now(),
            flush,
            |t| poll_ticket(model, t),
            |t| settle_ticket(model, t),
        );
        if report.expired > 0 {
            tracing::warn!(
                "kv_restore_async: {} restore(s) still copying after {}s; settled anyway",
                report.expired,
                MAX_DRAIN.as_secs()
            );
        }
        metrics::record(&report);
        q.sync_gauges();
        Some((report.readmit, q.core.parked()))
    });
    match polled {
        Some((readmit, parked_left)) => release(pending, readmit, idle, parked_left),
        None => false,
    }
}

/// Put `readmit` at the front of the pending queue (park order, ahead of newer arrivals); then,
/// on an idle tick with restores still parked and nothing else to do, wait up to [`IDLE_POLL`].
/// Returns whether it waited (the caller then skips the rest of the tick).
pub(super) fn release(
    pending: &Arc<(Mutex<PendingQueue>, Condvar)>,
    readmit: Vec<InferenceRequest>,
    idle: bool,
    parked_left: usize,
) -> bool {
    let released = !readmit.is_empty();
    let mut g = pending.0.lock();
    if released {
        let newer = std::mem::take(&mut g.requests);
        g.requests = readmit;
        g.requests.extend(newer);
    }
    let wait = idle
        && parked_left > 0
        && !released
        && g.requests.is_empty()
        && g.rotations.is_empty()
        && !g.closed;
    if wait {
        let _ = pending.1.wait_for(&mut g, IDLE_POLL);
    }
    wait
}
