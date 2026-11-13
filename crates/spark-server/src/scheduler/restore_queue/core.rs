// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/integration/vllm/lmcache_mp_connector.py:1360-1420 (commit 3aefae1, Apache-2.0: a failed asynchronous load is bypassed until the request is admitted for local computation, never re-entering the waiting state); no code copied.

//! Responsibility: the restore queue's logic (lane L27), generic over the parked item and the
//! ticket so it is tested without a GPU or a tier: parking and its refusals, the per-tick poll
//! with every exit (ready, failed, timeout, deadline, cancelled, flushed), the background drain
//! of tickets nobody waits for, and the no-second-park memory.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::{MAX_DRAIN, MAX_PARK, MAX_PARKED, READMIT_MEMORY_TICKS};

/// One non-blocking look at a parked restore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Poll {
    Pending,
    Ready,
    Failed,
}

/// Why a parked request left the queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Exit {
    Ready,
    Failed,
    Timeout,
    Deadline,
    Cancelled,
    Flushed,
}

impl Exit {
    pub(crate) const ALL: [Exit; 6] = [
        Exit::Ready,
        Exit::Failed,
        Exit::Timeout,
        Exit::Deadline,
        Exit::Cancelled,
        Exit::Flushed,
    ];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Exit::Ready => "ready",
            Exit::Failed => "failed",
            Exit::Timeout => "timeout",
            Exit::Deadline => "deadline",
            Exit::Cancelled => "cancelled",
            Exit::Flushed => "flushed",
        }
    }
}

/// Why `try_park` returned the item for synchronous handling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    Disabled,
    Readmitted,
    Full,
    HoldsSlot,
}

impl Refusal {
    pub(crate) const ALL: [Refusal; 4] = [
        Refusal::Disabled,
        Refusal::Readmitted,
        Refusal::Full,
        Refusal::HoldsSlot,
    ];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Refusal::Disabled => "disabled",
            Refusal::Readmitted => "readmitted",
            Refusal::Full => "full",
            Refusal::HoldsSlot => "holds_slot",
        }
    }
}

/// What the queue needs to know about a parked item.
pub(crate) trait Item {
    /// Stable while the item lives (the no-second-park rule).
    fn identity(&self) -> usize;
    /// The client went away (its response channel is closed).
    fn client_gone(&self) -> bool;
    /// The request's own deadline, if any.
    fn deadline(&self) -> Option<Instant>;
}

struct Entry<I, T> {
    item: I,
    ticket: T,
    since: Instant,
}

/// The items one poll released and why each parked entry left.
pub(crate) struct Report<I> {
    /// In park order; the caller puts them at the front of the pending queue.
    pub(crate) readmit: Vec<I>,
    /// One per entry that left, with how long it was parked.
    pub(crate) exits: Vec<(Exit, Duration)>,
    /// Drained tickets still pending after [`MAX_DRAIN`], settled anyway (a stuck copy).
    pub(crate) expired: usize,
}

/// Parked entries, drained tickets and the readmitted identities.
pub(crate) struct Core<I, T> {
    parked: VecDeque<Entry<I, T>>,
    draining: Vec<(T, Instant)>,
    readmitted: VecDeque<(usize, u64)>,
    ticks: u64,
}

impl<I: Item, T> Core<I, T> {
    pub(crate) fn new() -> Self {
        Self {
            parked: VecDeque::new(),
            draining: Vec::new(),
            readmitted: VecDeque::new(),
            ticks: 0,
        }
    }

    pub(crate) fn parked(&self) -> usize {
        self.parked.len()
    }

    pub(crate) fn draining(&self) -> usize {
        self.draining.len()
    }

    /// Nothing parked and nothing draining: [`Self::poll`] would do nothing.
    pub(crate) fn is_idle(&self) -> bool {
        self.parked.is_empty() && self.draining.is_empty()
    }

    /// Why `item` would be refused right now, if it would.
    pub(crate) fn refusal(&self, item: &I, enabled: bool) -> Option<Refusal> {
        if !enabled {
            Some(Refusal::Disabled)
        } else if self.readmitted.iter().any(|(id, _)| *id == item.identity()) {
            Some(Refusal::Readmitted)
        } else if self.parked.len() >= MAX_PARKED {
            Some(Refusal::Full)
        } else {
            None
        }
    }

    /// Park `item` until `ticket` completes. On refusal both come back, for the caller to finish
    /// the restore synchronously.
    pub(crate) fn park(
        &mut self,
        item: I,
        ticket: T,
        enabled: bool,
        now: Instant,
    ) -> Result<(), (I, T, Refusal)> {
        if let Some(r) = self.refusal(&item, enabled) {
            if r == Refusal::Readmitted {
                let id = item.identity();
                self.readmitted.retain(|(x, _)| *x != id);
            }
            return Err((item, ticket, r));
        }
        self.parked.push_back(Entry {
            item,
            ticket,
            since: now,
        });
        Ok(())
    }

    /// Poll every parked and draining ticket once (non-blocking). `flush` releases every parked
    /// item whatever its state (shutdown, a queued LoRA rotation). Every ticket that is done
    /// (not `Pending`) goes to `settle` exactly once; a released or cancelled item whose copy
    /// still runs leaves its ticket in the drain, polled on later ticks until it is done, or
    /// settled anyway after [`MAX_DRAIN`] (counted in [`Report::expired`]).
    pub(crate) fn poll(
        &mut self,
        now: Instant,
        flush: bool,
        mut poll: impl FnMut(&T) -> Poll,
        mut settle: impl FnMut(T),
    ) -> Report<I> {
        self.ticks += 1;
        let mut report = Report {
            readmit: Vec::new(),
            exits: Vec::new(),
            expired: 0,
        };
        let mut keep = VecDeque::with_capacity(self.parked.len());
        while let Some(e) = self.parked.pop_front() {
            let state = poll(&e.ticket);
            let waited = now.saturating_duration_since(e.since);
            let late = e.item.deadline().is_some_and(|d| now >= d);
            let exit = if e.item.client_gone() {
                Some(Exit::Cancelled)
            } else {
                match state {
                    Poll::Ready => Some(Exit::Ready),
                    Poll::Failed => Some(Exit::Failed),
                    Poll::Pending if flush => Some(Exit::Flushed),
                    Poll::Pending if late => Some(Exit::Deadline),
                    Poll::Pending if waited >= MAX_PARK => Some(Exit::Timeout),
                    Poll::Pending => None,
                }
            };
            let Some(exit) = exit else {
                keep.push_back(e);
                continue;
            };
            if state == Poll::Pending {
                self.draining.push((e.ticket, now));
            } else {
                settle(e.ticket);
            }
            report.exits.push((exit, waited));
            if exit != Exit::Cancelled {
                self.readmitted.push_back((e.item.identity(), self.ticks));
                report.readmit.push(e.item);
            }
        }
        self.parked = keep;
        for (t, since) in std::mem::take(&mut self.draining) {
            if now.saturating_duration_since(since) >= MAX_DRAIN {
                report.expired += 1;
                settle(t);
            } else if poll(&t) == Poll::Pending {
                self.draining.push((t, since));
            } else {
                settle(t);
            }
        }
        let ticks = self.ticks;
        while self
            .readmitted
            .front()
            .is_some_and(|(_, at)| ticks - at > READMIT_MEMORY_TICKS)
        {
            self.readmitted.pop_front();
        }
        report
    }
}
