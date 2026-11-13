// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/distributed/l1_manager/base.py (commit 3aefae1, Apache-2.0): a host copy is published only after the copy that writes it completed, and its memory is kept while any copy may still read it; no code copied.

//! Responsibility: the page API's mutable state and copy engine (lane L36): the page index,
//! the copy records, the zombie runs, the fence that orders the copy stream behind the caller's
//! stream, and the completion handling (`advance`) that publishes timings, releases busy pages
//! and frees zombies. Everything here runs under the page API's one lock.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};

use super::PageOp;
use super::stats::PageStats;
use crate::gpu::{CopyEntry, DevicePtr, GpuBackend};
use crate::kv_cache::block_io;
use crate::kv_host_tier::clock::Clock;
use crate::kv_host_tier::events::{EventPool, Timer};
use crate::kv_host_tier::{HostArena, HostRun, RestoreState};

/// A finished record no caller released is dropped this long after its copy completed.
const RECORD_TTL: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    Running,
    Done(RestoreState),
}

/// One host page: its run and the copies that may still touch it.
pub(super) struct HostPage {
    pub(super) run: HostRun,
    pub(super) busy: Vec<u64>,
}

/// A run whose page was released while a copy still touched it.
pub(super) struct Zombie {
    pub(super) run: HostRun,
    pub(super) waiting: Vec<u64>,
}

pub(super) struct Record {
    pub(super) op: PageOp,
    pub(super) event: u64,
    pub(super) timer: Timer,
    pub(super) bytes: u64,
    pub(super) pages: Vec<(u64, u32)>,
    pub(super) outcome: Outcome,
    pub(super) released: bool,
    pub(super) finished_at: Option<Duration>,
}

/// The immutable half of a page API.
pub(super) struct Core {
    pub(super) arena: Arc<dyn HostArena>,
    pub(super) block_bytes: usize,
    pub(super) copy_stream: u64,
    pub(super) clock: Arc<dyn Clock>,
    pub(super) stats: Arc<PageStats>,
}

/// The mutable half, behind the page API's lock.
pub(super) struct Inner {
    pub(super) pages: HashMap<(u64, u32), HostPage>,
    pub(super) layout: Option<Vec<usize>>,
    pub(super) records: HashMap<u64, Record>,
    pub(super) zombies: Vec<Zombie>,
    pub(super) events: EventPool,
    pub(super) fence: Option<u64>,
    next_id: u64,
}

pub(super) fn check_unique(pages: &[u32]) -> Result<()> {
    let mut seen = std::collections::HashSet::with_capacity(pages.len());
    for &p in pages {
        if !seen.insert(p) {
            bail!("kv pages: page {p} appears twice in one request");
        }
    }
    Ok(())
}

/// Copy entries of one block laid out from host address `base`, region by region.
pub(super) fn push_entries(out: &mut Vec<CopyEntry>, regions: &[(DevicePtr, usize)], base: usize) {
    let mut off = 0usize;
    for &(dev, bytes) in regions {
        out.push(CopyEntry {
            dev,
            host: base + off,
            bytes,
        });
        off += bytes;
    }
}

impl Inner {
    pub(super) fn new() -> Self {
        Self {
            pages: HashMap::new(),
            layout: None,
            records: HashMap::new(),
            zombies: Vec::new(),
            events: EventPool::new(),
            fence: None,
            next_id: 0,
        }
    }

    pub(super) fn next_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    pub(super) fn check_layout(
        &mut self,
        block_bytes: usize,
        regions: &[(DevicePtr, usize)],
    ) -> Result<()> {
        let total: usize = regions.iter().map(|r| r.1).sum();
        if total != block_bytes {
            bail!("kv pages: a block of {total} bytes for {block_bytes}-byte pages");
        }
        match &self.layout {
            Some(l) => {
                if l.len() != regions.len() || l.iter().zip(regions).any(|(a, r)| *a != r.1) {
                    bail!("kv pages: block region layout differs from the stored one");
                }
            }
            None => self.layout = Some(regions.iter().map(|r| r.1).collect()),
        }
        Ok(())
    }

    /// Make the copy stream wait for `src`, enqueue the copy, record its event. On an error after
    /// the fence the copy stream was synchronized (nothing of it still runs).
    pub(super) fn submit(
        &mut self,
        core: &Core,
        gpu: &dyn GpuBackend,
        src: u64,
        op: PageOp,
        entries: &[CopyEntry],
    ) -> Result<(u64, Timer)> {
        let cs = core.copy_stream;
        if src != cs {
            let ev = match self.fence {
                Some(e) => e,
                None => {
                    let e = gpu.create_event()?;
                    self.fence = Some(e);
                    e
                }
            };
            gpu.record_event(ev, src)?;
            gpu.stream_wait_event(cs, ev)?;
        }
        let mut timer = self.events.start_timer(gpu, cs, core.clock.now());
        let enqueued = match op {
            PageOp::Demote => block_io::submit_d2h(gpu, entries, cs),
            PageOp::Fetch => block_io::submit_h2d(gpu, entries, cs, true),
        };
        let submitted = enqueued.and_then(|()| {
            timer.stop(gpu, cs);
            let ev = self.events.take(gpu)?;
            match gpu.record_event(ev, cs) {
                Ok(()) => Ok(ev),
                Err(e) => {
                    self.events.give(ev);
                    Err(e)
                }
            }
        });
        match submitted {
            Ok(ev) => Ok((ev, timer)),
            Err(e) => {
                timer.discard(gpu);
                if let Err(se) = gpu.synchronize(cs) {
                    tracing::warn!("kv pages: copy stream sync after a failed submit: {se:#}");
                }
                Err(e.context("kv pages: enqueueing a page copy"))
            }
        }
    }

    /// Advance one record; `None` when the id is unknown.
    pub(super) fn advance(
        &mut self,
        core: &Core,
        gpu: &dyn GpuBackend,
        id: u64,
    ) -> Option<RestoreState> {
        let rec = self.records.get_mut(&id)?;
        if let Outcome::Done(s) = rec.outcome {
            return Some(s);
        }
        let state = match gpu.event_query(rec.event) {
            Ok(false) => return Some(RestoreState::Pending),
            Ok(true) => {
                let secs = rec.timer.elapsed_secs(gpu, core.clock.now());
                core.stats
                    .add_done(rec.op, rec.pages.len(), rec.bytes, secs);
                self.events.give(rec.event);
                RestoreState::Ready
            }
            Err(e) => {
                tracing::warn!("kv pages: copy {id} failed: {e:#}");
                if let Err(se) = gpu.synchronize(core.copy_stream) {
                    tracing::warn!("kv pages: copy stream sync after a failed copy: {se:#}");
                }
                rec.timer.discard(gpu);
                let _ = gpu.destroy_event(rec.event);
                core.stats.add_failed();
                RestoreState::Failed
            }
        };
        rec.outcome = Outcome::Done(state);
        rec.finished_at = Some(core.clock.now());
        let op = rec.op;
        let pages = std::mem::take(&mut rec.pages);
        if rec.released {
            self.records.remove(&id);
        }
        for key in &pages {
            if let Some(hp) = self.pages.get_mut(key) {
                hp.busy.retain(|&b| b != id);
            }
        }
        if op == PageOp::Demote && state == RestoreState::Failed {
            // The bytes these pages hold were never fully written: they are not a copy of anything.
            for key in &pages {
                if let Some(hp) = self.pages.remove(key) {
                    self.retire_run(core, hp);
                }
            }
        }
        self.free_zombies(core, id);
        Some(state)
    }

    /// Free a page's run now, or once the copies still touching it completed.
    pub(super) fn retire_run(&mut self, core: &Core, hp: HostPage) {
        core.stats.drop_held(core.block_bytes as u64);
        if hp.busy.is_empty() {
            core.arena.free(hp.run);
        } else {
            self.zombies.push(Zombie {
                run: hp.run,
                waiting: hp.busy,
            });
        }
    }

    fn free_zombies(&mut self, core: &Core, finished: u64) {
        let mut i = 0;
        while i < self.zombies.len() {
            self.zombies[i].waiting.retain(|&w| w != finished);
            if self.zombies[i].waiting.is_empty() {
                let z = self.zombies.swap_remove(i);
                core.arena.free(z.run);
            } else {
                i += 1;
            }
        }
    }

    /// Advance every record; drop finished ones that were released (or abandoned past the TTL).
    pub(super) fn sweep(&mut self, core: &Core, gpu: &dyn GpuBackend) {
        let ids: Vec<u64> = self.records.keys().copied().collect();
        for id in ids {
            let _ = self.advance(core, gpu, id);
        }
        let now = core.clock.now();
        self.records.retain(|_, r| match r.outcome {
            Outcome::Running => true,
            Outcome::Done(_) => {
                !r.released
                    && r.finished_at
                        .is_none_or(|t| now.saturating_sub(t) < RECORD_TTL)
            }
        });
    }

    /// Copies whose completion was not observed yet.
    pub(super) fn running(&self) -> usize {
        self.records
            .values()
            .filter(|r| r.outcome == Outcome::Running)
            .count()
    }

    /// Events of the copies still running (teardown waits on them).
    pub(super) fn running_events(&self) -> Vec<u64> {
        self.records
            .values()
            .filter(|r| r.outcome == Outcome::Running)
            .map(|r| r.event)
            .collect()
    }
}
