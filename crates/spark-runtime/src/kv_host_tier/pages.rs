// SPDX-License-Identifier: AGPL-3.0-only
// Design after the TierKV pager of arczhi/llama-tierkv (kv-paging-design.md; MIT, Copyright (c) 2023-2026 The ggml authors): a VRAM desk of head sinks, a recent window and recalled pages over a host-RAM store of every evicted page, taken from a written description of it; no code copied or fetched.
// Design after LMCache lmcache/v1/distributed/l1_manager/base.py (commit 3aefae1, Apache-2.0): a host copy is published only after the copy that writes it completed, and its memory is kept while any copy may still read it; no code copied.

//! Responsibility: page-granular demote and fetch for the middle KV blocks of a LIVE sequence
//! (the host-resident window desk; reached through `KvHostTier::pages`).
//!
//! The prefix tier (`KvHostTier`) keeps whole cached prefixes for a LATER request. This API
//! serves the request that is running: the window desk (`spark-model/src/port/kv_window/`)
//! moves the full blocks between the head sinks and the recent window of a long sequence to host
//! RAM ([`PageApi::demote`]), keeps their logical positions, and copies back the few pages its
//! scorer selects ([`PageApi::fetch`]) into fresh GPU blocks. A page is keyed by
//! `(sequence id, logical block index)`; its bytes are the block's regions (all layers, K then V,
//! the swap byte order of `kv_cache::block_io`).
//!
//! # Ordering
//! Every copy runs on the API's copy stream (a non-legacy stream from
//! `GpuBackend::create_copy_stream`). Before each copy the copy stream waits for the caller's
//! compute stream: a demote reads KV the compute stream wrote, a fetch overwrites blocks the
//! compute stream may have read (or zeroed) before they were handed out. Completion is one event
//! per copy, polled without blocking ([`PageApi::poll`]). A fetch of a page whose demote is still
//! running is correct (same stream, FIFO order); the caller must not read a fetched block or free
//! a demoted source block before the copy's ticket is [`RestoreState::Ready`].
//!
//! # Host memory
//! Runs come from a [`super::HostArena`]. A run that a copy may still touch is never freed: a page
//! released while busy waits in a zombie list until every copy that touches it completed. A
//! failed demote drops the pages it was writing (their bytes are not trustworthy). KV of a full
//! block never changes after it is written, so demoting a page that is already on the host (a
//! recalled page going back to the shelf) copies nothing.
//!
//! The desk builds its own instance over a [`slab::SlabArena`] (the prefix tier's pinned pool
//! serves only with the prefix cache on, which the desk refuses). Nothing here reads a row; with
//! the `kv_window_desk` row off no page API is ever used, and the instance every `KvHostTier`
//! holds stays empty. The state and copy engine live in `pages_engine.rs`, the counters in
//! `pages_stats.rs`.

use std::sync::Arc;

use anyhow::{Result, bail};
use parking_lot::Mutex;

use super::clock::{Clock, MonotonicClock};
use super::{HostRun, PageDeps, RestoreState};
use crate::gpu::GpuBackend;
use crate::kv_cache::block_io::BlockRegions;

#[path = "pages_engine.rs"]
mod engine;
/// A fixed-slot host arena (registered slabs) for the window desk.
#[path = "pages_slab.rs"]
pub mod slab;
#[path = "pages_stats.rs"]
mod stats;

#[cfg(test)]
#[path = "pages_tests.rs"]
mod tests;

use engine::{Core, HostPage, Inner, Outcome, Record, check_unique, push_entries};
pub use slab::SlabArena;
pub use stats::{PageSnapshot, PageStats};

/// Which way a page copy goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageOp {
    /// GPU block -> host page (D2H).
    Demote,
    /// Host page -> GPU block (H2D).
    Fetch,
}

/// A started page copy. Poll it with [`PageApi::poll`], then hand it back with
/// [`PageApi::release`].
#[derive(Debug, PartialEq, Eq)]
pub struct PageTicket {
    id: u64,
    event: u64,
    op: PageOp,
    pages: usize,
    bytes: u64,
}

impl PageTicket {
    pub fn id(&self) -> u64 {
        self.id
    }
    /// The completion event on the copy stream (a compute stream may `stream_wait_event` on it).
    pub fn event(&self) -> u64 {
        self.event
    }
    pub fn op(&self) -> PageOp {
        self.op
    }
    /// Pages the copy moves.
    pub fn pages(&self) -> usize {
        self.pages
    }
    /// Bytes the copy moves.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// What [`PageApi::demote`] did with each requested page.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct DemoteReport {
    /// The copy of `copied` (None when nothing needed copying).
    pub ticket: Option<PageTicket>,
    /// Pages whose copy started: their source blocks must stay alive until the ticket is Ready.
    pub copied: Vec<u32>,
    /// Pages already on the host (no copy; their source blocks may go at once).
    pub resident: Vec<u32>,
    /// Pages the arena had no room for: they stay on the GPU.
    pub refused: Vec<u32>,
}

/// Page-granular host storage for live sequences (see the module doc). `Send + Sync`.
pub struct PageApi {
    core: Core,
    inner: Mutex<Inner>,
}

impl PageApi {
    /// An API over the tier's arena and copy stream (one is built per `KvHostTier`).
    pub fn new(deps: PageDeps) -> Self {
        Self::with_parts(
            deps,
            Arc::new(MonotonicClock::new()),
            Arc::new(PageStats::default()),
        )
    }

    /// An API whose counters land in `stats` (shared by every instance the caller creates).
    pub fn with_stats(deps: PageDeps, stats: Arc<PageStats>) -> Self {
        Self::with_parts(deps, Arc::new(MonotonicClock::new()), stats)
    }

    pub(crate) fn with_parts(deps: PageDeps, clock: Arc<dyn Clock>, stats: Arc<PageStats>) -> Self {
        Self {
            core: Core {
                arena: deps.arena,
                block_bytes: deps.block_bytes,
                copy_stream: deps.copy_stream,
                clock,
                stats,
            },
            inner: Mutex::new(Inner::new()),
        }
    }

    pub fn block_bytes(&self) -> usize {
        self.core.block_bytes
    }

    pub fn copy_stream(&self) -> u64 {
        self.core.copy_stream
    }

    pub fn stats(&self) -> &Arc<PageStats> {
        &self.core.stats
    }

    /// Copy the GPU blocks `blocks` (one per page of `pages`, logical block indices of sequence
    /// `seq`) to host pages, after everything already enqueued on `src_stream`. Pages already on
    /// the host are reported `resident` and not copied; pages the arena cannot place are
    /// `refused`. On `Err` nothing was queued, no page was added and every GPU block is untouched.
    pub fn demote(
        &self,
        gpu: &dyn GpuBackend,
        seq: u64,
        pages: &[u32],
        blocks: &[BlockRegions],
        src_stream: u64,
    ) -> Result<DemoteReport> {
        if pages.len() != blocks.len() {
            bail!(
                "kv pages demote: {} pages for {} blocks",
                pages.len(),
                blocks.len()
            );
        }
        check_unique(pages)?;
        let core = &self.core;
        let mut inner = self.inner.lock();
        inner.sweep(core, gpu);
        for b in blocks {
            inner.check_layout(core.block_bytes, &b.regions)?;
        }
        let mut report = DemoteReport::default();
        let mut fresh: Vec<(u32, HostRun)> = Vec::new();
        let mut entries = Vec::new();
        for (&page, block) in pages.iter().zip(blocks) {
            if inner.pages.contains_key(&(seq, page)) {
                report.resident.push(page);
                continue;
            }
            let Some(run) = core.arena.alloc(core.block_bytes) else {
                report.refused.push(page);
                continue;
            };
            if run.bytes < core.block_bytes || run.ptr == 0 {
                tracing::warn!(
                    "kv pages: the arena returned {} bytes at {:#x} for a {}-byte page",
                    run.bytes,
                    run.ptr,
                    core.block_bytes
                );
                core.arena.free(run);
                report.refused.push(page);
                continue;
            }
            push_entries(&mut entries, &block.regions, run.ptr);
            fresh.push((page, run));
        }
        if !report.refused.is_empty() {
            core.stats.add_refused(report.refused.len());
        }
        if fresh.is_empty() {
            return Ok(report);
        }
        let (event, timer) = match inner.submit(core, gpu, src_stream, PageOp::Demote, &entries) {
            Ok(v) => v,
            Err(e) => {
                // The copy stream was synchronized: no copy can still write these runs.
                for (_, run) in fresh {
                    core.arena.free(run);
                }
                return Err(e);
            }
        };
        let id = inner.next_id();
        let bytes = (fresh.len() * core.block_bytes) as u64;
        for &(page, run) in &fresh {
            inner.pages.insert(
                (seq, page),
                HostPage {
                    run,
                    busy: vec![id],
                },
            );
        }
        core.stats.hold(bytes);
        inner.records.insert(
            id,
            Record {
                op: PageOp::Demote,
                event,
                timer,
                bytes,
                pages: fresh.iter().map(|&(p, _)| (seq, p)).collect(),
                outcome: Outcome::Running,
                released: false,
                finished_at: None,
            },
        );
        report.copied = fresh.iter().map(|&(p, _)| p).collect();
        report.ticket = Some(PageTicket {
            id,
            event,
            op: PageOp::Demote,
            pages: fresh.len(),
            bytes,
        });
        Ok(report)
    }

    /// Copy host pages `pages` of `seq` into the GPU blocks `dst` (fresh blocks, same region
    /// layout), after everything already enqueued on `stream`. Every page must be on the host.
    /// The caller reads `dst` only after the ticket is Ready. On `Err` nothing was queued.
    pub fn fetch(
        &self,
        gpu: &dyn GpuBackend,
        seq: u64,
        pages: &[u32],
        dst: &[BlockRegions],
        stream: u64,
    ) -> Result<PageTicket> {
        if pages.is_empty() || pages.len() != dst.len() {
            bail!(
                "kv pages fetch: {} pages for {} destination blocks",
                pages.len(),
                dst.len()
            );
        }
        check_unique(pages)?;
        let core = &self.core;
        let mut inner = self.inner.lock();
        inner.sweep(core, gpu);
        let mut entries = Vec::new();
        for (&page, block) in pages.iter().zip(dst) {
            inner.check_layout(core.block_bytes, &block.regions)?;
            let Some(hp) = inner.pages.get(&(seq, page)) else {
                bail!("kv pages fetch: page {page} of sequence {seq} is not on the host");
            };
            push_entries(&mut entries, &block.regions, hp.run.ptr);
        }
        let (event, timer) = inner.submit(core, gpu, stream, PageOp::Fetch, &entries)?;
        let id = inner.next_id();
        for &page in pages {
            if let Some(hp) = inner.pages.get_mut(&(seq, page)) {
                hp.busy.push(id);
            }
        }
        let bytes = (pages.len() * core.block_bytes) as u64;
        inner.records.insert(
            id,
            Record {
                op: PageOp::Fetch,
                event,
                timer,
                bytes,
                pages: pages.iter().map(|&p| (seq, p)).collect(),
                outcome: Outcome::Running,
                released: false,
                finished_at: None,
            },
        );
        Ok(PageTicket {
            id,
            event,
            op: PageOp::Fetch,
            pages: pages.len(),
            bytes,
        })
    }

    /// Non-blocking state of a copy. An unknown or reclaimed ticket answers `Failed`.
    pub fn poll(&self, gpu: &dyn GpuBackend, t: &PageTicket) -> Result<RestoreState> {
        let mut inner = self.inner.lock();
        Ok(inner
            .advance(&self.core, gpu, t.id)
            .unwrap_or(RestoreState::Failed))
    }

    /// Block until the copy completed, then [`Self::poll`].
    pub fn wait(&self, gpu: &dyn GpuBackend, t: &PageTicket) -> Result<RestoreState> {
        if let Err(e) = gpu.event_synchronize(t.event) {
            tracing::warn!("kv pages: waiting for copy {} failed: {e:#}", t.id);
        }
        self.poll(gpu, t)
    }

    /// Done with a ticket. A running copy keeps its pages busy until it completes.
    pub fn release(&self, t: PageTicket) {
        let mut inner = self.inner.lock();
        let done = match inner.records.get_mut(&t.id) {
            Some(r) => {
                r.released = true;
                matches!(r.outcome, Outcome::Done(_))
            }
            None => false,
        };
        if done {
            inner.records.remove(&t.id);
        }
    }

    /// Whether page `page` of `seq` is on the host.
    pub fn contains(&self, seq: u64, page: u32) -> bool {
        self.inner.lock().pages.contains_key(&(seq, page))
    }

    /// Host pages `seq` holds.
    pub fn seq_pages(&self, seq: u64) -> usize {
        self.inner
            .lock()
            .pages
            .keys()
            .filter(|(s, _)| *s == seq)
            .count()
    }

    /// Bytes of the page `page` of `seq` as stored (tests and oracles; `None` when absent). Only
    /// meaningful once every copy touching the page completed.
    pub fn read_page(&self, seq: u64, page: u32) -> Option<Vec<u8>> {
        let inner = self.inner.lock();
        let hp = inner.pages.get(&(seq, page))?;
        // SAFETY: the run is `block_bytes` long and stays allocated while the lock is held (it is
        // freed only under this lock); the caller's contract is that no copy writes it now.
        let bytes =
            unsafe { std::slice::from_raw_parts(hp.run.ptr as *const u8, self.core.block_bytes) };
        Some(bytes.to_vec())
    }

    /// Drop every host page of `seq` (the sequence ended or its desk was reset). A page a copy
    /// still touches is freed once that copy completed. Returns the pages dropped.
    pub fn release_seq(&self, gpu: &dyn GpuBackend, seq: u64) -> usize {
        let mut inner = self.inner.lock();
        inner.sweep(&self.core, gpu);
        let keys: Vec<(u64, u32)> = inner
            .pages
            .keys()
            .filter(|(s, _)| *s == seq)
            .copied()
            .collect();
        for k in &keys {
            if let Some(hp) = inner.pages.remove(k) {
                inner.retire_run(&self.core, hp);
            }
        }
        keys.len()
    }

    /// Advance every copy and free what finished copies no longer need.
    pub fn sweep(&self, gpu: &dyn GpuBackend) {
        self.inner.lock().sweep(&self.core, gpu);
    }

    /// Copies whose completion was not observed yet.
    pub fn running(&self) -> usize {
        self.inner.lock().running()
    }

    /// Wait for every copy, then free every page (teardown; the API is empty afterwards).
    pub fn quiesce(&self, gpu: &dyn GpuBackend) {
        let mut inner = self.inner.lock();
        if let Err(e) = gpu.synchronize(self.core.copy_stream) {
            tracing::warn!("kv pages: copy stream sync at teardown failed: {e:#}");
        }
        for ev in inner.running_events() {
            let _ = gpu.event_synchronize(ev);
        }
        inner.sweep(&self.core, gpu);
        let keys: Vec<(u64, u32)> = inner.pages.keys().copied().collect();
        for k in keys {
            if let Some(hp) = inner.pages.remove(&k) {
                inner.retire_run(&self.core, hp);
            }
        }
        inner.events.destroy_all(gpu);
        if let Some(ev) = inner.fence.take() {
            let _ = gpu.destroy_event(ev);
        }
    }
}

impl Drop for PageApi {
    fn drop(&mut self) {
        let inner = self.inner.get_mut();
        if inner.running() > 0 {
            // A copy may still touch these runs: leaking them is safe, freeing them is not.
            tracing::warn!(
                "kv pages: dropped with {} page(s) under a running copy; their host memory is leaked",
                inner.pages.len() + inner.zombies.len()
            );
            return;
        }
        let mut held = 0u64;
        for (_, hp) in inner.pages.drain() {
            held += self.core.block_bytes as u64;
            self.core.arena.free(hp.run);
        }
        for z in inner.zombies.drain(..) {
            self.core.arena.free(z.run);
        }
        self.core.stats.drop_held(held);
    }
}
