// SPDX-License-Identifier: AGPL-3.0-only
// Design after the TierKV pager of arczhi/llama-tierkv (kv-paging-design.md; MIT, Copyright (c) 2023-2026 The ggml authors): a VRAM desk of head sinks, a recent window and recalled pages, every other page on a host-RAM shelf, deltas only on recall, taken from a written description of the design; no code copied or fetched.

//! Responsibility: the window desk's per-sequence state machine (lane L36). One
//! [`Desk`] serves one KV cache; [`Desk::maintain`] runs from the decode block hook
//! (`model/block_mgmt.rs::ensure_blocks_through_decode`) before every decode or verify forward,
//! on the host, and:
//!
//! 1. settles the sequence's finished copies ([`super::fetch::settle`]);
//! 2. demotes, once a batch of them waits ([`policy::demote_batch`]), the full blocks between
//!    the head sinks and the recent window: D2H to host pages, and the table entry becomes the
//!    desk's HOLE block, so the logical length and every position stay as they were;
//! 3. every [`policy::STEP_TOKENS`] positions after an eviction, asks the recall scorer (if one
//!    is registered) to rank the demoted pages, retires recalled pages that fell out (no copy:
//!    the host copy is still there) and fetches only the new ones into fresh blocks.
//!
//! Sequences are keyed by their first physical block, which is a head sink (never demoted) and
//! is not shared between live sequences because the desk refuses to serve with the prefix
//! cache on. A key reused by a new sequence is caught by checking the stored slots against the
//! table; a state whose block is free again, or that was not seen for [`IDLE_CALLS`] hook calls,
//! is dropped by a periodic sweep. Every block reference stays balanced (see `fetch.rs`).
//!
//! No step here waits on the GPU. An `Err` leaves the table, the slots and every reference as
//! they were before the failing operation. The two moves (demote, recall) live in
//! `desk_ops.rs`.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::kv_host_tier::pages::PageStats;

use super::fetch::{self, HostSide, Pending, Settled};
use super::policy::{self, Geometry, Slot};
use super::selector::PageScorer;

#[path = "desk_ops.rs"]
mod ops;

/// Hook calls between two sweeps for dead sequence states.
pub const SWEEP_EVERY: u64 = 64;
/// A state not seen for this many hook calls is dropped (its sequence is gone or parked).
pub const IDLE_CALLS: u64 = 200_000;

/// The device half the desk needs: the view kernel and its hole slot (`selector::SeamDevice`
/// in production; tests drive the desk without the seam's process state).
pub(crate) trait DeviceHooks {
    /// The view kernel and its buffers exist on `gpu`, so hiding a page is safe.
    fn ready(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<bool>;
    /// Publish the hole block id the view kernel removes (`None`: nothing hidden).
    fn publish_hole(&self, gpu: &dyn GpuBackend, hole: Option<u32>, stream: u64) -> Result<()>;
}

/// The desk's sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeskConfig {
    pub(crate) head_tokens: usize,
    pub(crate) window_tokens: usize,
    pub(crate) recall_tokens: usize,
    pub(crate) host_budget_bytes: u64,
}

/// What one [`Desk::maintain`] call changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepReport {
    /// Blocks moved off the GPU by a copy (or re-hidden from the host copy).
    pub demoted: usize,
    /// Recalled pages put back on the shelf.
    pub rehidden: usize,
    /// Pages whose fetch started.
    pub fetch_started: usize,
    /// Fetched pages installed in the table.
    pub installed: usize,
    /// Original blocks released after their demote completed.
    pub freed: usize,
    /// Blocks put back after a failed demote.
    pub reverted: usize,
}

/// One sequence's desk.
pub(crate) struct SeqDesk {
    pub(crate) id: u64,
    pub(crate) slots: Vec<Slot>,
    pending: Vec<Pending>,
    last_round: Option<usize>,
    last_touch: u64,
    pub(crate) resident_tokens: usize,
}

impl SeqDesk {
    fn new(id: u64) -> Self {
        Self {
            id,
            slots: Vec::new(),
            pending: Vec::new(),
            last_round: None,
            last_touch: 0,
            resident_tokens: 0,
        }
    }

    /// Worth keeping: a page is off the GPU or a copy runs.
    fn keep(&self) -> bool {
        !self.pending.is_empty() || self.slots.iter().any(|s| s.demoted())
    }

    /// The stored slots describe `table` (the key was not reused by another sequence).
    fn matches(&self, table: &[u32], hole: Option<u32>) -> bool {
        for (i, s) in self.slots.iter().enumerate() {
            let Some(&b) = table.get(i) else {
                if s.demoted() {
                    return false;
                }
                continue;
            };
            let is_hole = Some(b) == hole;
            let hole_expected = matches!(s, Slot::Hidden | Slot::Fetching);
            if is_hole != hole_expected {
                return false;
            }
        }
        let tail = self.slots.len().min(table.len());
        table[tail..].iter().all(|&b| Some(b) != hole)
    }
}

/// The desk of one KV cache.
pub(crate) struct Desk {
    cfg: DeskConfig,
    stats: Arc<PageStats>,
    cache: Option<u64>,
    hole: Option<u32>,
    host: Option<HostSide>,
    seqs: HashMap<u32, SeqDesk>,
    orphans: Vec<Pending>,
    next_id: u64,
    calls: u64,
    warned_lost: bool,
}

/// Identity of a cache: its first pool address and block count (a model swap builds new ones).
fn cache_id(kv: &PagedKvCache) -> u64 {
    let base = if kv.num_layers() > 0 && kv.num_blocks() > 0 {
        kv.k_cache_ptr(0, 0).0
    } else {
        0
    };
    base ^ ((kv.num_blocks() as u64) << 48)
}

impl Desk {
    pub(crate) fn new(cfg: DeskConfig, stats: Arc<PageStats>) -> Self {
        Self {
            cfg,
            stats,
            cache: None,
            hole: None,
            host: None,
            seqs: HashMap::new(),
            orphans: Vec::new(),
            next_id: 0,
            calls: 0,
            warned_lost: false,
        }
    }

    pub(crate) fn geometry(&self, block_size: usize) -> Geometry {
        Geometry::new(
            block_size,
            self.cfg.head_tokens,
            self.cfg.window_tokens,
            self.cfg.recall_tokens,
        )
    }

    /// Sum of the positions the attention reads, over every sequence with a desk.
    pub(crate) fn resident_tokens(&self) -> usize {
        self.seqs.values().map(|s| s.resident_tokens).sum()
    }

    /// Any page of any sequence is off the GPU (or on its way back).
    pub(crate) fn any_hidden(&self) -> bool {
        self.seqs
            .values()
            .any(|s| s.slots.iter().any(|x| !x.on_gpu()))
    }

    /// The hole block, once one was taken.
    #[cfg(test)]
    pub(crate) fn hole(&self) -> Option<u32> {
        self.hole
    }

    /// The state of the sequence whose first block is `block0`.
    #[cfg(test)]
    pub(crate) fn seq(&self, block0: u32) -> Option<&SeqDesk> {
        self.seqs.get(&block0)
    }

    /// The page API, once the first demote created it.
    #[cfg(test)]
    pub(crate) fn host(&self) -> Option<&HostSide> {
        self.host.as_ref()
    }

    /// Run the desk for one sequence before its forward (see the module doc). `kv_tokens` is the
    /// number of positions already in the KV.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn maintain(
        &mut self,
        table: &mut [u32],
        kv_tokens: usize,
        kv: &mut PagedKvCache,
        gpu: &dyn GpuBackend,
        stream: u64,
        dev: &dyn DeviceHooks,
        scorer: Option<&dyn PageScorer>,
    ) -> Result<StepReport> {
        let mut rep = StepReport::default();
        self.calls += 1;
        self.bind(kv, gpu, stream, dev)?;
        if let Some(h) = &self.host
            && !self.orphans.is_empty()
        {
            self.orphans =
                fetch::settle_orphans(&h.api, gpu, std::mem::take(&mut self.orphans), kv);
        }
        if self.calls.is_multiple_of(SWEEP_EVERY) {
            self.sweep(gpu, kv);
        }
        let Some(&block0) = table.first() else {
            return Ok(rep);
        };
        let geom = self.geometry(kv.block_size());
        if self
            .seqs
            .get(&block0)
            .is_some_and(|sd| !sd.matches(table, self.hole))
            && let Some(sd) = self.seqs.remove(&block0)
        {
            tracing::debug!(
                "kv window desk: block {block0} starts another sequence now; dropping the old state"
            );
            self.retire_seq(sd, gpu);
        }
        let mut sd = match self.seqs.remove(&block0) {
            Some(sd) => sd,
            None => {
                if geom.demotable(kv_tokens).len() < geom.batch_blocks() {
                    return Ok(rep);
                }
                if self.hole.is_some_and(|h| table.contains(&h)) {
                    // Hidden entries whose state is gone (a sweep dropped it): nothing to restore
                    // them from, and demoting the hole would copy garbage. Leave the sequence be.
                    if !self.warned_lost {
                        self.warned_lost = true;
                        tracing::warn!(
                            "kv window desk: a sequence holds hidden pages the desk no longer tracks; \
                             its desk stays as it is"
                        );
                    }
                    return Ok(rep);
                }
                self.next_id += 1;
                SeqDesk::new(self.next_id)
            }
        };
        sd.last_touch = self.calls;
        sd.slots.resize(table.len(), Slot::Resident);
        let res = self.step(
            &mut sd, block0, table, kv_tokens, kv, gpu, stream, dev, scorer, &mut rep,
        );
        sd.resident_tokens = policy::resident_tokens(&geom, kv_tokens, &sd.slots);
        if sd.keep() {
            self.seqs.insert(block0, sd);
        }
        res.map(|()| rep)
    }

    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        sd: &mut SeqDesk,
        block0: u32,
        table: &mut [u32],
        kv_tokens: usize,
        kv: &mut PagedKvCache,
        gpu: &dyn GpuBackend,
        stream: u64,
        dev: &dyn DeviceHooks,
        scorer: Option<&dyn PageScorer>,
        rep: &mut StepReport,
    ) -> Result<()> {
        let geom = self.geometry(kv.block_size());
        if let (Some(h), Some(hole)) = (&self.host, self.hole)
            && !sd.pending.is_empty()
        {
            let mut s = Settled::default();
            let pending = std::mem::take(&mut sd.pending);
            sd.pending =
                fetch::settle(&h.api, gpu, pending, table, &mut sd.slots, hole, kv, &mut s);
            rep.freed += s.freed;
            rep.installed += s.installed;
            rep.reverted += s.reverted;
        }
        let batch = policy::demote_batch(&geom, kv_tokens, &sd.slots);
        if !batch.is_empty() {
            self.demote(sd, block0, table, &batch, kv, gpu, stream, dev, scorer, rep)?;
        }
        if let Some(sc) = scorer
            && policy::recall_due(&geom, kv_tokens, sd.last_round, &sd.slots)
        {
            sd.last_round = Some(kv_tokens);
            self.recall(sd, block0, table, kv_tokens, kv, gpu, stream, sc, rep)?;
        }
        Ok(())
    }

    /// Forget every state of a cache that is gone; the new cache starts with nothing hidden.
    fn bind(
        &mut self,
        kv: &PagedKvCache,
        gpu: &dyn GpuBackend,
        stream: u64,
        dev: &dyn DeviceHooks,
    ) -> Result<()> {
        let id = cache_id(kv);
        if self.cache == Some(id) {
            return Ok(());
        }
        if self.cache.is_some() {
            tracing::info!(
                "kv window desk: a new KV cache; every desk state of the old one is dropped"
            );
            if let Some(h) = self.host.take() {
                h.teardown(gpu);
            }
            self.seqs.clear();
            self.orphans.clear();
            if self.hole.take().is_some() {
                dev.publish_hole(gpu, None, stream)?;
            }
        }
        self.cache = Some(id);
        Ok(())
    }

    /// Drop the states whose sequence is gone (its first block is free) or idle too long.
    fn sweep(&mut self, gpu: &dyn GpuBackend, kv: &PagedKvCache) {
        let calls = self.calls;
        let dead: Vec<u32> = self
            .seqs
            .iter()
            .filter(|(b0, sd)| {
                (**b0 as usize) >= kv.num_blocks()
                    || kv.ref_count(**b0) == 0
                    || calls.saturating_sub(sd.last_touch) > IDLE_CALLS
            })
            .map(|(b0, _)| *b0)
            .collect();
        for b0 in dead {
            if let Some(sd) = self.seqs.remove(&b0) {
                self.retire_seq(sd, gpu);
            }
        }
    }

    fn retire_seq(&mut self, sd: SeqDesk, gpu: &dyn GpuBackend) {
        self.orphans.extend(sd.pending);
        if let Some(h) = &self.host {
            h.api.release_seq(gpu, sd.id);
        }
    }

    /// The sequence whose table is `table` ends now: drop its host pages and hand its running
    /// copies to the orphan list (their blocks are released when the copies finish).
    pub(crate) fn on_free(&mut self, table: &[u32], gpu: &dyn GpuBackend) {
        let Some(&b0) = table.first() else {
            return;
        };
        if self
            .seqs
            .get(&b0)
            .is_some_and(|sd| sd.matches(table, self.hole))
            && let Some(sd) = self.seqs.remove(&b0)
        {
            self.retire_seq(sd, gpu);
        }
    }
}
