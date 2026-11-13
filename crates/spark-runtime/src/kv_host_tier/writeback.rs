// SPDX-License-Identifier: AGPL-3.0-only
// Design translated from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08), vllm/v1/kv_offload/cpu/gpu_worker.py:543-731 (D2H on a dedicated stream that waits on the compute stream, one event per transfer, completion polled) and vllm/v1/simple_kv_offload/manager.py (offload_prompt_only, per-step store cap), Copyright contributors to the vLLM project, Apache-2.0; no code copied. Changes: Atlas block regions as the copy unit, one batched `copy_batch_d2h_async` per store call, staging entries in the tier index.
// Design after LMCache lmcache/v1/distributed/l1_manager/base.py:132-188 (commit 3aefae1, Apache-2.0; publish only after the write completed); no code copied.

//! Responsibility: the write-back path (host tier store). [`TierInner::start_copies`] turns
//! store requests into one batched D2H on the copy stream: per block it charges the per-step
//! byte budget and the staging cap, makes room (watermark eviction), allocates a host run and
//! adds a STAGING entry; then it fences the copy stream behind the source stream, enqueues every
//! region copy at once and records the batch's completion event. [`TierInner::poll_writebacks`]
//! publishes a batch (STAGING -> READY, into the eviction policy) only after its event
//! completed, and feeds the copy's duration to the cost model.
//!
//! Failure rule: a submission that fails synchronizes the copy stream (so no DMA can still write
//! a run), removes the batch's entries and frees their runs. Device memory is only ever read
//! here, so every GPU block stays valid.

use std::collections::{HashSet, VecDeque};

use anyhow::{Result, bail};

use super::events::Timer;
use super::fingerprint::BlockKey;
use super::index::Probe;
use super::{Shared, TierInner};
use crate::gpu::{CopyEntry, DevicePtr, GpuBackend};
use crate::kv_cache::block_io;

/// One block handed to the tier: key, tokens and its device regions (K then V per layer).
#[derive(Clone, Copy, Debug)]
pub(crate) struct StoreItem<'a> {
    pub(crate) key: BlockKey,
    pub(crate) tokens: &'a [u32],
    pub(crate) regions: &'a [(DevicePtr, usize)],
}

/// Per-block outcome of one store call (indices into the call's inputs).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StoreReport {
    /// D2H copy started. Each key comes back once from `poll_writebacks`.
    pub started: Vec<usize>,
    /// Lazy mode: buffered, the copy starts later in `drain_lazy`.
    pub buffered: Vec<usize>,
    /// Already stored (or buffered) with the same tokens: nothing to do.
    pub present: Vec<usize>,
    /// Not stored: step budget, staging cap, no host room, or a key held with other tokens.
    pub dropped: Vec<usize>,
}

impl StoreReport {
    /// Blocks the tier took on (started or buffered).
    pub fn accepted(&self) -> usize {
        self.started.len() + self.buffered.len()
    }
}

/// One in-flight D2H batch.
pub(crate) struct WritebackBatch {
    pub(crate) id: u64,
    pub(crate) keys: Vec<BlockKey>,
    pub(crate) event: u64,
    pub(crate) timer: Timer,
    pub(crate) bytes: u64,
}

impl TierInner {
    /// Every block must have the tier's size; the first store fixes the region layout.
    pub(crate) fn check_layout(
        &mut self,
        sh: &Shared,
        regions: &[(DevicePtr, usize)],
    ) -> Result<()> {
        let total: usize = regions.iter().map(|r| r.1).sum();
        if total != sh.block_bytes {
            bail!(
                "kv_host_tier: block of {total} bytes in {} regions, the tier holds {}-byte blocks",
                regions.len(),
                sh.block_bytes
            );
        }
        if regions.iter().any(|r| r.0.is_null() && r.1 > 0) {
            bail!("kv_host_tier: block region with a null device pointer");
        }
        let sizes: Vec<usize> = regions.iter().map(|r| r.1).collect();
        if self.layout.is_none() {
            self.layout = Some(sizes);
            return Ok(());
        }
        if self.layout.as_ref() != Some(&sizes) {
            bail!(
                "kv_host_tier: block region layout {sizes:?} differs from the first store's {:?}",
                self.layout
            );
        }
        Ok(())
    }

    /// Eager store: classify, then start one batched copy for every block that fits.
    pub(crate) fn store_eager(
        &mut self,
        sh: &Shared,
        gpu: &dyn GpuBackend,
        items: &[StoreItem<'_>],
    ) -> Result<StoreReport> {
        let mut report = StoreReport::default();
        let mut todo: Vec<(usize, StoreItem<'_>)> = Vec::with_capacity(items.len());
        let mut seen: HashSet<BlockKey> = HashSet::with_capacity(items.len());
        for (i, it) in items.iter().enumerate() {
            let dup = !seen.insert(it.key);
            match self.index.probe(&it.key, it.tokens) {
                Probe::Absent if !dup => todo.push((i, *it)),
                Probe::Absent | Probe::Ready | Probe::Staging => report.present.push(i),
                Probe::Mismatch => report.dropped.push(i),
            }
        }
        let (started, dropped) = self.start_copies(sh, gpu, &todo)?;
        report.started = started;
        report.dropped.extend(dropped);
        report.dropped.sort_unstable();
        let n_dropped = report.dropped.len() as u64;
        sh.stats.add_writeback_dropped(n_dropped);
        Ok(report)
    }

    /// Bytes this call may still start under the per-step write-back budget.
    fn step_budget_left(&self, sh: &Shared) -> u64 {
        let per_step = sh.cfg.writeback_bytes_per_step;
        if self.step_mode {
            per_step.saturating_sub(self.step_spent)
        } else {
            per_step
        }
    }

    /// Start the D2H copy of `todo` (absent keys, valid layouts). Returns the input indices
    /// started and the ones dropped. Does not count drops on `/metrics` (callers do).
    pub(crate) fn start_copies(
        &mut self,
        sh: &Shared,
        gpu: &dyn GpuBackend,
        todo: &[(usize, StoreItem<'_>)],
    ) -> Result<(Vec<usize>, Vec<usize>)> {
        let bb = sh.block_bytes as u64;
        let mut budget = self.step_budget_left(sh);
        let first_of_step = !self.step_mode || self.step_spent == 0;
        let cap = sh.inflight_cap();
        let batch = self.next_id();
        let mut started: Vec<(usize, StoreItem<'_>, usize)> = Vec::with_capacity(todo.len());
        let mut dropped = Vec::new();
        for (i, it) in todo {
            let over_budget = bb > budget && !(first_of_step && started.is_empty());
            let staging = self.index.staging_bytes();
            let over_cap = staging > 0 && staging + bb > cap;
            if bb == 0 || over_budget || over_cap {
                dropped.push(*i);
                continue;
            }
            let Some(run) = self.place(sh, bb) else {
                dropped.push(*i);
                continue;
            };
            match self.index.insert_staging(it.key, it.tokens, run, bb, batch) {
                Ok(()) => {
                    budget = budget.saturating_sub(bb);
                    started.push((*i, *it, run.ptr));
                }
                Err(run) => {
                    sh.arena.free(run);
                    dropped.push(*i);
                }
            }
        }
        if started.is_empty() {
            return Ok((Vec::new(), dropped));
        }
        if let Err(e) = self.submit_d2h(sh, gpu, batch, &started) {
            for (_, it, _) in &started {
                if let Some(entry) = self.index.remove_staging(&it.key, batch) {
                    sh.arena.free(entry.run);
                }
            }
            return Err(e);
        }
        let bytes = bb * started.len() as u64;
        if self.step_mode {
            self.step_spent += bytes;
        }
        Ok((started.into_iter().map(|(i, _, _)| i).collect(), dropped))
    }

    /// Fence, enqueue every region copy of `blocks` and record the batch event.
    fn submit_d2h(
        &mut self,
        sh: &Shared,
        gpu: &dyn GpuBackend,
        batch: u64,
        blocks: &[(usize, StoreItem<'_>, usize)],
    ) -> Result<()> {
        let mut entries = Vec::with_capacity(blocks.iter().map(|b| b.1.regions.len()).sum());
        for (_, it, host) in blocks {
            let mut off = 0usize;
            for &(dev, bytes) in it.regions {
                entries.push(CopyEntry {
                    dev,
                    host: host + off,
                    bytes,
                });
                off += bytes;
            }
        }
        let stream = sh.copy_stream;
        if let Err(e) = self.fence(gpu, stream) {
            return Err(e.context("kv_host_tier: fencing the copy stream for a write-back"));
        }
        let mut timer = self.events.start_timer(gpu, stream, sh.clock.now());
        match self.enqueue(gpu, &entries, stream, &mut timer) {
            Ok(event) => {
                self.writebacks.push_back(WritebackBatch {
                    id: batch,
                    keys: blocks.iter().map(|b| b.1.key).collect(),
                    event,
                    timer,
                    bytes: sh.block_bytes as u64 * blocks.len() as u64,
                });
                Ok(())
            }
            Err(e) => {
                timer.discard(gpu);
                if let Err(se) = gpu.synchronize(stream) {
                    tracing::warn!("kv_host_tier: copy stream sync after a failed store: {se:#}");
                }
                Err(e.context("kv_host_tier: enqueueing the write-back copy"))
            }
        }
    }

    /// Enqueue `entries` D2H, stop the timer and record a completion event (returned).
    fn enqueue(
        &mut self,
        gpu: &dyn GpuBackend,
        entries: &[CopyEntry],
        stream: u64,
        timer: &mut Timer,
    ) -> Result<u64> {
        block_io::submit_d2h(gpu, entries, stream)?;
        timer.stop(gpu, stream);
        let ev = self.events.take(gpu)?;
        if let Err(e) = gpu.record_event(ev, stream) {
            self.events.give(ev);
            return Err(e);
        }
        Ok(ev)
    }

    /// Publish completed write-back batches; return the keys of every finished batch.
    pub(crate) fn poll_writebacks(&mut self, sh: &Shared, gpu: &dyn GpuBackend) -> Vec<BlockKey> {
        let mut done = Vec::new();
        let mut pending = VecDeque::with_capacity(self.writebacks.len());
        while let Some(mut b) = self.writebacks.pop_front() {
            match gpu.event_query(b.event) {
                Ok(false) => pending.push_back(b),
                Ok(true) => {
                    let published: Vec<BlockKey> = b
                        .keys
                        .iter()
                        .copied()
                        .filter(|k| self.index.publish(k, b.id))
                        .collect();
                    self.policy.insert_batch(&published);
                    sh.stats.add_writeback_blocks(published.len() as u64);
                    let secs = b.timer.elapsed_secs(gpu, sh.clock.now());
                    self.cost.observe_copy(b.bytes as f64, secs);
                    self.events.give(b.event);
                    done.extend(b.keys);
                }
                Err(e) => {
                    tracing::warn!("kv_host_tier: write-back batch {} failed: {e:#}", b.id);
                    if let Err(se) = gpu.synchronize(sh.copy_stream) {
                        tracing::warn!(
                            "kv_host_tier: copy stream sync after a failed copy: {se:#}"
                        );
                    }
                    for k in &b.keys {
                        if let Some(entry) = self.index.remove_staging(k, b.id) {
                            sh.arena.free(entry.run);
                        }
                    }
                    sh.stats.add_writeback_dropped(b.keys.len() as u64);
                    b.timer.discard(gpu);
                    let _ = gpu.destroy_event(b.event);
                    done.extend(b.keys);
                }
            }
        }
        self.writebacks = pending;
        done
    }
}
