// SPDX-License-Identifier: AGPL-3.0-only
// Design translated from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08), vllm/v1/kv_offload/cpu/gpu_worker.py:543-731 (H2D on a dedicated stream with source access order "any", one event per transfer, completion polled), Copyright contributors to the vLLM project, Apache-2.0; no code copied.
// Design after LMCache lmcache/v1/distributed/l1_manager/base.py:132-188 and storage_controllers/prefetch_controller.py:1114-1160 (commit 3aefae1, Apache-2.0; read locks with expiry, in-flight admission watermark); no code copied.
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: restores (host tier -> fresh GPU blocks). [`TierInner::begin_restore`] leases
//! every source entry, fences the copy stream behind the compute stream (the destination blocks
//! may still be read there by their previous owner), enqueues one batched H2D with source access
//! order "any" and records the completion event. [`TierInner::poll_restore`] answers
//! [`RestoreState::Ready`] only after that event completed, and only then drops the leases, so a
//! leased run is never evicted, freed or overwritten while the copy engine reads it.
//!
//! Lifetime: a record lives until its ticket is released AND its copy finished. A ticket that is
//! never released is reclaimed `lease_ttl` after its copy finished (an abandoned request can not
//! pin host memory), but a lease is never dropped while its copy may still run. The in-flight
//! cap (`inflight_fraction` of the budget, LMCache's prefetch watermark) bounds leased bytes.

use std::time::Duration;

use anyhow::{Result, bail};

use super::events::Timer;
use super::fingerprint::BlockKey;
use super::index::EntryState;
use super::{Shared, TierInner};
use crate::gpu::{CopyEntry, DevicePtr, GpuBackend};
use crate::kv_cache::block_io;

/// State of one restore (contract 3.7.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestoreState {
    /// The copy may still run: the destination blocks must not be read yet.
    Pending,
    /// The copy completed: the destination blocks hold the stored bytes.
    Ready,
    /// The copy failed (or the ticket is unknown): free the destination blocks and recompute.
    Failed,
}

/// A started restore. Hand it back with `KvHostTier::release` once done with it.
#[derive(Debug, PartialEq, Eq)]
pub struct RestoreTicket {
    id: u64,
    event: u64,
    blocks: usize,
    tokens: usize,
}

impl RestoreTicket {
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The completion event on the copy stream. A compute stream may `stream_wait_event` on it
    /// instead of polling (the synchronous path) before it reads the destination blocks.
    pub fn event(&self) -> u64 {
        self.event
    }

    /// Blocks being restored.
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    /// Tokens covered by the restored blocks.
    pub fn tokens(&self) -> usize {
        self.tokens
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Running,
    Done(RestoreState),
}

/// The tier's side of one restore.
pub(crate) struct RestoreRecord {
    keys: Vec<BlockKey>,
    event: u64,
    timer: Timer,
    bytes: u64,
    tokens: usize,
    outcome: Outcome,
    released: bool,
    finished_at: Option<Duration>,
}

impl TierInner {
    pub(crate) fn begin_restore(
        &mut self,
        sh: &Shared,
        gpu: &dyn GpuBackend,
        keys: &[BlockKey],
        dst: &[&[(DevicePtr, usize)]],
    ) -> Result<RestoreTicket> {
        if keys.is_empty() || keys.len() != dst.len() {
            bail!(
                "kv_host_tier restore: {} keys for {} destination blocks",
                keys.len(),
                dst.len()
            );
        }
        let bb = sh.block_bytes as u64;
        let new_bytes = keys
            .iter()
            .filter(|k| self.index.get(k).is_some_and(|e| e.leases == 0))
            .count() as u64
            * bb;
        let cap = sh.inflight_cap();
        let leased = self.index.leased_bytes();
        if leased > 0 && leased + new_bytes > cap {
            bail!(
                "kv_host_tier restore: {new_bytes} more leased bytes would pass the in-flight cap \
                 ({leased} of {cap} leased)"
            );
        }
        let mut tokens = 0usize;
        let mut entries = Vec::new();
        for (k, regions) in keys.iter().zip(dst) {
            let Some(e) = self.index.get(k) else {
                bail!(
                    "kv_host_tier restore: block {:x} is not in host RAM",
                    k.short()
                );
            };
            if e.state != EntryState::Ready {
                bail!(
                    "kv_host_tier restore: block {:x} is still being written",
                    k.short()
                );
            }
            let total: usize = regions.iter().map(|r| r.1).sum();
            if total != sh.block_bytes {
                bail!(
                    "kv_host_tier restore: destination of {total} bytes for a {}-byte block",
                    sh.block_bytes
                );
            }
            let layout_differs = self.layout.as_ref().is_some_and(|l| {
                l.len() != regions.len() || l.iter().zip(regions.iter()).any(|(a, b)| *a != b.1)
            });
            if layout_differs {
                bail!(
                    "kv_host_tier restore: destination region layout differs from the stored one"
                );
            }
            let mut off = 0usize;
            for &(dev, bytes) in regions.iter() {
                entries.push(CopyEntry {
                    dev,
                    host: e.run.ptr + off,
                    bytes,
                });
                off += bytes;
            }
            tokens += e.tokens.len();
        }
        for k in keys {
            self.index.lease(k);
        }
        match self.submit_h2d(sh, gpu, &entries) {
            Ok((event, timer)) => {
                self.policy.touch_batch(keys);
                let id = self.next_id();
                self.restores.insert(
                    id,
                    RestoreRecord {
                        keys: keys.to_vec(),
                        event,
                        timer,
                        bytes: bb * keys.len() as u64,
                        tokens,
                        outcome: Outcome::Running,
                        released: false,
                        finished_at: None,
                    },
                );
                Ok(RestoreTicket {
                    id,
                    event,
                    blocks: keys.len(),
                    tokens,
                })
            }
            Err(e) => {
                for k in keys {
                    self.index.unlease(k);
                }
                sh.stats.add_restore_failed();
                Err(e)
            }
        }
    }

    /// Fence, enqueue the H2D, record the event. On error the copy stream was synchronized.
    fn submit_h2d(
        &mut self,
        sh: &Shared,
        gpu: &dyn GpuBackend,
        entries: &[CopyEntry],
    ) -> Result<(u64, Timer)> {
        let stream = sh.copy_stream;
        self.fence(gpu, stream)?;
        let mut timer = self.events.start_timer(gpu, stream, sh.clock.now());
        let submitted = block_io::submit_h2d(gpu, entries, stream, true).and_then(|()| {
            timer.stop(gpu, stream);
            let ev = self.events.take(gpu)?;
            match gpu.record_event(ev, stream) {
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
                if let Err(se) = gpu.synchronize(stream) {
                    tracing::warn!("kv_host_tier: copy stream sync after a failed restore: {se:#}");
                }
                Err(e.context("kv_host_tier: enqueueing the restore copy"))
            }
        }
    }

    /// Advance one record: poll its event while it runs, finish it when the event completed.
    fn advance(&mut self, sh: &Shared, gpu: &dyn GpuBackend, id: u64) -> Option<RestoreState> {
        let rec = self.restores.get_mut(&id)?;
        if let Outcome::Done(s) = rec.outcome {
            return Some(s);
        }
        let state = match gpu.event_query(rec.event) {
            Ok(false) => return Some(RestoreState::Pending),
            Ok(true) => {
                let secs = rec.timer.elapsed_secs(gpu, sh.clock.now());
                sh.stats
                    .add_restore(rec.keys.len() as u64, rec.tokens as u64, secs);
                self.cost.observe_copy(rec.bytes as f64, secs);
                self.events.give(rec.event);
                RestoreState::Ready
            }
            Err(e) => {
                tracing::warn!("kv_host_tier: restore {id} failed: {e:#}");
                if let Err(se) = gpu.synchronize(sh.copy_stream) {
                    tracing::warn!("kv_host_tier: copy stream sync after a failed restore: {se:#}");
                }
                rec.timer.discard(gpu);
                let _ = gpu.destroy_event(rec.event);
                sh.stats.add_restore_failed();
                RestoreState::Failed
            }
        };
        rec.outcome = Outcome::Done(state);
        rec.finished_at = Some(sh.clock.now());
        let keys = std::mem::take(&mut rec.keys);
        let released = rec.released;
        for k in &keys {
            self.index.unlease(k);
        }
        if released {
            self.restores.remove(&id);
        }
        Some(state)
    }

    pub(crate) fn poll_restore(
        &mut self,
        sh: &Shared,
        gpu: &dyn GpuBackend,
        id: u64,
    ) -> Result<RestoreState> {
        match self.advance(sh, gpu, id) {
            Some(s) => Ok(s),
            None => {
                tracing::debug!("kv_host_tier: poll of unknown or reclaimed restore {id}");
                Ok(RestoreState::Failed)
            }
        }
    }

    /// The caller is done with ticket `id`. A finished record goes now; a running one keeps its
    /// leases until its copy completes (the next sweep or poll reclaims it).
    pub(crate) fn release_restore(&mut self, id: u64) {
        let Some(rec) = self.restores.get_mut(&id) else {
            return;
        };
        rec.released = true;
        if matches!(rec.outcome, Outcome::Done(_)) {
            self.restores.remove(&id);
        }
    }

    /// Finish every restore whose copy completed and reclaim released or expired records.
    pub(crate) fn sweep_restores(&mut self, sh: &Shared, gpu: &dyn GpuBackend) {
        let ids: Vec<u64> = self.restores.keys().copied().collect();
        let now = sh.clock.now();
        for id in ids {
            let _ = self.advance(sh, gpu, id);
            let expired = self.restores.get(&id).is_some_and(|r| {
                matches!(r.outcome, Outcome::Done(_))
                    && r.finished_at
                        .is_some_and(|t| now.saturating_sub(t) >= sh.opts.lease_ttl)
            });
            if expired {
                tracing::debug!("kv_host_tier: reclaimed restore {id}, never released");
                self.restores.remove(&id);
            }
        }
    }

    /// Restores whose copy may still run (tests and teardown).
    pub(crate) fn running_restores(&self) -> usize {
        self.restores
            .values()
            .filter(|r| r.outcome == Outcome::Running)
            .count()
    }
}
