// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/integration/vllm/lazy_offload_policy/eviction_aware.py:60-140 (commit 3aefae1, Apache-2.0; buffer stores, release them when their GPU blocks come within `horizon_steps` of the free queue's eviction head, capped per step, optional deadline); no code copied.
// Design translated from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08), vllm/v1/simple_kv_offload/manager.py:644-690 (keep the store ahead of the allocator: aim at a multiple of the per-step block demand), Copyright contributors to the vLLM project, Apache-2.0; no code copied.

//! Responsibility: the lazy store mode (`--kv-tier-store lazy`). A store only buffers the block
//! (key, tokens, device regions); nothing is copied while the block sits safely in the GPU cache.
//! Each [`TierInner::drain_lazy`] asks the caller's [`EvictionView`] where each buffered block
//! stands in the GPU cache's eviction order and starts the copy of the blocks that will be
//! evicted soon: rank below the near-head window, which is the larger of LMCache's
//! `horizon_steps x blocks_per_step` and vLLM's store-ahead target
//! `lazy_target_factor x blocks_per_step`, closest to the head first, never more than
//! `max_drain_per_step`, and always within the write-back byte budget and the staging cap.
//! A block whose GPU copy is gone is dropped (counted). A block the budget could not start stays
//! buffered for the next drain. With `max_deferral` set, a block buffered longer starts anyway.
//!
//! Address rule: a buffered block's device regions stay valid exactly as long as the view
//! reports it present ([`GpuBlockState::InUse`] or [`GpuBlockState::Queued`]); the radix cache
//! never moves a cached block, so the caller answers [`GpuBlockState::Gone`] the moment it was
//! evicted or recycled.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use anyhow::Result;

use super::fingerprint::BlockKey;
use super::index::Probe;
use super::writeback::{StoreItem, StoreReport};
use super::{Shared, TierInner};
use crate::gpu::{DevicePtr, GpuBackend};

/// Where a buffered block stands in the GPU prefix cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBlockState {
    /// No longer cached on the GPU (evicted or recycled): its device regions are invalid.
    Gone,
    /// Cached and referenced by a live sequence: not evictable yet.
    InUse,
    /// Cached and evictable; `rank` blocks will be evicted before it (0 = next).
    Queued { rank: usize },
}

/// The GPU cache's eviction order, as the glue sees it.
pub trait EvictionView {
    fn state(&self, key: &BlockKey) -> GpuBlockState;
}

/// Per-drain demand signal from the glue.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DrainSignals {
    /// GPU blocks the cache hands out per scheduler step (a smoothed recent value).
    pub blocks_per_step: f64,
}

/// One buffered store.
#[derive(Debug)]
pub(crate) struct LazyEntry {
    key: BlockKey,
    tokens: Box<[u32]>,
    regions: Box<[(DevicePtr, usize)]>,
    at: Duration,
}

/// Buffered stores, in arrival order.
#[derive(Default)]
pub(crate) struct LazyQueue {
    entries: Vec<LazyEntry>,
    keys: HashSet<BlockKey>,
}

impl LazyQueue {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn pending_blocks(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.keys.clear();
    }

    fn push(&mut self, e: LazyEntry) {
        self.keys.insert(e.key);
        self.entries.push(e);
    }

    /// Remove and return the entries at `idx`, in the order of `idx` (no duplicates).
    fn take(&mut self, idx: &[usize]) -> Vec<LazyEntry> {
        let pos: HashMap<usize, usize> = idx.iter().enumerate().map(|(p, &i)| (i, p)).collect();
        let mut taken: Vec<(usize, LazyEntry)> = Vec::with_capacity(pos.len());
        let mut kept = Vec::with_capacity(self.entries.len());
        for (i, e) in self.entries.drain(..).enumerate() {
            match pos.get(&i) {
                Some(&p) => {
                    self.keys.remove(&e.key);
                    taken.push((p, e));
                }
                None => kept.push(e),
            }
        }
        self.entries = kept;
        taken.sort_by_key(|(p, _)| *p);
        taken.into_iter().map(|(_, e)| e).collect()
    }
}

impl TierInner {
    /// Lazy store: buffer every absent block.
    pub(crate) fn admit_lazy(&mut self, sh: &Shared, items: &[StoreItem<'_>]) -> StoreReport {
        let mut report = StoreReport::default();
        let cap = (sh.cfg.budget_bytes / sh.block_bytes.max(1) as u64) as usize;
        let now = sh.clock.now();
        for (i, it) in items.iter().enumerate() {
            match self.index.probe(&it.key, it.tokens) {
                Probe::Mismatch => report.dropped.push(i),
                Probe::Ready | Probe::Staging => report.present.push(i),
                Probe::Absent if self.lazy.keys.contains(&it.key) => report.present.push(i),
                Probe::Absent if self.lazy.entries.len() >= cap.max(1) => report.dropped.push(i),
                Probe::Absent => {
                    self.lazy.push(LazyEntry {
                        key: it.key,
                        tokens: it.tokens.into(),
                        regions: it.regions.into(),
                        at: now,
                    });
                    report.buffered.push(i);
                }
            }
        }
        sh.stats.add_writeback_dropped(report.dropped.len() as u64);
        report
    }

    /// Start the copies of buffered blocks that approach eviction. Returns the started keys.
    pub(crate) fn drain_lazy(
        &mut self,
        sh: &Shared,
        gpu: &dyn GpuBackend,
        signals: DrainSignals,
        view: &dyn EvictionView,
    ) -> Result<Vec<BlockKey>> {
        if self.lazy.entries.is_empty() {
            return Ok(Vec::new());
        }
        let now = sh.clock.now();
        let demand = if signals.blocks_per_step.is_finite() {
            signals.blocks_per_step.max(0.0)
        } else {
            0.0
        };
        let horizon = sh.opts.horizon_steps * demand;
        let target =
            ((sh.opts.lazy_target_factor * demand).ceil() as usize).min(sh.opts.max_drain_per_step);
        let states: Vec<GpuBlockState> = self
            .lazy
            .entries
            .iter()
            .map(|e| view.state(&e.key))
            .collect();
        let gone: Vec<usize> = (0..states.len())
            .filter(|&i| states[i] == GpuBlockState::Gone)
            .collect();
        let states: Vec<GpuBlockState> = if gone.is_empty() {
            states
        } else {
            sh.stats
                .add_writeback_dropped(self.lazy.take(&gone).len() as u64);
            states
                .into_iter()
                .filter(|s| *s != GpuBlockState::Gone)
                .collect()
        };
        // The near-head window: LMCache's horizon or vLLM's store-ahead target, the larger.
        let window = horizon.max(target as f64);
        let mut urgent: Vec<(usize, usize)> = Vec::new();
        for (i, (e, state)) in self.lazy.entries.iter().zip(&states).enumerate() {
            let overdue = sh
                .opts
                .max_deferral
                .is_some_and(|d| now.saturating_sub(e.at) >= d);
            match *state {
                GpuBlockState::Gone => {}
                GpuBlockState::InUse => {
                    if overdue {
                        urgent.push((usize::MAX, i));
                    }
                }
                GpuBlockState::Queued { rank } => {
                    if overdue || (rank as f64) < window {
                        urgent.push((rank, i));
                    }
                }
            }
        }
        urgent.sort_unstable();
        let pick: Vec<usize> = urgent
            .iter()
            .take(sh.opts.max_drain_per_step)
            .map(|&(_, i)| i)
            .collect();
        if pick.is_empty() {
            return Ok(Vec::new());
        }
        let taken = self.lazy.take(&pick);
        // A buffered key that meanwhile became present (or conflicting) needs no copy.
        let todo: Vec<(usize, StoreItem<'_>)> = taken
            .iter()
            .enumerate()
            .filter(|(_, e)| self.index.probe(&e.key, &e.tokens) == Probe::Absent)
            .map(|(j, e)| {
                (
                    j,
                    StoreItem {
                        key: e.key,
                        tokens: &e.tokens,
                        regions: &e.regions,
                    },
                )
            })
            .collect();
        let (started, dropped) = match self.start_copies(sh, gpu, &todo) {
            Ok(r) => r,
            Err(e) => {
                drop(todo);
                for t in taken {
                    self.lazy.push(t);
                }
                return Err(e);
            }
        };
        let started_keys: Vec<BlockKey> = started.iter().map(|&j| taken[j].key).collect();
        drop(todo);
        let retry: HashSet<usize> = dropped.into_iter().collect();
        for (j, e) in taken.into_iter().enumerate() {
            if retry.contains(&j) {
                self.lazy.push(e);
            }
        }
        Ok(started_keys)
    }
}
