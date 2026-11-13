// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/distributed/l1_manager/base.py, storage_controllers/eviction_controller.py and storage_controllers/prefetch_controller.py (commit 3aefae1, Apache-2.0); no code copied.
// Design translated from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08), vllm/v1/kv_offload/cpu/gpu_worker.py and vllm/v1/simple_kv_offload/manager.py, Copyright contributors to the vLLM project, Apache-2.0; no code copied. Changes: one in-process Rust tier over a pinned page pool, event-owned DMA on a dedicated copy stream, Atlas paged-KV block regions as the copy unit.
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the host-RAM KV tier core (port plan contract 3.7.5): a pinned host copy of
//! prefix-cached paged-KV blocks, keyed by [`BlockKey`], that a later request restores into
//! fresh GPU blocks by one batched H2D instead of re-running prefill.
//!
//! This crate never reads a `[defaults]` row. The spark-model glue (lane L26b,
//! `model/kv_tier_glue.rs`) resolves the `kv_host_tier` row, builds a [`TierConfig`] from the
//! serve flags (`port_apply` publishes it with [`config::set`]), creates one [`KvHostTier`] per
//! model and drives it. With no tier created, nothing in this module runs.
//!
//! # Lifecycle of one block
//!
//! 1. **Store** ([`KvHostTier::store`]): the glue hands over newly cached FULL blocks (key,
//!    tokens, device regions). Eager mode allocates a host page run per block, makes the copy
//!    stream wait on the source (compute) stream, enqueues one batched D2H and records an event.
//!    The entry is STAGING: invisible to [`KvHostTier::lookup`] and never evicted. Lazy mode
//!    (`--kv-tier-store lazy`) only buffers the request; [`KvHostTier::drain_lazy`] releases it
//!    when its GPU blocks come within the eviction horizon (after LMCache `eviction_aware.py`).
//! 2. **Publish** ([`KvHostTier::poll_writebacks`]): once the D2H event completed the entry
//!    becomes READY and enters the eviction policy. The returned keys tell the glue which
//!    source blocks it may release (it holds one extra block reference per started copy, so the
//!    radix tree can never recycle a block the DMA still reads).
//! 3. **Lookup** ([`KvHostTier::lookup`]): leading-hit count over READY entries whose stored
//!    tokens equal the probe's tokens (a forged or colliding key with other tokens misses).
//! 4. **Restore** ([`KvHostTier::begin_restore`], [`KvHostTier::poll`], [`KvHostTier::release`]):
//!    every source entry takes a read lease, one batched H2D (`src_any_order`) fills the caller's
//!    freshly allocated blocks, and the leases drop when the H2D event completed. A leased entry
//!    is never evicted, freed or overwritten.
//! 5. **Evict** (inside store): watermark batch eviction (trigger 0.8 of the budget, evict down to
//!    0.6) through the configured policy ([`TierPolicy::Lru`] tail-first, [`TierPolicy::Arc`],
//!    [`TierPolicy::Heat`]); an evicted entry is demoted to the cold store when one is set.
//!
//! # Invariants (tested in `tests/`)
//!
//! * an entry is visible only after its D2H event completed (staging -> ready);
//! * a restore holds a read lease on every source entry until its H2D event completed; leased and
//!   staging entries are never evicted or freed;
//! * a ticket that is never released is reclaimed after the lease TTL once its copy completed;
//! * publish happens only after the whole batch's copy completed;
//! * the key includes the model fingerprint (namespace), so a stale model misses;
//! * capacity keeps headroom: eviction triggers at the high watermark, staging and leased bytes
//!   are each capped at half the budget;
//! * a failed submission leaves every GPU block valid and every host run returned.
//!
//! Locking: one `parking_lot::Mutex` around the mutable state. Every GPU call made under it
//! only enqueues work or polls an event; none blocks on the device except the best-effort
//! `synchronize` after a failed submission, [`KvHostTier::wait`] and `shutdown`. Methods beyond
//! the contract (step budget, lazy drain, the cost gate, teardown) live in `control.rs`.

mod arena;
mod clock;
pub mod cold_store;
pub mod config;
mod control;
pub mod cost;
/// Disk tier and warm restart (lane L27): implements [`ColdStore`].
pub mod disk;
mod events;
mod evict;
pub mod fingerprint;
mod index;
pub mod lazy;
/// Page-granular demote and fetch for a live sequence's middle blocks (lane L36).
pub mod pages;
mod policy;
pub mod restore;
pub mod stats;
mod writeback;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use anyhow::{Result, bail};
use parking_lot::Mutex;

use crate::gpu::{DevicePtr, GpuBackend};
use crate::kv_cache::block_io::BlockRegions;
use crate::pinned_pool::PinnedPool;

pub use arena::{HostArena, HostRun, PoolArena};
pub use clock::{Clock, MonotonicClock};
pub use cold_store::ColdStore;
pub use config::{StoreMode, TierConfig, TierOptions, TierPolicy};
pub use cost::GateDecision;
pub use fingerprint::BlockKey;
pub use lazy::{DrainSignals, EvictionView, GpuBlockState};
pub use restore::{RestoreState, RestoreTicket};
pub use stats::{TierSnapshot, TierStats, render_prometheus};
pub use writeback::StoreReport;

/// What [`pages::PageApi::new`] receives from the tier (lane L36 builds its page API on the
/// tier's host arena and copy stream, so page traffic shares the tier's budget and DMA order).
pub struct PageDeps {
    pub arena: Arc<dyn HostArena>,
    pub block_bytes: usize,
    pub copy_stream: u64,
}

/// Immutable parts of a tier, shared by every operation.
pub(crate) struct Shared {
    pub(crate) cfg: TierConfig,
    pub(crate) opts: TierOptions,
    pub(crate) block_bytes: usize,
    pub(crate) copy_stream: u64,
    pub(crate) arena: Arc<dyn HostArena>,
    pub(crate) stats: Arc<TierStats>,
    pub(crate) clock: Arc<dyn Clock>,
}

impl Shared {
    /// Bytes the tier may hold in staging (in-flight D2H) or in leases (in-flight H2D).
    pub(crate) fn inflight_cap(&self) -> u64 {
        (self.cfg.budget_bytes as f64 * self.opts.inflight_fraction) as u64
    }
}

/// Mutable state, behind the tier's one lock.
pub(crate) struct TierInner {
    pub(crate) index: index::Index,
    pub(crate) policy: policy::Policy,
    /// Region sizes of one block (all layers, K then V), fixed by the first store.
    pub(crate) layout: Option<Vec<usize>>,
    pub(crate) writebacks: VecDeque<writeback::WritebackBatch>,
    pub(crate) lazy: lazy::LazyQueue,
    pub(crate) restores: HashMap<u64, restore::RestoreRecord>,
    pub(crate) step_spent: u64,
    pub(crate) step_mode: bool,
    pub(crate) cost: cost::CostModel,
    pub(crate) events: events::EventPool,
    pub(crate) next_id: u64,
    pub(crate) cold: Option<Arc<dyn ColdStore>>,
    pub(crate) cold_keys: HashSet<BlockKey>,
    pub(crate) namespace: u64,
    pub(crate) source_stream: Option<u64>,
    pub(crate) fence: Option<u64>,
    pub(crate) warned_demote: bool,
}

impl TierInner {
    pub(crate) fn next_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    /// Make `copy_stream` wait for everything already enqueued on the source stream (the KV a
    /// D2H reads was written there; the blocks an H2D overwrites may still be read there).
    pub(crate) fn fence(&mut self, gpu: &dyn GpuBackend, copy_stream: u64) -> Result<()> {
        let src = self.source_stream.unwrap_or_else(|| gpu.default_stream());
        if src == copy_stream {
            return Ok(());
        }
        let ev = match self.fence {
            Some(e) => e,
            None => {
                let e = gpu.create_event()?;
                self.fence = Some(e);
                e
            }
        };
        gpu.record_event(ev, src)?;
        gpu.stream_wait_event(copy_stream, ev)
    }

    /// Refresh the byte gauges after a state change.
    pub(crate) fn publish_gauges(&self, sh: &Shared) {
        sh.stats.set_resident(self.index.used_bytes());
    }
}

/// The host-RAM KV tier. One per model; `Send + Sync`, every method takes `&self`.
pub struct KvHostTier {
    sh: Shared,
    inner: Mutex<TierInner>,
    pages: pages::PageApi,
}

impl KvHostTier {
    /// Contract 3.7.5 constructor: a tier over `pool` (lane L25's pinned pool), whose blocks are
    /// `block_bytes` long (all layers, K then V), copying on `copy_stream` (a non-legacy stream
    /// from `GpuBackend::create_copy_stream`). Tunables come from [`TierOptions::from_globals`]
    /// (the store mode published by `port_apply`); the tier's `/metrics` series are registered.
    pub fn new(
        cfg: TierConfig,
        pool: Arc<PinnedPool>,
        block_bytes: usize,
        copy_stream: u64,
    ) -> Self {
        let arena: Arc<dyn HostArena> = Arc::new(PoolArena::new(pool));
        Self::with_arena(
            cfg,
            arena,
            block_bytes,
            copy_stream,
            TierOptions::from_globals(),
        )
    }

    /// A tier over any [`HostArena`] with explicit tunables.
    pub fn with_arena(
        cfg: TierConfig,
        arena: Arc<dyn HostArena>,
        block_bytes: usize,
        copy_stream: u64,
        opts: TierOptions,
    ) -> Self {
        Self::with_parts(
            cfg,
            arena,
            block_bytes,
            copy_stream,
            opts,
            Arc::new(MonotonicClock::new()),
        )
    }

    pub(crate) fn with_parts(
        cfg: TierConfig,
        arena: Arc<dyn HostArena>,
        block_bytes: usize,
        copy_stream: u64,
        opts: TierOptions,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let stats = Arc::new(TierStats::default());
        stats.set_capacity(cfg.budget_bytes);
        if opts.register_metrics {
            stats::register(stats.clone());
        }
        let capacity_blocks = if block_bytes == 0 {
            0
        } else {
            (cfg.budget_bytes / block_bytes as u64) as usize
        };
        let inner = TierInner {
            index: index::Index::new(),
            policy: policy::Policy::new(cfg.policy, capacity_blocks, opts.heat_halve_every),
            layout: None,
            writebacks: VecDeque::new(),
            lazy: lazy::LazyQueue::new(),
            restores: HashMap::new(),
            step_spent: 0,
            step_mode: false,
            cost: cost::CostModel::new(&opts),
            events: events::EventPool::new(),
            next_id: 0,
            cold: None,
            cold_keys: HashSet::new(),
            namespace: opts.namespace,
            source_stream: None,
            fence: None,
            warned_demote: false,
        };
        let pages = pages::PageApi::new(PageDeps {
            arena: arena.clone(),
            block_bytes,
            copy_stream,
        });
        Self {
            sh: Shared {
                cfg,
                opts,
                block_bytes,
                copy_stream,
                arena,
                stats,
                clock,
            },
            inner: Mutex::new(inner),
            pages,
        }
    }

    /// Queue newly cached full blocks for write-back. Returns how many were accepted (copy
    /// started in eager mode, buffered in lazy mode). See [`Self::store_detailed`].
    pub fn store(
        &self,
        gpu: &dyn GpuBackend,
        keys: &[BlockKey],
        tokens: &[&[u32]],
        regions: &[BlockRegions],
    ) -> Result<usize> {
        Ok(self.store_detailed(gpu, keys, tokens, regions)?.accepted())
    }

    /// [`Self::store`] with the per-block outcome. Every index in [`StoreReport::started`] comes
    /// back exactly once, as its key, from a later [`Self::poll_writebacks`]; a caller that
    /// protects the source blocks takes one reference per started key and drops it then.
    /// Buffered (lazy) blocks need no reference until [`Self::drain_lazy`] starts them.
    /// On `Err` nothing was queued and every GPU block is untouched.
    pub fn store_detailed(
        &self,
        gpu: &dyn GpuBackend,
        keys: &[BlockKey],
        tokens: &[&[u32]],
        regions: &[BlockRegions],
    ) -> Result<StoreReport> {
        if keys.len() != tokens.len() || keys.len() != regions.len() {
            bail!(
                "kv_host_tier store: {} keys, {} token chunks, {} region lists",
                keys.len(),
                tokens.len(),
                regions.len()
            );
        }
        let items: Vec<writeback::StoreItem<'_>> = keys
            .iter()
            .zip(tokens)
            .zip(regions)
            .map(|((k, t), r)| writeback::StoreItem {
                key: *k,
                tokens: t,
                regions: &r.regions,
            })
            .collect();
        let mut inner = self.inner.lock();
        for it in &items {
            if it.tokens.is_empty() {
                bail!(
                    "kv_host_tier store: block {:x} has no tokens",
                    it.key.short()
                );
            }
            inner.check_layout(&self.sh, it.regions)?;
        }
        let report = match self.sh.opts.store_mode {
            StoreMode::Eager => inner.store_eager(&self.sh, gpu, &items)?,
            StoreMode::Lazy => inner.admit_lazy(&self.sh, &items),
        };
        inner.publish_gauges(&self.sh);
        Ok(report)
    }

    /// Leading-hit count: how many of `keys` (in order) are READY with equal stored tokens.
    /// Takes no lease, changes no GPU state and no eviction order. A block demoted to the cold
    /// store is promoted back into host RAM here (a synchronous read) and counts as a hit.
    pub fn lookup(&self, keys: &[BlockKey], tokens: &[&[u32]]) -> usize {
        let mut inner = self.inner.lock();
        let mut hits = 0;
        // Hits already counted (keys[..covered]) are never evicted to make room for a promotion.
        let mut protect: HashSet<BlockKey> = HashSet::new();
        let mut covered = 0;
        for (i, (k, t)) in keys.iter().zip(tokens).enumerate() {
            match inner.index.probe(k, t) {
                index::Probe::Ready => hits += 1,
                index::Probe::Absent if inner.cold_keys.contains(k) => {
                    protect.extend(keys[covered..i].iter().copied());
                    covered = i;
                    if inner.promote_cold(&self.sh, k, t, &protect) {
                        hits += 1;
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        if hits > 0 {
            self.sh.stats.add_hit();
        } else {
            self.sh.stats.add_miss();
        }
        inner.publish_gauges(&self.sh);
        hits
    }

    /// Start restoring `keys` (all READY, from a [`Self::lookup`]) into `dst` (freshly allocated
    /// blocks, same region layout). Returns a ticket to [`Self::poll`] and then [`Self::release`].
    /// The compute stream must not read `dst` before the ticket is [`RestoreState::Ready`] (or
    /// before it waited on [`RestoreTicket::event`]). `Err`: nothing leased, nothing copied.
    /// Keep a restore within [`Self::max_restore_blocks`] (the in-flight cap).
    pub fn begin_restore(
        &self,
        gpu: &dyn GpuBackend,
        keys: &[BlockKey],
        dst: &[BlockRegions],
    ) -> Result<RestoreTicket> {
        let dst: Vec<&[(DevicePtr, usize)]> = dst.iter().map(|d| &d.regions[..]).collect();
        let mut inner = self.inner.lock();
        let t = inner.begin_restore(&self.sh, gpu, keys, &dst);
        inner.publish_gauges(&self.sh);
        t
    }

    /// Non-blocking state of a restore. `Ready` (or `Failed`) released the source leases.
    pub fn poll(&self, gpu: &dyn GpuBackend, t: &RestoreTicket) -> Result<RestoreState> {
        let mut inner = self.inner.lock();
        inner.poll_restore(&self.sh, gpu, t.id())
    }

    /// Block until the restore's copy completed, then [`Self::poll`] (the synchronous path).
    pub fn wait(&self, gpu: &dyn GpuBackend, t: &RestoreTicket) -> Result<RestoreState> {
        if let Err(e) = gpu.event_synchronize(t.event()) {
            tracing::warn!("kv_host_tier: waiting for restore {} failed: {e:#}", t.id());
        }
        self.poll(gpu, t)
    }

    /// Done with a ticket. A still-pending restore keeps its leases until its copy completes.
    pub fn release(&self, t: RestoreTicket) {
        self.inner.lock().release_restore(t.id());
    }

    /// Attach the cold store (lane L27's disk tier). Keys it already holds for this tier's
    /// namespace (set it first with [`Self::set_namespace`]) become promotable at once.
    pub fn set_cold_store(&self, s: Arc<dyn ColdStore>) {
        let mut inner = self.inner.lock();
        let ns = inner.namespace;
        match s.adopt_on_restart(ns) {
            Ok(keys) => {
                tracing::info!("kv_host_tier: cold store adopted {} block(s)", keys.len());
                inner.cold_keys.extend(keys);
            }
            Err(e) => tracing::warn!("kv_host_tier: cold store adopt failed: {e:#}"),
        }
        inner.cold = Some(s);
    }

    /// Page-granular API for live sequences (lane L36).
    pub fn pages(&self) -> &pages::PageApi {
        &self.pages
    }
}
