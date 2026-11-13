// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/cache_engine.py (store after the GPU cache insert, retrieve before the prefill) and lmcache/integration/vllm/vllm_v1_adapter.py (the hit counted in whole blocks, contiguous with the GPU match) (commit 3aefae1, Apache-2.0); no code copied.
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the host-RAM KV tier's model glue (port plan 3.7.5, lane L26b): the hooks
//! that copy newly cached prefix blocks into lane L26a's [`KvHostTier`] and the prefetch that
//! restores a prompt's host-held blocks into the radix tree before the normal prefix lookup.
//!
//! # When it runs
//!
//! Only when the `kv_host_tier` row is on AND `--kv-host-tier-gb` gave a budget (both published
//! by `port_apply` through `spark_runtime::kv_host_tier::config`) AND this model's prefix cache
//! is active (`--hermetic` closes it through the serve resolvers). With the row off every hook
//! returns after one atomic load, so every target that does not opt in behaves exactly as
//! before. The tier is built lazily, on the first hook that runs with the row on.
//!
//! # Store (write-through)
//!
//! After each `PrefixCache::insert*` of a prefill (`prefill_b/finalize_last.rs`,
//! `prefill_b/save_checkpoint.rs`) the blocks whose radix nodes that insert CREATED
//! (`InsertAcquired::blocks`) are stored: full blocks only, adapter 0 only (v1), prompt-only
//! (decode-time inserts are not hooked). Each started copy holds one extra block reference
//! until its D2H completed, so a radix eviction while the DMA reads the block can never hand
//! the block to another sequence ([`store`]).
//!
//! # Prefetch (restore)
//!
//! Before the normal lookup the scheduler calls [`TransformerModel::prefetch`] (or, with the
//! `kv_restore_async` row of lane L27, [`TransformerModel::restore_begin`] and parks the
//! request). It takes the GPU match `M`, walks the host keys for blocks `M..`, and restores
//! blocks `[M, d)` for the deepest `d` that the boundary rule allows ([`plan`]): on a hybrid
//! model the restore must end at the block holding an SSM snapshot (resident or tiered) past
//! `M`, and with no such snapshot nothing is loaded. The tier's cost gate, watermarks and
//! in-flight cap are lane L26a's. The restored blocks are allocated with
//! `alloc_block_evicting`, filled by one batched H2D, and re-inserted with
//! `PrefixCache::insert`, so the prefill's own lookup then sees an ordinary GPU hit.
//!
//! Children: `plan` (the pure boundary rule), `state` (the per-model slot, build and
//! teardown), `pool_gpu` (the pinned pool's handle to the model's backend), `store`
//! (write-through, reference bookkeeping, the lazy drain), `restore` (prefetch and the async
//! restore), `flags` (the serve flags' validation and publication).

pub mod flags;
mod plan;
mod pool_gpu;
mod restore;
mod state;
mod store;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_async;
#[cfg(test)]
mod tests_lazy;

use std::sync::Arc;

use anyhow::Result;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::kv_host_tier::{self, RestoreState, RestoreTicket};
use spark_runtime::prefix_cache::InsertAcquired;

pub use plan::PrefetchOutcome;
pub use state::TierState;

use super::types::TransformerModel;
use crate::traits::SequenceState;
use restore::Env;
use state::Live;

/// What one prefetch did (contract 3.7.5).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TierPrefetch {
    pub outcome: PrefetchOutcome,
    /// Prompt tokens the GPU prefix cache already held.
    pub gpu_tokens: usize,
    /// Tokens restored from the host tier (0 unless `outcome` is `Restored`).
    pub restored_tokens: usize,
    /// Blocks restored from the host tier.
    pub restored_blocks: usize,
}

impl TierPrefetch {
    pub(crate) fn nothing(outcome: PrefetchOutcome, gpu_tokens: usize) -> Self {
        Self {
            outcome,
            gpu_tokens,
            restored_tokens: 0,
            restored_blocks: 0,
        }
    }
}

impl TransformerModel {
    /// The serving tier, building it on first use. `None` (after one atomic load) whenever
    /// the row is off; `None` for good when the build was refused or failed.
    fn kv_tier_live(&self, kv: &PagedKvCache) -> Option<Arc<Live>> {
        if !kv_host_tier::config::row_enabled() {
            return None;
        }
        self.port.kv_tier.get_or_build(|| {
            state::build(
                self.gpu.as_ref(),
                &self.config,
                kv,
                self.prefix_cache.is_active(),
            )
        })
    }

    fn kv_tier_env<'a>(&'a self, kv: &'a mut PagedKvCache) -> Env<'a> {
        Env {
            gpu: self.gpu.as_ref(),
            kv,
            cache: self.prefix_cache.as_ref(),
            has_ssm: self.config.num_ssm_layers() > 0,
        }
    }

    /// Store hook, called right after each prefill `insert*` (one call per site). `acquired`
    /// is what that insert reported; `stream` is the stream the blocks' KV was written on.
    /// Never fails the prefill: a store error is logged and the blocks stay GPU-only.
    pub(in crate::model) fn kv_tier_store(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        acquired: &InsertAcquired,
        kv: &mut PagedKvCache,
        stream: u64,
    ) {
        if acquired.blocks.is_empty() {
            return;
        }
        let Some(live) = self.kv_tier_live(kv) else {
            return;
        };
        // `--high-speed-swap` keeps a parallel disk-id table the restore path cannot rebuild;
        // the tier stays out of HSS sequences.
        if !seq.disk_block_ids.is_empty() {
            return;
        }
        let gpu = self.gpu.as_ref();
        live.reap(gpu, kv);
        if let Err(e) = live.store_inserted(
            gpu,
            kv,
            tokens,
            &seq.block_table,
            seq.adapter_id,
            &acquired.blocks,
            stream,
        ) {
            live.warn_once(&format!(
                "kv_host_tier store failed (blocks stay GPU-only): {e:#}"
            ));
        }
    }

    /// Contract 3.7.5: restore the prompt's host-held prefix into the radix tree, waiting for
    /// the copy. Session-agnostic (tail SSM snapshots are not counted); the scheduler uses
    /// [`Self::tier_prefetch`], which passes the request's session.
    pub(crate) fn prefetch(&self, tokens: &[u32], adapter: u32) -> Result<TierPrefetch> {
        self.tier_prefetch(tokens, adapter, 0)
    }

    /// [`Self::prefetch`] with the request's session hash (the SSM snapshot session gate).
    pub(crate) fn tier_prefetch(
        &self,
        tokens: &[u32],
        adapter: u32,
        session_hash: u64,
    ) -> Result<TierPrefetch> {
        let mut kv = self.kv_cache.lock();
        let Some(live) = self.kv_tier_live(&kv) else {
            return Ok(TierPrefetch::default());
        };
        if self.tokens_have_vision_pad(tokens) {
            return Ok(TierPrefetch::default());
        }
        let mut env = self.kv_tier_env(&mut kv);
        live.reap(env.gpu, env.kv);
        live.prefetch(&mut env, tokens, adapter as u64, session_hash)
    }

    /// Contract 3.7.5: start the same restore without waiting. `Ok(None)`: nothing to restore
    /// (the request goes straight to its prefill). The ticket goes to
    /// [`Self::restore_poll`] until it is not `Pending`, then to [`Self::tier_settle`].
    pub(crate) fn restore_begin(
        &self,
        tokens: &[u32],
        adapter: u32,
    ) -> Result<Option<RestoreTicket>> {
        self.tier_restore_begin(tokens, adapter, 0)
    }

    /// [`Self::restore_begin`] with the request's session hash.
    pub(crate) fn tier_restore_begin(
        &self,
        tokens: &[u32],
        adapter: u32,
        session_hash: u64,
    ) -> Result<Option<RestoreTicket>> {
        let mut kv = self.kv_cache.lock();
        let Some(live) = self.kv_tier_live(&kv) else {
            return Ok(None);
        };
        if self.tokens_have_vision_pad(tokens) {
            return Ok(None);
        }
        let mut env = self.kv_tier_env(&mut kv);
        live.reap(env.gpu, env.kv);
        live.restore_begin(&mut env, tokens, adapter as u64, session_hash)
    }

    /// Contract 3.7.5: non-blocking state of a restore. On `Ready` the restored blocks enter the
    /// radix tree (the parked request's prefill then hits them); on `Failed` they are freed.
    pub(crate) fn restore_poll(&self, t: &RestoreTicket) -> Result<RestoreState> {
        let mut kv = self.kv_cache.lock();
        let Some(live) = self.port.kv_tier.live() else {
            return Ok(RestoreState::Failed);
        };
        let mut env = self.kv_tier_env(&mut kv);
        live.restore_poll(&mut env, t)
    }

    /// Done with a ticket: finish the restore (waiting for its copy if it still runs) and hand
    /// the ticket back to the tier. Every ticket from [`Self::restore_begin`] ends here,
    /// including one whose request was cancelled while parked.
    pub(crate) fn tier_settle(&self, t: RestoreTicket) {
        let mut kv = self.kv_cache.lock();
        let Some(live) = self.port.kv_tier.live() else {
            return;
        };
        let mut env = self.kv_tier_env(&mut kv);
        live.settle(&mut env, t);
    }

    /// Once per scheduler tick: reset the tier's per-step write-back budget, drop the block
    /// references of finished write-backs and, in lazy store mode, start the buffered stores
    /// whose blocks approach eviction.
    pub(crate) fn tier_tick(&self) {
        if !kv_host_tier::config::row_enabled() {
            return;
        }
        let mut kv = self.kv_cache.lock();
        let Some(live) = self.port.kv_tier.live() else {
            return;
        };
        live.tick(self.gpu.as_ref(), &mut kv);
    }

    /// Feed one measured prefill (`tokens` computed in `secs`) to the restore cost model.
    pub(crate) fn tier_observe_prefill(&self, tokens: usize, secs: f64) {
        if let Some(live) = self.port.kv_tier.live() {
            live.tier.observe_prefill(tokens, secs);
        }
    }

    /// Whether the tier serves (or will serve) this model, without building it.
    pub(crate) fn tier_active(&self) -> bool {
        kv_host_tier::config::active().is_some()
            && self.prefix_cache.is_active()
            && !self.port.kv_tier.refused()
    }

    /// Quiesce and drop the tier while the model's backend is still alive (called from
    /// `release_pools` and from `Drop`; idempotent).
    pub(in crate::model) fn tier_teardown(&self) {
        let mut kv = self.kv_cache.lock();
        self.port.kv_tier.teardown(self.gpu.as_ref(), &mut kv);
    }
}
