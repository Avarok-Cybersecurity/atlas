// SPDX-License-Identifier: AGPL-3.0-only

//! The READ half of the SSM spill tier: faulting a spilled snapshot back into
//! a pool slot, reclaiming a slot from the prefix cache, and retiring index
//! entries whose blob is gone.
//!
//! Split out of `ssm_snapshot_spill.rs` (the WRITE half: gather + spill) to
//! keep both files under the repo's 500-LoC cap.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::prefix_cache::{PrefixCache, TierEvict};

use spark_runtime::kv_cache::PagedKvCache;

use super::ssm_snapshot::SsmSnapshotPool;
use super::ssm_spill_gate::spill_min_tokens;

impl SsmSnapshotPool {
    /// **Fault in** a spilled snapshot for `key` into Marconi slot `snap_slot`:
    /// fetch the host blob and scatter it H2D back into the slot's per-layer
    /// `(h,conv)` chunks. Returns `false` if the tier has no blob for `key`
    /// (caller recomputes) — the correct miss degradation.
    ///
    /// Shares `spill_slot`'s reusable page-locked staging buffer: this path was
    /// already async+one-sync, but it re-allocated, zeroed and first-touched a
    /// fresh 66 MB `Vec` per fault-in, and a pageable destination keeps
    /// `cuMemcpyHtoDAsync_v2` off the DMA fast path. Sharing is safe because
    /// the model is single-threaded post-construction and
    /// `acquire_or_spill_slot → spill_slot → (return) → fault_in_slot` is
    /// sequential, never nested; the staging `Mutex` makes that a checked
    /// invariant rather than a comment.
    ///
    /// A trailing `synchronize(stream)` guarantees the H2D scatter has
    /// committed before the caller issues a `restore` (D2D slot→main pool) that
    /// reads this slot — the write-direction half of the ordering hazard.
    pub(super) fn fault_in_slot(
        &self,
        snap_slot: usize,
        key: u64,
        store: &dyn super::ssm_tier::SnapshotBlobStore,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<bool> {
        if !self.is_enabled() {
            return Ok(false);
        }
        let timing = std::env::var_os("ATLAS_SSM_TIER_TIMING").is_some();
        let t0 = std::time::Instant::now();
        let bytes = self.spill_blob_bytes();
        let mut guard = self.spill_staging.acquire(gpu, bytes)?;
        let kind = guard.kind();
        let blob = guard.as_mut_slice();
        let hit = store.get(key, blob)?;
        let get_us = t0.elapsed().as_micros();
        if !hit {
            return Ok(false);
        }
        let scatter = self.scatter_async(snap_slot, blob, gpu, stream);
        // Commit before the caller's restore reads the slot — and, as in
        // `spill_slot`, before the shared staging buffer can be re-acquired
        // while enqueued chunks are still reading it.
        gpu.synchronize(stream)?;
        scatter?;
        if timing {
            tracing::info!(
                "SSM fault-in: {} B  store.get(RDMA read)={}us  scatter+sync={}us  total={}us  \
                 staging={}",
                bytes,
                get_us,
                t0.elapsed().as_micros() - get_us,
                t0.elapsed().as_micros(),
                kind,
            );
        }
        Ok(true)
    }

    /// Try to reclaim a snapshot slot by evicting a snapshot from the prefix
    /// cache's snapshot index. Snapshots are decoupled from tree nodes, so this
    /// directly frees a slot without evicting KV blocks.
    ///
    /// Phase 1b: when `tier` is `Some` (`ATLAS_SSM_TIER`), a victim deep enough
    /// to repay the spill cost is **spilled** — its bytes moved to the tier and
    /// its index entry kept (findable), so a warm turn faults it back instead of
    /// recomputing — before the slot is freed for reuse. A victim below
    /// `ATLAS_SSM_SPILL_MIN_TOKENS` is dropped instead (see
    /// [`super::ssm_spill_gate`]). When `tier` is `None` the victim is dropped
    /// exactly as before (byte-identical default path). Returns whether a slot
    /// was reclaimed.
    pub(super) fn reclaim_from_cache(
        &self,
        prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
        _kv_cache: &mut PagedKvCache,
        tier: Option<&dyn super::ssm_tier::SnapshotBlobStore>,
        gpu: &dyn GpuBackend,
    ) -> bool {
        if let Some(store) = tier {
            // Spill-not-drop. Marconi saves are enqueued on the default stream,
            // so draining it inside `spill_slot` guarantees we never D2H a
            // half-written victim slot (the read half of the ordering hazard).
            let Some(evict) = prefix_cache.evict_snapshot_to_tier(spill_min_tokens()) else {
                return false;
            };
            match evict {
                TierEvict::Spill { slot, key, .. } => {
                    let stream = gpu.default_stream();
                    match self.spill_slot(slot, key, store, gpu, stream) {
                        Ok(true) => {}
                        Ok(false) => {
                            // Unbounded tier never rejects; a bounded one could.
                            // The entry would otherwise stay marked `tiered`
                            // holding no bytes — a stale tier key manufactured
                            // eagerly — so retire it here instead of leaving the
                            // next warm turn to rediscover it the expensive way.
                            retire_refused_spill(prefix_cache, store, key);
                        }
                        Err(e) => {
                            // Key RETAINED on purpose: an error is not evidence
                            // the bytes are absent, and the fault-in miss arm
                            // retires it after exactly one doomed cycle.
                            tracing::warn!(
                                "SSM spill failed ({e:#}); freeing slot, key {key} retained — \
                                 entry will miss on fault-in and be retired there"
                            );
                        }
                    }
                }
                TierEvict::Drop { .. } => log_spill_gate_skip(&evict),
            }
            self.free(evict.slot()); // slot reusable regardless of the arm taken
            return true;
        }
        if let Some(snap) = prefix_cache.evict_snapshot_lru() {
            self.free(snap);
            true
        } else {
            false
        }
    }
}

/// The spill-REFUSAL twin of the fault-in miss reap. `Ok(false)` from
/// `spill_slot` means the tier took no bytes, so `evict_to_tier` has already
/// marked the entry `tiered` holding nothing — a stale tier key manufactured
/// eagerly. Retire it now rather than leave it findable-but-empty for the next
/// warm turn to rediscover the expensive way: spill a LIVE 66 MB snapshot D2H →
/// allocate a slot → miss → free, once per turn, forever.
///
/// Index FIRST and the blob only if a TIERED entry was really removed — a
/// resident entry's `snapshot_id` is a live pool slot, so a by-key remove of one
/// would leak it. `Err` at spill time deliberately KEEPS the key (an error is
/// not evidence of absence); the fault-in miss arm retires it after exactly one
/// doomed cycle.
pub(super) fn retire_refused_spill(
    prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
    store: &dyn super::ssm_tier::SnapshotBlobStore,
    key: u64,
) {
    if prefix_cache.forget_snapshot_tier_key(key) {
        store.remove(key);
        tracing::warn!(
            "SSM spill tier refused a blob for key {key}; retired the entry rather than \
             leaving it findable-but-empty — a dead tier key costs one live-snapshot spill \
             per warm turn to rediscover"
        );
    }
}

/// The spill-side twin of `ssm_fault_in`'s gate log, worded alike so both
/// halves of the cost model read the same way.
pub(super) fn log_spill_gate_skip(evict: &TierEvict) {
    if let TierEvict::Drop { depth, .. } = *evict {
        tracing::info!(
            "SSM spill SKIPPED (cost gate): victim depth {depth} < \
             ATLAS_SSM_SPILL_MIN_TOKENS={} — dropped instead; a ~45ms spill cannot repay \
             {depth} tokens of prefill",
            spill_min_tokens(),
        );
    }
}

#[cfg(test)]
#[path = "ssm_snapshot_spill_tests.rs"]
mod tier_tests;

#[cfg(test)]
#[path = "ssm_snapshot_reap_tests.rs"]
mod reap_tests;
