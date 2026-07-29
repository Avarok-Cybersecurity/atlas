// SPDX-License-Identifier: AGPL-3.0-only

// The `#![allow]` on `ssm_snapshot.rs` does not cross into this sibling module
// file. `tag_session`/`try_pop_free_slot`/`acquire_or_spill_slot`/`fault_in_slot`
// have no non-test caller until the Phase-1b fault-in serving wiring (a later
// PR), so they are dead here — exercised only by `ssm_snapshot_spill_tests`.
#![allow(unused_imports, dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::prefix_cache::TierEvict;

use super::ssm_snapshot::SsmSnapshotPool;
use super::ssm_spill_gate::spill_min_tokens;

/// One-shot latch for the "this is the new spill shape" info line. Steady state
/// must not flood, but the first spill has to say on the record which path is
/// live, because the whole change is invisible in the byte counts.
static SPILL_SHAPE_LOGGED: AtomicBool = AtomicBool::new(false);

/// Phase-1 snapshot **spill** (spill-not-drop) and fault-in primitives, plus the
/// tier-aware [`reclaim_from_cache`](SsmSnapshotPool::reclaim_from_cache). Split
/// out of `ssm_snapshot.rs` (500-LoC cap) as a second impl block over the same
/// fields; default-off (`tier == None`) keeps every path byte-identical.
impl SsmSnapshotPool {
    /// Tag Marconi slot `snap_slot` as owned by `session_hash` (0 ⇒ untagged).
    pub(super) fn tag_session(&self, snap_slot: usize, session_hash: u64) {
        if session_hash != 0 {
            self.session_tags.lock().insert(snap_slot, session_hash);
        }
    }

    /// Claim an immediately-free Marconi slot (no eviction). `None` when the
    /// pool is full. The claimed slot must be `free`d if the caller doesn't use
    /// it (e.g. a fault-in miss).
    pub(super) fn try_pop_free_slot(&self) -> Option<usize> {
        if !self.is_enabled() {
            return None;
        }
        self.free_slots.lock().pop()
    }

    /// Acquire a Marconi slot for a **fault-in target** (Phase 1b), spilling a
    /// resident victim to make room when the pool is full. Under a small pool +
    /// heavy churn the free list is usually empty at warm-turn lookup time, so
    /// without this the fault-in silently degrades to recompute (measured: only
    /// 13 of 43 tiered hits completed a fault-in at `--ssm-cache-slots 4`).
    ///
    /// Order: pop a free slot; else evict the session-aware victim. A victim
    /// deep enough to repay the spill is SPILLED (`evict_snapshot_to_tier`
    /// keeps its entry findable → still faultable later); a shallow one is
    /// DROPPED by the cost gate. Either way its slot is freed and popped. The
    /// victim is always a RESIDENT entry (`skip_tiered`), never the tiered
    /// entry we're about to fault in. `None` only if nothing is resident to
    /// evict (every slot mid-flight).
    pub(super) fn acquire_or_spill_slot(
        &self,
        prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
        store: &dyn super::ssm_tier::SnapshotBlobStore,
        gpu: &dyn GpuBackend,
    ) -> Option<usize> {
        if let Some(s) = self.try_pop_free_slot() {
            return Some(s);
        }
        let evict = prefix_cache.evict_snapshot_to_tier(spill_min_tokens())?;
        if let TierEvict::Spill { slot, key, .. } = evict {
            let stream = gpu.default_stream();
            if let Err(e) = self.spill_slot(slot, key, store, gpu, stream) {
                tracing::warn!(
                    "SSM spill during fault-in acquire failed ({e:#}); freeing slot anyway"
                );
            }
        } else {
            log_spill_gate_skip(&evict);
        }
        self.free(evict.slot());
        self.try_pop_free_slot()
    }

    /// Bytes in one slot's full spill blob: every SSM layer's `h` + `conv`
    /// state, laid out `[h_0 conv_0 h_1 conv_1 … h_{L-1} conv_{L-1}]`.
    pub(super) fn spill_blob_bytes(&self) -> usize {
        self.num_ssm_layers * (self.h_bytes + self.conv_bytes)
    }

    /// Release the shared spill/fault-in staging buffer. Called from
    /// `TransformerModel::drop` beside `drop_pinned_staging` — the pool holds no
    /// `gpu` handle of its own.
    pub(crate) fn free_staging(&self, gpu: &dyn GpuBackend) {
        self.spill_staging.free(gpu);
    }

    /// **Spill** Marconi slot `snap_slot` to the byte tier (Phase 1,
    /// spill-not-drop): gather the slot's scattered per-layer `(h,conv)` device
    /// chunks D2H into one host blob and `put` it under `key` (the snapshot's
    /// prefix hash). Returns whether the tier accepted the blob — `false` (tier
    /// full / disabled pool) means the caller should fall back to a plain drop.
    ///
    /// ## Shape (the fix, 2026-07)
    ///
    /// This used to issue `2 × num_ssm_layers` **blocking** `copy_d2h` calls
    /// into a freshly `vec![0u8; …]`-allocated blob. `CudaBackend::copy_d2h`
    /// synchronizes the stream INSIDE every call, so the 30-layer Holo case
    /// paid 60 full stream drains plus a 66 MB zero-fill per eviction:
    /// measured `gather+sync=392936us … total=412334us`, i.e. ~400 ms to move
    /// 66,846,720 B (~165 MB/s) of which the disk write was 19 ms. The gather
    /// now enqueues the same 60 chunks with `copy_d2h_async` into a REUSED
    /// page-locked buffer and drains the stream exactly once — the identical
    /// shape `fault_in_slot` has always used for the H2D direction, which moves
    /// the same bytes in ~28 ms. Target: ~40-50 ms total, ~8-10×.
    ///
    /// Ordering (unchanged, not weakened): a leading `synchronize(stream)`
    /// drains any in-flight D2D `save` into this slot before the D2H read, so
    /// we never spill a half-written snapshot; the trailing `synchronize` then
    /// commits all chunks before `store.put` reads the host bytes and before
    /// the caller's `free(slot)`. `stream` is now genuinely honoured — the old
    /// `copy_d2h` discarded it and enqueued on the default stream, so the
    /// doc-comment's ordering claim held only because callers happen to pass
    /// `default_stream()`.
    pub(super) fn spill_slot(
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
        gpu.synchronize(stream)?; // drain the pending save into this slot
        let bytes = self.spill_blob_bytes();
        let mut guard = self.spill_staging.acquire(gpu, bytes)?;
        let kind = guard.kind();
        self.log_spill_shape_once(bytes, kind);
        let blob = guard.as_mut_slice(); // NOT zeroed — fully overwritten below
        let gather = self.gather_async(snap_slot, blob, gpu, stream);
        // One drain for all chunks — and it must run even when an enqueue
        // failed mid-loop: the chunks already enqueued keep DMA-ing into the
        // SHARED staging buffer, so returning early would let the next
        // spill/fault-in gather tear against them.
        gpu.synchronize(stream)?;
        gather?;
        let t_put = std::time::Instant::now();
        let r = store.put(key, blob)?;
        if timing {
            tracing::info!(
                "SSM spill: {} B  gather+sync={}us  store.put={}us  total={}us  staging={}",
                bytes,
                t_put.duration_since(t0).as_micros(),
                t_put.elapsed().as_micros(),
                t0.elapsed().as_micros(),
                kind,
            );
        }
        Ok(r)
    }

    /// Enqueue the per-layer D2H chunks of `snap_slot` into `blob`. Enqueue
    /// only — the caller owns the single trailing `synchronize`.
    fn gather_async(
        &self,
        snap_slot: usize,
        blob: &mut [u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let per_layer = self.h_bytes + self.conv_bytes;
        for i in 0..self.num_ssm_layers {
            let off = i * per_layer;
            gpu.copy_d2h_async(
                self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                &mut blob[off..off + self.h_bytes],
                stream,
            )?;
            gpu.copy_d2h_async(
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                &mut blob[off + self.h_bytes..off + per_layer],
                stream,
            )?;
        }
        Ok(())
    }

    /// Enqueue the per-layer H2D chunks of `blob` into `snap_slot`. Enqueue
    /// only — the caller owns the single trailing `synchronize`.
    fn scatter_async(
        &self,
        snap_slot: usize,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let per_layer = self.h_bytes + self.conv_bytes;
        for i in 0..self.num_ssm_layers {
            let off = i * per_layer;
            gpu.copy_h2d_async(
                &blob[off..off + self.h_bytes],
                self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                stream,
            )?;
            gpu.copy_h2d_async(
                &blob[off + self.h_bytes..off + per_layer],
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                stream,
            )?;
        }
        Ok(())
    }

    fn log_spill_shape_once(&self, bytes: usize, kind: &str) {
        if SPILL_SHAPE_LOGGED.swap(true, Ordering::Relaxed) {
            return;
        }
        tracing::info!(
            "SSM spill path: {} async D2H chunks + 1 stream sync into a reusable {bytes} B \
             {kind} staging buffer (was {} blocking copies + a fresh heap blob, measured \
             ~400ms/spill)",
            2 * self.num_ssm_layers,
            2 * self.num_ssm_layers,
        );
    }

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
                            // Unbounded tier never rejects; a bounded one could. The
                            // entry is now marked tiered but holds no bytes → a later
                            // fault-in cleanly misses (recompute). Bounded-tier
                            // drop-on-reject is a follow-up.
                            tracing::warn!(
                                "SSM spill tier refused a blob; entry will miss on fault-in"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                "SSM spill failed ({e:#}); freeing slot, entry will miss"
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

/// The spill-side twin of `ssm_fault_in`'s gate log, worded alike so both
/// halves of the cost model read the same way.
fn log_spill_gate_skip(evict: &TierEvict) {
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
