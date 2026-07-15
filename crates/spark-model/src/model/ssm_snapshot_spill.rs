// SPDX-License-Identifier: AGPL-3.0-only

// The `#![allow]` on `ssm_snapshot.rs` does not cross into this sibling module
// file. `tag_session`/`try_pop_free_slot`/`acquire_or_spill_slot`/`fault_in_slot`
// have no non-test caller until the Phase-1b fault-in serving wiring (a later
// PR), so they are dead here — exercised only by `ssm_snapshot_spill_tests`.
#![allow(unused_imports, dead_code)]

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;

use super::ssm_snapshot::SsmSnapshotPool;

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
    /// Order: pop a free slot; else spill the session-aware victim
    /// (`evict_snapshot_to_tier` keeps its entry findable → still faultable
    /// later) to free its slot, then pop that. The victim is always a RESIDENT
    /// entry (`skip_tiered`), never the tiered entry we're about to fault in.
    /// `None` only if nothing is resident to evict (every slot mid-flight).
    pub(super) fn acquire_or_spill_slot(
        &self,
        prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
        store: &dyn super::ssm_tier::SnapshotBlobStore,
        gpu: &dyn GpuBackend,
    ) -> Option<usize> {
        if let Some(s) = self.try_pop_free_slot() {
            return Some(s);
        }
        let (victim_slot, key) = prefix_cache.evict_snapshot_to_tier()?;
        let stream = gpu.default_stream();
        if let Err(e) = self.spill_slot(victim_slot, key, store, gpu, stream) {
            tracing::warn!("SSM spill during fault-in acquire failed ({e:#}); freeing slot anyway");
        }
        self.free(victim_slot);
        self.try_pop_free_slot()
    }

    /// Bytes in one slot's full spill blob: every SSM layer's `h` + `conv`
    /// state, laid out `[h_0 conv_0 h_1 conv_1 … h_{L-1} conv_{L-1}]`.
    pub(super) fn spill_blob_bytes(&self) -> usize {
        self.num_ssm_layers * (self.h_bytes + self.conv_bytes)
    }

    /// **Spill** Marconi slot `snap_slot` to the byte tier (Phase 1,
    /// spill-not-drop): gather the slot's scattered per-layer `(h,conv)` device
    /// chunks D2H into one host blob and `put` it under `key` (the snapshot's
    /// prefix hash). Returns whether the tier accepted the blob — `false` (tier
    /// full / disabled pool) means the caller should fall back to a plain drop.
    ///
    /// Ordering: a single `synchronize(stream)` first drains any in-flight D2D
    /// `save` into this slot (which the caller enqueued on `stream`) before the
    /// D2H read, so we never spill a half-written snapshot — the read-direction
    /// half of the cross-stream hazard the plan flags. (The caller still owns
    /// ordering the *slot reuse* after this spill; see Phase 1b.)
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
        let mut blob = vec![0u8; self.spill_blob_bytes()];
        let per_layer = self.h_bytes + self.conv_bytes;
        for i in 0..self.num_ssm_layers {
            let off = i * per_layer;
            gpu.copy_d2h(
                self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                &mut blob[off..off + self.h_bytes],
            )?;
            gpu.copy_d2h(
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                &mut blob[off + self.h_bytes..off + per_layer],
            )?;
        }
        let t_put = std::time::Instant::now();
        let r = store.put(key, &blob)?;
        if timing {
            tracing::info!(
                "SSM spill: {} B  gather+sync={}us  store.put={}us  total={}us",
                blob.len(),
                t_put.duration_since(t0).as_micros(),
                t_put.elapsed().as_micros(),
                t0.elapsed().as_micros(),
            );
        }
        Ok(r)
    }

    /// **Fault in** a spilled snapshot for `key` into Marconi slot `snap_slot`:
    /// fetch the host blob and scatter it H2D back into the slot's per-layer
    /// `(h,conv)` chunks. Returns `false` if the tier has no blob for `key`
    /// (caller recomputes) — the correct miss degradation.
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
        let mut blob = vec![0u8; self.spill_blob_bytes()];
        let hit = store.get(key, &mut blob)?;
        let get_us = t0.elapsed().as_micros();
        if !hit {
            return Ok(false);
        }
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
        gpu.synchronize(stream)?; // commit before caller's restore reads the slot
        if timing {
            tracing::info!(
                "SSM fault-in: {} B  store.get(RDMA read)={}us  scatter+sync={}us  total={}us",
                blob.len(),
                get_us,
                t0.elapsed().as_micros() - get_us,
                t0.elapsed().as_micros(),
            );
        }
        Ok(true)
    }

    /// Try to reclaim a snapshot slot by evicting a snapshot from the prefix
    /// cache's snapshot index. Snapshots are decoupled from tree nodes, so this
    /// directly frees a slot without evicting KV blocks.
    ///
    /// Phase 1b: when `tier` is `Some` (`ATLAS_SSM_TIER`), the victim is
    /// **spilled** — its bytes moved to the tier and its index entry kept
    /// (findable), so a warm turn faults it back instead of recomputing — before
    /// the slot is freed for reuse. When `tier` is `None` the victim is dropped
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
            if let Some((slot, key)) = prefix_cache.evict_snapshot_to_tier() {
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
                        tracing::warn!("SSM spill failed ({e:#}); freeing slot, entry will miss");
                    }
                }
                self.free(slot); // slot reusable regardless; bytes are (or aren't) in the tier
                return true;
            }
            return false;
        }
        if let Some(snap) = prefix_cache.evict_snapshot_lru() {
            self.free(snap);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
#[path = "ssm_snapshot_spill_tests.rs"]
mod tier_tests;
