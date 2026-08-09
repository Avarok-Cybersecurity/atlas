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
use super::ssm_snapshot_faultin::{log_spill_gate_skip, retire_refused_spill};
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
            match self.spill_slot(slot, key, store, gpu, stream) {
                Ok(true) => {}
                Ok(false) => retire_refused_spill(prefix_cache, store, key),
                Err(e) => tracing::warn!(
                    "SSM spill during fault-in acquire failed ({e:#}); freeing slot anyway, \
                     key {key} RETAINED (an error is not evidence of absence)"
                ),
            }
        } else {
            log_spill_gate_skip(&evict);
        }
        self.free(evict.slot());
        self.try_pop_free_slot()
    }

    /// One warm-turn fault-in cycle for a TIERED anchor: acquire a slot
    /// (spilling a live victim when the pool is full), read the blob back, and
    /// on success re-home the index entry onto the fresh slot and re-tag its
    /// session. `None` means nothing was restored and the caller recomputes.
    ///
    /// Lives on the pool rather than inline in
    /// `TransformerModel::try_fault_in_ssm_snapshot` because this is the ONLY
    /// place that observes a tier MISS while still holding both the prefix
    /// cache and the store — i.e. the only place that can ever retire a stale
    /// tier key — and a `TransformerModel` cannot be built without a GPU. As a
    /// pool method the whole cycle is reachable from a CPU-only test
    /// (`MockGpuBackend` + `RadixTree` + `MemBlobStore`), which is what pins
    /// the miss-arm behaviour. The caller keeps the gates (resident-hit,
    /// tier-store-present, `ATLAS_SSM_FAULT_MIN_TOKENS` depth); this owns the
    /// acquire → fault → promote/miss sequence.
    pub(in crate::model) fn fault_in_for_key(
        &self,
        prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
        store: &dyn super::ssm_tier::SnapshotBlobStore,
        gpu: &dyn GpuBackend,
        key: u64,
        session_hash: u64,
        depth: usize,
        stream: u64,
    ) -> Option<usize> {
        // `acquire_or_spill_slot` spills a resident victim to make room when the
        // pool is full, so a warm hit isn't lost to a busy pool; `None` only if
        // every slot is mid-flight.
        let slot = self.acquire_or_spill_slot(prefix_cache, store, gpu)?;
        match self.fault_in_slot(slot, key, store, gpu, stream) {
            Ok(true) => {
                // A `false` here means no index entry owns this key any more,
                // so the slot we are about to hand back is referenced by
                // nothing — harmless this turn (the caller uses it, then frees
                // it), but it means the bytes we just restored will never be
                // found again. Worth seeing: the reap below is what makes this
                // newly reachable at all.
                if !prefix_cache.promote_snapshot(key, slot) {
                    tracing::warn!(
                        "SSM tier fault-in restored key {key} but no index entry accepted the \
                         promotion — this prefix will recompute next turn"
                    );
                }
                // Re-home the session owner onto the fresh slot. Without this
                // the slot is untagged (or carries a spill victim's stale tag)
                // and the `session_matches` gate at the call site rejects the
                // just-faulted state → full recompute. `lookup`/`lookup_tiered`
                // already filtered by session, so `session_hash` is the
                // rightful owner.
                self.tag_session(slot, session_hash);
                tracing::info!(
                    "SSM tier fault-in: restored spilled snapshot at token {depth} into slot {slot}"
                );
                Some(slot)
            }
            // MISS: the store reported no bytes for this key — under
            // `ATLAS_SSM_TIER_DISK_GB` that is `make_disk_room` having unmapped
            // it. RETIRE the index entry so this prefix recomputes ONCE, rather
            // than re-running this whole doomed cycle (spill a LIVE 66 MB
            // snapshot D2H → allocate a slot → miss → free) on every warm turn,
            // each doomed spill evicting one more tier record: the cap's own
            // pressure would otherwise keep manufacturing more cap pressure.
            Ok(false) => {
                self.free(slot);
                // Ordering is load-bearing: index FIRST (under its lock), store
                // second, store GATED on the index result. Only after we really
                // removed a TIERED entry may the blob go — if it came back
                // resident meanwhile (a concurrent promote), the bytes may
                // belong to a live entry again and must survive.
                if prefix_cache.forget_snapshot_tier_key(key) {
                    // On the cap path this is a no-op (the key is already
                    // unmapped), but `get` also reports a miss for a LENGTH
                    // MISMATCH while KEEPING the record — without this that
                    // blob would be unreachable forever yet still consume
                    // ATLAS_SSM_TIER_DISK_GB budget.
                    store.remove(key);
                    tracing::info!(
                        "SSM tier reap: no blob for key {key} (depth {depth} tok) — retired the \
                         index entry; this prefix now recomputes once instead of re-spilling a \
                         live snapshot every turn. Sustained reaps mean ATLAS_SSM_TIER_DISK_GB \
                         is undersized for the working set."
                    );
                }
                None
            }
            // ERROR: NOT proof the blob is gone. A failed record read leaves it
            // on disk and still mapped (`Residency` restores `disk_lru` and
            // returns Err), and `acquire`/`synchronize`/`scatter` all fail
            // AFTER the key lookup. Reaping here would destroy a live 66 MB
            // snapshot to save one retry; keeping the key costs one wasted
            // cycle, and the miss arm above is the backstop that retires it for
            // good once absence is actually proven.
            Err(e) => {
                self.free(slot);
                tracing::warn!(
                    "SSM tier fault-in failed ({e:#}); key {key} RETAINED (an error is not \
                     evidence of absence) — recomputing this turn, will retry next turn"
                );
                None
            }
        }
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
        // `store.put` is now ~93% of a ~19 ms spill: a single-threaded host
        // memcpy of the 66,846,720 B blob into the tier's arena slot. A
        // `put_with(key, |slot: &mut [u8]| …)` API letting the D2H land
        // STRAIGHT in that slot was analysed and **REJECTED** — do not
        // re-litigate without new measurements:
        //   * It trades away the PINNED destination. `VecSlotArena::buf` is an
        //     ordinary heap `Vec` and `atlas-tier`'s dependency budget forbids
        //     GPU/RDMA deps, so it can never pin. Pinned, this 60-chunk gather
        //     is 1.38-1.42 ms; the ~28 ms pageable figure is INFERRED from the
        //     old H2D path, which also paid a fresh alloc + zero-fill, so it
        //     OVERSTATES the pure pinned-vs-pageable delta — it is not a
        //     like-for-like measurement. Even so the trade is buy back
        //     ~17-19 ms of memcpy, pay back an unmeasured pageable penalty:
        //     net ≈ zero at best. This bullet alone is NOT decisive; the two
        //     below are structural and each sufficient on its own.
        //   * Only 2 of the 4 production `SnapshotBlobStore` implementors could
        //     support it (`MemBlobStore`, and `UnifiedSnapshotStore` only over
        //     `VecSlotArena`); RDMA/paging have no host-addressable slot, and
        //     even the supportable arm needs a new borrow-out method on
        //     `SlotArena` (which exposes only read_slot/write_slot). So it
        //     needs a permanent capability branch through the one
        //     correctness-critical D2H in the tier.
        //   * It would hold the residency `Mutex` across the caller's 60
        //     enqueues + stream sync — flag-ON blocker #1 in `ssm_tier/unified`.
        // Pinning the arena from THIS crate with `cuMemHostRegister` (no
        // `atlas-tier` dep) is the obvious escape and is also rejected: the
        // default hot arena is 64 x 66,846,720 B ~= 4.3 GB, and page-locking
        // that on a UMA box costs far more than the ~17 ms it saves.
        // Likely the real cost here is FIRST TOUCH, not memcpy bandwidth:
        // 66,846,720 B = 16,320 fresh 4 KiB pages ≈ 16 ms of minor faults, and
        // with the default 64 slots every measured put landed in a never-written
        // slot. If so this decays to ~0 in steady state with no code at all.
        // Measure that (put wall time vs put ordinal 1..128) before touching
        // the trait.
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
    ///
    /// `blob` is the `SpillStaging` buffer, which IS page-locked when the box
    /// allows it, so these copies are genuinely asynchronous and the bytes are
    /// read after each call returns. That is exactly the contract
    /// `copy_h2d_async_retained` names: the caller holds the `StagingGuard`
    /// across the whole scatter and synchronises before releasing it. Using the
    /// transient `copy_h2d_async` here would be correct but would reintroduce
    /// one stream drain per chunk — the ~400 ms shape this path exists to
    /// escape.
    pub(super) fn scatter_async(
        &self,
        snap_slot: usize,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let per_layer = self.h_bytes + self.conv_bytes;
        for i in 0..self.num_ssm_layers {
            let off = i * per_layer;
            gpu.copy_h2d_async_retained(
                &blob[off..off + self.h_bytes],
                self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                stream,
            )?;
            gpu.copy_h2d_async_retained(
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
}
