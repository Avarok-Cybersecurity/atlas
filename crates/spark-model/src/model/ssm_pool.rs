// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Pre-allocated contiguous GPU memory pool for SSM layer states.
///
/// Each pool slot has fixed GPU addresses for h_state and conv_state across
/// all SSM layers. This enables CUDA graph capture at batch sizes > 1 because
/// the graph embeds memory addresses that remain stable across replays.
pub(crate) struct SsmStatePool {
    pub(super) h_state_pools: Vec<DevicePtr>,
    pub(super) conv_state_pools: Vec<DevicePtr>,
    /// Per-slot K=3 intermediate checkpoint pools (only allocated when has_mtp).
    /// Layout: `[num_ssm_layers]`, each allocation = max_slots * 3 * h_bytes.
    pub(super) h_intermediate_pools: Vec<DevicePtr>,
    pub(super) conv_intermediate_pools: Vec<DevicePtr>,
    /// Per-slot SSM state checkpoint pools (only allocated when has_mtp).
    pub(super) h_checkpoint_pools: Vec<DevicePtr>,
    pub(super) conv_checkpoint_pools: Vec<DevicePtr>,
    pub(super) h_bytes: usize,
    pub(super) conv_bytes: usize,
    /// Number of CLAIMABLE slots (excludes the reserved dummy slot at
    /// index `max_slots`). All claim_slot/release_slot operations work
    /// in `[0, max_slots)`.
    pub(super) max_slots: usize,
    pub(super) num_ssm_layers: usize,
    pub(super) has_mtp: bool,
    pub(super) num_intermediates: usize,
    pub(super) free_slots: Mutex<Vec<usize>>,
}

impl SsmStatePool {
    pub(super) fn new(
        config: &ModelConfig,
        max_slots: usize,
        has_mtp: bool,
        num_intermediates: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let _d_conv = config.linear_conv_kernel_dim;

        let h_bytes = config.ssm_h_state_bytes();
        let conv_bytes = config.ssm_conv_state_bytes();
        let num_ssm_layers = config.num_ssm_layers();

        // Reserve one extra slot at index `max_slots` as a dedicated
        // dummy used by `decode_batch` / `mixed_forward` padding (see
        // `dummy_slot()` below). Without this, pad positions write to
        // pool slot indices `n..padded_n` which can collide with
        // claimed slots if the scheduler invariant ("active sequences
        // occupy contiguous slots [0..n)") is ever broken — silent SSM
        // state corruption. Costs `(h_bytes + conv_bytes) *
        // num_ssm_layers` extra GPU memory (~kilobytes per pool).
        let total_slots = max_slots + 1;

        let mut h_state_pools = Vec::with_capacity(num_ssm_layers);
        let mut conv_state_pools = Vec::with_capacity(num_ssm_layers);
        let mut h_intermediate_pools = Vec::new();
        let mut conv_intermediate_pools = Vec::new();
        let mut h_checkpoint_pools = Vec::new();
        let mut conv_checkpoint_pools = Vec::new();

        for _ in 0..num_ssm_layers {
            let h_pool = gpu.alloc(total_slots * h_bytes)?;
            gpu.memset(h_pool, 0, total_slots * h_bytes)?;
            h_state_pools.push(h_pool);

            let conv_pool = gpu.alloc(total_slots * conv_bytes)?;
            gpu.memset(conv_pool, 0, total_slots * conv_bytes)?;
            conv_state_pools.push(conv_pool);
        }

        if has_mtp {
            let ni = num_intermediates;
            for _ in 0..num_ssm_layers {
                let h_inter = gpu.alloc(total_slots * ni * h_bytes)?;
                gpu.memset(h_inter, 0, total_slots * ni * h_bytes)?;
                h_intermediate_pools.push(h_inter);

                let conv_inter = gpu.alloc(total_slots * ni * conv_bytes)?;
                gpu.memset(conv_inter, 0, total_slots * ni * conv_bytes)?;
                conv_intermediate_pools.push(conv_inter);

                // 1 checkpoint per slot per layer
                let h_ckpt = gpu.alloc(total_slots * h_bytes)?;
                gpu.memset(h_ckpt, 0, total_slots * h_bytes)?;
                h_checkpoint_pools.push(h_ckpt);

                let conv_ckpt = gpu.alloc(total_slots * conv_bytes)?;
                gpu.memset(conv_ckpt, 0, total_slots * conv_bytes)?;
                conv_checkpoint_pools.push(conv_ckpt);
            }

            let mtp_mb = num_ssm_layers
                * total_slots
                * (ni * h_bytes + ni * conv_bytes + h_bytes + conv_bytes)
                / (1024 * 1024);
            tracing::info!("SSM MTP pools ({ni} intermediates + checkpoints): {mtp_mb} MB");
        }

        // free_slots holds claimable indices only; the dummy at index
        // `max_slots` is permanently reserved.
        let free_slots: Vec<usize> = (0..max_slots).rev().collect();

        let total_mb = num_ssm_layers * max_slots * (h_bytes + conv_bytes) / (1024 * 1024);
        tracing::info!(
            "SSM state pool: {max_slots} slots × {num_ssm_layers} layers = {total_mb} MB",
        );

        Ok(Self {
            h_state_pools,
            conv_state_pools,
            h_intermediate_pools,
            conv_intermediate_pools,
            h_checkpoint_pools,
            conv_checkpoint_pools,
            h_bytes,
            conv_bytes,
            max_slots,
            num_ssm_layers,
            has_mtp,
            num_intermediates,
            free_slots: Mutex::new(free_slots),
        })
    }

    pub(super) fn claim_slot(&self) -> Result<usize> {
        self.free_slots.lock().pop().ok_or_else(|| {
            anyhow::anyhow!("SSM state pool exhausted (max {} slots)", self.max_slots)
        })
    }

    /// Claim a slot and wrap it in a [`SlotGuard`] that returns the slot to the
    /// free list when dropped. This is the leak-safe claim API: the guard is
    /// stored on the owning [`SequenceState`], so the slot is released on EVERY
    /// sequence-exit path — normal completion, abort/cancel, decode error,
    /// swap-out failure, and panic/unwind — not only the explicit
    /// `free_sequence`/`compact_sequence` sites. The explicit sites neutralize
    /// the guard via [`SlotGuard::take`]/[`SlotGuard::migrate`] so a slot is
    /// released EXACTLY once (a double `push` would corrupt `free_slots` and
    /// hand the same index to two sequences → SSM state corruption).
    ///
    /// `self: &Arc<Self>` so the guard can hold an owning handle to the pool.
    pub(super) fn claim_guarded(self: &Arc<Self>) -> Result<SlotGuard> {
        let idx = self.claim_slot()?;
        Ok(SlotGuard {
            pool: Arc::clone(self),
            idx: Some(idx),
        })
    }

    pub(super) fn release_slot(&self, idx: usize) {
        self.free_slots.lock().push(idx);
    }

    /// Reserved pool slot used by `decode_batch` / `mixed_forward` padding.
    /// Never claimed by `claim_slot()`, never released. SSM kernels are
    /// free to read/write this slot's pool memory without affecting any
    /// active sequence.
    #[inline]
    pub(super) fn dummy_slot(&self) -> usize {
        self.max_slots
    }

    /// Zero h_state and conv_state for a slot across all SSM layers.
    /// Must be called on slot allocation to prevent stale SSM state
    /// from prior sequences from corrupting new prefill output.
    pub(super) fn zero_slot(&self, idx: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.memset_async(self.h_state(i, idx), 0, self.h_bytes, stream)?;
            gpu.memset_async(self.conv_state(i, idx), 0, self.conv_bytes, stream)?;
        }
        Ok(())
    }

    pub(super) fn h_state(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.h_state_pools[ssm_layer_idx].offset(slot * self.h_bytes)
    }

    pub(super) fn conv_state(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.conv_state_pools[ssm_layer_idx].offset(slot * self.conv_bytes)
    }

    /// DEBUG (env-gated): PER-LAYER fingerprint of h_state + conv_state for a
    /// pool slot, used to prove restore/recompute state divergence. States are
    /// FP32 (`ssm_h_state_bytes`/`ssm_conv_state_bytes`). For each SSM layer we
    /// emit three reductions so per-element divergence cannot cancel:
    ///   - `sum`   (signed sum — catches gross errors / sign flips)
    ///   - `ssq`   (sum of squares — magnitude-weighted, cancellation-free)
    ///   - `sabs`  (sum of absolute values — cancellation-free L1)
    ///
    /// A global `(sum, ssq, sabs)` triple is also logged for a quick gate.
    pub(super) fn debug_state_checksum(
        &self,
        slot: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
        tag: &str,
    ) {
        gpu.synchronize(stream).ok();
        let mut g_h_sum = 0f64;
        let mut g_h_ssq = 0f64;
        let mut g_h_sabs = 0f64;
        let mut g_c_sum = 0f64;
        let mut g_c_ssq = 0f64;
        let mut g_c_sabs = 0f64;
        for i in 0..self.num_ssm_layers {
            let mut hb = vec![0u8; self.h_bytes];
            let mut cb = vec![0u8; self.conv_bytes];
            if gpu.copy_d2h(self.h_state(i, slot), &mut hb).is_err() {
                return;
            }
            if gpu.copy_d2h(self.conv_state(i, slot), &mut cb).is_err() {
                return;
            }
            let (mut h_sum, mut h_ssq, mut h_sabs) = (0f64, 0f64, 0f64);
            for c in hb.chunks_exact(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64;
                h_sum += v;
                h_ssq += v * v;
                h_sabs += v.abs();
            }
            let (mut c_sum, mut c_ssq, mut c_sabs) = (0f64, 0f64, 0f64);
            for c in cb.chunks_exact(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64;
                c_sum += v;
                c_ssq += v * v;
                c_sabs += v.abs();
            }
            g_h_sum += h_sum;
            g_h_ssq += h_ssq;
            g_h_sabs += h_sabs;
            g_c_sum += c_sum;
            g_c_ssq += c_ssq;
            g_c_sabs += c_sabs;
            tracing::warn!(
                "ATLAS_SSM_CKSUM[{tag}] slot={slot} L{i} \
                 h_sum={h_sum:.6} h_ssq={h_ssq:.6} h_sabs={h_sabs:.6} \
                 c_sum={c_sum:.6} c_ssq={c_ssq:.6} c_sabs={c_sabs:.6}"
            );
        }
        tracing::warn!(
            "ATLAS_SSM_CKSUM[{tag}] slot={slot} GLOBAL \
             h_sum={g_h_sum:.6} h_ssq={g_h_ssq:.6} h_sabs={g_h_sabs:.6} \
             c_sum={g_c_sum:.6} c_ssq={g_c_ssq:.6} c_sabs={g_c_sabs:.6}"
        );
    }

    /// Get fixed-address intermediate h_state for K=2/3/4 verify.
    /// `token_idx` is 0..3 (which token in the verify pass).
    pub(super) fn h_intermediate(
        &self,
        ssm_layer_idx: usize,
        slot: usize,
        token_idx: usize,
    ) -> DevicePtr {
        let ni = self.num_intermediates;
        self.h_intermediate_pools[ssm_layer_idx].offset((slot * ni + token_idx) * self.h_bytes)
    }

    pub(super) fn conv_intermediate(
        &self,
        ssm_layer_idx: usize,
        slot: usize,
        token_idx: usize,
    ) -> DevicePtr {
        let ni = self.num_intermediates;
        self.conv_intermediate_pools[ssm_layer_idx]
            .offset((slot * ni + token_idx) * self.conv_bytes)
    }

    pub(super) fn h_checkpoint(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.h_checkpoint_pools[ssm_layer_idx].offset(slot * self.h_bytes)
    }

    pub(super) fn conv_checkpoint(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.conv_checkpoint_pools[ssm_layer_idx].offset(slot * self.conv_bytes)
    }

    pub(super) fn reset_slot(&self, slot: usize, gpu: &dyn GpuBackend) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.memset(self.h_state(i, slot), 0, self.h_bytes)?;
            gpu.memset(self.conv_state(i, slot), 0, self.conv_bytes)?;
            if self.has_mtp {
                for t in 0..self.num_intermediates {
                    gpu.memset(self.h_intermediate(i, slot, t), 0, self.h_bytes)?;
                    gpu.memset(self.conv_intermediate(i, slot, t), 0, self.conv_bytes)?;
                }
                gpu.memset(self.h_checkpoint(i, slot), 0, self.h_bytes)?;
                gpu.memset(self.conv_checkpoint(i, slot), 0, self.conv_bytes)?;
            }
        }
        Ok(())
    }

    pub(super) fn copy_slot(
        &self,
        from: usize,
        to: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                self.h_state(i, from),
                self.h_state(i, to),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.conv_state(i, from),
                self.conv_state(i, to),
                self.conv_bytes,
                stream,
            )?;
            if self.has_mtp {
                for t in 0..self.num_intermediates {
                    gpu.copy_d2d_async(
                        self.h_intermediate(i, from, t),
                        self.h_intermediate(i, to, t),
                        self.h_bytes,
                        stream,
                    )?;
                    gpu.copy_d2d_async(
                        self.conv_intermediate(i, from, t),
                        self.conv_intermediate(i, to, t),
                        self.conv_bytes,
                        stream,
                    )?;
                }
                gpu.copy_d2d_async(
                    self.h_checkpoint(i, from),
                    self.h_checkpoint(i, to),
                    self.h_bytes,
                    stream,
                )?;
                gpu.copy_d2d_async(
                    self.conv_checkpoint(i, from),
                    self.conv_checkpoint(i, to),
                    self.conv_bytes,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}

/// RAII owner of a claimed SSM pool slot.
///
/// Stored on the owning [`crate::traits::SequenceState`]. While `idx` is
/// `Some(i)`, this guard is responsible for returning slot `i` to the pool's
/// free list. The slot is released on `Drop` UNLESS the explicit teardown path
/// has already neutralized the guard via [`take`](Self::take) (normal
/// `free_sequence`) or transferred it via [`migrate`](Self::migrate)
/// (slot-migration in `compact_sequence`). This makes the release happen
/// EXACTLY once on every exit path:
///   - normal finish / error / cancel / swap-out → `free_sequence` calls
///     `take()` then releases explicitly (one push);
///   - slot-migration → `compact_sequence` releases the OLD slot explicitly and
///     calls `migrate(new)` so the guard tracks the NEW slot;
///   - abort/early-return/panic where `free_sequence` is never reached →
///     `Drop` releases the still-`Some` slot (one push).
///
/// Because the explicit sites `take()` the idx before pushing, and `Drop` only
/// pushes when the idx is still `Some`, the same slot index is never pushed
/// twice — no double-release / `free_slots` corruption. `free_slots` is a
/// `parking_lot::Mutex`; the scheduler is single-threaded, so claim and release
/// never race, but the mutex keeps the EP-worker path sound regardless.
pub(crate) struct SlotGuard {
    pool: Arc<SsmStatePool>,
    idx: Option<usize>,
}

impl SlotGuard {
    /// A guard that owns no slot (released/migrated, or a placeholder for the
    /// reserved-dummy / sentinel paths). Holds an `Arc` to the pool but its
    /// `Drop` is a no-op while `idx` is `None`.
    pub(crate) fn empty(pool: Arc<SsmStatePool>) -> Self {
        Self { pool, idx: None }
    }

    /// The currently-owned claimable slot index, if any.
    #[inline]
    pub(crate) fn idx(&self) -> Option<usize> {
        self.idx
    }

    /// Neutralize the guard, returning the owned slot index (if any) WITHOUT
    /// releasing it. The caller becomes responsible for releasing exactly once
    /// (the explicit `free_sequence` path). After this the guard's `Drop` is a
    /// no-op, so there is no double-release.
    #[inline]
    pub(crate) fn take(&mut self) -> Option<usize> {
        self.idx.take()
    }

    /// Slot-migration: the guard's OLD slot has already been released by the
    /// caller (`compact_sequence`); point the guard at the NEW slot it now
    /// owns. Asserts the old slot was already taken so a stale idx cannot be
    /// silently leaked or double-released.
    #[inline]
    pub(crate) fn migrate(&mut self, new_idx: usize) {
        debug_assert!(
            self.idx.is_none(),
            "SlotGuard::migrate called before the old slot was released/taken"
        );
        self.idx = Some(new_idx);
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        if let Some(idx) = self.idx.take() {
            // Reached only when the sequence exited WITHOUT the explicit
            // teardown path neutralizing the guard (abort, early-return after
            // an owned `ActiveSeq` move, panic/unwind). Returns the slot to the
            // free list so the pool cannot leak itself into exhaustion.
            tracing::debug!("SlotGuard::drop releasing un-freed SSM slot {idx}");
            self.pool.release_slot(idx);
        }
    }
}

#[cfg(test)]
mod slot_guard_tests {
    use super::*;

    /// Build a bare pool that touches ONLY the CPU-side slot bookkeeping
    /// (`free_slots`/`max_slots`). All GPU pointer vectors are empty; the guard
    /// path and `claim_slot`/`release_slot` never dereference them, so no GPU is
    /// required to validate the exactly-once release invariant.
    fn bare_pool(max_slots: usize) -> Arc<SsmStatePool> {
        Arc::new(SsmStatePool {
            h_state_pools: Vec::new(),
            conv_state_pools: Vec::new(),
            h_intermediate_pools: Vec::new(),
            conv_intermediate_pools: Vec::new(),
            h_checkpoint_pools: Vec::new(),
            conv_checkpoint_pools: Vec::new(),
            h_bytes: 0,
            conv_bytes: 0,
            max_slots,
            num_ssm_layers: 0,
            has_mtp: false,
            num_intermediates: 0,
            free_slots: Mutex::new((0..max_slots).rev().collect()),
        })
    }

    fn free_count(pool: &SsmStatePool) -> usize {
        pool.free_slots.lock().len()
    }

    #[test]
    fn guard_releases_on_drop() {
        let pool = bare_pool(2);
        let claimed;
        {
            let g = pool.claim_guarded().unwrap();
            // free_slots is `(0..max).rev()`, so `pop()` returns the LOWEST
            // index first (0) — matching the original `claim_slot` behavior.
            claimed = g.idx().expect("guard owns a slot");
            assert_eq!(claimed, 0);
            assert_eq!(free_count(&pool), 1);
        } // guard dropped (abort/panic surrogate) → slot returned
        assert_eq!(
            free_count(&pool),
            2,
            "drop must return the slot exactly once"
        );
        // The released slot is back in the free list (no phantom indices).
        assert!(pool.free_slots.lock().contains(&claimed));
    }

    #[test]
    fn take_neutralizes_drop_no_double_release() {
        let pool = bare_pool(2);
        let mut g = pool.claim_guarded().unwrap();
        let idx = g.take().expect("guard owns a slot");
        // Explicit teardown releases exactly once...
        pool.release_slot(idx);
        assert_eq!(free_count(&pool), 2);
        drop(g); // ...and the now-empty guard's Drop is a no-op (no double push)
        assert_eq!(
            free_count(&pool),
            2,
            "take() must make Drop a no-op (no double-release)"
        );
    }

    #[test]
    fn migration_releases_old_once_then_owns_new() {
        // Two live sequences so the migration target is a genuinely-claimed
        // slot (as in production), not one still sitting in the free list.
        let pool = bare_pool(2); // {0,1}
        let mut survivor = pool.claim_guarded().unwrap(); // owns 0 (pop → 0)
        let donor = pool.claim_guarded().unwrap(); // owns 1
        assert_eq!(free_count(&pool), 0);
        let donor_slot = donor.idx().unwrap();

        // Simulate compact_sequence(survivor, donor_slot): release survivor's
        // OLD slot and migrate it onto the donor's slot.
        let old = survivor.take().unwrap();
        pool.release_slot(old); // survivor's old slot released once
        // donor is being torn down; disown its slot WITHOUT releasing (survivor
        // takes it over). Mirrors detach_slot_for_reuse.
        let mut donor = donor;
        let _ = donor.take();
        drop(donor); // empty guard → no release
        survivor.migrate(donor_slot);
        assert_eq!(survivor.idx(), Some(donor_slot));

        // Free the survivor later: releases donor_slot exactly once.
        let final_idx = survivor.take().unwrap();
        pool.release_slot(final_idx);
        drop(survivor);

        let free = pool.free_slots.lock();
        let mut sorted = free.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1], "both slots free exactly once, no dupes");
    }

    #[test]
    fn retire_with_migration_is_double_free_free() {
        // Full scenario: retired R owns slot i; survivor S owns slot j; S is
        // compacted onto i; R is then disowned (detach_slot_for_reuse) and
        // freed. Then S is freed. Every slot must be released exactly once.
        let pool = bare_pool(2); // slots {0,1}
        let mut r = pool.claim_guarded().unwrap(); // R owns 1
        let mut s = pool.claim_guarded().unwrap(); // S owns 0
        assert_eq!(free_count(&pool), 0);
        let r_slot = r.idx().unwrap();
        let s_slot = s.idx().unwrap();

        // compact_sequence(S, r_slot): release S's old slot, migrate to R's slot.
        let old = s.take().unwrap();
        assert_eq!(old, s_slot);
        pool.release_slot(old); // j released once
        s.migrate(r_slot); // S now owns i

        // detach_slot_for_reuse(R): take WITHOUT release (S owns it now).
        let _ = r.take();
        drop(r); // R's guard is empty → no release of i

        // free_sequence(S) later: release i exactly once.
        let i = s.take().unwrap();
        pool.release_slot(i);
        drop(s);

        let free = pool.free_slots.lock();
        let mut sorted = free.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1], "both slots free, exactly once each");
    }
}
