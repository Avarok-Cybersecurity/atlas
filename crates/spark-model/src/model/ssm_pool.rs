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
    /// Base pointers returned by `GpuBackend::alloc`, owned by this pool.
    /// The per-family vectors below may be interior layer views into one
    /// contiguous allocation and therefore must never be freed directly.
    owned_allocations: Vec<DevicePtr>,
    pub(super) h_state_pools: Vec<DevicePtr>,
    pub(super) conv_state_pools: Vec<DevicePtr>,
    /// Per-slot K=3 intermediate checkpoint pools (only allocated when has_mtp).
    /// Layout: `[num_ssm_layers]`, each allocation = max_slots * 3 * h_bytes.
    pub(super) h_intermediate_pools: Vec<DevicePtr>,
    pub(super) conv_intermediate_pools: Vec<DevicePtr>,
    /// Per-slot SSM state checkpoint pools (only allocated when has_mtp).
    pub(super) h_checkpoint_pools: Vec<DevicePtr>,
    pub(super) conv_checkpoint_pools: Vec<DevicePtr>,
    /// FP32-width h blob bytes per layer (`config.ssm_h_state_bytes()`) —
    /// the ELEMENT-count authority (elems = h_bytes / 4) and the width of
    /// everything outside this pool (Marconi snapshots stay FP32).
    pub(super) h_bytes: usize,
    /// STORAGE width of one h blob inside this pool — what the h pools
    /// allocate, offset and copy by. Equals `h_bytes` except under the
    /// stage-3 f16-SIZED pool (`ssm_reserve::ssm_h_stored_bytes`).
    pub(super) h_stored_bytes: usize,
    /// Stage-3 f16-SIZED pool ONLY: one FP32 h-state staging blob per slot,
    /// laid out `[total_slots] × h_bytes` in a single allocation. `None`
    /// under an FP32-sized pool, where prefill writes the slot in place.
    ///
    /// Shared across LAYERS on purpose — see
    /// [`crate::ssm_reserve::ssm_h_prefill_stage_bytes`] for why one blob per
    /// slot is sufficient and why per-slot (rather than per-co-dispatched-
    /// sequence) is the sizing that needs no width assumption.
    pub(super) h_prefill_stage_pool: Option<DevicePtr>,
    pub(super) conv_bytes: usize,
    /// Number of CLAIMABLE slots (excludes the reserved dummy slot at
    /// index `max_slots`). All claim_slot/release_slot operations work
    /// in `[0, max_slots)`.
    pub(super) max_slots: usize,
    /// Number of claimable slots COVERED by the MTP intermediate/checkpoint
    /// pools (`ssm_reserve::mtp_state_slots(max_slots)`; 0 when `!has_mtp`).
    /// Those pools allocate `mtp_slots + 1` slots — their own dummy lives at
    /// index `mtp_slots`. At `max_slots <= 32` this equals `max_slots`
    /// (sizing byte-identical to the pre-diet layout); above 32 the
    /// scheduler's spec dispatch guard (`slot_idx < mtp_state_slots(bs)`)
    /// keeps uncovered slots out of every verify path, and
    /// [`Self::mtp_slot`] clamps a stray access onto the MTP dummy so it is
    /// memory-safe.
    pub(super) mtp_slots: usize,
    pub(super) num_ssm_layers: usize,
    pub(super) has_mtp: bool,
    /// UNIFORM per-slot count of the CONV intermediates (and the H count of
    /// every slot when the tiers are off — DFlash-γ pools, kill switch,
    /// ladder disabled). Conv never tiers: the batched conv verify kernel
    /// needs a uniform cross-sequence snapshot stride and writes all K
    /// snapshots (see `ssm_reserve::verify_slot_h_intermediates`).
    pub(super) num_intermediates: usize,
    /// TIERED per-slot H-intermediate counts, one entry per MTP slot plus
    /// the MTP dummy at index `mtp_slots` (always full-width — pad rows may
    /// write any token index). SSOT for the values:
    /// `ssm_reserve::verify_slot_h_intermediates`. Empty when `!has_mtp`.
    pub(super) h_inter_counts: Vec<usize>,
    /// Prefix sums over `h_inter_counts` (len + 1): slot `s`'s H
    /// intermediates start at element `h_inter_offsets[s]` of each layer's
    /// pool; the last entry is the per-layer pool size in h_state units.
    /// Fixed at allocation ⇒ per-slot addresses stay CUDA-graph-stable.
    pub(super) h_inter_offsets: Vec<usize>,
    /// Verify-rollback mode (`--ssm-rollback-mode`, EXPERIMENTAL). Under
    /// `Replay` the per-token intermediate pools are NOT allocated (one
    /// checkpoint blob per slot instead) and every speculative verify entry
    /// refuses through [`Self::require_verify_rollback_supported`] — the
    /// capture/replay device path is scaffold-only.
    pub(super) rollback_mode: crate::ssm_reserve::SsmRollbackMode,
    /// Replay-mode verify-window input ring, one flat region per SSM layer
    /// (`(mtp_slots + 1) × (K-1) rows` of qkvz+gates each — layout
    /// `ssm_reserve::ssm_replay_ring_bytes` / `ssm_replay_row_bytes`).
    /// Empty in snapshot mode. Allocated so boot sizing is honest; the
    /// capture that would fill it is not wired yet.
    pub(super) replay_input_rings: Vec<DevicePtr>,
    pub(super) free_slots: Mutex<Vec<usize>>,
}

/// Prefix-sum layout for tiered per-slot H intermediates: returns
/// `(offsets, total)` where `offsets[s]` is slot `s`'s first element index
/// and `total` the pool size in elements. Pure — the offset arithmetic is
/// the failure mode of a tiered pool, so it is factored out and tested.
fn h_inter_layout(counts: &[usize]) -> (Vec<usize>, usize) {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    let mut acc = 0usize;
    for &c in counts {
        offsets.push(acc);
        acc += c;
    }
    offsets.push(acc);
    (offsets, acc)
}

/// Allocate one zeroed region of `bytes` per SSM layer, PREFERRING a single
/// contiguous block so `pools[l] == pools[0] + l * bytes`.
///
/// The uniform layer stride is what lets the verify-state copy sets collapse
/// from `2 × num_ssm_layers` eager `copy_d2d_async` launches per sequence to
/// two pitched 2-D copies (`model::ssm_batched_copy`). Nothing depends on the
/// stride existing: if the single block cannot be allocated — a fragmented
/// device, or simply a pool large enough that one VA range is not available
/// — this falls back to the original per-layer allocations, the strided-run
/// detector declines, and the copies run as the same per-layer loop they
/// always did. Bytes allocated, and the addresses every accessor derives, are
/// identical either way.
fn alloc_layer_pools(
    gpu: &dyn GpuBackend,
    num_ssm_layers: usize,
    bytes: usize,
) -> Result<(Vec<DevicePtr>, Vec<DevicePtr>)> {
    if num_ssm_layers == 0 || bytes == 0 {
        return Ok((vec![DevicePtr::NULL; num_ssm_layers], Vec::new()));
    }
    if let Some(total) = bytes.checked_mul(num_ssm_layers)
        && let Ok(base) = gpu.alloc(total)
    {
        gpu.memset(base, 0, total)?;
        let layers = (0..num_ssm_layers)
            .map(|l| base.offset(l * bytes))
            .collect();
        return Ok((layers, vec![base]));
    }
    tracing::warn!(
        "SSM pool: {num_ssm_layers} × {bytes} B did not fit one contiguous block — \
         falling back to per-layer allocations (verify-state rollback keeps the \
         per-layer copy loop)"
    );
    let mut pools = Vec::with_capacity(num_ssm_layers);
    for _ in 0..num_ssm_layers {
        let p = gpu.alloc(bytes)?;
        gpu.memset(p, 0, bytes)?;
        pools.push(p);
    }
    Ok((pools.clone(), pools))
}

impl SsmStatePool {
    /// `h_f16_pool` is `gdn_flags::ssm_h_f16_pool_enabled()` at the one
    /// production call site (`impl_a1.rs`) — a parameter, not a flag read,
    /// so pool geometry stays testable without the process-global cell.
    pub(super) fn new(
        config: &ModelConfig,
        max_slots: usize,
        has_mtp: bool,
        num_intermediates: usize,
        num_drafts: usize,
        h_f16_pool: bool,
        rollback_mode: crate::ssm_reserve::SsmRollbackMode,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let _d_conv = config.linear_conv_kernel_dim;

        let h_bytes = config.ssm_h_state_bytes();
        let h_stored_bytes = crate::ssm_reserve::ssm_h_stored_bytes(h_bytes, h_f16_pool);
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

        let mut owned_allocations = Vec::new();
        let mut h_intermediate_pools = Vec::new();
        let mut conv_intermediate_pools = Vec::new();
        let mut h_checkpoint_pools = Vec::new();
        let mut conv_checkpoint_pools = Vec::new();

        let (h_state_pools, allocations) =
            alloc_layer_pools(gpu, num_ssm_layers, total_slots * h_stored_bytes)?;
        owned_allocations.extend(allocations);
        let (conv_state_pools, allocations) =
            alloc_layer_pools(gpu, num_ssm_layers, total_slots * conv_bytes)?;
        owned_allocations.extend(allocations);

        // Stage-3 f16-SIZED pool: the FP32 prefill staging arena. Allocated
        // ONLY when the h slots actually narrowed — an FP32-sized pool needs
        // no staging and must not pay for it (flag-off is byte-identical).
        let h_prefill_stage_pool = {
            let bytes = crate::ssm_reserve::ssm_h_prefill_stage_bytes(
                total_slots,
                h_bytes,
                h_stored_bytes < h_bytes,
            );
            if bytes == 0 {
                None
            } else {
                let p = gpu.alloc(bytes)?;
                gpu.memset(p, 0, bytes)?;
                owned_allocations.push(p);
                tracing::info!(
                    "SSM f16-sized h pool: FP32 prefill staging arena {} MB ({total_slots} slots × {h_bytes} B)",
                    bytes / (1024 * 1024)
                );
                Some(p)
            }
        };

        // MTP verify pools cover only the slots spec dispatch can reach
        // (SSOT: `ssm_reserve::mtp_state_slots` — the same number preflight
        // reserves and the scheduler guard enforces). Equals `max_slots` at
        // bs<=32; above that it caps at the spec dispatch width (floor 32),
        // saving `(max_slots - mtp_slots) × (ni+1) × blob` — 25.4 GB at
        // bs=64/K=4 on the 27B. `+1` = the MTP pools' own dummy slot.
        let mtp_slots = if has_mtp {
            crate::ssm_reserve::mtp_state_slots(max_slots)
        } else {
            0
        };
        // TIERED per-slot H-intermediate capacity (2026-08-16). SSOT:
        // `ssm_reserve::verify_slot_h_intermediates` — the same numbers
        // preflight reserves and the scheduler's dispatch clamp enforces
        // (`Model::mtp_slot_draft_capacity`). Uniform (every slot at
        // `num_intermediates`) when the pools are DFlash-γ sized
        // (`num_intermediates != num_drafts + 1` — DFlash verify width does
        // not follow the MTP ladder); the kill switch and a disabled ladder
        // make `verify_slot_h_intermediates` itself uniform. The MTP dummy
        // at index `mtp_slots` is ALWAYS full-width (K-1): batched-verify
        // pad rows may write any of the K-1 live token indices regardless
        // of the active slots' tiers.
        // H side allocates K-1 intermediates per K-row verify (index K-1 is
        // never written or read — audit in `verify_slot_h_intermediates`);
        // the CONV side keeps all K because the fused conv kernels write the
        // dead K-1 snapshot on-device. `num_intermediates` remains the K
        // ceiling (conv count); the uniform H count is `num_intermediates-1`.
        let replay = rollback_mode == crate::ssm_reserve::SsmRollbackMode::Replay;
        // DFlash's K=γ verify hits EVERY slot at the full width, so its
        // pools must be uniform — and that must be stated, not inferred.
        // The old inference (`num_intermediates != num_drafts + 1`) only
        // held by accident while the DFlash ceiling was a hardcoded 17:
        // once the pools sized from the real γ (config.dflash_gamma), a
        // γ+1 at or below num_drafts+1 made the widths EQUAL, the inference
        // flipped to the MTP ladder tiers, and slots with 3-6 h
        // intermediates 500'd the first K=8 verify ("SSM MTP intermediate
        // buffers not allocated", 2026-08-19 C=1/2/4 report run).
        let uniform_h =
            !config.dflash_capture_layers.is_empty() || num_intermediates != num_drafts + 1;
        let h_inter_counts: Vec<usize> = if has_mtp && replay {
            // Replay: no per-token snapshots exist — every slot's count is 0
            // (the vec stays populated so accessors keep their shape).
            vec![0; mtp_slots + 1]
        } else if has_mtp {
            (0..=mtp_slots)
                .map(|s| {
                    if s == mtp_slots || uniform_h {
                        num_intermediates.saturating_sub(1)
                    } else {
                        crate::ssm_reserve::verify_slot_h_intermediates(s, num_drafts, false)
                            .min(num_intermediates.saturating_sub(1))
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        let (h_inter_offsets, h_inter_total) = h_inter_layout(&h_inter_counts);
        let mut replay_input_rings = Vec::new();
        if has_mtp {
            let ni = num_intermediates;
            let mtp_total = mtp_slots + 1;
            if !replay {
                let (layers, allocations) =
                    alloc_layer_pools(gpu, num_ssm_layers, h_inter_total * h_stored_bytes)?;
                h_intermediate_pools = layers;
                owned_allocations.extend(allocations);
                let (layers, allocations) =
                    alloc_layer_pools(gpu, num_ssm_layers, mtp_total * ni * conv_bytes)?;
                conv_intermediate_pools = layers;
                owned_allocations.extend(allocations);
            } else {
                // Replay: verify-window INPUT rows instead of state
                // snapshots — (mtp_total slots incl. dummy) × (K-1)
                // rows of qkvz+gates per layer. Sized by the same SSOT
                // preflight reserves through.
                let row = crate::ssm_reserve::ssm_replay_row_bytes(
                    config.ssm_qkvz_size(),
                    config.linear_num_value_heads,
                );
                let ring = crate::ssm_reserve::ssm_replay_ring_bytes(1, row, ni, mtp_total);
                let (layers, allocations) = alloc_layer_pools(gpu, num_ssm_layers, ring)?;
                replay_input_rings = layers;
                owned_allocations.extend(allocations);
            }

            // 1 checkpoint per slot per layer (BOTH modes: replay's
            // reconstruction base is exactly this blob).
            let (layers, allocations) =
                alloc_layer_pools(gpu, num_ssm_layers, mtp_total * h_stored_bytes)?;
            h_checkpoint_pools = layers;
            owned_allocations.extend(allocations);
            let (layers, allocations) =
                alloc_layer_pools(gpu, num_ssm_layers, mtp_total * conv_bytes)?;
            conv_checkpoint_pools = layers;
            owned_allocations.extend(allocations);

            let mtp_mb = num_ssm_layers
                * (h_inter_total * h_stored_bytes
                    + mtp_total * (ni * conv_bytes + h_stored_bytes + conv_bytes))
                / (1024 * 1024);
            // Baseline for the log: FULL-WIDTH UNIFORM sizing at today's
            // per-slot shape ((ni-1) H + ni conv + checkpoint).
            let full_h = mtp_total * ni.saturating_sub(1);
            if mtp_slots < max_slots || h_inter_total < full_h {
                let saved_mb = (num_ssm_layers
                    * (max_slots - mtp_slots)
                    * (ni.saturating_sub(1) * h_stored_bytes
                        + ni * conv_bytes
                        + h_stored_bytes
                        + conv_bytes)
                    + num_ssm_layers * (full_h - h_inter_total) * h_stored_bytes)
                    / (1024 * 1024);
                tracing::info!(
                    "SSM MTP pools (conv {ni}/slot, h tiered {:?}..{:?} + checkpoints): \
                     {mtp_mb} MB, covering {mtp_slots}/{max_slots} slots (spec dispatch \
                     width; saves {saved_mb} MB vs full-width uniform; kill switch \
                     ATLAS_MTP_POOL_FULL_WIDTH)",
                    h_inter_counts.iter().min(),
                    h_inter_counts.iter().max(),
                );
            } else {
                tracing::info!("SSM MTP pools ({ni} intermediates + checkpoints): {mtp_mb} MB");
            }
        }

        // free_slots holds claimable indices only; the dummy at index
        // `max_slots` is permanently reserved.
        let free_slots: Vec<usize> = (0..max_slots).rev().collect();

        let total_mb = num_ssm_layers * max_slots * (h_stored_bytes + conv_bytes) / (1024 * 1024);
        tracing::info!(
            "SSM state pool: {max_slots} slots × {num_ssm_layers} layers = {total_mb} MB",
        );

        Ok(Self {
            owned_allocations,
            h_state_pools,
            conv_state_pools,
            h_intermediate_pools,
            conv_intermediate_pools,
            h_checkpoint_pools,
            conv_checkpoint_pools,
            h_bytes,
            h_stored_bytes,
            h_prefill_stage_pool,
            conv_bytes,
            max_slots,
            mtp_slots,
            num_ssm_layers,
            has_mtp,
            num_intermediates,
            h_inter_counts,
            h_inter_offsets,
            rollback_mode,
            replay_input_rings,
            free_slots: Mutex::new(free_slots),
        })
    }

    /// Map a pool slot onto the (possibly narrower) MTP verify pools.
    ///
    /// Covered slots (`< mtp_slots`) map 1:1 — at `max_slots <= 32` that is
    /// every slot AND the base dummy (`max_slots == mtp_slots` ⇒ dummy maps
    /// to the MTP dummy at the same index), so addressing is byte-identical
    /// to the pre-diet layout. Uncovered slots (only possible at bs>32)
    /// clamp onto the MTP dummy at index `mtp_slots`: their pointers are
    /// computed for every sequence's `SsmLayerState` at alloc/compact time
    /// but are only ever DEREFERENCED by verify paths, which the scheduler
    /// guard (`slot_idx < ssm_reserve::mtp_state_slots(bs)`) keeps away
    /// from uncovered slots — the clamp just makes a missed guard
    /// memory-safe (scratch write) instead of out-of-bounds.
    #[inline]
    fn mtp_slot(&self, slot: usize) -> usize {
        if slot < self.mtp_slots {
            slot
        } else {
            self.mtp_slots
        }
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
        let mut free = self.free_slots.lock();
        debug_assert!(
            !free.contains(&idx),
            "release_slot: slot {idx} already free (double-release hands it to two seqs)"
        );
        free.push(idx);
    }

    /// Remove a SPECIFIC slot from the free list if present, returning whether
    /// it was. Used by `compact_sequence` to claim a known-free migration
    /// target EXCLUSIVELY, so a slot is never simultaneously owned and free —
    /// the bug-2 invariant (an owned-and-free slot gets handed to two sequences
    /// by `claim_slot`, sharing GDN state → cross-stream corruption).
    pub(super) fn claim_specific(&self, slot: usize) -> bool {
        let mut free = self.free_slots.lock();
        if let Some(pos) = free.iter().position(|&s| s == slot) {
            free.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// True when `idx` is currently on the free list — claimable, owned by no
    /// live sequence. Drain-tail graph borrowing (`graph_borrow.rs`) uses this
    /// to prove a borrowed graph's tail rows only scribble on unowned state.
    /// The scheduler thread is the only claimer/releaser, so the answer is
    /// stable for the duration of a dispatch it performs itself.
    pub(super) fn slot_is_free(&self, idx: usize) -> bool {
        self.free_slots.lock().contains(&idx)
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
            gpu.memset_async(self.h_state(i, idx), 0, self.h_stored_bytes, stream)?;
            gpu.memset_async(self.conv_state(i, idx), 0, self.conv_bytes, stream)?;
        }
        Ok(())
    }

    pub(super) fn h_state(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.h_state_pools[ssm_layer_idx].offset(slot * self.h_stored_bytes)
    }

    /// This slot's FP32 prefill staging blob (stage-3 f16-SIZED pool only).
    ///
    /// `None` under an FP32-sized pool — the ONLY signal the prefill path
    /// needs to know that it must widen/narrow, and the reason flag-off
    /// prefill is byte-identical (there is nothing to convert through).
    /// Layer-independent by construction: see the field's doc.
    pub(super) fn h_prefill_stage(&self, slot: usize) -> Option<DevicePtr> {
        self.h_prefill_stage_pool
            .map(|p| p.offset(slot * self.h_bytes))
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
            // Storage width, not FP32 width: reading h_bytes off a stage-3
            // f16-sized slot would run past the slot. The FP32 reduction
            // below is only meaningful over an FP32-holding slot; on an FP16
            // slot the triple still flags divergence (same bits => same
            // sums), it just isn't interpretable as values.
            let mut hb = vec![0u8; self.h_stored_bytes];
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
        let slot = self.mtp_slot(slot);
        debug_assert!(
            token_idx < self.h_inter_counts[slot],
            "h_intermediate: token_idx {token_idx} >= slot {slot}'s tiered capacity {}",
            self.h_inter_counts[slot],
        );
        self.h_intermediate_pools[ssm_layer_idx]
            .offset((self.h_inter_offsets[slot] + token_idx) * self.h_stored_bytes)
    }

    /// Number of H intermediates allocated for `slot` (tiered — see
    /// `h_inter_counts`). Uncovered slots clamp onto the full-width MTP
    /// dummy, mirroring the pointer accessors. 0 when `!has_mtp`.
    #[inline]
    pub(super) fn h_inter_count(&self, slot: usize) -> usize {
        if !self.has_mtp {
            return 0;
        }
        self.h_inter_counts[self.mtp_slot(slot)]
    }

    /// Verify DRAFT capacity of `slot`'s H-intermediate allocation: the
    /// deepest `num_drafts` a speculative step may dispatch to a sequence
    /// occupying it. The scheduler's dispatch clamp
    /// (`Model::mtp_slot_draft_capacity`) reads THIS — the allocated
    /// geometry, not a re-derivation of the policy — so pool and dispatch
    /// cannot disagree. `usize::MAX` when there are no verify pools to
    /// constrain; 0 for uncovered slots (spec dispatch is already gated off
    /// them by the `spec_slots_covered` guard).
    #[inline]
    pub(crate) fn verify_draft_capacity(&self, slot: usize) -> usize {
        if !self.has_mtp {
            return usize::MAX;
        }
        // Replay scaffold: capacity is NOT snapshot-bounded (no per-token
        // snapshots exist to overflow). Report unconstrained so the
        // scheduler does not silently zero out speculation — dispatch then
        // hits `require_verify_rollback_supported`'s LOUD refusal instead.
        if self.rollback_mode == crate::ssm_reserve::SsmRollbackMode::Replay {
            return usize::MAX;
        }
        if slot >= self.mtp_slots {
            return 0;
        }
        // K-1 sizing: the count IS the draft capacity (a K-row verify
        // writes K-1 H snapshots, indices 0..K-2).
        self.h_inter_counts[slot]
    }

    /// Refuse a speculative VERIFY under the replay scaffold. The replay
    /// mode's device path (verify-window input capture + checkpoint-replay
    /// reconstruction) is not wired; running a verify would either index
    /// empty intermediate vecs (opaque) or, worse, roll back to nothing
    /// (silent corruption). Called by every `decode_verify*` entry.
    pub(crate) fn require_verify_rollback_supported(&self) -> Result<()> {
        if self.rollback_mode == crate::ssm_reserve::SsmRollbackMode::Replay
            && self.num_ssm_layers > 0
        {
            bail!(
                "--ssm-rollback-mode replay is an EXPERIMENTAL scaffold: the verify-window \
                 input capture and checkpoint-replay reconstruction are not wired yet, so \
                 speculative verify cannot run. The serve boots (reserve sizing shows the \
                 replay capacity win) but --speculative traffic must use \
                 --ssm-rollback-mode snapshot."
            );
        }
        Ok(())
    }

    pub(super) fn conv_intermediate(
        &self,
        ssm_layer_idx: usize,
        slot: usize,
        token_idx: usize,
    ) -> DevicePtr {
        let ni = self.num_intermediates;
        let slot = self.mtp_slot(slot);
        self.conv_intermediate_pools[ssm_layer_idx]
            .offset((slot * ni + token_idx) * self.conv_bytes)
    }

    pub(super) fn h_checkpoint(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.h_checkpoint_pools[ssm_layer_idx].offset(self.mtp_slot(slot) * self.h_stored_bytes)
    }

    pub(super) fn conv_checkpoint(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.conv_checkpoint_pools[ssm_layer_idx].offset(self.mtp_slot(slot) * self.conv_bytes)
    }

    pub(super) fn reset_slot(&self, slot: usize, gpu: &dyn GpuBackend) -> Result<()> {
        // MTP pools only cover slots < mtp_slots (bs>32 diet); an uncovered
        // slot has no MTP state of its own to reset (its accessors clamp to
        // the shared MTP dummy — zeroing that would be wasted work).
        let reset_mtp = self.has_mtp && slot < self.mtp_slots;
        for i in 0..self.num_ssm_layers {
            gpu.memset(self.h_state(i, slot), 0, self.h_stored_bytes)?;
            gpu.memset(self.conv_state(i, slot), 0, self.conv_bytes)?;
            if reset_mtp {
                for t in 0..self.h_inter_count(slot) {
                    gpu.memset(self.h_intermediate(i, slot, t), 0, self.h_stored_bytes)?;
                }
                for t in 0..self.num_intermediates {
                    gpu.memset(self.conv_intermediate(i, slot, t), 0, self.conv_bytes)?;
                }
                gpu.memset(self.h_checkpoint(i, slot), 0, self.h_stored_bytes)?;
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
        // Compaction can migrate FROM an uncovered slot (bs>32: source slot
        // >= mtp_slots) — such a sequence has never speculated (the dispatch
        // guard requires slot < mtp_slots to verify), so it has no live MTP
        // checkpoint/intermediate state to carry; copying the clamped dummy
        // would smear scratch bytes over the target. Targets are always
        // in-range (`compact_survivors_into_range` picks targets < n), so
        // `to` uncovered can only pair with `from` uncovered.
        let copy_mtp = self.has_mtp && from < self.mtp_slots && to < self.mtp_slots;
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                self.h_state(i, from),
                self.h_state(i, to),
                self.h_stored_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.conv_state(i, from),
                self.conv_state(i, to),
                self.conv_bytes,
                stream,
            )?;
            if copy_mtp {
                // Tiered H: compaction migrates DOWNWARD (targets < n <=
                // source), so `to`'s tier is >= `from`'s and the prefix copy
                // is lossless. The intermediates are per-verify-step scratch
                // regardless — nothing in them survives a step — so even a
                // truncated copy could not lose live state.
                for t in 0..self.h_inter_count(from).min(self.h_inter_count(to)) {
                    gpu.copy_d2d_async(
                        self.h_intermediate(i, from, t),
                        self.h_intermediate(i, to, t),
                        self.h_stored_bytes,
                        stream,
                    )?;
                }
                for t in 0..self.num_intermediates {
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
                    self.h_stored_bytes,
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

/// Release every per-layer state pool.
///
/// The intermediate and checkpoint pools are only allocated when MTP is on, so
/// the vectors are empty otherwise — draining handles both without a branch.
impl atlas_core::scope::ModelResource<dyn GpuBackend> for SsmStatePool {
    fn label(&self) -> &'static str {
        "ssm state pool"
    }

    fn release(&mut self, gpu: &dyn GpuBackend) -> anyhow::Result<()> {
        let mut first_error = None;
        for ptr in self.owned_allocations.drain(..) {
            if let Err(e) = gpu.free(ptr)
                && first_error.is_none()
            {
                first_error = Some(e);
            }
        }
        self.h_state_pools.clear();
        self.conv_state_pools.clear();
        self.h_intermediate_pools.clear();
        self.conv_intermediate_pools.clear();
        self.h_checkpoint_pools.clear();
        self.conv_checkpoint_pools.clear();
        self.h_prefill_stage_pool = None;
        self.replay_input_rings.clear();
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod h_inter_layout_tests {
    use super::h_inter_layout;

    #[test]
    fn offsets_are_prefix_sums_and_total_is_the_sum() {
        // The default-ladder tier shape at nd=3, 32 covered slots + dummy,
        // K-1 sizing: 8 full slots (3), 24 low slots (1), dummy full (3).
        let mut counts = vec![3usize; 8];
        counts.extend(std::iter::repeat_n(1usize, 24));
        counts.push(3);
        let (offsets, total) = h_inter_layout(&counts);
        assert_eq!(offsets.len(), counts.len() + 1);
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[8], 24); // tier-1 region: 8 × 3
        assert_eq!(offsets[32], 24 + 24); // + tier-2 region: 24 × 1
        assert_eq!(total, 51); // + dummy 3
        for s in 0..counts.len() {
            assert_eq!(offsets[s + 1] - offsets[s], counts[s]);
        }
        // Uniform counts reproduce the legacy `slot * ni` addressing.
        let (uni, uni_total) = h_inter_layout(&[5usize; 33]);
        assert_eq!(uni_total, 165);
        for (s, off) in uni.iter().take(33).enumerate() {
            assert_eq!(*off, s * 5);
        }
        // Empty (no MTP pools) is a valid degenerate layout.
        assert_eq!(h_inter_layout(&[]), (vec![0], 0));
    }
}

#[cfg(test)]
mod h_stored_geometry_tests {
    use super::*;
    use atlas_core::config::ModelConfig;
    use atlas_core::scope::ModelResource;
    use spark_runtime::gpu::mock::MockGpuBackend;

    /// Claimable slots in the test pool (the pool allocates `SLOTS + 1`,
    /// the extra one being the padding dummy).
    const SLOTS: usize = 4;

    /// Build the pool on the mock backend (only `alloc`/`memset` are hit).
    fn pool(h_f16_pool: bool) -> SsmStatePool {
        let config = ModelConfig::qwen3_next_80b_nvfp4();
        let gpu = MockGpuBackend::new();
        SsmStatePool::new(
            &config,
            SLOTS,
            true,
            4,
            3,
            h_f16_pool,
            crate::ssm_reserve::SsmRollbackMode::Snapshot,
            &gpu,
        )
        .unwrap()
    }

    /// Every pool family is ONE contiguous block with a uniform per-layer
    /// stride. This is the enabling precondition for the batched verify-state
    /// rollback (`model::ssm_batched_copy`): without it the 2 × num_ssm_layers
    /// copy plan cannot collapse to two pitched 2-D transfers and the verify
    /// path silently keeps its 96-launch-per-sequence loop. Pinned here
    /// because nothing else would notice the regression — the addresses stay
    /// correct, only the launch count changes.
    #[test]
    fn layer_pools_are_one_contiguous_block_per_family() {
        let p = pool(false);
        assert_eq!(
            p.owned_allocations.len(),
            6,
            "each family must own one bulk allocation"
        );
        let families: [(&str, &[DevicePtr], usize); 6] = [
            (
                "h_state",
                &p.h_state_pools,
                (p.max_slots + 1) * p.h_stored_bytes,
            ),
            (
                "conv_state",
                &p.conv_state_pools,
                (p.max_slots + 1) * p.conv_bytes,
            ),
            (
                "h_intermediate",
                &p.h_intermediate_pools,
                *p.h_inter_offsets.last().unwrap() * p.h_stored_bytes,
            ),
            (
                "conv_intermediate",
                &p.conv_intermediate_pools,
                (p.mtp_slots + 1) * p.num_intermediates * p.conv_bytes,
            ),
            (
                "h_checkpoint",
                &p.h_checkpoint_pools,
                (p.mtp_slots + 1) * p.h_stored_bytes,
            ),
            (
                "conv_checkpoint",
                &p.conv_checkpoint_pools,
                (p.mtp_slots + 1) * p.conv_bytes,
            ),
        ];
        for (name, pools, stride) in families {
            assert_eq!(
                pools.len(),
                p.num_ssm_layers,
                "{name}: one region per layer"
            );
            assert!(stride > 0, "{name}: degenerate stride");
            for (l, ptr) in pools.iter().enumerate() {
                assert_eq!(
                    ptr.0,
                    pools[0].0 + (l * stride) as u64,
                    "{name}: layer {l} is not at base + {l}*{stride}"
                );
            }
        }
    }

    #[test]
    fn rejected_bulk_allocation_falls_back_to_owned_layer_allocations() {
        let gpu = MockGpuBackend::new();
        gpu.set_max_allocation_bytes(1024);
        let (layers, owners) = alloc_layer_pools(&gpu, 4, 512).unwrap();
        assert_eq!(layers, owners, "fallback views must be allocation bases");
        assert_eq!(gpu.alloc_count(), 4);
        for ptr in owners {
            assert_eq!(gpu.read_alloc(ptr).unwrap(), vec![0; 512]);
            gpu.free(ptr).unwrap();
        }
        assert_eq!(gpu.alloc_count(), 0);
    }

    #[test]
    fn release_frees_backing_allocations_not_layer_views() {
        let config = ModelConfig::qwen3_next_80b_nvfp4();
        for (mode, narrowed) in [
            (crate::ssm_reserve::SsmRollbackMode::Snapshot, true),
            (crate::ssm_reserve::SsmRollbackMode::Replay, false),
        ] {
            let gpu = MockGpuBackend::new();
            let mut p =
                SsmStatePool::new(&config, SLOTS, true, 4, 3, narrowed, mode, &gpu).unwrap();
            assert!(gpu.alloc_count() > 0, "fixture must own device allocations");
            p.release(&gpu).unwrap();
            assert_eq!(gpu.alloc_count(), 0, "mode={mode:?}, narrowed={narrowed}");
            assert!(p.owned_allocations.is_empty());
            assert!(p.h_prefill_stage_pool.is_none());
            assert!(p.replay_input_rings.is_empty());
        }
    }

    /// Replay-scaffold pool geometry: no per-token intermediates, checkpoints
    /// and the input ring allocated, verify refused loudly, dispatch capacity
    /// unconstrained (the refusal must be LOUD, never a silent zero-draft).
    #[test]
    fn replay_pool_has_checkpoints_and_ring_but_no_intermediates() {
        let config = ModelConfig::qwen3_next_80b_nvfp4();
        let gpu = MockGpuBackend::new();
        let p = SsmStatePool::new(
            &config,
            4,
            true,
            4,
            3,
            false,
            crate::ssm_reserve::SsmRollbackMode::Replay,
            &gpu,
        )
        .unwrap();
        assert!(p.h_intermediate_pools.is_empty());
        assert!(p.conv_intermediate_pools.is_empty());
        assert_eq!(p.h_inter_counts, vec![0; p.mtp_slots + 1]);
        assert_eq!(p.h_checkpoint_pools.len(), p.num_ssm_layers);
        assert_eq!(p.replay_input_rings.len(), p.num_ssm_layers);
        assert_eq!(p.verify_draft_capacity(0), usize::MAX);
        let err = p.require_verify_rollback_supported().unwrap_err();
        assert!(err.to_string().contains("EXPERIMENTAL"), "{err}");
        // Snapshot mode: supported, and its geometry untouched.
        let snap = pool(false);
        assert!(snap.require_verify_rollback_supported().is_ok());
        assert!(snap.replay_input_rings.is_empty());
        assert!(!snap.h_intermediate_pools.is_empty());
    }

    /// Stage-3 sizing is a STORAGE-width change, not a layout change: every
    /// h family (base slots, tiered intermediates, checkpoints) strides by
    /// `h_stored_bytes`, and flag-off that width IS `h_bytes` — pinning
    /// that the default geometry did not move by a byte.
    #[test]
    fn h_families_stride_by_the_stored_width() {
        for f16 in [false, true] {
            let p = pool(f16);
            let expect = if f16 { p.h_bytes / 2 } else { p.h_bytes };
            assert_eq!(p.h_stored_bytes, expect, "f16={f16}");
            assert_eq!(
                p.h_state(0, 1).0 - p.h_state(0, 0).0,
                expect as u64,
                "base slot stride (f16={f16})"
            );
            assert_eq!(
                p.h_intermediate(0, 0, 1).0 - p.h_intermediate(0, 0, 0).0,
                expect as u64,
                "intermediate stride (f16={f16})"
            );
            assert_eq!(
                p.h_checkpoint(0, 1).0 - p.h_checkpoint(0, 0).0,
                expect as u64,
                "checkpoint stride (f16={f16})"
            );
            // Conv never narrows — its kernels are FP32 writers in every mode.
            assert_eq!(
                p.conv_state(0, 1).0 - p.conv_state(0, 0).0,
                p.conv_bytes as u64,
                "conv stride (f16={f16})"
            );
        }
    }

    /// The FP32 element-count authority (`h_bytes`) must NOT narrow with the
    /// storage width — converters and kernels derive `n = h_bytes / 4` from
    /// it in both modes.
    #[test]
    fn h_bytes_stays_the_fp32_width() {
        let p32 = pool(false);
        let p16 = pool(true);
        assert_eq!(p16.h_bytes, p32.h_bytes);
        assert_eq!(p16.h_stored_bytes * 2, p16.h_bytes);
    }

    /// The FP32 prefill staging arena exists ONLY under the narrowed pool,
    /// is FP32-wide (never `h_stored_bytes` — staging the FP32 kernels write
    /// is the entire point), and strides ONE blob per SLOT, not per slot per
    /// layer. A per-layer arena would be `num_ssm_layers ×` bigger and would
    /// eat the win this mode exists for, so the stride is pinned, not
    /// assumed.
    #[test]
    fn prefill_staging_is_one_fp32_blob_per_slot_and_only_when_narrowed() {
        // Flag off: not allocated, and every slot answers `None` — the
        // signal the prefill path takes its historical in-place arm on.
        let p32 = pool(false);
        assert!(p32.h_prefill_stage_pool.is_none());
        assert!(p32.h_prefill_stage(0).is_none());
        assert!(p32.h_prefill_stage(SLOTS - 1).is_none());

        let p16 = pool(true);
        assert!(p16.h_prefill_stage_pool.is_some());
        let s0 = p16.h_prefill_stage(0).expect("narrowed pool stages");
        let s1 = p16.h_prefill_stage(1).expect("narrowed pool stages");
        // FP32 pitch — twice the (narrowed) slot pitch it feeds.
        assert_eq!(s1.0 - s0.0, p16.h_bytes as u64);
        assert_eq!(s1.0 - s0.0, 2 * p16.h_stored_bytes as u64);
        // The dummy slot is staged too: `SsmStatePool::new` allocates
        // `max_slots + 1` blobs, so `dummy_slot()` is addressable rather
        // than one blob past the end.
        let dummy = p16.h_prefill_stage(p16.dummy_slot()).unwrap();
        assert_eq!(dummy.0 - s0.0, (p16.dummy_slot() * p16.h_bytes) as u64);

        // SSOT with the preflight reserve: same function, same numbers.
        assert_eq!(
            crate::ssm_reserve::ssm_h_prefill_stage_bytes(SLOTS + 1, p16.h_bytes, true),
            (SLOTS + 1) * p16.h_bytes
        );
        assert_eq!(
            crate::ssm_reserve::ssm_h_prefill_stage_bytes(SLOTS + 1, p16.h_bytes, false),
            0
        );
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
            owned_allocations: Vec::new(),
            h_state_pools: Vec::new(),
            conv_state_pools: Vec::new(),
            h_intermediate_pools: Vec::new(),
            conv_intermediate_pools: Vec::new(),
            h_checkpoint_pools: Vec::new(),
            conv_checkpoint_pools: Vec::new(),
            h_bytes: 0,
            h_stored_bytes: 0,
            h_prefill_stage_pool: None,
            conv_bytes: 0,
            max_slots,
            mtp_slots: 0,
            num_ssm_layers: 0,
            has_mtp: false,
            num_intermediates: 0,
            h_inter_counts: Vec::new(),
            h_inter_offsets: Vec::new(),
            rollback_mode: crate::ssm_reserve::SsmRollbackMode::Snapshot,
            replay_input_rings: Vec::new(),
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
}
