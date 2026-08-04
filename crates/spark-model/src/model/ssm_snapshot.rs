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
use super::ssm_pool::SsmStatePool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Pre-allocated GPU memory pool for SSM state snapshots.
///
/// Each snapshot slot stores a copy of h_state + conv_state for all SSM layers
/// at a specific point in a token sequence.
///
/// The pool serves **two** independent consumers from one set of GPU
/// allocations (SSOT — one snapshot mechanism, one D2D copy primitive):
///
/// 1. **Marconi prefix caching** — the LRU-managed `[0, num_slots)` slot
///    region, allocated/freed via [`save`](Self::save) / [`free`](Self::free)
///    against the `free_slots` list. When a prefix cache hit occurs the
///    snapshot is restored to skip SSM recompute for cached tokens.
///
/// 2. **Phase-C decode-time boundary rollback** — a *separate*,
///    deterministically-addressed `[0, decode_ring_slots)` region (per
///    active sequence). No free list: ring slot `r` for SSM-pool
///    sequence slot `s` lives at flat index `s * ring_slots + r`, so a
///    sequence's snapshots never collide with another's and never
///    contend with Marconi's LRU slots. Sized for `max_batch_size`
///    sequences so the watchdog rollback always has capacity.
pub(crate) struct SsmSnapshotPool {
    pub(super) h_snapshots: Vec<DevicePtr>,
    pub(super) conv_snapshots: Vec<DevicePtr>,
    pub(super) free_slots: Mutex<Vec<usize>>,
    pub(super) num_slots: usize,
    pub(super) h_bytes: usize,
    pub(super) conv_bytes: usize,
    pub(super) num_ssm_layers: usize,
    /// Maps snapshot_slot_id → session_hash for session-scoped isolation.
    /// When restoring, skip snapshots that belong to a different session.
    pub(super) session_tags: Mutex<std::collections::HashMap<usize, u64>>,
    /// Decode-rollback region: `h_snapshots` for the Phase-C ring.
    /// Layout per layer: `[max_batch_size * decode_ring_slots * h_bytes]`.
    /// Empty when `decode_ring_slots == 0`.
    pub(super) decode_h_snapshots: Vec<DevicePtr>,
    /// Decode-rollback region: `conv_snapshots` for the Phase-C ring.
    pub(super) decode_conv_snapshots: Vec<DevicePtr>,
    /// Number of decode-rollback ring slots reserved per active sequence.
    /// 0 disables the decode-rollback region entirely.
    pub(super) decode_ring_slots: usize,
    /// Number of active-sequence slots the decode region is sized for
    /// (equals `max_batch_size`). A sequence's SSM-pool `slot_idx` must
    /// be `< decode_max_seqs` to use the decode region.
    pub(super) decode_max_seqs: usize,
    /// Last-token post-final-norm hidden state for each Marconi snapshot
    /// slot. Single buffer of `num_slots * hidden_bytes`; slot `s` lives
    /// at `offset(s * hidden_bytes)`. NULL when Marconi is disabled.
    ///
    /// Marconi's leaf snapshot stores SSM recurrent state *after* the last
    /// token (state@N). On an exact full-prompt hit the engine must
    /// produce the first generated token's logits — which normally come
    /// from re-running the last prompt token's forward. For SSM layers
    /// that re-run would apply the last token's recurrent update a second
    /// time on top of state@N (double-advance → corruption). Instead we
    /// stash the last token's post-norm hidden here at save time and feed
    /// it straight to `lm_head` on the hit, skipping any SSM re-run.
    pub(super) hidden_snapshot: DevicePtr,
    /// Byte size of one slot's last-token hidden (`hidden_size * 2`, BF16).
    pub(super) hidden_bytes: usize,
    /// Marconi slots that currently hold a valid `hidden_snapshot` entry
    /// (only leaf saves populate it; intermediate checkpoints do not).
    pub(super) slot_has_hidden: Mutex<std::collections::HashSet<usize>>,
}

impl SsmSnapshotPool {
    /// Build the snapshot pool.
    ///
    /// `num_slots` sizes the Marconi LRU region; `decode_ring_slots` ×
    /// `decode_max_seqs` sizes the Phase-C decode-rollback region. A
    /// pool with `num_slots == 0` but `decode_ring_slots > 0` is valid
    /// (decode rollback enabled, Marconi caching disabled) and vice
    /// versa — the two regions are independent.
    pub(super) fn new(
        num_slots: usize,
        h_bytes: usize,
        conv_bytes: usize,
        num_ssm_layers: usize,
        decode_ring_slots: usize,
        decode_max_seqs: usize,
        hidden_bytes: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let decode_enabled = num_ssm_layers > 0 && decode_ring_slots > 0 && decode_max_seqs > 0;
        let marconi_enabled = num_ssm_layers > 0 && num_slots > 0;

        if !marconi_enabled && !decode_enabled {
            return Ok(Self {
                h_snapshots: Vec::new(),
                conv_snapshots: Vec::new(),
                free_slots: Mutex::new(Vec::new()),
                num_slots: 0,
                h_bytes,
                conv_bytes,
                num_ssm_layers,
                session_tags: Mutex::new(std::collections::HashMap::new()),
                decode_h_snapshots: Vec::new(),
                decode_conv_snapshots: Vec::new(),
                decode_ring_slots: 0,
                decode_max_seqs: 0,
                hidden_snapshot: DevicePtr::NULL,
                hidden_bytes,
                slot_has_hidden: Mutex::new(std::collections::HashSet::new()),
            });
        }

        let mut h_snapshots = Vec::new();
        let mut conv_snapshots = Vec::new();
        let mut hidden_snapshot = DevicePtr::NULL;
        if marconi_enabled {
            for _ in 0..num_ssm_layers {
                h_snapshots.push(gpu.alloc(num_slots * h_bytes)?);
                conv_snapshots.push(gpu.alloc(num_slots * conv_bytes)?);
            }
            hidden_snapshot = gpu.alloc(num_slots * hidden_bytes)?;
        }

        let mut decode_h_snapshots = Vec::new();
        let mut decode_conv_snapshots = Vec::new();
        let decode_region = if decode_enabled {
            decode_max_seqs * decode_ring_slots
        } else {
            0
        };
        if decode_enabled {
            for _ in 0..num_ssm_layers {
                decode_h_snapshots.push(gpu.alloc(decode_region * h_bytes)?);
                decode_conv_snapshots.push(gpu.alloc(decode_region * conv_bytes)?);
            }
        }

        let free_slots: Vec<usize> = if marconi_enabled {
            (0..num_slots).rev().collect()
        } else {
            Vec::new()
        };
        let marconi_mb = num_ssm_layers * num_slots * (h_bytes + conv_bytes) / (1024 * 1024);
        let decode_mb = num_ssm_layers * decode_region * (h_bytes + conv_bytes) / (1024 * 1024);
        tracing::info!(
            "SSM snapshot pool: Marconi {num_slots} slots ({marconi_mb} MB), \
             decode-rollback {decode_ring_slots} slots × {decode_max_seqs} seqs \
             ({decode_mb} MB), {num_ssm_layers} layers",
        );

        Ok(Self {
            h_snapshots,
            conv_snapshots,
            free_slots: Mutex::new(free_slots),
            num_slots: if marconi_enabled { num_slots } else { 0 },
            h_bytes,
            conv_bytes,
            num_ssm_layers,
            session_tags: Mutex::new(std::collections::HashMap::new()),
            decode_h_snapshots,
            decode_conv_snapshots,
            decode_ring_slots: if decode_enabled { decode_ring_slots } else { 0 },
            decode_max_seqs: if decode_enabled { decode_max_seqs } else { 0 },
            hidden_snapshot,
            hidden_bytes,
            slot_has_hidden: Mutex::new(std::collections::HashSet::new()),
        })
    }

    /// Marconi prefix-cache region availability.
    pub(super) fn is_enabled(&self) -> bool {
        self.num_slots > 0
    }

    /// Phase-C decode-rollback region availability.
    pub(super) fn decode_rollback_enabled(&self) -> bool {
        self.decode_ring_slots > 0 && !self.decode_h_snapshots.is_empty()
    }

    /// Save the SSM state of pool slot `ssm_slot` into the decode-rollback
    /// ring slot `(ssm_slot, ring_slot)`. Deterministic addressing — no
    /// free list, no eviction. Errors if the decode region is disabled
    /// or the indices are out of the reserved range (fail fast — a
    /// silent skip would leave the watchdog rollback unable to undo SSM
    /// state, corrupting every subsequent decode).
    pub(super) fn save_decode(
        &self,
        ssm_slot: usize,
        ring_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let flat = self.decode_flat_index(ssm_slot, ring_slot)?;
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                main_pool.h_state(i, ssm_slot),
                self.decode_h_snapshots[i].offset(flat * self.h_bytes),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                main_pool.conv_state(i, ssm_slot),
                self.decode_conv_snapshots[i].offset(flat * self.conv_bytes),
                self.conv_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// Restore the SSM state of pool slot `ssm_slot` from the
    /// decode-rollback ring slot `(ssm_slot, ring_slot)`.
    pub(super) fn restore_decode(
        &self,
        ssm_slot: usize,
        ring_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let flat = self.decode_flat_index(ssm_slot, ring_slot)?;
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                self.decode_h_snapshots[i].offset(flat * self.h_bytes),
                main_pool.h_state(i, ssm_slot),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.decode_conv_snapshots[i].offset(flat * self.conv_bytes),
                main_pool.conv_state(i, ssm_slot),
                self.conv_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// Flat index into the decode-rollback region, with bounds checks.
    fn decode_flat_index(&self, ssm_slot: usize, ring_slot: usize) -> Result<usize> {
        if !self.decode_rollback_enabled() {
            bail!("SSM decode-rollback region not allocated");
        }
        if ssm_slot >= self.decode_max_seqs {
            bail!(
                "SSM decode-rollback: ssm_slot {ssm_slot} >= reserved {} seqs",
                self.decode_max_seqs
            );
        }
        if ring_slot >= self.decode_ring_slots {
            bail!(
                "SSM decode-rollback: ring_slot {ring_slot} >= reserved {} slots",
                self.decode_ring_slots
            );
        }
        Ok(ssm_slot * self.decode_ring_slots + ring_slot)
    }

    /// Save SSM state from active pool slot into a snapshot slot.
    /// Returns `None` if no free snapshot slots are available.
    /// Tags the snapshot with `session_hash` for session-scoped isolation.
    pub(super) fn save(
        &self,
        ssm_slot: usize,
        session_hash: u64,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<usize>> {
        if !self.is_enabled() {
            return Ok(None);
        }
        let snap_slot = match self.free_slots.lock().pop() {
            Some(s) => s,
            None => return Ok(None),
        };
        // Reusing a freed slot: drop any stale last-token hidden tag. The
        // caller re-populates it via `save_hidden` for leaf snapshots only.
        self.slot_has_hidden.lock().remove(&snap_slot);
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                main_pool.h_state(i, ssm_slot),
                self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                main_pool.conv_state(i, ssm_slot),
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                self.conv_bytes,
                stream,
            )?;
        }
        if session_hash != 0 {
            self.session_tags.lock().insert(snap_slot, session_hash);
        }
        Ok(Some(snap_slot))
    }

    /// Check if a snapshot belongs to the given session.
    /// Returns true if: session tracking is disabled (hash=0), no tag exists, or tags match.
    pub(super) fn session_matches(&self, snap_slot: usize, session_hash: u64) -> bool {
        if session_hash == 0 {
            return true;
        } // Legacy: no session tracking
        let tags = self.session_tags.lock();
        match tags.get(&snap_slot) {
            None => true, // Untagged snapshot (pre-session-manager) — allow
            Some(&tag) => tag == session_hash,
        }
    }

    /// Restore SSM state from a snapshot slot into an active pool slot.
    pub(super) fn restore(
        &self,
        snap_slot: usize,
        ssm_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                main_pool.h_state(i, ssm_slot),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                main_pool.conv_state(i, ssm_slot),
                self.conv_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// Return a snapshot slot to the free list. Clears the slot's session
    /// tag: a freed slot carries no restorable state, so leaving the tag
    /// would make [`Self::session_has_history`] report phantom history.
    pub(super) fn free(&self, snap_slot: usize) {
        self.slot_has_hidden.lock().remove(&snap_slot);
        self.session_tags.lock().remove(&snap_slot);
        self.free_slots.lock().push(snap_slot);
    }

    /// Whether any LIVE snapshot slot is tagged with `session_hash` — i.e.
    /// this session has produced at least one snapshot before (a prior turn
    /// finished, or a prior tail was captured). Used by the mid-chunk tail
    /// capture to skip sessions on first sight: `session_hash` is a hash of
    /// the first ≤1024 prompt tokens, so single-turn traffic gets a unique
    /// hash per request and can never reuse a captured tail, while
    /// multi-turn agents (stable long system prompt) match from their
    /// second request onward — exactly when tail reuse begins.
    pub(crate) fn session_has_history(&self, session_hash: u64) -> bool {
        session_hash != 0
            && self
                .session_tags
                .lock()
                .values()
                .any(|&s| s == session_hash)
    }

    /// Reserve a Marconi snapshot slot for an in-pass MID-CHUNK tail capture.
    /// Pops a free slot, tags it with `session_hash`, and clears any stale
    /// last-token-hidden marker (a tail snapshot is never a leaf). Returns
    /// `None` when the pool is exhausted; the caller may `reclaim_from_cache`
    /// and retry, or skip capture. Mirrors the bookkeeping `save` performs so
    /// `restore` and session isolation behave identically for this slot.
    pub(crate) fn reserve_tail_slot(&self, session_hash: u64) -> Option<usize> {
        if !self.is_enabled() {
            return None;
        }
        let snap_slot = self.free_slots.lock().pop()?;
        self.slot_has_hidden.lock().remove(&snap_slot);
        if session_hash != 0 {
            self.session_tags.lock().insert(snap_slot, session_hash);
        }
        Some(snap_slot)
    }

    /// Per-SSM-layer h_state snapshot destination for `snap_slot`
    /// (byte offset into the layer's slot region already applied).
    pub(crate) fn tail_h_dst(&self, ssm_layer: usize, snap_slot: usize) -> DevicePtr {
        self.h_snapshots[ssm_layer].offset(snap_slot * self.h_bytes)
    }

    /// Per-SSM-layer conv_state snapshot destination for `snap_slot`.
    pub(crate) fn tail_conv_dst(&self, ssm_layer: usize, snap_slot: usize) -> DevicePtr {
        self.conv_snapshots[ssm_layer].offset(snap_slot * self.conv_bytes)
    }

    /// Bytes per layer of a snapshot's h_state.
    pub(crate) fn h_bytes(&self) -> usize {
        self.h_bytes
    }

    /// Bytes per layer of a snapshot's conv_state.
    pub(crate) fn conv_bytes(&self) -> usize {
        self.conv_bytes
    }

    /// Number of SSM layers (== length of the per-layer dst vectors).
    pub(crate) fn num_ssm_layers(&self) -> usize {
        self.num_ssm_layers
    }

    /// Stash the last-token post-final-norm hidden (`hidden_bytes`, BF16)
    /// for a leaf snapshot slot. Used so an exact full-prompt hit can emit
    /// the first token's logits via `lm_head` without re-running the last
    /// token through the SSM layers (which would double-advance the
    /// recurrent state). Only leaf saves call this; intermediate
    /// checkpoints leave the slot untagged.
    pub(super) fn save_hidden(
        &self,
        snap_slot: usize,
        last_hidden: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if !self.is_enabled() || self.hidden_snapshot.is_null() {
            return Ok(());
        }
        gpu.copy_d2d_async(
            last_hidden,
            self.hidden_snapshot.offset(snap_slot * self.hidden_bytes),
            self.hidden_bytes,
            stream,
        )?;
        self.slot_has_hidden.lock().insert(snap_slot);
        Ok(())
    }

    /// Whether `snap_slot` holds a valid last-token hidden (leaf snapshot).
    pub(super) fn has_hidden(&self, snap_slot: usize) -> bool {
        self.slot_has_hidden.lock().contains(&snap_slot)
    }

    /// Restore the stashed last-token hidden of `snap_slot` into `dst`
    /// (the `norm_output` buffer), ready for `lm_head`.
    pub(super) fn restore_hidden(
        &self,
        snap_slot: usize,
        dst: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if self.hidden_snapshot.is_null() {
            bail!("SSM hidden snapshot region not allocated");
        }
        gpu.copy_d2d_async(
            self.hidden_snapshot.offset(snap_slot * self.hidden_bytes),
            dst,
            self.hidden_bytes,
            stream,
        )?;
        Ok(())
    }

    // `reclaim_from_cache` (spill-or-drop) and the Phase-1 spill/fault-in
    // primitives live in the sibling `ssm_snapshot_spill` module to keep this
    // file under the 500-LoC cap. They are a second `impl SsmSnapshotPool`
    // block over the same fields.
}
