// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

mod state_io;

impl TransformerModel {
    pub(super) fn cache_sequence_dispatch(&self, seq: &SequenceState) {
        let bs = self.kv_cache.lock().block_size();
        // Only cache if the sequence has block-aligned content worth caching.
        // Sequences shorter than one block have no reusable KV blocks.
        if seq.tokens.len() >= bs && !seq.block_table.is_empty() {
            // Prompt tokens were already inserted + ref-bumped by prefill.
            // Only generated tokens past `prompt_len` are "newly seq-owned"
            // at this point — pass prompt_len as matched_tokens so insert
            // skips re-bumping the prompt portion.
            // Phase 6.3 sliding-window: when HSS has slid older blocks out,
            // `block_table` no longer parallels `tokens` from index 0 — the
            // physical IDs at the front of block_table now hold WRITES for
            // recent positions, not the historical prompt. Skip cache_sequence
            // insert in that case to avoid populating the radix tree with
            // mis-correlated entries. (Disk-side ref counting via
            // `apply_evicted_blocks` keeps the disk_block_ids alive
            // independently when the prefix cache later evicts.)
            // Skip when the prefix cache is a no-op (`--enable-prefix-caching`
            // off): the manual inc_ref below would never get a paired dec_ref
            // from cache eviction, leaking the seq's blocks every request.
            // Also skip on HSS-slid (front of block_table no longer parallels
            // tokens) and vision prompts — both handled by the guard below.
            if self.prefix_cache.is_active()
                && !self.tokens_have_vision_pad(&seq.tokens)
                && seq.hss_window_start() == 0
            {
                // #155: leaf snapshot at FULL length (prompt + generated) so
                // the next warm hit restores at this turn's END and replays
                // ~nothing. Save logic + the secondary-stream ordering guard
                // live in decode_checkpoint.rs (finish_leaf_snapshot).
                let finish_snap = self.finish_leaf_snapshot(seq);
                let acquired = if let Some(snap_id) = finish_snap {
                    let (displaced, acquired) = self.prefix_cache.insert_with_snapshot(
                        &seq.tokens,
                        &seq.block_table,
                        &seq.disk_block_ids,
                        bs,
                        snap_id,
                        seq.session_hash,
                        seq.prompt_len,
                        seq.adapter_id,
                    );
                    if let Some(old) = displaced {
                        self.ssm_snapshots.free(old);
                    }
                    acquired
                } else {
                    self.prefix_cache.insert(
                        &seq.tokens,
                        &seq.block_table,
                        &seq.disk_block_ids,
                        bs,
                        seq.prompt_len,
                        seq.adapter_id,
                    )
                };
                super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
                // Bump KV block ref_counts so the prefix cache "owns" a reference.
                // This keeps blocks alive after free_sequence drops the sequence's ref.
                // Eviction (return_evicted_block) releases these refs when nodes are removed.
                let mut kv = self.kv_cache.lock();
                let num_cached_blocks = (seq.tokens.len() / bs).min(seq.block_table.len());
                for &block_idx in &seq.block_table[..num_cached_blocks] {
                    kv.inc_ref(block_idx);
                }
            }
        }
    }

    pub(super) fn free_sequence_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        // Release prefix cache refs before freeing blocks.
        // dec_ref will only actually free blocks whose ref_count hits 0
        // CRITICAL: release SSM slot FIRST to prevent slot leak if later
        // operations fail (e.g. after sticky CUDA error 700). The slot is a
        // CPU-side resource; its release must not be gated on GPU success.
        //
        // Slot-reuse sentinel: the scheduler sets `slot_idx = usize::MAX` on a
        // retired sequence AFTER `compact_sequence` migrated this sequence's
        // pool slot to the surviving (swapped-in) sequence. In that case THIS
        // sequence no longer owns the slot — the survivor's guard does — so we
        // must NOT release it (that would be a double-release: the survivor's
        // guard still owns the same index). We still `take()` the guard to make
        // its Drop a no-op, but discard the index without pushing it back.
        //
        // On the normal teardown path (`slot_idx < max_slots`), `take()` yields
        // the owned index and we release it exactly once. `take()` also makes
        // the guard's Drop a no-op so abort/panic cannot double-release.
        let slot_reused_by_compact = seq.slot_idx >= self.ssm_pool.max_slots;
        let taken = seq.ssm_slot.as_mut().and_then(|g| g.take());
        let slot_to_release = if slot_reused_by_compact { None } else { taken };
        if let Some(slot) = slot_to_release {
            let stream = self.gpu.default_stream();
            if let Err(e) = self.ssm_pool.zero_slot(slot, self.gpu.as_ref(), stream) {
                tracing::error!("free_sequence: ssm_pool.zero_slot({slot}): {e:#}");
            }
            if let Err(e) = self.gpu.synchronize(stream) {
                tracing::error!("free_sequence: gpu.synchronize after zero_slot({slot}): {e:#}");
            }
            self.ssm_pool.release_slot(slot);
        }

        // Task #25: release this sequence's LoRA slot ref (the single terminal
        // chokepoint every stamped seq routes through — normal stop/EOS/length,
        // error/abort, prefill-error frees, and swap-out spill). Guarded by the
        // RESOLVED `acquired_adapter_slot` (`-1` = never acquired: the non-
        // scheduler alloc paths and the base no-LoRA path skip this) and zeroed
        // so it fires exactly once per acquire, idempotent against a double free.
        if seq.acquired_adapter_slot >= 0 {
            self.release_adapter_slot(seq.acquired_adapter_slot);
            seq.acquired_adapter_slot = -1;
        }

        // Release prefix cache refs before freeing blocks.
        // (i.e., blocks not shared with the prefix cache).
        self.prefix_cache.release(
            &seq.tokens,
            self.kv_cache.lock().block_size(),
            seq.adapter_id,
        );
        if !seq.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&seq.block_table);
            seq.block_table.clear();
        }

        // --high-speed-swap: release disk-side refs for every block this
        // sequence ever held (Phase 6.1.c). disk_block_ids are layer-
        // agnostic (each ID indexes a slot in *every* layer's file), so
        // one dec_disk_ref per ID covers all layers' data simultaneously.
        // The orchestrator's free list only reclaims an ID when its
        // refcount hits 0, so sequences sharing a prefix correctly keep
        // each other's disk blocks alive via ref-counting.
        if !seq.disk_block_ids.is_empty() {
            // with_local returns Option<Result>: None when HSS isn't engaged
            // (no-op, fine), Some(Err) when the closure failed (advisory).
            if let Some(Err(e)) = spark_storage::with_local(|hss| {
                for &disk_id in &seq.disk_block_ids {
                    hss.dec_disk_ref(disk_id);
                }
                Ok(())
            }) {
                tracing::error!("free_sequence: spark_storage dec_disk_ref batch: {e:#}");
            }
            seq.disk_block_ids.clear();
            for v in seq.disk_last_offloaded_per_layer.iter_mut() {
                *v = 0;
            }
        }

        // All SSM buffers (h_state, conv_state, checkpoints, intermediates) belong
        // to the pool — do NOT gpu.free() them. Just clear the references.
        for state in &mut seq.layer_states {
            if let Some(ssm) = state.as_any_mut().downcast_mut::<SsmLayerState>() {
                ssm.h_state = DevicePtr(0);
                ssm.conv_state = DevicePtr(0);
                ssm.h_state_checkpoint = None;
                ssm.conv_state_checkpoint = None;
                ssm.h_state_intermediates.clear();
                ssm.conv_state_intermediates.clear();
            }
        }

        // Invalidate cached CUDA graphs that reference this sequence's slot
        // — the graph was captured with this slot's KV/SSM pointers baked in,
        // and replaying after the slot is freed would read stale data.
        // decode_graph is keyed by slot, so drop only this slot's entry.
        // (parking_lot::Mutex::lock() never poisons, so the previous `if let
        // Ok(...) = .lock()` graceful-recovery branch is unreachable.)
        if let Some(graph) = self.decode_graph.lock().remove(&seq.slot_idx)
            && let Err(e) = self.gpu.destroy_graph(graph)
        {
            tracing::error!(
                "free_sequence: destroy_graph(decode_graph[{}]): {e:#}",
                seq.slot_idx
            );
        }
        // batch_decode_graphs is keyed by padded_n, not slot — but the captured
        // graphs DO contain per-slot SSM pointers from the active set at capture
        // time. Drop them all (they'll be re-captured on next batched decode).
        for (_, graph) in self.batch_decode_graphs.lock().drain() {
            if let Err(e) = self.gpu.destroy_graph(graph) {
                tracing::error!("free_sequence: destroy_graph(batch_decode_graphs entry): {e:#}");
            }
        }
        // Verify graphs are now slot-keyed (sibling of decode_graph fix).
        // Drop only this slot's entry to preserve other concurrent seqs' graphs.
        for graph_mutex in [
            &self.verify2_graph,
            &self.verify3_graph,
            &self.verify4_graph,
        ] {
            if let Some(graph) = graph_mutex.lock().remove(&seq.slot_idx)
                && let Err(e) = self.gpu.destroy_graph(graph)
            {
                tracing::error!(
                    "free_sequence: destroy_graph(verify[{}]): {e:#}",
                    seq.slot_idx
                );
            }
        }
        // verify_kgamma_graph + fused_graph are keyed by (slot, K). They now
        // capture the LoRA bgmv-vs-installed-pair branch and read the per-seq
        // seq_slot buffer, so a freed slot's entries MUST be destroyed — else a
        // reused slot replays a stale adapter index (multi-adapter + DFlash
        // spec-decode output corruption). Drop every K for this slot.
        for graph_map in [&self.verify_kgamma_graph, &self.fused_graph] {
            let mut cache = graph_map.lock();
            let keys: Vec<(usize, usize)> = cache
                .keys()
                .filter(|k| k.0 == seq.slot_idx)
                .copied()
                .collect();
            for k in keys {
                if let Some(graph) = cache.remove(&k)
                    && let Err(e) = self.gpu.destroy_graph(graph)
                {
                    tracing::error!(
                        "free_sequence: destroy_graph(kgamma/fused[{},{}]): {e:#}",
                        k.0,
                        k.1
                    );
                }
            }
        }

        // ATLAS_MTP_CARRY_DRAFTER: hand this turn's drafter KV to the model's
        // single carry slot BEFORE `free_state`, so the next turn of the same
        // session can adopt it instead of starting blind. `take_drafter_kv`
        // empties the proposer state, so the `free_state` below then releases
        // nothing — the blocks are owned by the carry slot XOR by a live
        // sequence, never both.
        if crate::model::mtp_carry::mtp_carry_drafter_enabled()
            && let Some(ref proposer) = self.proposer
            && let Some(ref mut pstate) = seq.proposer_state
            && let Some((blocks, rows, last_pair_key)) = proposer.take_drafter_kv(pstate.as_mut())
        {
            let entry = crate::model::mtp_carry::CarriedDrafter {
                block_table: blocks,
                rows,
                last_pair_key,
                tokens: seq.tokens.clone(),
            };
            let previous = self.mtp_carry.lock().replace(entry);
            if let Some(old) = previous {
                proposer.free_drafter_kv(&old.block_table);
            }
            if crate::model::mtp_carry::mtp_carry_debug() {
                tracing::info!(
                    "MTP_CARRY store: rows={rows} last_pair_key={last_pair_key:?} \
                     seq_tokens={}",
                    seq.tokens.len(),
                );
            }
        }

        // Free proposer state (KV cache blocks + per-seq device buffers).
        if let Some(ref proposer) = self.proposer
            && let Some(ref mut pstate) = seq.proposer_state
        {
            proposer.free_state(self.gpu.as_ref(), pstate.as_mut())?;
        }

        self.free_chunked_prefill_meta(seq)?;

        Ok(())
    }

    /// Disown a retired sequence's SSM slot because `compact_sequence` migrated
    /// it to a surviving sequence. Takes the slot out of this sequence's RAII
    /// guard WITHOUT releasing it (the survivor's guard now owns it) and sets
    /// the `slot_idx = usize::MAX` reuse sentinel. Must be called by the
    /// scheduler immediately after a successful `compact_sequence` that reuses
    /// THIS sequence's slot, and BEFORE any fallible step (e.g. swap-out
    /// `save_sequence_state`) that could drop the sequence early — otherwise the
    /// guard's Drop would re-release the migrated slot (double-release).
    pub(super) fn detach_slot_for_reuse_dispatch(&self, seq: &mut SequenceState) {
        if let Some(g) = seq.ssm_slot.as_mut() {
            // Discard the owned index without pushing it to the free list.
            let _ = g.take();
        }
        seq.slot_idx = usize::MAX;
    }

    pub(super) fn compact_sequence_dispatch(
        &self,
        seq: &mut SequenceState,
        new_slot: usize,
    ) -> Result<()> {
        let old_slot = seq.slot_idx;
        if old_slot == new_slot {
            return Ok(());
        }

        let stream = self.gpu.default_stream();
        self.ssm_pool
            .copy_slot(old_slot, new_slot, self.gpu.as_ref(), stream)?;

        // Update ALL SsmLayerState pool pointers to point at the new slot.
        // BUG FIX: previously only h_state and conv_state were repointed, leaving
        // the MTP checkpoint and intermediate pointers aimed at the OLD slot.
        // After release_slot, that old slot is reallocatable to a NEW sequence,
        // and any subsequent MTP save_hidden / start_checkpoint_async on this seq
        // would write into the new occupant's pool memory — cross-seq corruption.
        let has_mtp = self.ssm_pool.has_mtp;
        let num_intermediates = self.ssm_pool.num_intermediates;
        let mut ssm_layer_idx = 0usize;
        for (i, state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                if let Some(ssm) = state.as_any_mut().downcast_mut::<SsmLayerState>() {
                    ssm.h_state = self.ssm_pool.h_state(ssm_layer_idx, new_slot);
                    ssm.conv_state = self.ssm_pool.conv_state(ssm_layer_idx, new_slot);
                    if has_mtp {
                        if ssm.h_state_checkpoint.is_some() {
                            ssm.h_state_checkpoint =
                                Some(self.ssm_pool.h_checkpoint(ssm_layer_idx, new_slot));
                        }
                        if ssm.conv_state_checkpoint.is_some() {
                            ssm.conv_state_checkpoint =
                                Some(self.ssm_pool.conv_checkpoint(ssm_layer_idx, new_slot));
                        }
                        if !ssm.h_state_intermediates.is_empty() {
                            ssm.h_state_intermediates.clear();
                            for t in 0..num_intermediates {
                                ssm.h_state_intermediates.push(self.ssm_pool.h_intermediate(
                                    ssm_layer_idx,
                                    new_slot,
                                    t,
                                ));
                            }
                        }
                        if !ssm.conv_state_intermediates.is_empty() {
                            ssm.conv_state_intermediates.clear();
                            for t in 0..num_intermediates {
                                ssm.conv_state_intermediates
                                    .push(self.ssm_pool.conv_intermediate(
                                        ssm_layer_idx,
                                        new_slot,
                                        t,
                                    ));
                            }
                        }
                    }
                }
                ssm_layer_idx += 1;
            }
        }

        seq.slot_idx = new_slot;
        // BUG FIX: synchronize before releasing the old slot. copy_slot is async
        // (queued D2D), so without this barrier, claim_slot() in the next request
        // could hand the old_slot back to a new sequence while the copy's source
        // reads are still in flight — cross-seq race that produces partial data.
        self.gpu.synchronize(stream)?;
        // Slot-migration is an ownership TRANSFER, not a free: this sequence
        // keeps a live slot (the NEW one). Take the old idx out of the guard so
        // its Drop won't re-release it, release the old slot exactly once, then
        // re-point the guard at the new slot it now owns. This preserves the
        // exactly-once invariant: old_slot is pushed here (once) and new_slot
        // will be pushed by whichever path later frees THIS sequence (once).
        // Claim the NEW slot EXCLUSIVELY (bug-2 fix): if the migration target
        // is on the free list (a slot freed by a retiring sequence in the
        // two-phase retire compaction), remove it so it is never simultaneously
        // owned (by this guard) and free. Without this, a later release of this
        // slot double-pushes it and `claim_slot` hands the same slot to two
        // sequences → shared GDN state → cross-stream content bleed. A no-op
        // for the ownership-TRANSFER caller (lifecycle swap-out), where the
        // target is owned by the retiring victim and not on the free list.
        self.ssm_pool.claim_specific(new_slot);
        if let Some(g) = seq.ssm_slot.as_mut() {
            // Guard owned `old_slot`; drop that ownership before releasing.
            let owned = g.take();
            debug_assert_eq!(
                owned,
                Some(old_slot),
                "compact_sequence: guard owned {owned:?}, expected old_slot {old_slot}"
            );
            self.ssm_pool.release_slot(old_slot);
            g.migrate(new_slot);
        } else {
            // No guard (e.g. mock model with no SSM pool): preserve the legacy
            // explicit release so behavior is unchanged where there is no guard.
            self.ssm_pool.release_slot(old_slot);
        }
        Ok(())
    }

    pub(super) fn num_free_blocks_dispatch(&self) -> usize {
        self.kv_cache.lock().num_free_blocks()
    }
}
