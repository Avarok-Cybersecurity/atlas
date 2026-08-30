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
                // Take the cache's KV ref on exactly the blocks whose radix nodes
                // this insert created — reported by the insert itself.
                //
                // This used to `inc_ref` a RANGE of the finishing sequence's
                // `block_table` (everything past the matched prefix) on the
                // assumption that node i holds `block_table[i]`. It does not: when
                // a node for a token chunk already exists, `insert` keeps that
                // node's original block, so a sequence that did not get its block
                // from the cache (no match, or restored from a swap file) has a
                // DIFFERENT block at that position. The ref then landed on a block
                // no node referenced, while the node's own block carried none — so
                // evicting it decremented a ref belonging to a live sequence,
                // returning an in-use block to the free list to be handed out
                // again. Reporting the blocks keeps ref lifetime == node lifetime.
                super::super::block_mgmt::cache_acquires_refs(&acquired, &mut self.kv_cache.lock());
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

        // Release per-sequence layer state that is NOT pooled: the QSA indexer
        // carry (12 full-attention layers x ~61.6 MB at 200K ctx) and the PLE
        // conv carry. Both are bare `DevicePtr`s inside the layer state, so
        // dropping `seq.layer_states` reclaims the host structs and leaks the
        // device buffers — ~739 MB per request, invisible to RSS on unified
        // memory and reported as N/A by `nvidia-smi`, which is why it read as
        // "the box is growing" with no process to blame.
        //
        // Errors are logged, not propagated: this runs on the teardown path,
        // and a sequence that cannot free its state is still finished. Bailing
        // here would strand the KV blocks and prefix refs released below —
        // trading a leak for a worse one.
        for (layer_idx, ls) in seq.layer_states.iter_mut().enumerate() {
            if let Some(layer) = self.layers.get(layer_idx)
                && let Err(e) = layer.release_state(ls.as_mut(), self.gpu.as_ref())
            {
                tracing::error!("free_sequence: release_state(layer {layer_idx}): {e:#}");
            }
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
        //
        // Normally `seq.tokens` (prompt + generated) fully covers the matched
        // prefix, so releasing over it undoes the lookup's radix inc_refs. But a
        // prefill that matched a prefix then FAILED to allocate its suffix never
        // populated `seq.tokens` (that happens in a later finalize phase), so
        // `release(&seq.tokens)` would be a no-op and the matched radix nodes
        // would stay pinned forever → the pool wedges. When `seq.tokens` is too
        // short to cover the matched prefix, release over the stashed prefix
        // tokens instead. Exactly one of the two covers the matched nodes, so
        // they are released once (never double-released).
        let release_tokens = if seq.tokens.len() >= seq.cached_prefix_tokens {
            &seq.tokens
        } else {
            &seq.prefix_ref_tokens
        };
        self.prefix_cache.release(
            release_tokens,
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
                ssm.h_prefill_stage = None;
                ssm.h_state_checkpoint = None;
                ssm.conv_state_checkpoint = None;
                ssm.h_state_intermediates.clear();
                ssm.conv_state_intermediates.clear();
            }
        }

        // Slot-keyed decode / K=2/3/4 verify / batched-decode graphs bake SSM
        // pool addresses that are a function of the SLOT (fixed for process
        // lifetime) and read dynamic KV/metadata from staging refreshed before
        // every replay. A new occupant of this slot can replay them; LRU in
        // `insert_batch_decode_graph` bounds batched-graph memory. Recapturing
        // on every completion was an extra eager step per request. Policy is
        // covered at the cache-key/LRU layer in `decode_graph_key`.
        //
        // verify_kgamma / fused still drop below: they bake a per-occupant
        // LoRA adapter index, so they must be dropped for this slot.
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
        if crate::model::mtp_carry::mtp_carry_drafter_enabled(&self.levers)
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
        // Tiered pools: the H-intermediate count is a property of the SLOT
        // (h_inter_count), the conv count is uniform (num_intermediates).
        let num_intermediates = self.ssm_pool.num_intermediates;
        let h_intermediates = self.ssm_pool.h_inter_count(new_slot);
        let mut ssm_layer_idx = 0usize;
        for (i, state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                if let Some(ssm) = state.as_any_mut().downcast_mut::<SsmLayerState>() {
                    ssm.h_state = self.ssm_pool.h_state(ssm_layer_idx, new_slot);
                    ssm.conv_state = self.ssm_pool.conv_state(ssm_layer_idx, new_slot);
                    // Stage-3 f16-SIZED pool: the FP32 prefill staging blob is
                    // per-SLOT, so compaction must repoint it for the same
                    // reason the checkpoint/intermediate families below are
                    // repointed — the old slot becomes reallocatable, and a
                    // continuation chunk staging through the new occupant's
                    // blob is cross-sequence corruption. `None` stays `None`
                    // (FP32-sized pool: no staging exists).
                    ssm.h_prefill_stage = self.ssm_pool.h_prefill_stage(new_slot);
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
                            for t in 0..h_intermediates {
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

    pub(super) fn num_total_blocks_dispatch(&self) -> usize {
        self.kv_cache.lock().num_blocks()
    }

    pub(super) fn reclaim_prefix_blocks_dispatch(&self, num_blocks: usize) -> usize {
        if num_blocks == 0 || !self.prefix_cache.is_active() {
            return 0;
        }
        let evicted = self.prefix_cache.evict(num_blocks);
        if evicted.is_empty() {
            return 0;
        }
        let mut kv = self.kv_cache.lock();
        let before = kv.num_free_blocks();
        super::super::block_mgmt::apply_evicted_blocks(evicted, &mut kv);
        kv.num_free_blocks().saturating_sub(before)
    }
}
