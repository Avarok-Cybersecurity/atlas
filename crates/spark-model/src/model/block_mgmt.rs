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

use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Apply an `EvictedBlocks` result to the production cache and the HSS
/// orchestrator. Physical blocks return to the free list; disk-block IDs get
/// `dec_disk_ref`'d (Phase 6.1.e). When HSS isn't engaged the disk vec is
/// empty and this becomes a thin loop over the physical blocks.
pub(crate) fn apply_evicted_blocks(
    evicted: spark_runtime::prefix_cache::EvictedBlocks,
    kv_cache: &mut PagedKvCache,
) {
    for block in &evicted.physical {
        kv_cache.return_evicted_block(*block);
    }
    if !evicted.disk_block_ids.is_empty()
        && let Some(res) = spark_storage::with_local(|hss| {
            for id in &evicted.disk_block_ids {
                // dec_disk_ref returns the new refcount; discarded here.
                let _new_refcount = hss.dec_disk_ref(*id);
            }
            Ok(())
        })
        && let Err(e) = res
    {
        // Errors here are advisory — orchestrator absent shouldn't block
        // the cache eviction path. Log and continue.
        tracing::debug!("apply_evicted_blocks: spark_storage::with_local closure: {e:#}");
    }
}

/// Phase 6.1.e: bump disk-side refcounts for blocks reused from a prefix-cache
/// hit, and push the disk_block_ids onto the sequence's history. The cache's
/// own ref keeps these slots alive across eviction; we add the seq's ref so
/// `free_sequence` can dec_disk_ref it on exit.
///
/// `matched_disk_block_ids` parallels `matched_blocks` when the entries were
/// inserted under HSS (every entry is a live disk_id). When HSS wasn't
/// engaged at insert time the slice is empty — the per-layer offload helper
/// will alloc fresh disk_ids and stream the data to disk on the first decode
/// step that touches each block.
pub(crate) fn reuse_prefix_match_disk_ids(
    matched_disk_block_ids: &[u32],
    seq_disk_block_ids: &mut Vec<u32>,
) {
    if matched_disk_block_ids.is_empty() {
        return;
    }
    if let Some(res) = spark_storage::with_local(|hss| {
        for &id in matched_disk_block_ids {
            if id == u32::MAX {
                // Mixed-mode entry — skip; the catch-up offload will populate.
                continue;
            }
            hss.inc_disk_ref(id);
            seq_disk_block_ids.push(id);
        }
        Ok(())
    }) && let Err(e) = res
    {
        tracing::debug!("reuse_prefix_match_disk_ids: spark_storage::with_local: {e:#}");
    }
}

/// Phase 6.3 — Sliding-window allocation helper (decode path).
///
/// Ensures `seq.physical_block_for(abs_block_idx)` is `Some` after this call
/// returns. With HSS off, this is a thin wrapper over `kv_cache.alloc_block()`.
/// With HSS on (`cache_blocks_per_seq` set), it slides the rolling window
/// when at cap, allocates the new physical block (recycling the freed one),
/// zeroes it, and pushes a parallel disk_block_id onto `seq.disk_block_ids`.
///
/// Pre-condition (debug-asserted before each slide): every attention layer
/// has already offloaded everything in `seq.disk_block_ids` — i.e.,
/// `disk_last_offloaded_per_layer[L] == disk_block_ids.len()` for all L.
/// This holds at decode-step boundaries because every layer's
/// `attention_forward` calls `high_speed_swap_offload_new_blocks` before
/// returning.
pub(crate) fn ensure_blocks_through_decode(
    seq: &mut SequenceState,
    abs_block_idx: usize,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    let cap = kv_cache.config().cache_blocks_per_seq.map(|c| c as usize);
    // Loop invariant: each iter either slides (frees a block) or grows
    // block_table by one. Terminates when the highest needed logical block
    // is in-window.
    let mut slide_count = 0usize;
    let mut alloc_count = 0usize;
    loop {
        let ws = seq.hss_window_start();
        let bt_len = seq.block_table.len();
        // Highest in-window logical block (inclusive). If empty, treat as
        // "none yet" — keep the loop going until we've allocated.
        let in_window = bt_len > 0 && abs_block_idx < ws + bt_len;
        if in_window {
            if slide_count > 0 || alloc_count > 0 {
                tracing::trace!(
                    "ensure_blocks_through_decode: abs={} ws={} bt_len={} slid={} alloc'd={}",
                    abs_block_idx,
                    ws,
                    bt_len,
                    slide_count,
                    alloc_count
                );
            }
            return Ok(());
        }
        // We need to extend block_table by at least one. If at cap, slide.
        if let Some(c) = cap
            && bt_len >= c
        {
            debug_assert!(
                seq.disk_last_offloaded_per_layer
                    .iter()
                    .all(|&n| n as usize == seq.disk_block_ids.len()),
                "Phase 6.3 invariant: all attention layers must offload before slide. \
                     disk_block_ids.len()={}, per-layer cursors={:?}",
                seq.disk_block_ids.len(),
                seq.disk_last_offloaded_per_layer
            );
            let evicted = seq.block_table.remove(0);
            kv_cache.free_block(evicted);
            slide_count += 1;
            continue;
        }
        // F77 (2026-04-30): same try_alloc → evict prefix cache → retry
        // pattern as ensure_blocks_through_prefill. Without the
        // eviction fallback, multi-turn opencode sessions exhaust the
        // KV pool because every completed turn leaves prefix-cached
        // blocks alive — error observed live:
        // "alloc failed in ensure_blocks_through_decode: abs=590 ...
        //  free_blocks=0". The prefill helper already had this; the
        // decode helper diverged.
        let blk = match kv_cache.try_alloc_block() {
            Some(b) => b,
            None => {
                let evicted = prefix_cache.evict(1);
                apply_evicted_blocks(evicted, kv_cache);
                kv_cache.alloc_block().map_err(|e| {
                    anyhow::anyhow!(
                        "alloc failed in ensure_blocks_through_decode: abs={} ws={} bt_len={} \
                         cap={:?} free_blocks={} slid={} alloc'd={}: {}",
                        abs_block_idx,
                        ws,
                        bt_len,
                        cap,
                        kv_cache.num_free_blocks(),
                        slide_count,
                        alloc_count,
                        e
                    )
                })?
            }
        };
        kv_cache.zero_block(blk, gpu, stream)?;
        seq.block_table.push(blk);
        alloc_count += 1;
        if cap.is_some() {
            let id = spark_storage::with_local(|hss| {
                hss.alloc_disk_block_id().ok_or_else(|| {
                    anyhow::anyhow!(
                        "high-speed-swap: disk-block-id pool exhausted; \
                         increase --high-speed-swap-bytes or shorten --max-seq-len"
                    )
                })
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "high-speed-swap: orchestrator not installed but cache_blocks_per_seq is set"
                )
            })??;
            seq.disk_block_ids.push(id);
        }
    }
}

/// Phase 6.3 — Sliding-window allocation helper (prefill path).
///
/// Same as `ensure_blocks_through_decode` but with a prefix-cache eviction
/// fallback: if `kv_cache.alloc_block()` would fail (no free physical
/// blocks AND not at HSS cap), it asks the prefix cache to evict LRU
/// entries before retrying. With HSS off, this matches the existing
/// `try_alloc_block` → evict-prefix-cache → `alloc_block` pattern at
/// every prefill site. With HSS on, the slide path takes priority over
/// the prefix-cache evict (cap is the binding constraint).
pub(crate) fn ensure_blocks_through_prefill(
    seq: &mut SequenceState,
    abs_block_idx: usize,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    let cap = kv_cache.config().cache_blocks_per_seq.map(|c| c as usize);
    loop {
        let ws = seq.hss_window_start();
        let bt_len = seq.block_table.len();
        let in_window = bt_len > 0 && abs_block_idx < ws + bt_len;
        if in_window {
            return Ok(());
        }
        // Slide first when at HSS cap.
        if let Some(c) = cap
            && bt_len >= c
        {
            debug_assert!(
                seq.disk_last_offloaded_per_layer
                    .iter()
                    .all(|&n| n as usize == seq.disk_block_ids.len()),
                "Phase 6.3 invariant violated in prefill"
            );
            let evicted = seq.block_table.remove(0);
            kv_cache.free_block(evicted);
            continue;
        }
        // Try alloc; on failure, evict prefix-cache entries and retry once.
        let blk = match kv_cache.try_alloc_block() {
            Some(b) => b,
            None => {
                // Ask the prefix cache to free a block via LRU eviction.
                let evicted = prefix_cache.evict(1);
                apply_evicted_blocks(evicted, kv_cache);
                kv_cache.alloc_block()?
            }
        };
        kv_cache.zero_block(blk, gpu, stream)?;
        seq.block_table.push(blk);
        if cap.is_some() {
            let id = spark_storage::with_local(|hss| {
                hss.alloc_disk_block_id().ok_or_else(|| {
                    anyhow::anyhow!(
                        "high-speed-swap: disk-block-id pool exhausted; \
                         increase --high-speed-swap-bytes or shorten --max-seq-len"
                    )
                })
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "high-speed-swap: orchestrator not installed but cache_blocks_per_seq is set"
                )
            })??;
            seq.disk_block_ids.push(id);
        }
    }
}

/// Extract mutable references to a single layer's state across N sequences.
///
/// The explicit lifetime `'a` ties the returned refs to the borrow of `all`,
/// so the compiler knows the borrow is released when the returned Vec is dropped.
/// Uses a for loop instead of iter_mut().map() because FnMut closures cannot
/// express that returned references outlive the closure invocation.
pub(crate) fn extract_layer_refs<'a>(
    all: &'a mut [Vec<Box<dyn LayerState>>],
    layer_idx: usize,
) -> Vec<&'a mut (dyn LayerState + 'static)> {
    let mut refs = Vec::with_capacity(all.len());
    for seq_states in all.iter_mut() {
        refs.push(seq_states[layer_idx].as_mut());
    }
    refs
}
