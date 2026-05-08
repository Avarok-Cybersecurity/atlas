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

/// Apply the disk-ref obligation reported by a `prefix_cache.insert*` call.
/// The cache returns the disk_block_ids on which it newly took an ownership
/// ref (a node was created OR an existing node had its disk_block_id
/// populated for the first time). The caller `inc_disk_ref`s each one so
/// the swap allocator's refcount matches the cache's reachability.
///
/// Without this, the cache stores disk_block_ids whose only live ref is
/// the sequence's; when `free_sequence` decs, the ID is reclaimed by the
/// swap allocator while the cache still references it — the next prefix
/// hit then trips `inc_disk_ref` on a freed ID and panics the scheduler
/// thread (Issue #17, panic at `high_speed_swap.rs:167`).
pub(crate) fn cache_acquires_disk_refs(newly_acquired: &[u32]) {
    if newly_acquired.is_empty() {
        return;
    }
    if let Some(res) = spark_storage::with_local(|hss| {
        for &id in newly_acquired {
            if id != u32::MAX {
                hss.inc_disk_ref(id);
            }
        }
        Ok(())
    }) && let Err(e) = res
    {
        tracing::debug!("cache_acquires_disk_refs: spark_storage::with_local: {e:#}");
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

/// Issue #31: validate that the block being evicted at logical position
/// `evict_pos` has been offloaded by every attention layer. The slide
/// is safe iff `disk_last_offloaded[L] > evict_pos` for all L (strictly
/// greater because `disk_last_offloaded[L]` is the count of offloaded
/// blocks, and a block at position N is "offloaded" iff the count is at
/// least N+1, i.e., > N).
///
/// Returns `Err` describing the first lagging layer if any layer hasn't
/// caught up, else `Ok(())`. Pure function — no side effects.
pub(crate) fn check_safe_to_evict(
    disk_last_offloaded_per_layer: &[u32],
    evict_pos: usize,
) -> Result<()> {
    for (layer_idx, &cursor) in disk_last_offloaded_per_layer.iter().enumerate() {
        if (cursor as usize) <= evict_pos {
            bail!(
                "high-speed-swap: attempting to evict block at logical position {} \
                 from HBM, but attention layer {} only offloaded up to position {}. \
                 Eviction would lose K/V data. Per-layer cursors: {:?}",
                evict_pos,
                layer_idx,
                cursor,
                disk_last_offloaded_per_layer,
            );
        }
    }
    Ok(())
}

/// Issue #31: after a successful slide advances `window_start` to
/// `new_window_start`, advance every attention layer's offload cursor
/// to keep pace. Layers whose cursor was already ≥ `new_window_start`
/// (e.g. they offloaded more recently in this chunk) are left alone.
/// Pure mutation on a `&mut [u32]`.
pub(crate) fn advance_layer_cursors_after_slide(
    disk_last_offloaded_per_layer: &mut [u32],
    new_window_start: usize,
) {
    let new_ws = new_window_start as u32;
    for cursor in disk_last_offloaded_per_layer.iter_mut() {
        if *cursor < new_ws {
            *cursor = new_ws;
        }
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
            // Issue #31: the safety condition for evicting block_table[0]
            // (logical position `ws`) is that EVERY attention layer has
            // already offloaded that block. The previous code asserted
            // the stricter (and overly-strict) condition
            // `disk_last_offloaded[L] == disk_block_ids.len()`, which
            // fails immediately after the first alloc within this loop
            // bumps `disk_block_ids.len()` past the cursors. The
            // assertion was `debug_assert!` only — release builds slid
            // past anyway, then `offload_layer_kv` bailed downstream.
            check_safe_to_evict(&seq.disk_last_offloaded_per_layer, ws).map_err(|e| {
                anyhow::anyhow!(
                    "{e} (decode path; disk_block_ids.len()={}, block_table.len()={}, \
                     slid={}, alloc'd={})",
                    seq.disk_block_ids.len(),
                    seq.block_table.len(),
                    slide_count,
                    alloc_count,
                )
            })?;
            let evicted = seq.block_table.remove(0);
            kv_cache.free_block(evicted);
            // After the slide, advance every layer cursor so the next
            // `offload_layer_kv` doesn't see `start < window_start`.
            // The blocks now outside the window were already on disk
            // (the safety check above guaranteed it).
            advance_layer_cursors_after_slide(&mut seq.disk_last_offloaded_per_layer, ws + 1);
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
            // Issue #31: see `ensure_blocks_through_decode`. Bail in
            // release mode if any attention layer hasn't offloaded the
            // block being evicted; advance every cursor after a
            // successful slide.
            check_safe_to_evict(&seq.disk_last_offloaded_per_layer, ws).map_err(|e| {
                anyhow::anyhow!(
                    "{e} (prefill path; disk_block_ids.len()={}, block_table.len()={})",
                    seq.disk_block_ids.len(),
                    seq.block_table.len(),
                )
            })?;
            let evicted = seq.block_table.remove(0);
            kv_cache.free_block(evicted);
            advance_layer_cursors_after_slide(&mut seq.disk_last_offloaded_per_layer, ws + 1);
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

#[cfg(test)]
mod tests {
    use super::*;

    // Issue #31 — `check_safe_to_evict` enforces the slide invariant.

    #[test]
    fn safe_to_evict_when_all_layers_caught_up() {
        // Every layer has offloaded position 5 (cursor=6 means 0..6 are on disk),
        // evicting position 5 is safe because cursor > 5 for every layer.
        let cursors = vec![6, 6, 6];
        assert!(check_safe_to_evict(&cursors, 5).is_ok());
    }

    #[test]
    fn unsafe_to_evict_when_a_layer_lags() {
        // Layer 1 has only offloaded up to position 4 (cursor=5 = positions 0..5
        // means position 4 is the last offloaded; cursor=5 means the LIMIT is 5,
        // i.e. cursor > 5 means position 5 is offloaded). Strict-greater comparison.
        let cursors = vec![10, 5, 10];
        let err = check_safe_to_evict(&cursors, 5).unwrap_err().to_string();
        assert!(err.contains("attention layer 1"), "got: {err}");
        assert!(err.contains("position 5"), "got: {err}");
    }

    #[test]
    fn unsafe_to_evict_when_a_layer_never_offloaded() {
        // Cursor=0 means the layer has offloaded NOTHING. Evicting any
        // position fails the check.
        let cursors = vec![10, 10, 0];
        let err = check_safe_to_evict(&cursors, 0).unwrap_err().to_string();
        assert!(err.contains("attention layer 2"), "got: {err}");
    }

    #[test]
    fn safe_to_evict_with_empty_cursor_vec_is_vacuously_true() {
        // A sequence whose `disk_last_offloaded_per_layer` hasn't been
        // populated yet (e.g. fresh sequence with no attn layers run)
        // can't have un-offloaded blocks because no layer has run. The
        // production `meta.rs:180` initializes this vec to `vec![0; n_attn]`
        // so this case shouldn't fire in real workloads, but the helper
        // should be vacuously correct.
        let cursors: Vec<u32> = vec![];
        assert!(check_safe_to_evict(&cursors, 100).is_ok());
    }

    // Issue #31 — `advance_layer_cursors_after_slide` keeps cursors ≥ window_start.

    #[test]
    fn advance_after_slide_promotes_lagging_cursors() {
        let mut cursors = vec![10, 5, 8];
        advance_layer_cursors_after_slide(&mut cursors, 9);
        // Layer 0 was already at 10 ≥ 9, unchanged. Layer 1 was at 5 < 9, bumped.
        // Layer 2 was at 8 < 9, bumped.
        assert_eq!(cursors, vec![10, 9, 9]);
    }

    #[test]
    fn advance_after_slide_never_moves_cursor_backward() {
        let mut cursors = vec![100, 100, 100];
        advance_layer_cursors_after_slide(&mut cursors, 50);
        // All cursors ≥ 50 → no change.
        assert_eq!(cursors, vec![100, 100, 100]);
    }

    #[test]
    fn advance_after_slide_idempotent() {
        let mut cursors = vec![5, 5, 5];
        advance_layer_cursors_after_slide(&mut cursors, 10);
        advance_layer_cursors_after_slide(&mut cursors, 10);
        assert_eq!(cursors, vec![10, 10, 10]);
    }

    // Round-trip: a slide loop pattern — for each slide, check then advance.
    // Models the cap=4 / chunk crossing case described in issue #31.

    #[test]
    fn slide_loop_round_trip_chunk_transition() {
        // After chunk N, all 3 attn layers have offloaded blocks 0..64.
        let mut cursors = vec![64u32, 64, 64];

        // Chunk N+1's bulk alloc loop: simulate 64 slides + 64 allocs with
        // window_start advancing one step per slide. cap = 64.
        let cap = 64;
        for slide_idx in 0..cap {
            let ws_before = slide_idx; // prior to this slide, ws = slide_idx
            // Safety check: every cursor > ws_before? Initial cursors are 64.
            // All slides up to slide_idx=63 have cursors > slide_idx → safe.
            assert!(
                check_safe_to_evict(&cursors, ws_before).is_ok(),
                "slide {slide_idx} should be safe with cursors {cursors:?}"
            );
            // Advance after the slide.
            advance_layer_cursors_after_slide(&mut cursors, ws_before + 1);
        }

        // After 64 slides (ws now 64), cursors should still be [64; 3] because
        // none of the advances moved them past 64 (each step advanced ws by 1
        // up to 64, and cursors started at 64 ≥ each new_ws).
        assert_eq!(cursors, vec![64, 64, 64]);
    }
}
