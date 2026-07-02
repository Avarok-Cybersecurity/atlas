// SPDX-License-Identifier: AGPL-3.0-only

//! Q12 batched-prefill metadata staging.
//!
//! Builds the [`BatchedAttnMetadata`] struct that the model-level batched
//! per-layer dispatchers (`prefill_attn_batched_layer`,
//! `prefill_ssm_batched_layer`) consume. Called once per
//! `prefill_batch_chunk_dispatch` after the per-stream Phase 1-3 setup
//! has run and each stream has its `chunked_prefill_meta.block_table`
//! and `seq_len` device pointers populated.
//!
//! Layout produced (all in scratch buffer, distinct offsets):
//!   - positions_stacked       `[total_tokens]`           u32 = 4 B
//!   - positions_h_stacked     `[total_tokens]`           u32 (MRoPE only)
//!   - positions_w_stacked     `[total_tokens]`           u32 (MRoPE only)
//!   - slot_stacked            `[total_tokens]`           i64 = 8 B
//!   - block_table_ptrs        `[batch_size]`             DevicePtr = 8 B
//!   - seq_len_ptrs            `[batch_size]`             DevicePtr = 8 B
//!
//! `h_state_ptrs` is NOT staged here — it's per-layer (each SSM layer
//! owns its own h_state allocations across streams), so
//! `prefill_ssm_batched_layer` builds it just-in-time per-layer-call.
//!
//! The scratch buffer is sized for arena_cap × per-token-metadata size.
//! Q12 batched dispatch's fit-check (`total_tokens > arena_cap`) is done
//! upstream in `prefill_batch_chunk_dispatch` so this helper can assume
//! the scratch is large enough.
//!
//! Validation status: device-side correctness pending hardware run.
//! Compile-only verified.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::layer::BatchedAttnMetadata;
use crate::traits::{PrefillSlice, SequenceState};

/// Per-stream metadata needed to stage one entry of the batched arrays.
/// Built by the per-stream Phase 1-3 setup and consumed by
/// `stage_batched_attn_metadata`.
pub(in crate::model) struct PerStreamStageInfo<'a> {
    /// Absolute starting position of this stream's chunk in its sequence
    /// (`proc_start` from `prefill_b_proc_range`).
    pub proc_start: usize,
    /// Number of tokens this stream contributes — SAME across batched
    /// streams (scheduler-enforced via `can_batch_prefill_only`).
    pub proc_count: usize,
    /// Already-uploaded block_table device pointer for this stream.
    pub block_table_dev: DevicePtr,
    /// Already-uploaded seq_len device pointer for this stream.
    pub seq_len_dev: DevicePtr,
    /// Number of block-table entries this stream uses (for max_blocks_per_seq).
    pub num_blocks: usize,
    /// Reference to the stream's seq for slot-table construction.
    pub seq: &'a SequenceState,
}

impl TransformerModel {
    /// Build BatchedAttnMetadata for N streams. See module docs.
    ///
    /// `scratch_offset_bytes` is the byte offset within `self.buffers.scratch()`
    /// where staging starts. The caller is responsible for ensuring no
    /// concurrent usage of the same scratch region during this call.
    pub(in crate::model) fn stage_batched_attn_metadata(
        &self,
        streams_info: &[PerStreamStageInfo<'_>],
        kv_cache: &PagedKvCache,
        use_mrope: bool,
        scratch_offset_bytes: usize,
        stream: u64,
    ) -> Result<BatchedAttnMetadata> {
        let n = streams_info.len();
        if n == 0 {
            anyhow::bail!("stage_batched_attn_metadata called with empty streams_info");
        }
        // Constraint: same chunk_len across batched streams.
        let chunk_len = streams_info[0].proc_count;
        for (i, info) in streams_info.iter().enumerate() {
            if info.proc_count != chunk_len {
                anyhow::bail!(
                    "stage_batched_attn_metadata: stream {i} proc_count={} \
                     != batch chunk_len={chunk_len} (scheduler gate broken)",
                    info.proc_count
                );
            }
        }
        let total_tokens = n * chunk_len;

        // Layout offsets within scratch (relative to scratch_offset_bytes).
        let pos_bytes = total_tokens * 4;
        let pos_aligned = (pos_bytes + 7) & !7;
        let positions_off = scratch_offset_bytes;
        let positions_h_off = if use_mrope {
            positions_off + pos_aligned
        } else {
            positions_off
        };
        let positions_w_off = if use_mrope {
            positions_h_off + pos_aligned
        } else {
            positions_off
        };
        let after_positions = if use_mrope {
            positions_w_off + pos_aligned
        } else {
            positions_off + pos_aligned
        };
        let slot_off = after_positions;
        let slot_bytes = total_tokens * 8;
        let slot_aligned = (slot_bytes + 7) & !7;
        let block_ptrs_off = slot_off + slot_aligned;
        let block_ptrs_bytes = n * std::mem::size_of::<DevicePtr>();
        let block_ptrs_aligned = (block_ptrs_bytes + 7) & !7;
        let seq_len_ptrs_off = block_ptrs_off + block_ptrs_aligned;
        let seq_len_ptrs_aligned = (block_ptrs_bytes + 7) & !7;
        let total_meta_bytes = seq_len_ptrs_off + seq_len_ptrs_aligned - scratch_offset_bytes;

        // #110 defense-in-depth: bounds-check the metadata footprint against
        // the scratch allocation. This should NEVER fire — the dispatch-entry
        // predicate `check_kernel_batched_eligible` now pre-flights the full
        // batched footprint (`q12_batched_scratch_bytes`) against scratch_cap
        // BEFORE any stream state is mutated, and the scratch buffer is sized
        // for `Q12_SIZING_STREAMS` streams. If this guard ever trips, the
        // eligibility predicate and the staging layout have diverged (an SSOT
        // violation in `q12_batched_scratch_bytes`): bail rather than overrun
        // (out-of-range HtoD → corruption → sticky CUDA-700). Note this bail
        // is mid-Phase-A, so the per-stream fallback re-runs setup on
        // partially-mutated streams — strictly worse than the clean pre-flight
        // bail, hence "should never reach here".
        let scratch_cap = self.buffers.scratch_bytes();
        if scratch_offset_bytes + total_meta_bytes > scratch_cap {
            tracing::error!(
                "stage_batched_attn_metadata SSOT violation: n={n} chunk_len={chunk_len} \
                 meta_bytes={total_meta_bytes} scratch_offset={scratch_offset_bytes} \
                 scratch_cap={scratch_cap} — eligibility pre-flight should have prevented this"
            );
            anyhow::bail!(
                "batched attn metadata footprint {} B at offset {} exceeds scratch \
                 capacity {} B (n={n}, chunk_len={chunk_len}) — fall back to per-stream",
                total_meta_bytes,
                scratch_offset_bytes,
                scratch_cap,
            );
        }

        // Build host-side staging buffers, then upload via the model's
        // pinned-staging area.
        // SAFETY: same-thread invariant across TransformerModel.
        let stg = unsafe { &mut *self.pinned_staging.get() };
        stg.positions.clear();
        let mut max_blocks: u32 = 0;
        for info in streams_info.iter() {
            for t in 0..chunk_len {
                stg.positions.push((info.proc_start + t) as u32);
            }
            max_blocks = max_blocks.max(info.num_blocks as u32);
        }
        // MRoPE H/W streams reuse positions when MRoPE is disabled
        // (positions_h_stacked == positions_stacked etc.) so we only need
        // to stage them when use_mrope is true.
        if use_mrope {
            stg.positions_h.clear();
            stg.positions_w.clear();
            // For now: assume no vision pads in batched prefill (Q12 isn't
            // gated to vision-prompt batching). T = H = W = current_pos.
            // The scheduler can refuse to batch vision-pad-containing prompts
            // via `can_batch_prefill_only`.
            stg.positions_h.extend_from_slice(&stg.positions);
            stg.positions_w.extend_from_slice(&stg.positions);
        }

        // Slot table: each stream's slots = block_idx * block_size + offset.
        // Block index uses `seq.physical_block_for(token_pos / block_size)`
        // and falls back to dummy_kv_block when evicted/absent.
        let bs = kv_cache.block_size();
        stg.slots.clear();
        for info in streams_info.iter() {
            for t in 0..chunk_len {
                let pos = info.proc_start + t;
                let block_idx = info
                    .seq
                    .physical_block_for(pos / bs)
                    .unwrap_or(self.dummy_kv_block);
                let slot = (block_idx as i64) * (bs as i64) + ((pos % bs) as i64);
                stg.slots.push(slot);
            }
        }

        // Pointer arrays for block_table and seq_len.
        let mut bt_ptrs: Vec<u64> = Vec::with_capacity(n);
        let mut sl_ptrs: Vec<u64> = Vec::with_capacity(n);
        for info in streams_info.iter() {
            // DevicePtr is a transparent wrapper around u64 in spark-runtime;
            // we serialise by raw value to avoid Cargo cycles.
            bt_ptrs.push(info.block_table_dev.0);
            sl_ptrs.push(info.seq_len_dev.0);
        }

        // Single H2D copy of all metadata. The pinned-staging buffer holds
        // positions, MRoPE H/W (optional), slots, block_ptrs, seq_len_ptrs
        // packed contiguously. Upload to scratch at scratch_offset_bytes.
        let scratch_base = self.buffers.scratch().offset(scratch_offset_bytes);

        // Host-side pack into pinned buffer at the correct relative offsets.
        let pinned = stg.ptr;
        unsafe {
            let mut cursor = 0usize;
            // positions
            std::ptr::copy_nonoverlapping(
                stg.positions.as_ptr() as *const u8,
                pinned.add(cursor),
                pos_bytes,
            );
            cursor = pos_aligned;
            if use_mrope {
                std::ptr::copy_nonoverlapping(
                    stg.positions_h.as_ptr() as *const u8,
                    pinned.add(cursor),
                    pos_bytes,
                );
                cursor += pos_aligned;
                std::ptr::copy_nonoverlapping(
                    stg.positions_w.as_ptr() as *const u8,
                    pinned.add(cursor),
                    pos_bytes,
                );
                cursor += pos_aligned;
            }
            // slots
            std::ptr::copy_nonoverlapping(
                stg.slots.as_ptr() as *const u8,
                pinned.add(cursor),
                slot_bytes,
            );
            cursor += slot_aligned;
            // block_table_ptrs
            std::ptr::copy_nonoverlapping(
                bt_ptrs.as_ptr() as *const u8,
                pinned.add(cursor),
                block_ptrs_bytes,
            );
            cursor += block_ptrs_aligned;
            // seq_len_ptrs
            std::ptr::copy_nonoverlapping(
                sl_ptrs.as_ptr() as *const u8,
                pinned.add(cursor),
                block_ptrs_bytes,
            );
            cursor += seq_len_ptrs_aligned;
            assert!(
                cursor <= stg.bytes,
                "stage_batched_attn_metadata: pinned overflow {cursor} > {}",
                stg.bytes
            );
            let pinned_slice = std::slice::from_raw_parts(pinned, cursor);
            self.gpu
                .copy_h2d_async(pinned_slice, scratch_base, stream)?;
        }

        Ok(BatchedAttnMetadata {
            positions_stacked: scratch_base,
            positions_h_stacked: scratch_base.offset(positions_h_off - positions_off),
            positions_w_stacked: scratch_base.offset(positions_w_off - positions_off),
            slot_stacked: scratch_base.offset(slot_off - positions_off),
            block_table_ptrs: scratch_base.offset(block_ptrs_off - positions_off),
            seq_len_ptrs: scratch_base.offset(seq_len_ptrs_off - positions_off),
            batch_size: n as u32,
            chunk_len: chunk_len as u32,
            total_tokens: total_tokens as u32,
            max_blocks_per_seq: max_blocks,
            staged_bytes: total_meta_bytes,
        })
    }
}
