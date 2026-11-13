// SPDX-License-Identifier: AGPL-3.0-only
// Copy-on-write partial-hit scheme translated from vLLM (https://github.com/vllm-project/vllm,
// commit 8a36f08), vllm/v1/core/single_type_kv_cache_manager.py (`allocate_new_blocks`,
// `_apply_cow`), vllm/v1/core/kv_cache_manager.py:879-894 and the copy-before-forward order of
// vllm/v1/worker/gpu/model_runner.py:1382-1389, Copyright contributors to the vLLM project,
// Apache-2.0. Changes: the pending copy is applied at the prefill's prefix lookup instead of
// draining into a scheduler output; one D2D per attention layer and side on the prefill's own
// stream; the source is held by a KV reference instead of a second block-pool retain; a failed
// allocation falls back to the block-aligned match; by Atlas contributors. See
// VLLM-COW-ATTRIBUTION.md.

//! Responsibility: apply a copy-on-write prefix tail (row `prefix_cow_tail`,
//! issue #1193) at the chunked prefill's prefix lookup: choose the CoW-aware
//! lookup, allocate the private block, copy every attention layer's K and V
//! from the source on the prefill stream, and fall back to the block-aligned
//! match when the copy cannot be made.
//!
//! Ordering. The copies are enqueued on the same stream as every kernel of
//! this prefill, before the block table is uploaded or read, so the forward
//! sees the copied rows (vLLM's copy-before-forward). Rows `[0, rows)` of the
//! source were written by the donor's prefill before it published the block,
//! exactly as for a shared full block. Rows past `rows` are copied too and are
//! overwritten by this sequence before it reads them.
//!
//! References. The tree took no reference on the source for this sequence.
//! A KV reference held across the allocation keeps the source's bytes alive
//! even if `alloc_block_evicting` evicts it from the tree; it is released
//! right after the enqueue. When that release frees the block (the tree let
//! go of it meanwhile), the stream is drained first, so no later owner of the
//! block can write it while the copy still reads it. The private block belongs
//! to the sequence like any block it allocates and is freed with its table.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::prefix_cache::{PrefixCache, PrefixMatch};

use super::super::super::block_mgmt::alloc_block_evicting;
use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// One device-to-device copy of a CoW plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::model) struct CowCopy {
    /// Attention layer index in the KV cache.
    pub(in crate::model) layer: usize,
    /// `false` = K, `true` = V.
    pub(in crate::model) v: bool,
    pub(in crate::model) src: DevicePtr,
    pub(in crate::model) dst: DevicePtr,
    pub(in crate::model) bytes: usize,
}

/// Whether this prefill may take a sub-block match: never in a multi-rank
/// world (ranks must agree on one `matched` through F83, which is
/// block-granular) and never under `--high-speed-swap` (a private copy has no
/// disk image).
pub(in crate::model) fn cow_lookup_allowed(multi_rank: bool, kv_cache: &PagedKvCache) -> bool {
    !multi_rank && kv_cache.config().cache_blocks_per_seq.is_none()
}

/// The copy plan from block `src` to block `dst`: every attention layer, K
/// then V, the whole block (per-layer strides, so mixed dtypes and per-layer
/// dims are covered).
pub(in crate::model) fn copy_plan(kv_cache: &PagedKvCache, src: u32, dst: u32) -> Vec<CowCopy> {
    let mut plan = Vec::with_capacity(2 * kv_cache.num_layers());
    for layer in 0..kv_cache.num_layers() {
        plan.push(CowCopy {
            layer,
            v: false,
            src: kv_cache.k_cache_ptr(layer, src),
            dst: kv_cache.k_cache_ptr(layer, dst),
            bytes: kv_cache.k_block_stride_bytes_for_layer(layer),
        });
        plan.push(CowCopy {
            layer,
            v: true,
            src: kv_cache.v_cache_ptr(layer, src),
            dst: kv_cache.v_cache_ptr(layer, dst),
            bytes: kv_cache.v_block_stride_bytes_for_layer(layer),
        });
    }
    plan
}

/// Copy block `src` into a freshly allocated private block on `stream`.
///
/// `Ok(Some(dst))`: `dst` holds one reference (the caller's sequence) and
/// the copies are enqueued. `Ok(None)`: no copy was made and every reference
/// is as it was (the source is not live, or no block could be freed). `Err`:
/// an enqueue failed; `dst` is freed and the source hold released.
pub(in crate::model) fn copy_tail_block(
    gpu: &dyn GpuBackend,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
    src: u32,
    stream: u64,
) -> Result<Option<u32>> {
    if src as usize >= kv_cache.num_blocks() || kv_cache.ref_count(src) == 0 {
        // A 0-ref block is on the free list: its bytes are not the cached
        // rows any more, and a reference taken now would alias the allocator.
        return Ok(None);
    }
    kv_cache.inc_ref(src);
    let Some(dst) = alloc_block_evicting(kv_cache, prefix_cache) else {
        kv_cache.dec_ref(src);
        return Ok(None);
    };
    let copied = copy_plan(kv_cache, src, dst).iter().try_for_each(|c| {
        gpu.copy_d2d_async(c.src, c.dst, c.bytes, stream)
            .with_context(|| {
                format!(
                    "attention layer {} {} ({} bytes)",
                    c.layer,
                    if c.v { "V" } else { "K" },
                    c.bytes
                )
            })
    });
    // The hold is the last reference when the allocation evicted the source
    // from the tree: drain the copy before the block can be handed out again.
    let drained = if kv_cache.ref_count(src) == 1 {
        gpu.synchronize(stream)
    } else {
        Ok(())
    };
    kv_cache.dec_ref(src);
    if let Err(e) = copied.and(drained) {
        kv_cache.dec_ref(dst);
        return Err(e).context(format!("copy-on-write prefix tail {src} -> {dst}"));
    }
    Ok(Some(dst))
}

impl TransformerModel {
    /// The chunk-0 radix lookup of the per-stream prefill: the CoW-aware
    /// lookup where a sub-block match is allowed. With the row off the tree
    /// answers `lookup_cow` exactly as `lookup`.
    pub(in crate::model) fn prefill_b_lookup_maybe_cow(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &PagedKvCache,
    ) -> PrefixMatch {
        let bs = kv_cache.block_size();
        if cow_lookup_allowed(self.multi_rank_protocol_active(), kv_cache) {
            self.prefix_cache
                .lookup_cow(tokens, bs, seq.session_hash, seq.adapter_id)
        } else {
            self.prefix_cache
                .lookup(tokens, bs, seq.session_hash, seq.adapter_id)
        }
    }

    /// Apply `prefix_match.cow_tail`, if any: returns the private block the
    /// caller appends to the block table right after the shared blocks.
    ///
    /// When no copy can be made, the CoW lookup is undone and replaced by the
    /// block-aligned `lookup`, so `prefix_match` (its SSM snapshot depth
    /// included) never claims rows the sequence does not hold. On an enqueue
    /// error the lookup's references are released before the error returns
    /// (the caller has not recorded them yet).
    pub(in crate::model) fn prefill_b_apply_cow_tail(
        &self,
        prefix_match: &mut PrefixMatch,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) -> Result<Option<u32>> {
        let Some((src, rows)) = prefix_match.cow_tail else {
            return Ok(None);
        };
        let bs = kv_cache.block_size();
        debug_assert_eq!(
            prefix_match.matched_tokens,
            prefix_match.matched_blocks.len() * bs + rows,
            "a CoW match is the shared full blocks plus the tail rows"
        );
        let undo = |m: &PrefixMatch| {
            self.prefix_cache
                .release_matched(tokens, bs, m.matched_tokens, seq.adapter_id);
        };
        match copy_tail_block(
            self.gpu.as_ref(),
            kv_cache,
            self.prefix_cache.as_ref(),
            src,
            stream,
        ) {
            Ok(Some(dst)) => {
                spark_runtime::radix_tree::record_cow_copy(rows);
                tracing::debug!(
                    "prefix CoW tail: block {src} -> private {dst} ({rows} rows, {} layers) \
                     after {} shared blocks, slot={}",
                    kv_cache.num_layers(),
                    prefix_match.matched_blocks.len(),
                    seq.slot_idx,
                );
                Ok(Some(dst))
            }
            Ok(None) => {
                spark_runtime::radix_tree::record_cow_fallback();
                tracing::debug!(
                    "prefix CoW tail dropped (source {src} not live or no free block): \
                     block-aligned match instead, slot={}",
                    seq.slot_idx
                );
                undo(prefix_match);
                *prefix_match =
                    self.prefix_cache
                        .lookup(tokens, bs, seq.session_hash, seq.adapter_id);
                Ok(None)
            }
            Err(e) => {
                undo(prefix_match);
                Err(e)
            }
        }
    }
}

#[cfg(test)]
#[path = "prefix_cow_tests.rs"]
mod tests;
