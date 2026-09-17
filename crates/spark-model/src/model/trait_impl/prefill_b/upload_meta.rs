// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 3: stage and upload positions (T,H,W for MRoPE) + slot table
//! into the per-chunk metadata buffer carved out of `scratch`. Returns
//! the layout descriptor used by phase 3b/4.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

pub(in crate::model) struct MetaLayout {
    pub meta_base: DevicePtr,
    pub slot_offset: usize,
    pub pos_stream_bytes: usize,
    pub use_mrope: bool,
    pub needs_paged: bool,
}

impl TransformerModel {
    pub(super) fn prefill_b_upload_meta(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        proc_start: usize,
        proc_count: usize,
        effective_seq_len_start: usize,
        kv_cache: &PagedKvCache,
        stream: u64,
    ) -> Result<MetaLayout> {
        // Single-stream entry point: lay metadata at the default offset
        // after the MoE topk staging area.
        let moe_scratch_bytes = proc_count * self.config.num_experts_per_tok * 4 * 2;
        let meta_offset = (moe_scratch_bytes + 7) & !7;
        let meta_base = self.buffers.scratch().offset(meta_offset);
        self.prefill_b_upload_meta_at(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            proc_start,
            proc_count,
            effective_seq_len_start,
            kv_cache,
            meta_base,
            self.buffers.scratch_bytes().saturating_sub(meta_offset),
            stream,
        )
    }

    /// Build positions + slots metadata for `proc_count` tokens and upload
    /// to the caller-provided `meta_base` device pointer.
    ///
    /// `meta_region_bytes` is how much room this metadata block owns AT
    /// `meta_base` — the tail of the scratch arena for the single-stream entry
    /// point, one per-stream slice for the Q12 batched one. It is a required
    /// parameter because `meta_base` is a bare `DevicePtr` that carries no size,
    /// so without it the pack can only be bounded against the HOST staging
    /// buffer and would happily run off the end of the DEVICE allocation. Used by both the
    /// single-stream entry point above and Q12 batched prefill (multiple
    /// per-stream metadata blocks concatenated in one big scratch region).
    pub(in crate::model) fn prefill_b_upload_meta_at(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        proc_start: usize,
        proc_count: usize,
        effective_seq_len_start: usize,
        kv_cache: &PagedKvCache,
        meta_base: DevicePtr,
        meta_region_bytes: usize,
        stream: u64,
    ) -> Result<MetaLayout> {
        // MRoPE-interleaved packs three u32 position streams (T, H, W).
        let use_mrope = self.config.mrope_interleaved;
        let pos_stream_bytes = proc_count * 4;
        let slot_offset = if use_mrope {
            (pos_stream_bytes * 3 + 7) & !7
        } else {
            (pos_stream_bytes + 7) & !7
        };
        let needs_paged = effective_seq_len_start > 0;

        // Lock staging, build positions plus non-paged slots, and upload.
        {
            // SAFETY: Single-threaded scheduler access (see TransformerModel Send/Sync docs).
            let stg = unsafe { &mut *self.pinned_staging.get() };
            stg.positions.clear();
            stg.positions
                .extend(proc_start as u32..(proc_start + proc_count) as u32);

            // Build (T, H, W) streams matching HF Qwen3-VL's
            // `get_rope_index`/`get_vision_position_ids`:
            //   - text token: T = H = W = current_pos, increment current_pos by 1.
            //   - vision-pad run (item of t_len groups over a post-merge grid
            //     gh×gw, at base = current_pos): token k of group g gets
            //       T = base + g, H = base + k/gw, W = base + k%gw.
            //     After the run, current_pos += max(t_len, gh, gw).
            // This matters because Qwen3-VL/3.6 was trained with T constant
            // across one image and subsequent text tokens shifted by the
            // image's max spatial extent — Atlas's previous "T=linear over
            // all tokens" scheme produced out-of-distribution position IDs
            // for every post-image token.
            //
            // An IMAGE is the t_len = 1 case: `base + g` collapses to `base`,
            // and `max(1, gh, gw)` to `max(gh, gw)`, so the arithmetic below
            // is byte-identical to the image-only version it replaces. A
            // VIDEO is where the two diverge — its T advances once per
            // temporal group, which is the whole reason the run cannot be
            // treated as t_len separate images.
            if use_mrope {
                stg.positions_h.clear();
                stg.positions_w.clear();
                let grids = self.vision_image_grids.lock().clone();
                // Both pad tokens: consumed identically here — the item's
                // t_len already says which it is — but the scan has to
                // recognize both, or a video run would be walked one text
                // token at a time.
                let (pad_id, video_pad_id) = self.vision_pad_ids();
                let is_pad = |tok: u32| tok == pad_id || tok == video_pad_id;
                let chunk_tokens = &tokens[chunk_start..chunk_start + chunk_len];
                let have_vision = !grids.is_empty() && chunk_tokens.iter().copied().any(is_pad);

                if have_vision {
                    stg.positions.clear();
                    let current_pos: u32 = proc_start as u32;
                    // Co-dispatch: this request owns grids[grid_base .. grid_base+owned]
                    // of the shared packed vision_image_grids (0/all for legacy).
                    let grid_base = *self.vision_grid_base.lock();
                    let owned = *self.vision_owned_images.lock();
                    let grid_hi = if owned > 0 {
                        (grid_base + owned).min(grids.len())
                    } else {
                        grids.len()
                    };
                    mrope_pos::build(
                        chunk_tokens,
                        &grids,
                        grid_base,
                        grid_hi,
                        current_pos,
                        pad_id,
                        video_pad_id,
                        &mut stg.positions,
                        &mut stg.positions_h,
                        &mut stg.positions_w,
                    );
                } else {
                    stg.positions_h.extend_from_slice(&stg.positions);
                    stg.positions_w.extend_from_slice(&stg.positions);
                }
            }

            // Build the slot table before packing: the packer borrows the
            // staging struct, so every reusable `Vec` has to be final first.
            if !needs_paged {
                let bs = kv_cache.block_size();
                stg.slots.clear();
                stg.slots
                    .extend((proc_start..proc_start + proc_count).map(|i| {
                        let block_idx = seq
                            .physical_block_for(i / bs)
                            .unwrap_or(self.dummy_kv_block);
                        (block_idx as i64) * (bs as i64) + ((i % bs) as i64)
                    }));
            }

            // The MRoPE+vision arm above REBUILDS `positions` from the chunk's
            // pad-token runs (one entry per text token, `gh*gw` per image run)
            // instead of from `proc_count`, so its length is data-dependent —
            // it tracks `chunk_len`, and `proc_count <= chunk_len` only because
            // every `ProcRange::Compute` arm caps it there. `put_prefix_at`
            // carries that check: each stream contributes exactly `proc_count`
            // elements or the pack is refused.
            //
            // The DESTINATION bound used to be missing here entirely — the only
            // check was an `assert!(cursor <= stg.bytes)` AFTER the writes had
            // already landed, which is too late to keep them inside the
            // allocation. The packer checks each field before writing it.
            //
            // Rounding `slot_offset` up to 8 leaves up to 4 pad bytes after the
            // position streams that no copy writes; they are still initialised
            // (see the `pinned_pack` module docs).
            let mut pack = stg.packer_for(meta_region_bytes);
            pack.put_prefix_at("positions", 0, &stg.positions, proc_count)?;
            if use_mrope {
                let h_at = pos_stream_bytes;
                let w_at = h_at + pos_stream_bytes;
                pack.put_prefix_at("positions_h", h_at, &stg.positions_h, proc_count)?;
                pack.put_prefix_at("positions_w", w_at, &stg.positions_w, proc_count)?;
            }
            if !needs_paged {
                pack.put_prefix_at("slots", slot_offset, &stg.slots, proc_count)?;
            }
            self.gpu
                .copy_h2d_async_retained(pack.packed(), meta_base, stream)?;
        }

        Ok(MetaLayout {
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
        })
    }
}

#[path = "mrope_pos.rs"]
pub(crate) mod mrope_pos;
