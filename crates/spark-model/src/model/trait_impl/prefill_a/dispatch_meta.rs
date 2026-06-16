// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill phase A helpers split out of `prefill_a.rs` to keep both files
//! under the 500-LoC cap: the vision-embed dispatch and the single-pass
//! attention-metadata pinned-staging upload (original "Phase 3").
//!
//! Same `unsafe { from_raw_parts(...) }` H2D-staging contract as the
//! `verify_*.rs` files; see `verify_c.rs` module docs.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::traits::SequenceState;

impl TransformerModel {
    pub(in crate::model) fn prepare_vision_embed_dispatch(
        &self,
        images: &[(Vec<f32>, usize, usize)],
    ) -> Result<()> {
        let ve = match &self.vision_encoder {
            Some(ve) => ve,
            None => return Ok(()),
        };
        let stream = self.gpu.default_stream();
        let mut total_patches = 0usize;
        let mut post_merge_grids: Vec<(usize, usize)> = Vec::with_capacity(images.len());
        let sms = ve.spatial_merge_size.max(1);
        for (pixels, grid_h, grid_w) in images {
            let p = ve.forward(pixels, *grid_h, *grid_w, self.gpu.as_ref(), stream)?;
            total_patches += p;
            // Record post-merge dimensions for downstream MRoPE position
            // computation. The ViT folds `sms × sms` pre-merge patches into
            // a single output embedding, so the effective spatial grid
            // shrinks by that factor in each axis.
            post_merge_grids.push((grid_h / sms, grid_w / sms));
        }
        *self.vision_embed_patches.lock() = total_patches;
        *self.vision_image_grids.lock() = post_merge_grids;
        tracing::info!("Vision encoder: {} patches encoded", total_patches);
        Ok(())
    }

    /// Single-pass prefill "Phase 3": upload positions + slot table (and, when
    /// `marconi_skip`, the paged block_table + seq_len) via one pinned H2D copy,
    /// then return the assembled [`AttnMetadataDev`] for the layer forward pass.
    ///
    /// Pure I/O/staging phase lifted verbatim from `prefill_dispatch`; the
    /// kernel/state ordering of the caller is unchanged.
    pub(in crate::model) fn prefill_dispatch_upload_meta(
        &self,
        seq: &SequenceState,
        n: usize,
        proc_count: usize,
        seq_len_start: usize,
        bs: usize,
        marconi_skip: bool,
        stream: u64,
    ) -> Result<AttnMetadataDev> {
        // ── 3. Upload attention metadata via pinned staging (one H2D copy) ──
        let moe_scratch_bytes = proc_count * self.config.num_experts_per_tok * 4 * 2;
        let meta_offset = (moe_scratch_bytes + 7) & !7;
        let meta_base = self.buffers.scratch().offset(meta_offset);

        let slot_offset = (proc_count * 4 + 7) & !7;

        // Lock staging, build metadata, pack, single H2D copy
        let (block_table_dev, seq_len_dev) = {
            // SAFETY: Single-threaded scheduler access (see TransformerModel Send/Sync docs).
            let stg = unsafe { &mut *self.pinned_staging.get() };
            stg.positions.clear();
            stg.positions
                .extend(seq_len_start as u32..(seq_len_start + proc_count) as u32);
            stg.slots.clear();
            stg.slots
                .extend((seq_len_start..seq_len_start + proc_count).map(|i| {
                    let block_idx = seq
                        .physical_block_for(i / bs)
                        .unwrap_or(self.dummy_kv_block);
                    (block_idx as i64) * (bs as i64) + ((i % bs) as i64)
                }));

            let pinned = stg.ptr;
            let mut cursor = 0usize;

            unsafe {
                std::ptr::copy_nonoverlapping(
                    stg.positions.as_ptr() as *const u8,
                    pinned.add(cursor),
                    proc_count * 4,
                );
            }
            cursor = slot_offset;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    stg.slots.as_ptr() as *const u8,
                    pinned.add(cursor),
                    proc_count * 8,
                );
            }
            cursor += proc_count * 8;

            let devs = if marconi_skip {
                let bt_start = (cursor + 3) & !3;
                let bt_len = seq.block_table.len() * 4;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        seq.block_table.as_ptr() as *const u8,
                        pinned.add(bt_start),
                        bt_len,
                    );
                }
                let sl_start = (bt_start + bt_len + 3) & !3;
                let seq_len_val = n as u32;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &seq_len_val as *const u32 as *const u8,
                        pinned.add(sl_start),
                        4,
                    );
                }
                cursor = sl_start + 4;
                (meta_base.offset(bt_start), meta_base.offset(sl_start))
            } else {
                (DevicePtr::NULL, DevicePtr::NULL)
            };

            assert!(cursor <= stg.bytes, "prefill metadata overflow");
            let pinned_slice = unsafe { std::slice::from_raw_parts(pinned, cursor) };
            self.gpu.copy_h2d_async(pinned_slice, meta_base, stream)?;
            devs
        };

        Ok(AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(slot_offset),
            seq_len: seq_len_dev,
            block_table: block_table_dev,
            max_blocks_per_seq: seq.block_table.len() as u32,
            num_seqs: 1,
        })
    }
}
