// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill phase A — vision-embed dispatch helpers.
//!
//! Extracted from `prefill_a.rs` to keep each file under the 500-LoC
//! file-size cap. These methods drive the ViT encoder (per-request and
//! cross-request batched) and stage the packed patch embeddings + grids
//! that the chunk-0 splice/MRoPE later consume.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;

use super::super::super::types::TransformerModel;

impl TransformerModel {
    pub(in crate::model) fn prepare_vision_embed_dispatch(
        &self,
        images: &[crate::VisionItem],
    ) -> Result<()> {
        let ve = match &self.vision_encoder {
            Some(ve) => ve,
            None => return Ok(()),
        };
        let stream = self.gpu.default_stream();
        // ONE batched ViT forward over all images in this request — block GEMM
        // weights read once over Σpatches instead of N× (the per-image loop also
        // overwrote buf_out row 0 every call, corrupting multi-image requests).
        // Each returned (post_h, post_w, merged_p) preserves image order, so the
        // packed buf_out matches the pad-token splice order downstream.
        // FLATTEN to the encoder's unit: one row-block per temporal group.
        // The ViT has no notion of time, so a clip's groups ride the same path
        // as separate stills; `item_groups` remembers which belong together so
        // the grids below can carry the item's temporal extent.
        let mut img_refs: Vec<(&[f32], usize, usize)> = Vec::new();
        let mut item_groups: Vec<usize> = Vec::with_capacity(images.len());
        for it in images {
            item_groups.push(it.t_len());
            for g in &it.groups {
                img_refs.push((g.as_slice(), it.grid_h, it.grid_w));
            }
        }
        let _vt0 = std::time::Instant::now();
        let per_image = ve.forward_batched(&img_refs, self.gpu.as_ref(), stream)?;
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            self.gpu.synchronize(stream).ok();
            tracing::info!(
                "VIT_TIMING self-encode {} imgs: {:.1}ms",
                images.len(),
                _vt0.elapsed().as_secs_f64() * 1000.0
            );
        }
        // Collapse the encoder's per-GROUP output back to per-ITEM grids,
        // carrying each item's temporal extent. A still yields (1, h, w),
        // which is what this produced before video existed.
        let post_merge_grids: Vec<(usize, usize, usize)> = {
            let mut out = Vec::with_capacity(item_groups.len());
            let mut row = 0usize;
            for t_len in &item_groups {
                let (h, w, _) = per_image[row];
                out.push((*t_len, h, w));
                row += t_len;
            }
            out
        };
        let total_merged: usize = per_image.iter().map(|(_, _, mp)| *mp).sum();
        *self.vision_embed_patches.lock() = total_merged;
        *self.vision_image_grids.lock() = post_merge_grids;
        tracing::info!(
            "Vision encoder (batched): {} images, {} merged patches encoded",
            images.len(),
            total_merged
        );
        Ok(())
    }

    /// Cross-request batched encode: flatten every request's images into ONE
    /// `forward_batched` call so block GEMM weights are read once over Σpatches
    /// across the whole tick (the concurrent-image win). `per_request[i]` holds
    /// request i's images. Fills the shared packed `buf_out` + `vision_image_grids`
    /// (in request-then-image order) and returns one
    /// `(patch_row_offset, grid_index_offset, num_images, patch_row_count)` per
    /// request locating its slice. Each request's chunk-0 splice/MRoPE then reads
    /// its slice via `set_vision_slice_base`.
    pub(in crate::model) fn prepare_vision_embed_batched_dispatch(
        &self,
        per_request: &[Vec<crate::VisionItem>],
    ) -> Result<Vec<(usize, usize, usize, usize)>> {
        let ve = match &self.vision_encoder {
            Some(ve) => ve,
            None => return Ok(Vec::new()),
        };
        let stream = self.gpu.default_stream();
        // Flatten all requests' images, recording each request's (start, count).
        let mut flat: Vec<(&[f32], usize, usize)> = Vec::new();
        let mut req_bounds: Vec<(usize, usize)> = Vec::with_capacity(per_request.len());
        let mut per_req_groups: Vec<Vec<usize>> = Vec::with_capacity(per_request.len());
        for imgs in per_request {
            let start = flat.len();
            let mut groups = Vec::with_capacity(imgs.len());
            for it in imgs {
                groups.push(it.t_len());
                for g in &it.groups {
                    flat.push((g.as_slice(), it.grid_h, it.grid_w));
                }
            }
            // Bounds are in ENCODER ROWS, which is what buf_out is indexed by.
            req_bounds.push((start, flat.len() - start));
            per_req_groups.push(groups);
        }
        let per_image = ve.forward_batched(&flat, self.gpu.as_ref(), stream)?;
        let grids: Vec<(usize, usize, usize)> = {
            let mut out = Vec::new();
            let mut row = 0usize;
            for groups in &per_req_groups {
                for t_len in groups {
                    let (h, w, _) = per_image[row];
                    out.push((*t_len, h, w));
                    row += t_len;
                }
            }
            out
        };
        let total_merged: usize = per_image.iter().map(|(_, _, mp)| *mp).sum();
        *self.vision_embed_patches.lock() = total_merged;
        *self.vision_image_grids.lock() = grids;
        // Per-request slice descriptors (request order matches the flatten order,
        // so row offsets accumulate Σ merged_p of earlier requests).
        // (buf_out row offset, GRID index offset, grid count, merged rows).
        // The grid offset/count are in ITEMS — `vision_image_grids` is now one
        // entry per item, not per encoder row — while the row offset stays in
        // merged rows, which is what the splice indexes.
        let mut out = Vec::with_capacity(per_request.len());
        let mut row_cursor = 0usize;
        let mut grid_cursor = 0usize;
        for ((enc_start, n_rows), groups) in req_bounds.iter().zip(&per_req_groups) {
            let row_count: usize = per_image[*enc_start..*enc_start + *n_rows]
                .iter()
                .map(|(_, _, mp)| *mp)
                .sum();
            out.push((row_cursor, grid_cursor, groups.len(), row_count));
            row_cursor += row_count;
            grid_cursor += groups.len();
        }
        tracing::info!(
            "Vision encoder (co-dispatch): {} requests, {} images, {} merged patches",
            per_request.len(),
            flat.len(),
            total_merged
        );
        Ok(out)
    }
}
