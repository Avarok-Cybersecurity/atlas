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
        images: &[(Vec<f32>, usize, usize)],
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
        let img_refs: Vec<(&[f32], usize, usize)> = images
            .iter()
            .map(|(px, gh, gw)| (px.as_slice(), *gh, *gw))
            .collect();
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
        let post_merge_grids: Vec<(usize, usize)> =
            per_image.iter().map(|(h, w, _)| (*h, *w)).collect();
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
        per_request: &[Vec<(Vec<f32>, usize, usize)>],
    ) -> Result<Vec<(usize, usize, usize, usize)>> {
        let ve = match &self.vision_encoder {
            Some(ve) => ve,
            None => return Ok(Vec::new()),
        };
        let stream = self.gpu.default_stream();
        // Flatten all requests' images, recording each request's (start, count).
        let mut flat: Vec<(&[f32], usize, usize)> = Vec::new();
        let mut req_bounds: Vec<(usize, usize)> = Vec::with_capacity(per_request.len());
        for imgs in per_request {
            req_bounds.push((flat.len(), imgs.len()));
            for (px, gh, gw) in imgs {
                flat.push((px.as_slice(), *gh, *gw));
            }
        }
        let per_image = ve.forward_batched(&flat, self.gpu.as_ref(), stream)?;
        let grids: Vec<(usize, usize)> = per_image.iter().map(|(h, w, _)| (*h, *w)).collect();
        let total_merged: usize = per_image.iter().map(|(_, _, mp)| *mp).sum();
        *self.vision_embed_patches.lock() = total_merged;
        *self.vision_image_grids.lock() = grids;
        // Per-request slice descriptors (request order matches the flatten order,
        // so row offsets accumulate Σ merged_p of earlier requests).
        let mut out = Vec::with_capacity(per_request.len());
        let mut row_cursor = 0usize;
        for (img_start, n_img) in req_bounds {
            let row_count: usize = per_image[img_start..img_start + n_img]
                .iter()
                .map(|(_, _, mp)| *mp)
                .sum();
            out.push((row_cursor, img_start, n_img, row_count));
            row_cursor += row_count;
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
