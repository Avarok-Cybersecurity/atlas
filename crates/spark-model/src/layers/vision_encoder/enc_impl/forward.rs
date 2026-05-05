// SPDX-License-Identifier: AGPL-3.0-only

//! Top-level `VisionEncoder::forward`: drives the full image → token
//! pipeline (pos_embed → RoPE → patch embed → 27 ViT blocks with
//! deepstack merger taps → final merger).

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::super::VisionEncoder;

impl VisionEncoder {
    /// Full forward pass: pixels → [num_patches, out_hidden_size] BF16 in buf_out.
    ///
    /// Returns the number of output patches (= num_patches = grid_h × grid_w).
    pub fn forward(
        &self,
        pixels: &[f32], // [P, C*T*Hp*Wp = 1536]
        grid_h: usize,
        grid_w: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<usize> {
        let p = grid_h * grid_w;
        let merged_p = p / (self.spatial_merge_size * self.spatial_merge_size);

        // Per-image prep: interpolate learned pos_embed grid + build 2D
        // rotary cos/sin for attention. A/B toggles:
        //   ATLAS_VISION_POSINTERP=0  → copy raw pos_embed[0..p*h] instead
        //   ATLAS_VISION_ROPE=0       → upload cos=1, sin=0 (identity)
        let pos_interp_on = std::env::var("ATLAS_VISION_POSINTERP")
            .map(|v| v != "0")
            .unwrap_or(true);
        if pos_interp_on {
            self.resample_pos_embed(grid_h, grid_w, gpu, stream)?;
        } else {
            // Legacy path: copy first p*hidden_size entries of raw pos_embed.
            self.gpu_copy_bf16(
                gpu,
                self.pos_embed,
                self.buf_pos_resampled,
                p * self.hidden_size * 2,
                stream,
            )?;
        }
        self.build_rope_cossin(grid_h, grid_w, gpu, stream)?;

        self.patch_embed(pixels, p, gpu, stream)?;
        Self::maybe_dump_buf(
            gpu,
            self.buf_h1,
            p * self.hidden_size,
            "patch_embed",
            stream,
        )?;

        // Output layout in buf_out:
        //   rows [0 .. merged_p)                 = final merger (block 27)
        //   rows [k*merged_p .. (k+1)*merged_p)  = deepstack merger k
        // The splicer reads the FIRST merged_p rows per image — those must
        // be the final merger output (which replaces <|image_pad|> in the
        // LLM stream). Deepstack features come after and are unused until
        // the LLM-side injection path is wired (TODO).
        let n_h_bytes = p * self.hidden_size * 2;
        let mut deepstack_iter = self.deepstack_indexes.iter().enumerate();
        let mut next_ds = deepstack_iter.next(); // (merger_idx, &block_1indexed)

        for (block_idx, blk) in self.blocks.iter().enumerate() {
            self.vit_block(blk, p, gpu, stream)?;
            Self::maybe_dump_buf(
                gpu,
                self.buf_h1,
                p * self.hidden_size,
                &format!("block{block_idx:02}"),
                stream,
            )?;

            let block_1indexed = block_idx + 1;
            // Deepstack extraction point: snapshot buf_h1 → buf_h2 first,
            // then apply the merger out-of-place so the residual stream
            // into the next ViT block stays unperturbed.
            if let Some((ds_idx, &ds_block)) = next_ds
                && block_1indexed == ds_block
            {
                self.gpu_copy_bf16(gpu, self.buf_h1, self.buf_h2, n_h_bytes, stream)?;
                let offset_rows = (ds_idx + 1) * merged_p;
                let out_slice = self.buf_out.offset(offset_rows * self.out_hidden_size * 2);
                self.apply_merger(
                    &self.deepstack[ds_idx],
                    p,
                    grid_h,
                    grid_w,
                    self.buf_h2,
                    out_slice,
                    gpu,
                    stream,
                )?;
                next_ds = deepstack_iter.next();
            }
        }

        // Final merger runs on buf_h1 in-place (no subsequent block to protect).
        let out_slice = self.buf_out.offset(0);
        self.apply_merger(
            &self.merger,
            p,
            grid_h,
            grid_w,
            self.buf_h1,
            out_slice,
            gpu,
            stream,
        )?;
        let total_rows = (1 + self.deepstack_indexes.len()) * merged_p;
        Self::maybe_dump_buf(
            gpu,
            self.buf_out,
            total_rows * self.out_hidden_size,
            "final",
            stream,
        )?;

        Ok(total_rows)
    }
}
