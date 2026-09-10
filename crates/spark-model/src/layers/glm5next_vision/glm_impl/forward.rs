// SPDX-License-Identifier: AGPL-3.0-only

//! Top-level GLM ViT drive: rope prep → patch embed → 24 blocks → merger.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::super::Glm5NextVisionEncoder;
use crate::layers::vision_encoder::enc_impl::forward::check_packed_rows;
use crate::layers::vision_encoder::enc_impl::utils::maybe_dump_buf;

impl Glm5NextVisionEncoder {
    /// Single-image forward. Returns the number of merged rows written.
    ///
    /// Note the return differs from the Qwen tower's `(1 + deepstack) *
    /// merged_p`: GLM has no DeepStack, so `buf_out` holds exactly the merged
    /// rows and nothing else.
    pub fn forward(
        &self,
        pixels: &[f32],
        grid_h: usize,
        grid_w: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<usize> {
        let images = [(pixels, grid_h, grid_w)];
        Ok(self.forward_batched(&images, gpu, stream)?[0].2)
    }

    /// Batched forward over N images. Patch embed and all 24 blocks' GEMMs,
    /// norms and residuals run ONCE over `M = Σpᵢ`; the per-image-geometry
    /// stages (rope prep, attention, merger) loop per image.
    ///
    /// `buf_out` holds `[0 .. Σmerged_p)` in image order — the splice reads it
    /// through `VisionTower::out_row`.
    pub fn forward_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        // First image pays for the scratch; a text-only serve never gets here.
        self.scratch_init(gpu)?;
        let sms = self.spatial_merge_size.max(1);
        let sms2 = sms * sms;

        let mut p_i = Vec::with_capacity(images.len());
        let mut p_off = Vec::with_capacity(images.len());
        let mut mp_i = Vec::with_capacity(images.len());
        let mut mp_off = Vec::with_capacity(images.len());
        let (mut p_total, mut mp_total) = (0usize, 0usize);
        for (_px, gh, gw) in images.iter() {
            let p = gh * gw;
            let mp = p / sms2;
            p_off.push(p_total);
            mp_off.push(mp_total);
            p_i.push(p);
            mp_i.push(mp);
            p_total += p;
            mp_total += mp;
        }

        // Callers cap Σp ≤ p_max; a batch that defeats that cap (a video's
        // temporal groups arrive as ONE media item) falls back to per-image
        // encoding rather than overrunning the shared buffers.
        if p_total > self.p_max {
            return self.forward_oversized_fallback(images, &p_i, &mp_i, &mp_off, gpu, stream);
        }

        for (i, (_px, gh, gw)) in images.iter().enumerate() {
            let cos = self
                .scratch()
                .buf_rope_cos
                .offset(p_off[i] * self.head_dim * 2);
            let sin = self
                .scratch()
                .buf_rope_sin
                .offset(p_off[i] * self.head_dim * 2);
            self.build_rope_cossin_into(*gh, *gw, cos, sin, gpu, stream)?;
        }

        self.patch_embed_batched(images, &p_off, p_total, None, gpu, stream)?;
        maybe_dump_buf(
            gpu,
            self.scratch().buf_h1,
            p_total * self.hidden_size,
            "glm_patch_embed",
            stream,
        )?;

        for (block_idx, blk) in self.blocks.iter().enumerate() {
            self.glm_block_batched(blk, p_total, &p_i, &p_off, gpu, stream)?;
            maybe_dump_buf(
                gpu,
                self.scratch().buf_h1,
                p_total * self.hidden_size,
                &format!("glm_block{block_idx:02}"),
                stream,
            )?;
        }

        for (i, (_px, gh, gw)) in images.iter().enumerate() {
            let src = self
                .scratch()
                .buf_h1
                .offset(p_off[i] * self.hidden_size * 2);
            let out_slice = self
                .scratch()
                .buf_out
                .offset(mp_off[i] * self.out_hidden_size * 2);
            self.apply_merger(p_i[i], *gh, *gw, src, out_slice, gpu, stream)?;
        }
        maybe_dump_buf(
            gpu,
            self.scratch().buf_out,
            mp_total * self.out_hidden_size,
            "glm_final",
            stream,
        )?;

        Ok(images
            .iter()
            .map(|(_px, gh, gw)| (gh / sms, gw / sms, (gh * gw) / sms2))
            .collect())
    }

    /// Fallback for `Σp > p_max`: encode each image alone into the SAME packed
    /// `buf_out` layout. There is no separate single-image kernel path — the
    /// batched block at N=1 emits an identical kernel stream, and keeping one
    /// path removes the "two implementations that must stay in sync" hazard
    /// that produced the `check_pixel_len` bug next door.
    fn forward_oversized_fallback(
        &self,
        images: &[(&[f32], usize, usize)],
        p_i: &[usize],
        mp_i: &[usize],
        mp_off: &[usize],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        check_packed_rows(mp_i, mp_off, self.p_max)?;
        let sms = self.spatial_merge_size.max(1);
        for (i, (pixels, gh, gw)) in images.iter().enumerate() {
            let p = p_i[i];
            self.build_rope_cossin(*gh, *gw, gpu, stream)?;
            self.patch_embed(pixels, p, gpu, stream)?;
            for blk in self.blocks.iter() {
                self.glm_block_batched(blk, p, &[p], &[0], gpu, stream)?;
            }
            let out_slice = self
                .scratch()
                .buf_out
                .offset(mp_off[i] * self.out_hidden_size * 2);
            self.apply_merger(p, *gh, *gw, self.scratch().buf_h1, out_slice, gpu, stream)?;
        }
        Ok(images
            .iter()
            .map(|(_px, gh, gw)| (gh / sms, gw / sms, (gh * gw) / (sms * sms)))
            .collect())
    }
}
