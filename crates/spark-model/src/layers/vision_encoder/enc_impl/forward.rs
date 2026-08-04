// SPDX-License-Identifier: AGPL-3.0-only

//! Top-level `VisionEncoder::forward` / `forward_batched`: drives the full
//! image → token pipeline (pos_embed → RoPE → patch embed → 27 ViT blocks
//! with deepstack merger taps → final merger). The batched form runs the
//! weight-bound block GEMMs ONCE over Σpatches across N images so concurrent
//! image requests stop serializing; per-image-geometry stages (host pos/rope
//! prep, attention, mergers) loop per image.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::super::VisionEncoder;

impl VisionEncoder {
    /// Single-image forward (back-compat shim). For N=1 this issues the SAME
    /// kernels with the SAME args in the SAME order as the old per-image path
    /// → byte-identical output. Returns `total_rows = (1+n_deepstack)*merged_p`
    /// (the OLD return value; only ever tested `> 0` downstream).
    pub fn forward(
        &self,
        pixels: &[f32], // [P, C*T*Hp*Wp = 1536]
        grid_h: usize,
        grid_w: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<usize> {
        let images = [(pixels, grid_h, grid_w)];
        let per_image = self.forward_batched(&images, gpu, stream)?;
        let merged_p = per_image[0].2;
        Ok((1 + self.deepstack_indexes.len()) * merged_p)
    }

    /// Batched forward over N images. M-agnostic ops (patch_embed + all 27
    /// blocks' GEMMs/norms/gelu/residuals) run ONCE over M=Σpᵢ; per-image-
    /// geometry stages (host pos/rope prep, attention, mergers) loop per image.
    ///
    /// `buf_out` layout (rows of out_hidden_size BF16):
    ///   [0 .. Σmerged_p)            = final merger, IMAGE-ORDER packed ← splicer reads here
    ///   [(k+1)*Σmerged_p .. )       = deepstack-k, image-order packed   ← LLM-unused
    ///
    /// IN-BOUNDS INVARIANT: the deepstack high-water row is 4·Σmerged_p = Σp ≤
    /// p_max (all four mergers emit exactly merged_p rows each), and buf_out is
    /// p_max rows → no realloc, no overrun.
    ///
    /// Returns per-image `(post_h, post_w, merged_p)` in image order.
    pub fn forward_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        let sms2 = self.spatial_merge_size * self.spatial_merge_size;
        let sms = self.spatial_merge_size.max(1);
        let n_img = images.len();

        // Per-image pre-merge patch counts (p_i) and post-merge counts (mp_i),
        // with running row offsets into the shared buffers (p_off / mp_off).
        let mut p_i = Vec::with_capacity(n_img);
        let mut p_off = Vec::with_capacity(n_img);
        let mut mp_i = Vec::with_capacity(n_img);
        let mut mp_off = Vec::with_capacity(n_img);
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

        // Callers cap Σp ≤ p_max; defend here with a still-correct per-image
        // fallback that packs buf_out the same way.
        if p_total > self.p_max {
            return self.forward_oversized_fallback(images, &p_i, &mp_i, &mp_off, sms, gpu, stream);
        }

        let _sec0 = std::time::Instant::now();
        // 1. Per-image host prep, packed into the SHARED buffers at p_off[i].
        let pos_interp_on = std::env::var("ATLAS_VISION_POSINTERP")
            .map(|v| v != "0")
            .unwrap_or(true);
        for (i, (_px, gh, gw)) in images.iter().enumerate() {
            let p = p_i[i];
            let pos_dst = self
                .buf_pos_resampled
                .offset(p_off[i] * self.hidden_size * 2);
            if pos_interp_on {
                self.resample_pos_embed_into(*gh, *gw, pos_dst, gpu, stream)?;
            } else {
                self.gpu_copy_bf16(
                    gpu,
                    self.pos_embed,
                    pos_dst,
                    p * self.hidden_size * 2,
                    stream,
                )?;
            }
            let cos_dst = self.buf_rope_cos.offset(p_off[i] * self.head_dim * 2);
            let sin_dst = self.buf_rope_sin.offset(p_off[i] * self.head_dim * 2);
            self.build_rope_cossin_into(*gh, *gw, cos_dst, sin_dst, gpu, stream)?;
        }

        let timing = std::env::var("ATLAS_VISION_TIMING").is_ok();
        if timing {
            gpu.synchronize(stream).ok();
            tracing::info!(
                "VIT_SEC host_prep({n_img} imgs): {:.1}ms",
                _sec0.elapsed().as_secs_f64() * 1000.0
            );
        }
        let _sec1 = std::time::Instant::now();
        // 2. Patch embed over M=Σp.
        self.patch_embed_batched(images, &p_off, p_total, gpu, stream)?;
        Self::maybe_dump_buf(
            gpu,
            self.buf_h1,
            p_total * self.hidden_size,
            "patch_embed",
            stream,
        )?;

        // 3. 27 blocks: M-agnostic ops once, attention per image, deepstack per image.
        let n_h_bytes = p_total * self.hidden_size * 2;
        let mut deepstack_iter = self.deepstack_indexes.iter().enumerate();
        let mut next_ds = deepstack_iter.next(); // (merger_idx, &block_1indexed)
        for (block_idx, blk) in self.blocks.iter().enumerate() {
            self.vit_block_batched(blk, p_total, &p_i, &p_off, gpu, stream)?;
            Self::maybe_dump_buf(
                gpu,
                self.buf_h1,
                p_total * self.hidden_size,
                &format!("block{block_idx:02}"),
                stream,
            )?;
            if let Some((ds_idx, &ds_block)) = next_ds
                && block_idx + 1 == ds_block
            {
                // snapshot buf_h1 → buf_h2 (out-of-place merger; residual stream
                // into the next block stays intact), then merge each image's slice.
                self.gpu_copy_bf16(gpu, self.buf_h1, self.buf_h2, n_h_bytes, stream)?;
                let ds_region_base = (ds_idx + 1) * mp_total;
                for (i, (_px, gh, gw)) in images.iter().enumerate() {
                    let src = self.buf_h2.offset(p_off[i] * self.hidden_size * 2);
                    let out_rows = ds_region_base + mp_off[i];
                    let out_slice = self.buf_out.offset(out_rows * self.out_hidden_size * 2);
                    self.apply_merger(
                        &self.deepstack[ds_idx],
                        p_i[i],
                        *gh,
                        *gw,
                        src,
                        out_slice,
                        gpu,
                        stream,
                    )?;
                }
                next_ds = deepstack_iter.next();
            }
        }

        if timing {
            gpu.synchronize(stream).ok();
            tracing::info!(
                "VIT_SEC patch+27blocks(M={p_total}): {:.1}ms",
                _sec1.elapsed().as_secs_f64() * 1000.0
            );
        }
        let _sec2 = std::time::Instant::now();
        // 4. Final merger per image → packed [0 .. Σmerged_p).
        for (i, (_px, gh, gw)) in images.iter().enumerate() {
            let src = self.buf_h1.offset(p_off[i] * self.hidden_size * 2);
            let out_slice = self.buf_out.offset(mp_off[i] * self.out_hidden_size * 2);
            self.apply_merger(&self.merger, p_i[i], *gh, *gw, src, out_slice, gpu, stream)?;
        }
        if timing {
            gpu.synchronize(stream).ok();
            tracing::info!(
                "VIT_SEC mergers(final+{} ds): {:.1}ms",
                self.deepstack_indexes.len(),
                _sec2.elapsed().as_secs_f64() * 1000.0
            );
        }
        // Dump the full packed region (final + deepstack) so N=1 == the old
        // `total_rows` span exactly (byte-identity validation).
        let dump_rows = (1 + self.deepstack_indexes.len()) * mp_total;
        Self::maybe_dump_buf(
            gpu,
            self.buf_out,
            dump_rows * self.out_hidden_size,
            "final",
            stream,
        )?;

        Ok(images
            .iter()
            .map(|(_px, gh, gw)| (gh / sms, gw / sms, (gh * gw) / sms2))
            .collect())
    }

    /// Fallback for Σp > p_max: encode each image alone (full single-image
    /// kernel sequence) writing its final-merger rows into the PACKED buf_out
    /// at mp_off[i]. NO deepstack write (LLM-unused; a packed deepstack region
    /// could overrun under an oversized batch). The scheduler caps Σp ≤ p_max
    /// so this is normally unreachable — a correctness guard only.
    #[allow(clippy::too_many_arguments)]
    fn forward_oversized_fallback(
        &self,
        images: &[(&[f32], usize, usize)],
        p_i: &[usize],
        mp_i: &[usize],
        mp_off: &[usize],
        sms: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        debug_assert!(
            mp_off.last().map(|o| o + mp_i.last().unwrap()).unwrap_or(0) <= self.p_max,
            "oversized vision batch: Σmerged_p exceeds buf_out rows"
        );
        let pos_interp_on = std::env::var("ATLAS_VISION_POSINTERP")
            .map(|v| v != "0")
            .unwrap_or(true);
        for (i, (pixels, gh, gw)) in images.iter().enumerate() {
            let p = p_i[i];
            if pos_interp_on {
                self.resample_pos_embed(*gh, *gw, gpu, stream)?;
            } else {
                self.gpu_copy_bf16(
                    gpu,
                    self.pos_embed,
                    self.buf_pos_resampled,
                    p * self.hidden_size * 2,
                    stream,
                )?;
            }
            self.build_rope_cossin(*gh, *gw, gpu, stream)?;
            self.patch_embed(pixels, p, gpu, stream)?;
            for blk in self.blocks.iter() {
                self.vit_block(blk, p, gpu, stream)?;
            }
            let out_slice = self.buf_out.offset(mp_off[i] * self.out_hidden_size * 2);
            self.apply_merger(
                &self.merger,
                p,
                *gh,
                *gw,
                self.buf_h1,
                out_slice,
                gpu,
                stream,
            )?;
        }
        Ok(images
            .iter()
            .map(|(_px, gh, gw)| (gh / sms, gw / sms, (gh * gw) / (sms * sms)))
            .collect())
    }
}
