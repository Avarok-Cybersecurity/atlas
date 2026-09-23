// SPDX-License-Identifier: AGPL-3.0-only

//! `GlmVit::forward_batched` — pixels → merged LM-splice embeddings.
//!
//! `buf_out` holds the merger's output, packed in image order and nothing
//! else. GLM's tower is single-scale (there is no `deepstack_visual_indexes`
//! key in its `vision_config` and no deepstack merger in the checkpoint), so
//! unlike the Qwen encoder there is no second region to pack behind it.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::GlmVit;

/// The host pixel buffer for one image, checked against BOTH the geometry the
/// encoder was built for and the device rows this upload will touch.
///
/// Two separate bounds because either alone is insufficient — the same pair the
/// Qwen encoder learned the hard way (a `patch_size: 14` checkpoint reading 360
/// floats per patch past the end of a `Vec`, and a width-consistent buffer whose
/// patch count ran past the allocation). Prose is not a bound; this is.
fn check_pixels(
    pixels: &[f32],
    patches: usize,
    end_row: usize,
    patch_dim: usize,
    p_max: usize,
) -> Result<()> {
    let want = patches
        .checked_mul(patch_dim)
        .ok_or_else(|| anyhow::anyhow!("glm_vit: patch count {patches} overflows"))?;
    anyhow::ensure!(
        pixels.len() == want,
        "glm_vit: pixel buffer is {} floats for {patches} patches, but this encoder is built \
         for {patch_dim} floats per patch ({want}). The checkpoint's vision_config \
         patch_size/temporal_patch_size do not match the tower that was bound.",
        pixels.len()
    );
    anyhow::ensure!(
        end_row <= p_max,
        "glm_vit: this upload ends at patch row {end_row} but the encoder's buffers hold \
         {p_max} rows ({patches} patches in this image)."
    );
    Ok(())
}

impl GlmVit {
    /// Encode N images. Returns per-image `(post_h, post_w, merged_p)` in image
    /// order, matching the Qwen encoder's contract so the prefill splice is
    /// shared.
    ///
    /// Images are encoded in greedy chunks whose Σpatches fits `p_max`; the
    /// merged rows of every chunk land contiguously in `buf_out` at the global
    /// image order, so a batch that overflows the scratch degrades to more
    /// launches rather than to a wrong answer or an out-of-bounds write.
    pub fn forward_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        self.scratch_init(gpu)?;
        let sms = self.spatial_merge_size;
        let sms2 = sms * sms;

        for (_, gh, gw) in images {
            anyhow::ensure!(
                gh % sms == 0 && gw % sms == 0 && *gh > 0 && *gw > 0,
                "glm_vit: patch grid {gh}x{gw} is not a multiple of the {sms}x{sms} merge — \
                 the downsample reshapes four CONSECUTIVE tokens into one block and an odd \
                 grid silently mixes rows"
            );
            anyhow::ensure!(
                gh * gw <= self.p_max,
                "glm_vit: one image of {} patches exceeds the encoder capacity of {} rows",
                gh * gw,
                self.p_max
            );
        }

        let mut total_merged = 0usize;
        for (_, gh, gw) in images {
            total_merged += (gh * gw) / sms2;
        }
        anyhow::ensure!(
            total_merged <= self.mp_max(),
            "glm_vit: this batch packs {total_merged} merged rows into an output buffer of {} \
             rows. Send fewer images, or allocate a larger vision scratch.",
            self.mp_max()
        );

        let mut mp_base = 0usize;
        let mut start = 0usize;
        while start < images.len() {
            let mut end = start;
            let mut p_sum = 0usize;
            while end < images.len() {
                let (_, gh, gw) = images[end];
                if p_sum + gh * gw > self.p_max {
                    break;
                }
                p_sum += gh * gw;
                end += 1;
            }
            debug_assert!(end > start, "capacity was checked per image above");
            self.encode_chunk(&images[start..end], mp_base, gpu, stream)?;
            mp_base += p_sum / sms2;
            start = end;
        }

        Ok(images
            .iter()
            .map(|(_, gh, gw)| (gh / sms, gw / sms, (gh * gw) / sms2))
            .collect())
    }

    /// One chunk whose Σpatches fits `p_max`. Writes `Σmerged` rows of
    /// `buf_out` starting at `mp_base`.
    fn encode_chunk(
        &self,
        images: &[(&[f32], usize, usize)],
        mp_base: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let s = self.scratch();
        let sms2 = self.spatial_merge_size * self.spatial_merge_size;
        let mut p_i = Vec::with_capacity(images.len());
        let mut p_off = Vec::with_capacity(images.len());
        let mut p_total = 0usize;
        for (_, gh, gw) in images {
            p_off.push(p_total);
            p_i.push(gh * gw);
            p_total += gh * gw;
        }

        // 1. Per-image host prep: the axial RoPE tables, packed at p_off[i].
        for (i, (_, gh, gw)) in images.iter().enumerate() {
            self.upload_rope_tables(
                *gh,
                *gw,
                s.buf_rope_cos.offset(p_off[i] * self.head_dim * 2),
                s.buf_rope_sin.offset(p_off[i] * self.head_dim * 2),
                gpu,
                stream,
            )?;
        }

        // 2. Patch embed over M = Σp. The Conv3d degenerates to one dot product
        //    per patch (kernel == input block, no sliding), so it IS a GEMM
        //    against the weight flattened as (in_channel, temporal, py, px) —
        //    the same order the host preprocessor lays each patch out in.
        for (i, (pixels, gh, gw)) in images.iter().enumerate() {
            let p = gh * gw;
            let end_row = p_off[i]
                .checked_add(p)
                .ok_or_else(|| anyhow::anyhow!("glm_vit: patch row offset overflows"))?;
            check_pixels(pixels, p, end_row, self.patch_dim, self.p_max)?;
            // SAFETY: `pixels` is a live `&[f32]` and the byte length is taken
            // from that same slice, so the view never leaves the allocation.
            // `f32` has no invalid bit patterns and `u8` has alignment 1.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(pixels.as_ptr().cast::<u8>(), pixels.len() * 4)
            };
            gpu.copy_h2d_async(
                bytes,
                s.buf_f32.offset(p_off[i] * self.patch_dim * 4),
                stream,
            )?;
        }
        let n_f32 = (p_total * self.patch_dim) as u32;
        KernelLaunch::new(gpu, self.k_f32_bf16)
            .grid([div_ceil(n_f32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.buf_f32)
            .arg_ptr(s.buf_gate)
            .arg_u32(n_f32)
            .launch(stream)?;
        self.gemm(
            gpu,
            s.buf_gate,
            self.patch_embed_w,
            Some(self.patch_embed_b),
            s.buf_h1,
            p_total as u32,
            self.hidden_size as u32,
            self.patch_dim as u32,
            stream,
        )?;
        self.maybe_dump(
            gpu,
            s.buf_h1,
            p_total * self.hidden_size,
            "patch_embed",
            stream,
        )?;

        // 3. There is NO learned positional embedding to add here — the
        //    checkpoint ships no `pos_embed` tensor and the config declares
        //    none. All position information enters through the axial RoPE
        //    inside each block's attention.
        for (idx, blk) in self.blocks.iter().enumerate() {
            self.block(blk, p_total, &p_i, &p_off, gpu, stream)?;
            self.maybe_dump(
                gpu,
                s.buf_h1,
                p_total * self.hidden_size,
                &format!("block{idx:02}"),
                stream,
            )?;
        }

        // 4. post_layernorm (weight-only RMSNorm) → conv downsample → merger.
        self.rmsnorm(
            gpu,
            s.buf_h1,
            self.merger.post_layernorm_w,
            p_total as u32,
            self.hidden_size as u32,
            stream,
        )?;
        let out = s.buf_out.offset(mp_base * self.out_hidden_size * 2);
        self.downsample_and_merge(p_total / sms2, out, gpu, stream)?;
        self.maybe_dump(
            gpu,
            out,
            (p_total / sms2) * self.out_hidden_size,
            "merged",
            stream,
        )
    }

    /// `AVAROK_DUMP_GLM_VIT=<dir>` snapshots a BF16 stage to
    /// `<dir>/<label>.bin` (raw little-endian, no header) for a tap-by-tap diff
    /// against the Python reference. Off by default.
    pub(super) fn maybe_dump(
        &self,
        gpu: &dyn GpuBackend,
        ptr: spark_runtime::gpu::DevicePtr,
        n_elements: usize,
        label: &str,
        stream: u64,
    ) -> Result<()> {
        let Ok(dir) = std::env::var("AVAROK_DUMP_GLM_VIT") else {
            return Ok(());
        };
        if dir.is_empty() {
            return Ok(());
        }
        gpu.synchronize(stream)?;
        let mut buf = vec![0u8; n_elements * 2];
        gpu.copy_d2h(ptr, &mut buf)?;
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(
            std::path::Path::new(&dir).join(format!("{label}.bin")),
            &buf,
        )?;
        tracing::info!("AVAROK_DUMP_GLM_VIT: wrote {label} ({n_elements} elements)");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::check_pixels;

    /// GLM's geometry: 3 × 2 × 14² = 1176 floats per patch.
    #[test]
    fn accepts_the_glm_patch_width() {
        assert!(check_pixels(&vec![0.0; 64 * 1176], 64, 64, 1176, 6400).is_ok());
    }

    /// A Qwen-shaped buffer (1536 floats per patch) against a GLM encoder is
    /// the mirror image of the bug that made the Qwen encoder unable to bind
    /// this checkpoint — and it must be refused, not DMA'd.
    #[test]
    fn rejects_a_qwen_width_buffer() {
        let err = check_pixels(&vec![0.0; 64 * 1536], 64, 64, 1176, 6400)
            .unwrap_err()
            .to_string();
        assert!(err.contains("1176"), "{err}");
        assert!(err.contains("patch_size"), "{err}");
    }

    /// The bound on the DESTINATION row, not the per-image count: a small
    /// image placed near the end of the packed buffer still overruns.
    #[test]
    fn rejects_a_small_image_placed_past_the_end() {
        let pixels = vec![0.0f32; 8 * 1176];
        assert!(check_pixels(&pixels, 8, 6399 + 8, 1176, 6400).is_err());
        assert!(check_pixels(&pixels, 8, 100 + 8, 1176, 6400).is_ok());
    }

    #[test]
    fn rejects_a_patch_count_that_overflows() {
        let err = check_pixels(&[], usize::MAX / 2, 0, 1176, 6400)
            .unwrap_err()
            .to_string();
        assert!(err.contains("overflow"), "{err}");
    }
}
