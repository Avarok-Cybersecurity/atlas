// SPDX-License-Identifier: AGPL-3.0-only

//! Patch-embed step: f32 pixels → BF16 → patch_embed GEMM → +pos_embed.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::{PATCH_DIM, VisionEncoder};

/// Check a host pixel buffer against BOTH the geometry and the capacity the
/// encoder was built for, before its bytes are reinterpreted and DMA'd.
///
/// Two independent bounds, and each one alone is insufficient:
///
/// 1. **Width.** `pixels` is sized by the CPU preprocessor from the
///    checkpoint's `vision_config` (`3 × temporal_patch_size × patch_size²` per
///    patch), while the encoder's device buffer and GEMM are fixed at
///    [`PATCH_DIM`]. A checkpoint declaring e.g. `patch_size: 14` yields 1176
///    floats per patch, so a `p * PATCH_DIM * 4` byte length ran 360 floats per
///    patch PAST the end of the `Vec` — an out-of-bounds read that then went to
///    the GPU.
///
/// 2. **Count.** `end_row` (the last device row this upload touches) against
///    `p_max`, the row capacity `buf_f32` and every downstream buffer were
///    allocated for. This check did not exist. A checkpoint whose
///    `vision_config` happens to yield exactly `PATCH_DIM` floats per patch on a
///    finer grid — `patch_size: 4, temporal_patch_size: 32` is one — passes
///    bound 1 with a CONSISTENT buffer while `p` runs far past `p_max`: at that
///    geometry one image is 102400 patches, a 629 MB H2D into a 39 MB
///    allocation. The batched entry point carried a comment asserting callers
///    cap `Σp ≤ p_max`, but the single-image path's only caller is inside
///    `forward_oversized_fallback`, which bounds Σ*merged* p and not per-image
///    `p`. Prose is not a bound; this is.
fn check_pixel_len(pixels: &[f32], patches: usize, end_row: usize, p_max: usize) -> Result<()> {
    let want = patches
        .checked_mul(PATCH_DIM)
        .ok_or_else(|| anyhow::anyhow!("vision: patch count {patches} overflows"))?;
    anyhow::ensure!(
        pixels.len() == want,
        "vision: pixel buffer is {} floats for {patches} patches, but this encoder is built \
         for {PATCH_DIM} floats per patch ({want}). The checkpoint's vision_config \
         patch_size/temporal_patch_size do not match the compiled ViT.",
        pixels.len()
    );
    anyhow::ensure!(
        end_row <= p_max,
        "vision: this upload ends at patch row {end_row} but the encoder's buffers hold \
         {p_max} rows ({patches} patches in this image). The checkpoint's vision_config \
         yields a finer patch grid than the compiled ViT was allocated for."
    );
    Ok(())
}

impl VisionEncoder {
    /// Upload f32 pixels → convert to BF16 → patch embed GEMM → add pos_embed.
    pub(super) fn patch_embed(
        &self,
        pixels: &[f32],
        p: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        // Single image at row 0, so the last row touched is `p`.
        check_pixel_len(pixels, p, p, self.p_max)?;
        let n_f32 = pixels.len();
        // SAFETY: `pixels` is a live `&[f32]`; the byte length is taken from
        // that same slice (`len() * 4`), so the view never leaves the
        // allocation. f32 has no padding or invalid bit patterns, and u8 has
        // alignment 1, so every byte of it is a valid `u8`. The view is
        // read-only and dies at the end of this function, before `pixels`.
        let f32_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u8, n_f32 * 4) };
        gpu.copy_h2d_async(f32_bytes, self.scratch().buf_f32, stream)?;
        // f32 → bf16 (result in buf_wide[0..p*PATCH_DIM])
        KernelLaunch::new(gpu, self.k_f32_bf16)
            .grid([div_ceil(n_f32 as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_f32)
            .arg_ptr(self.scratch().buf_wide)
            .arg_u32(n_f32 as u32)
            .launch(stream)?;
        // patch_embed GEMM: buf_wide[p,K] @ patch_embed_w[1152,K]^T + b → buf_h1[p,1152]
        self.vit_gemm_bias(
            gpu,
            self.scratch().buf_wide,
            self.patch_embed_w,
            self.patch_embed_b,
            self.scratch().buf_h1,
            p as u32,
            self.hidden_size as u32,
            PATCH_DIM as u32,
            stream,
        )?;
        // add the image-specific bilinear-interpolated pos_embed to buf_h1.
        // (Source was prepared by `resample_pos_embed()` in forward().)
        let n_pe = p * self.hidden_size;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_pe as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_h1)
            .arg_ptr(self.scratch().buf_pos_resampled)
            .arg_u32(n_pe as u32)
            .launch(stream)
    }

    /// Batched patch-embed over N images packed at `p_off[i]` (rows).
    /// Uploads each image's f32 pixels into `buf_f32` at its row offset, then
    /// runs ONE f32→bf16, ONE patch_embed GEMM (M=p_total), and ONE pos_embed
    /// add over the whole batch. `buf_pos_resampled` must already hold each
    /// image's per-row pos embed (filled by `resample_pos_embed_into`). For
    /// N=1 (p_off=[0]) this is byte-identical to `patch_embed`.
    pub(super) fn patch_embed_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        p_off: &[usize],
        p_total: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        // Upload each image's pixels into its row slice of buf_f32.
        // `check_pixel_len` bounds both ends: the host `Vec` (width) and the
        // `p_max × PATCH_DIM` device allocation (the row this image ends at).
        for (i, (pixels, gh, gw)) in images.iter().enumerate() {
            // Each image lands at row `p_off[i]`, so its last row is
            // `p_off[i] + gh*gw` — the exact bound on the destination, rather
            // than the `Σp ≤ p_max` the caller was trusted to have applied.
            let p_i = gh * gw;
            let end_row = p_off[i]
                .checked_add(p_i)
                .ok_or_else(|| anyhow::anyhow!("vision: patch row offset overflows"))?;
            check_pixel_len(pixels, p_i, end_row, self.p_max)?;
            // SAFETY: `pixels` is a live `&[f32]` and the byte length is
            // derived from that same slice, so the view stays inside its
            // allocation. `f32` has no invalid bit patterns and `u8` has
            // alignment 1, so the reinterpretation is valid for every byte.
            let f32_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4)
            };
            gpu.copy_h2d_async(
                f32_bytes,
                self.scratch().buf_f32.offset(p_off[i] * PATCH_DIM * 4),
                stream,
            )?;
        }
        let n_f32 = p_total * PATCH_DIM;
        // f32 → bf16 (result in buf_wide[0..p_total*PATCH_DIM])
        KernelLaunch::new(gpu, self.k_f32_bf16)
            .grid([div_ceil(n_f32 as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_f32)
            .arg_ptr(self.scratch().buf_wide)
            .arg_u32(n_f32 as u32)
            .launch(stream)?;
        // patch_embed GEMM over M=p_total → buf_h1
        self.vit_gemm_bias(
            gpu,
            self.scratch().buf_wide,
            self.patch_embed_w,
            self.patch_embed_b,
            self.scratch().buf_h1,
            p_total as u32,
            self.hidden_size as u32,
            PATCH_DIM as u32,
            stream,
        )?;
        // add the per-image interpolated pos_embed (packed in buf_pos_resampled).
        let n_pe = p_total * self.hidden_size;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_pe as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_h1)
            .arg_ptr(self.scratch().buf_pos_resampled)
            .arg_u32(n_pe as u32)
            .launch(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::{PATCH_DIM, check_pixel_len};

    /// The shipped Qwen3-VL geometry: patch_size 16, temporal_patch_size 2 →
    /// 3 × 2 × 16 × 16 = 1536 floats per patch.
    #[test]
    fn accepts_the_geometry_the_encoder_was_built_for() {
        assert_eq!(PATCH_DIM, 3 * 2 * 16 * 16);
        let pixels = vec![0.0f32; 64 * PATCH_DIM];
        assert!(check_pixel_len(&pixels, 64, 64, 6400).is_ok());
        // Zero patches (an image that scaled to nothing) is consistent, not a
        // slice-length hazard.
        assert!(check_pixel_len(&[], 0, 0, 6400).is_ok());
    }

    /// A checkpoint declaring `patch_size: 14` (the Qwen2-VL geometry) makes
    /// the CPU preprocessor emit 3 × 2 × 14 × 14 = 1176 floats per patch. The
    /// old code formed a `p * 1536 * 4`-byte view over that buffer and DMA'd
    /// it — reading 360 floats per patch past the end of the allocation.
    #[test]
    fn rejects_narrower_patch_dim_instead_of_reading_past_the_buffer() {
        let narrow = 3 * 2 * 14 * 14;
        assert!(narrow < PATCH_DIM, "this test must model an UNDER-run");
        let pixels = vec![0.0f32; 64 * narrow];
        let err = check_pixel_len(&pixels, 64, 64, 6400)
            .unwrap_err()
            .to_string();
        assert!(err.contains("patch_size"), "{err}");
        assert!(err.contains(&format!("{}", 64 * narrow)), "{err}");
    }

    /// The other direction: a wider patch_dim fits the host `Vec` but overruns
    /// the fixed-size device `buf_f32` on the H2D copy.
    #[test]
    fn rejects_wider_patch_dim() {
        let wide = 3 * 2 * 32 * 32;
        assert!(wide > PATCH_DIM);
        let pixels = vec![0.0f32; 4 * wide];
        assert!(check_pixel_len(&pixels, 4, 4, 6400).is_err());
    }

    /// A patch count large enough to wrap the multiply must be an error, not a
    /// wrapped-around "expected length" that some buffer accidentally matches.
    #[test]
    fn rejects_patch_count_that_overflows() {
        let err = check_pixel_len(&[], usize::MAX / 2, usize::MAX / 2, 6400)
            .unwrap_err()
            .to_string();
        assert!(err.contains("overflow"), "{err}");
    }

    /// The bound that did not exist. A checkpoint can declare a `vision_config`
    /// that yields exactly `PATCH_DIM` floats per patch on a much finer grid —
    /// `patch_size: 4, temporal_patch_size: 32` gives 3×32×4×4 = 1536 — so the
    /// pixel buffer is CONSISTENT and the width check passes, while the patch
    /// count runs far past the `p_max` rows every device buffer was sized for.
    #[test]
    fn rejects_a_consistent_buffer_with_too_many_patches() {
        assert_eq!(
            3 * 32 * 4 * 4,
            PATCH_DIM,
            "the hostile geometry is width-consistent"
        );
        let p_max = 6400;
        // Exactly p_max rows is the last admissible image.
        assert!(check_pixel_len(&vec![0.0f32; p_max * PATCH_DIM], p_max, p_max, p_max).is_ok());
        // One row more is refused rather than DMA'd past buf_f32.
        let over = p_max + 1;
        let err = check_pixel_len(&vec![0.0f32; over * PATCH_DIM], over, over, p_max)
            .unwrap_err()
            .to_string();
        assert!(err.contains("6400 rows"), "{err}");
    }

    /// The batched path places each image at its own row offset, so a batch
    /// whose images each fit can still overrun once packed. The bound is the
    /// END row, not the per-image count.
    #[test]
    fn rejects_a_small_image_placed_past_the_end() {
        let p_max = 6400;
        let pixels = vec![0.0f32; 8 * PATCH_DIM];
        // 8 patches is tiny, but landing at row 6399 ends at 6407.
        let err = check_pixel_len(&pixels, 8, 6399 + 8, p_max)
            .unwrap_err()
            .to_string();
        assert!(err.contains("6407"), "{err}");
        // The same image at a row that leaves space is fine.
        assert!(check_pixel_len(&pixels, 8, 100 + 8, p_max).is_ok());
    }
}
