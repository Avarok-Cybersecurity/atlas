// SPDX-License-Identifier: AGPL-3.0-only

//! Patch-embed step: f32 pixels → BF16 → patch_embed GEMM → +pos_embed.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::{PATCH_DIM, VisionEncoder};

/// Check a host pixel buffer against the geometry the encoder was built for
/// before its bytes are reinterpreted and DMA'd.
///
/// `pixels` is sized by the CPU preprocessor from the checkpoint's
/// `vision_config` (`3 × temporal_patch_size × patch_size²` per patch), while
/// the encoder's device buffer and GEMM are fixed at [`PATCH_DIM`]. A
/// checkpoint declaring e.g. `patch_size: 14` yields 1176 floats per patch, so
/// the old `p * PATCH_DIM * 4` byte length ran 360 floats per patch PAST the
/// end of the `Vec` — an out-of-bounds read that then went to the GPU. A
/// larger `patch_size` overruns in the other direction, over the fixed
/// `buf_f32` allocation on the device.
fn check_pixel_len(pixels: &[f32], patches: usize) -> Result<()> {
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
        check_pixel_len(pixels, p)?;
        let n_f32 = pixels.len();
        // SAFETY: `pixels` is a live `&[f32]`; the byte length is taken from
        // that same slice (`len() * 4`), so the view never leaves the
        // allocation. f32 has no padding or invalid bit patterns, and u8 has
        // alignment 1, so every byte of it is a valid `u8`. The view is
        // read-only and dies at the end of this function, before `pixels`.
        let f32_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u8, n_f32 * 4) };
        gpu.copy_h2d_async(f32_bytes, self.buf_f32, stream)?;
        // f32 → bf16 (result in buf_wide[0..p*PATCH_DIM])
        KernelLaunch::new(gpu, self.k_f32_bf16)
            .grid([div_ceil(n_f32 as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_f32)
            .arg_ptr(self.buf_wide)
            .arg_u32(n_f32 as u32)
            .launch(stream)?;
        // patch_embed GEMM: buf_wide[p,K] @ patch_embed_w[1152,K]^T + b → buf_h1[p,1152]
        self.vit_gemm_bias(
            gpu,
            self.buf_wide,
            self.patch_embed_w,
            self.patch_embed_b,
            self.buf_h1,
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
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_pos_resampled)
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
        // Upload each image's pixels into its row slice of buf_f32. The length
        // check is what keeps the destination in bounds too: `buf_f32` holds
        // `p_max × PATCH_DIM` floats and callers cap Σp ≤ p_max, so an image
        // whose host buffer is WIDER than PATCH_DIM per patch would run past
        // the device allocation as surely as a narrower one runs past the Vec.
        for (i, (pixels, gh, gw)) in images.iter().enumerate() {
            check_pixel_len(pixels, gh * gw)?;
            // SAFETY: `pixels` is a live `&[f32]` and the byte length is
            // derived from that same slice, so the view stays inside its
            // allocation. `f32` has no invalid bit patterns and `u8` has
            // alignment 1, so the reinterpretation is valid for every byte.
            let f32_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4)
            };
            gpu.copy_h2d_async(
                f32_bytes,
                self.buf_f32.offset(p_off[i] * PATCH_DIM * 4),
                stream,
            )?;
        }
        let n_f32 = p_total * PATCH_DIM;
        // f32 → bf16 (result in buf_wide[0..p_total*PATCH_DIM])
        KernelLaunch::new(gpu, self.k_f32_bf16)
            .grid([div_ceil(n_f32 as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_f32)
            .arg_ptr(self.buf_wide)
            .arg_u32(n_f32 as u32)
            .launch(stream)?;
        // patch_embed GEMM over M=p_total → buf_h1
        self.vit_gemm_bias(
            gpu,
            self.buf_wide,
            self.patch_embed_w,
            self.patch_embed_b,
            self.buf_h1,
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
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_pos_resampled)
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
        assert!(check_pixel_len(&pixels, 64).is_ok());
        // Zero patches (an image that scaled to nothing) is consistent, not a
        // slice-length hazard.
        assert!(check_pixel_len(&[], 0).is_ok());
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
        let err = check_pixel_len(&pixels, 64).unwrap_err().to_string();
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
        assert!(check_pixel_len(&pixels, 4).is_err());
    }

    /// A patch count large enough to wrap the multiply must be an error, not a
    /// wrapped-around "expected length" that some buffer accidentally matches.
    #[test]
    fn rejects_patch_count_that_overflows() {
        let err = check_pixel_len(&[], usize::MAX / 2)
            .unwrap_err()
            .to_string();
        assert!(err.contains("overflow"), "{err}");
    }
}
