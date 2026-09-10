// SPDX-License-Identifier: AGPL-3.0-only

//! Patch embed: host f32 pixels → BF16 → one GEMM.
//!
//! GLM's `patch_embed.proj` is a conv3d `[1024, 3, 2, 14, 14]` with `t = 2` and
//! a 14×14 kernel applied at stride 14, so it is a plain GEMM over
//! `GLM_PATCH_DIM = 1176`.
//!
//! C-T10: the weight needs NO permutation. `view(1024, 1176)` flattens
//! `(c, t, kh, kw)` as `c*392 + t*196 + kh*14 + kw`, which is exactly the order
//! the host patch loop writes (`vision_preprocess.rs:307-310`,
//! `off = c*(tp*ps*ps) + t*(ps*ps) + py*ps + px`). A plain `[1024, 1176]` fed to
//! `glm_gemm_bias` — which takes `B[n,k]` row-major and transposes internally —
//! is the whole job.
//!
//! There is NO pos_embed add here and NO post_conv_layernorm: GLM-4.1V had
//! both, 5.3 dropped both.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::{GLM_PATCH_DIM, Glm5NextVisionEncoder};
use crate::layers::vision_encoder::enc_impl::patch_embed::check_pixel_len;

impl Glm5NextVisionEncoder {
    /// Single-image patch embed at row 0 (the oversized fallback's path).
    pub(super) fn patch_embed(
        &self,
        pixels: &[f32],
        p: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.patch_embed_batched(&[(pixels, 0, 0)], &[0], p, Some(&[p]), gpu, stream)
    }

    /// Batched patch embed over N images packed at `p_off[i]` rows.
    ///
    /// `p_i` overrides the per-image patch count derived from the grids; the
    /// single-image shim passes it because it has no grid to derive from.
    pub(super) fn patch_embed_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        p_off: &[usize],
        p_total: usize,
        p_i: Option<&[usize]>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        for (i, (pixels, gh, gw)) in images.iter().enumerate() {
            let p = match p_i {
                Some(counts) => counts[i],
                None => gh * gw,
            };
            // Bounds BOTH ends: the host Vec's width (a checkpoint whose
            // patch/temporal sizes disagree with GLM_PATCH_DIM makes a buffer
            // of a different length) and the device row this image ends at.
            let end_row = p_off[i]
                .checked_add(p)
                .ok_or_else(|| anyhow::anyhow!("glm vision: patch row offset overflows"))?;
            check_pixel_len(pixels, p, end_row, self.p_max, GLM_PATCH_DIM)?;
            // SAFETY: `pixels` is a live `&[f32]` and the byte length is
            // derived from that same slice, so the view stays inside its
            // allocation. `f32` has no invalid bit patterns and `u8` has
            // alignment 1, so every byte of the reinterpretation is valid. The
            // view is read-only and dies before `pixels`.
            let f32_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4)
            };
            gpu.copy_h2d_async(
                f32_bytes,
                self.scratch().buf_f32.offset(p_off[i] * GLM_PATCH_DIM * 4),
                stream,
            )?;
        }
        let n_f32 = p_total * GLM_PATCH_DIM;
        KernelLaunch::new(gpu, self.k_f32_bf16)
            .grid([div_ceil(n_f32 as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_f32)
            .arg_ptr(self.scratch().buf_wide)
            .arg_u32(n_f32 as u32)
            .launch(stream)?;
        self.glm_gemm_bias(
            gpu,
            self.scratch().buf_wide,
            self.patch_embed_w,
            self.patch_embed_b,
            self.scratch().buf_h1,
            p_total as u32,
            self.hidden_size as u32,
            GLM_PATCH_DIM as u32,
            stream,
        )
    }
}
