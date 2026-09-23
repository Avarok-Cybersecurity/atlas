// SPDX-License-Identifier: AGPL-3.0-only

//! The post-block tail: 2×2 strided-conv downsample → merger MLP.
//!
//! ```text
//! h = h.view(-1, 2, 2, C).permute(0, 3, 1, 2)      # (blocks, C, 2, 2)
//! h = downsample(h).view(-1, out)                   # Conv2d(C→out, k=2, s=2)
//! h = proj(h)                                       # Linear(out, out, bias=False)
//! h = GELU(post_projection_norm(h))                 # LayerNorm(out) — NOT RMSNorm
//! out = down_proj(silu(clamp(gate)) * clamp(up))    # SwiGLU through 10240
//! ```
//!
//! The `view(-1, 2, 2, C)` is only free because the token stream is block-major
//! (see `rope::block_major_position_ids`). Reorder the tokens anywhere upstream
//! and this reshape silently scrambles each 2×2 block's four patches.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::GlmVit;

/// `nn.LayerNorm` default eps. The checkpoint's `rms_norm_eps` (1e-5) governs
/// every RMSNorm in the tower; this LayerNorm is a plain torch module and takes
/// torch's default, which happens to be the same number — stated separately so
/// a checkpoint that moves one does not silently move the other.
const LAYERNORM_EPS: f32 = 1e-5;

impl GlmVit {
    /// Consume `merged_p * 4` rows of `buf_h1` and write `merged_p` rows of
    /// `out` (`[merged_p, out_hidden_size]`).
    pub(super) fn downsample_and_merge(
        &self,
        merged_p: usize,
        out: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let s = self.scratch();
        let mp = merged_p as u32;
        let hidden = self.hidden_size as u32;
        let out_h = self.out_hidden_size as u32;
        let pi = self.projection_intermediate_size as u32;
        // The conv's K: one 2×2 block of `hidden` channels, flattened in the
        // weight's own (in_channel, kh, kw) order.
        let conv_k = hidden * 4;

        // im2col: buf_h1 [4*mp, hidden] → buf_merge_a [mp, 4*hidden].
        let n_elems = mp * conv_k;
        KernelLaunch::new(gpu, self.k_im2col)
            .grid([div_ceil(n_elems, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.buf_h1)
            .arg_ptr(s.buf_merge_a)
            .arg_u32(mp)
            .arg_u32(hidden)
            .launch(stream)?;

        // downsample conv == GEMM against the flattened weight, bias included.
        self.gemm(
            gpu,
            s.buf_merge_a,
            self.merger.downsample_w,
            Some(self.merger.downsample_b),
            s.buf_merge_b,
            mp,
            out_h,
            conv_k,
            stream,
        )?;

        // merger.proj — bias=False, like every Linear in the merger.
        self.gemm(
            gpu,
            s.buf_merge_b,
            self.merger.proj_w,
            None,
            s.buf_merge_a,
            mp,
            out_h,
            out_h,
            stream,
        )?;
        self.layernorm(
            gpu,
            s.buf_merge_a,
            self.merger.norm_w,
            self.merger.norm_b,
            mp,
            out_h,
            LAYERNORM_EPS,
            stream,
        )?;
        // Exact (erf) GELU: `act1 = nn.GELU()`, approximate='none'.
        KernelLaunch::new(gpu, self.k_gelu)
            .grid([div_ceil(mp * out_h, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.buf_merge_a)
            .arg_u32(mp * out_h)
            .launch(stream)?;

        self.gemm(
            gpu,
            s.buf_merge_a,
            self.merger.gate_w,
            None,
            s.buf_merge_g,
            mp,
            pi,
            out_h,
            stream,
        )?;
        self.gemm(
            gpu,
            s.buf_merge_a,
            self.merger.up_w,
            None,
            s.buf_merge_u,
            mp,
            pi,
            out_h,
            stream,
        )?;
        self.swiglu(
            gpu,
            s.buf_merge_g,
            s.buf_merge_u,
            s.buf_merge_g,
            mp * pi,
            stream,
        )?;
        self.gemm(
            gpu,
            s.buf_merge_g,
            self.merger.down_w,
            None,
            out,
            mp,
            out_h,
            pi,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    /// CPU reference for `glm_vit_im2col_2x2`, mirroring the kernel's index
    /// arithmetic: `dst[b, c*4 + t] = src[b*4 + t, c]`.
    fn im2col_ref(src: &[f32], merged_p: usize, c_dim: usize) -> Vec<f32> {
        let mut dst = vec![0.0f32; merged_p * 4 * c_dim];
        for b in 0..merged_p {
            for t in 0..4 {
                for c in 0..c_dim {
                    dst[b * 4 * c_dim + c * 4 + t] = src[(b * 4 + t) * c_dim + c];
                }
            }
        }
        dst
    }

    /// The layout claim in one assertion: a 2×2 block's four tokens land
    /// CONTIGUOUSLY within each channel's slot, in (kh, kw) order — which is
    /// what makes the conv weight's `(in_c, kh, kw)` flattening a plain GEMM.
    #[test]
    fn im2col_groups_by_channel_then_kernel_position() {
        let c_dim = 3;
        // Token t of block 0 carries the value 10*t + c.
        let src: Vec<f32> = (0..4)
            .flat_map(|t| (0..c_dim).map(move |c| (10 * t + c) as f32))
            .collect();
        let dst = im2col_ref(&src, 1, c_dim);
        // Channel 0's four kernel positions: tokens 0..4, channel 0.
        assert_eq!(&dst[0..4], &[0.0, 10.0, 20.0, 30.0]);
        // Channel 1's four.
        assert_eq!(&dst[4..8], &[1.0, 11.0, 21.0, 31.0]);
        assert_eq!(&dst[8..12], &[2.0, 12.0, 22.0, 32.0]);
    }

    /// Every input element appears exactly once — an im2col for a kernel equal
    /// to its stride is a permutation, not a gather with overlap.
    #[test]
    fn im2col_is_a_permutation() {
        let (mp, c_dim) = (5usize, 7usize);
        let src: Vec<f32> = (0..mp * 4 * c_dim).map(|i| i as f32).collect();
        let mut dst = im2col_ref(&src, mp, c_dim);
        assert_eq!(dst.len(), src.len());
        dst.sort_by(f32::total_cmp);
        assert_eq!(dst, src);
    }
}
