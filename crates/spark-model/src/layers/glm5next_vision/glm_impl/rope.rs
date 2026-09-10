// SPDX-License-Identifier: AGPL-3.0-only

//! 2D axial rotary tables. GLM-5.3's tower has NO learned position embedding
//! (`glm4v.py:107-110` sets `pos_embedding = False` for `glm5_next_vision`, and
//! the checkpoint carries no `embeddings.position_embedding.*`), so this is the
//! tower's only positional signal.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::Glm5NextVisionEncoder;
use super::f32_to_bf16_bits;

/// Fill `cos`/`sin` for one image's `[row(16); col(16); row(16); col(16)]`
/// axial table, in RASTER patch order.
///
/// Pure and host-side so the layout can be pinned by a test. `hd` is head_dim
/// (64) and `inv` the 16 frequencies; `out` is `p * hd` elements of each.
fn fill_rope_tables(
    grid_h: usize,
    grid_w: usize,
    hd: usize,
    inv: &[f32],
    cos: &mut [u16],
    sin: &mut [u16],
) {
    let half = hd / 2;
    let inv_n = inv.len();
    debug_assert_eq!(inv_n * 2, half, "the axial table fills exactly head_dim/2");
    for gh in 0..grid_h {
        for gw in 0..grid_w {
            // RASTER, not merge-major. Patchify (vision_preprocess.rs:299),
            // this table, and vision_spatial_merge's gather are all raster and
            // must stay that way together; ViT attention is full non-causal so
            // the only thing that must line up is these three with each other.
            let p_idx = gh * grid_w + gw;
            let off = p_idx * hd;
            for (k, &f) in inv.iter().enumerate() {
                let (rs, rc) = (gh as f32 * f).sin_cos();
                let (cs, cc) = (gw as f32 * f).sin_cos();
                cos[off + k] = f32_to_bf16_bits(rc);
                sin[off + k] = f32_to_bf16_bits(rs);
                cos[off + inv_n + k] = f32_to_bf16_bits(cc);
                sin[off + inv_n + k] = f32_to_bf16_bits(cs);
                // The second half duplicates the first: NEOX pairs (d, d+half),
                // so the rotation of element d+half needs the same angle as d.
                cos[off + half + k] = f32_to_bf16_bits(rc);
                sin[off + half + k] = f32_to_bf16_bits(rs);
                cos[off + half + inv_n + k] = f32_to_bf16_bits(cc);
                sin[off + half + inv_n + k] = f32_to_bf16_bits(cs);
            }
        }
    }
}

impl Glm5NextVisionEncoder {
    /// Zero-offset shim used by the oversized fallback.
    pub(super) fn build_rope_cossin(
        &self,
        grid_h: usize,
        grid_w: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.build_rope_cossin_into(
            grid_h,
            grid_w,
            self.scratch().buf_rope_cos,
            self.scratch().buf_rope_sin,
            gpu,
            stream,
        )
    }

    /// Build and upload one image's rotary tables at `cos_dst`/`sin_dst`.
    pub(super) fn build_rope_cossin_into(
        &self,
        grid_h: usize,
        grid_w: usize,
        cos_dst: DevicePtr,
        sin_dst: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let p = grid_h * grid_w;
        // THE capacity guard for an oversized image, and it lives here because
        // this is the first device write of the encode. The Qwen tower carried
        // the same `ensure!` inside `resample_pos_embed_into`; GLM deletes that
        // function outright (no learned position embedding), so the bound had
        // to move rather than disappear. Without it an oversized image
        // surfaced as `cuMemcpyHtoDAsync_v2 failed: status 1` from inside the
        // scheduler, naming neither vision nor a size.
        anyhow::ensure!(
            p <= self.p_max,
            "glm vision: image is {grid_h}x{grid_w} patches ({p} total) but this encoder holds \
             {} — raise --vision-max-pixels only if the encoder was built for it, since its \
             buffers are sized from that same bound and the ViT score matrix is O(patches^2)",
            self.p_max
        );
        let hd = self.head_dim;
        let mut cos_bf16 = vec![0u16; p * hd];
        let mut sin_bf16 = vec![0u16; p * hd];
        fill_rope_tables(
            grid_h,
            grid_w,
            hd,
            &self.rope_inv_freq,
            &mut cos_bf16,
            &mut sin_bf16,
        );
        // SAFETY (both): each Vec is live for the call, every element was
        // written or zeroed by `vec!`, `u16` has no invalid bit patterns and
        // `u8` has alignment 1, and each byte length comes from that same Vec's
        // `len()`, so neither read-only view can leave its allocation.
        let cos_b: &[u8] = unsafe {
            std::slice::from_raw_parts(cos_bf16.as_ptr() as *const u8, cos_bf16.len() * 2)
        };
        let sin_b: &[u8] = unsafe {
            std::slice::from_raw_parts(sin_bf16.as_ptr() as *const u8, sin_bf16.len() * 2)
        };
        gpu.copy_h2d_async(cos_b, cos_dst, stream)?;
        gpu.copy_h2d_async(sin_b, sin_dst, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::fill_rope_tables;

    fn bf16(v: u16) -> f32 {
        f32::from_bits((v as u32) << 16)
    }

    /// C-T12. Pins BOTH the raster `(h, w)` assignment and the
    /// `[row; col; row; col]` table layout against a hand-computed 4×4 grid.
    ///
    /// A merge-major patch order would put patch index 1 at `(0,1)` of the
    /// FIRST 2×2 block in a different sequence, and the resulting encoder is
    /// fluent and wrong with nothing logged — so the ordering is asserted, not
    /// assumed.
    #[test]
    fn raster_order_and_axial_layout_on_a_4x4_grid() {
        let hd = 8usize; // toy head_dim: half = 4, inv_n = 2
        let inv = [1.0f32, 0.5];
        let (gh, gw) = (4usize, 4usize);
        let p = gh * gw;
        let mut cos = vec![0u16; p * hd];
        let mut sin = vec![0u16; p * hd];
        fill_rope_tables(gh, gw, hd, &inv, &mut cos, &mut sin);

        // Patch 6 is raster (row 1, col 2), NOT merge-major.
        let (row, col) = (1.0f32, 2.0f32);
        let off = 6 * hd;
        let approx = |a: f32, b: f32| (a - b).abs() < 5e-3;
        // [0..2) row freqs, [2..4) col freqs, then the halves repeat.
        assert!(approx(bf16(cos[off]), (row * inv[0]).cos()));
        assert!(approx(bf16(sin[off + 1]), (row * inv[1]).sin()));
        assert!(approx(bf16(cos[off + 2]), (col * inv[0]).cos()));
        assert!(approx(bf16(sin[off + 3]), (col * inv[1]).sin()));
        for d in 0..hd / 2 {
            assert_eq!(cos[off + d], cos[off + hd / 2 + d], "d={d}");
            assert_eq!(sin[off + d], sin[off + hd / 2 + d], "d={d}");
        }
        // Patch 0 is the grid origin: every angle is 0, so cos=1, sin=0.
        for d in 0..hd {
            assert!(approx(bf16(cos[d]), 1.0));
            assert!(approx(bf16(sin[d]), 0.0));
        }
    }

    /// The last patch of the grid must be `(grid_h-1, grid_w-1)` — the check
    /// that catches a transposed loop, which a square test grid would hide.
    #[test]
    fn a_non_square_grid_indexes_row_major() {
        let hd = 8usize;
        let inv = [1.0f32, 0.5];
        let (gh, gw) = (2usize, 5usize);
        let mut cos = vec![0u16; gh * gw * hd];
        let mut sin = vec![0u16; gh * gw * hd];
        fill_rope_tables(gh, gw, hd, &inv, &mut cos, &mut sin);
        // Patch 5 is (1, 0): row angle 1.0, col angle 0 → col cos is exactly 1.
        let off = 5 * hd;
        assert!((bf16(cos[off]) - 1.0f32.cos()).abs() < 5e-3);
        assert!((bf16(cos[off + 2]) - 1.0).abs() < 1e-6);
    }
}
