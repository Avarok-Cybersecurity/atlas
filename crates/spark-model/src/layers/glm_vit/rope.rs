// SPDX-License-Identifier: AGPL-3.0-only

//! Block-major patch ids and the pure 2D axial RoPE tables built from them.
//!
//! Both halves of this file are pure CPU functions with no device state,
//! because both encode an ORDER that nothing downstream can check. The 2×2
//! downsample later reshapes four CONSECUTIVE tokens into one spatial block;
//! that is only correct while the token stream is block-major, and a raster
//! ordering would produce a perfectly plausible — and wrong — image embedding.
//! So the order is stated once, here, and tested.

use super::GlmVit;

/// Per-token `(h_id, w_id)` in the reference's block-major order.
///
/// `get_vision_position_ids` builds `meshgrid(arange(h), arange(w))`, reshapes
/// to `(h/m, m, w/m, m)`, `transpose(1, 2)` and flattens — i.e. the outer loop
/// walks 2×2 merge blocks in raster order and the inner loop walks the four
/// positions inside each block. The image processor's `patchify()` uses the
/// identical nesting (`permute(0, 2, 5, 3, 6, 1, 4, 7)`), which is what lets
/// the tower's `view(-1, 2, 2, C)` be a free reshape.
pub fn block_major_position_ids(grid_h: usize, grid_w: usize, merge: usize) -> Vec<(u32, u32)> {
    let m = merge.max(1);
    let (bh, bw) = (grid_h / m, grid_w / m);
    let mut ids = Vec::with_capacity(bh * bw * m * m);
    for a in 0..bh {
        for c in 0..bw {
            for b in 0..m {
                for d in 0..m {
                    ids.push(((a * m + b) as u32, (c * m + d) as u32));
                }
            }
        }
    }
    ids
}

/// `(cos, sin)` tables of `[num_tokens, head_dim]` f32, in the layout the
/// rotate-half kernel reads.
///
/// `inv_freq` has `spatial_dim/2 = head_dim/4` entries. The per-token frequency
/// row is `cat([h*inv_freq, w*inv_freq])` (length `head_dim/2`) duplicated once
/// to fill `head_dim` — the reference's `cat([freq_hw, freq_hw], -1)`. There is
/// no partial-rotary split: every dimension of the 64-wide head is rotated.
pub fn build_axial_rope_tables(
    ids: &[(u32, u32)],
    inv_freq: &[f32],
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let n_freq = inv_freq.len();
    let mut cos = vec![0.0f32; ids.len() * head_dim];
    let mut sin = vec![0.0f32; ids.len() * head_dim];
    for (t, &(h_id, w_id)) in ids.iter().enumerate() {
        for d in 0..half {
            // `freq_hw[d]`: the first half indexes the h axis, the second the w.
            let angle = if d < n_freq {
                h_id as f32 * inv_freq[d]
            } else {
                w_id as f32 * inv_freq[d - n_freq]
            };
            let (s, c) = angle.sin_cos();
            cos[t * head_dim + d] = c;
            cos[t * head_dim + half + d] = c;
            sin[t * head_dim + d] = s;
            sin[t * head_dim + half + d] = s;
        }
    }
    (cos, sin)
}

impl GlmVit {
    /// Build this image's rope tables on the host and upload them as BF16.
    pub(super) fn upload_rope_tables(
        &self,
        grid_h: usize,
        grid_w: usize,
        cos_dst: spark_runtime::gpu::DevicePtr,
        sin_dst: spark_runtime::gpu::DevicePtr,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        stream: u64,
    ) -> anyhow::Result<()> {
        let ids = block_major_position_ids(grid_h, grid_w, self.spatial_merge_size);
        let (cos, sin) = build_axial_rope_tables(&ids, &self.rope_inv_freq, self.head_dim);
        gpu.copy_h2d_async(&to_bf16_bytes(&cos), cos_dst, stream)?;
        gpu.copy_h2d_async(&to_bf16_bytes(&sin), sin_dst, stream)?;
        Ok(())
    }
}

/// Round-to-nearest-even f32 → BF16, little-endian. The tables are consumed by
/// a BF16 kernel, and truncation here would bias every angle one way.
pub(super) fn to_bf16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        let bits = x.to_bits();
        let lsb = (bits >> 16) & 1;
        let rounded = bits.wrapping_add(0x7fff + lsb);
        out.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The nesting the 2×2 downsample depends on: four CONSECUTIVE tokens are
    /// one spatial block, and the blocks themselves walk in raster order.
    #[test]
    fn position_ids_are_block_major_not_raster() {
        let ids = block_major_position_ids(4, 4, 2);
        assert_eq!(ids.len(), 16);
        // Block (0,0) occupies tokens 0..4.
        assert_eq!(&ids[0..4], &[(0, 0), (0, 1), (1, 0), (1, 1)]);
        // Block (0,1) — the NEXT block along w — occupies 4..8.
        assert_eq!(&ids[4..8], &[(0, 2), (0, 3), (1, 2), (1, 3)]);
        // Block (1,0) starts the second row of blocks.
        assert_eq!(&ids[8..12], &[(2, 0), (2, 1), (3, 0), (3, 1)]);
        // Raster order would have put (0,2) at index 2; it is at index 4.
        assert_ne!(ids[2], (0, 2));
    }

    /// Non-square grids are the case that catches an h/w transposition, which
    /// on a square test image is invisible.
    #[test]
    fn position_ids_handle_a_non_square_grid() {
        // The golden's third image: 26x46 unmerged patches → 13x23 blocks.
        let ids = block_major_position_ids(26, 46, 2);
        assert_eq!(ids.len(), 26 * 46);
        assert_eq!(ids[0], (0, 0));
        // Last token of the first block row: block (0, 22), inner (1, 1).
        assert_eq!(ids[23 * 4 - 1], (1, 45));
        assert_eq!(*ids.last().unwrap(), (25, 45));
        // Every id is in range, and each appears exactly once.
        let mut seen: Vec<(u32, u32)> = ids.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), ids.len(), "block-major must be a permutation");
    }

    /// The table layout the rotate-half kernel assumes: the second half of each
    /// row REPEATS the first, and inside the first half the h frequencies come
    /// before the w ones.
    #[test]
    fn axial_tables_duplicate_the_half_row() {
        let head_dim = 8; // spatial_dim 4 → 2 inv_freq entries
        let inv_freq = vec![1.0f32, 0.5];
        let ids = [(3u32, 5u32)];
        let (cos, sin) = build_axial_rope_tables(&ids, &inv_freq, head_dim);
        let expect: [f32; 4] = [3.0 * 1.0, 3.0 * 0.5, 5.0 * 1.0, 5.0 * 0.5];
        for (d, &angle) in expect.iter().enumerate() {
            assert!((cos[d] - angle.cos()).abs() < 1e-6, "cos[{d}]");
            assert!((sin[d] - angle.sin()).abs() < 1e-6, "sin[{d}]");
            // Duplicated into the upper half.
            assert_eq!(cos[d], cos[head_dim / 2 + d]);
            assert_eq!(sin[d], sin[head_dim / 2 + d]);
        }
    }

    /// Token 0 sits at (0, 0), so every angle is zero: cos 1, sin 0. A table
    /// that came out otherwise would mean the ids were built off-by-one.
    #[test]
    fn the_origin_token_is_an_identity_rotation() {
        let inv_freq: Vec<f32> = (0..16)
            .map(|k| 10_000f32.powf(-2.0 * k as f32 / 32.0))
            .collect();
        let ids = block_major_position_ids(32, 32, 2);
        let (cos, sin) = build_axial_rope_tables(&ids, &inv_freq, 64);
        for d in 0..64 {
            assert_eq!(cos[d], 1.0);
            assert_eq!(sin[d], 0.0);
        }
    }

    #[test]
    fn bf16_conversion_rounds_to_nearest_even() {
        // 1.0 is exact in BF16.
        assert_eq!(to_bf16_bytes(&[1.0]), 0x3f80u16.to_le_bytes().to_vec());
        // A value exactly between two BF16s rounds to the even mantissa.
        let halfway = f32::from_bits(0x3f80_8000); // 1 + 2^-8, tie
        assert_eq!(to_bf16_bytes(&[halfway]), 0x3f80u16.to_le_bytes().to_vec());
        // Truncation would have kept 0x3f80 here too; this one must round UP.
        let above = f32::from_bits(0x3f80_8001);
        assert_eq!(to_bf16_bytes(&[above]), 0x3f81u16.to_le_bytes().to_vec());
    }
}
