// SPDX-License-Identifier: AGPL-3.0-only

//! Positional embedding helpers: bilinear interpolation of learned
//! pos_embed grid + per-patch 2D rotary cos/sin builder.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::super::VisionEncoder;
use super::f32_to_bf16_bits;

impl VisionEncoder {
    /// Bilinear interpolate the learned pos_embed grid
    /// `[num_grid_per_side × num_grid_per_side, hidden_size]` down to
    /// `[grid_h × grid_w, hidden_size]` in row-major order, convert to
    /// BF16, upload to `buf_pos_resampled`.  Mirrors HF's
    /// `fast_pos_embed_interpolate` (same index/weight formulas).
    pub(super) fn resample_pos_embed(
        &self,
        grid_h: usize,
        grid_w: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size;
        let n = self.num_grid_per_side;
        let p = grid_h * grid_w;
        let mut out_bf16 = vec![0u16; p * h];

        // HF uses torch.linspace(0, n-1, grid_dim), so endpoints hit the
        // grid corners exactly.
        let denom_h = (grid_h.max(2) - 1) as f32;
        let denom_w = (grid_w.max(2) - 1) as f32;
        for gh in 0..grid_h {
            let fy = if grid_h <= 1 {
                0.0
            } else {
                gh as f32 * (n - 1) as f32 / denom_h
            };
            let y_f = fy.floor() as i32;
            let y_c = (y_f + 1).min(n as i32 - 1);
            let dy = fy - y_f as f32;
            let y_f_u = y_f.clamp(0, n as i32 - 1) as usize;
            let y_c_u = y_c.clamp(0, n as i32 - 1) as usize;
            for gw in 0..grid_w {
                let fx = if grid_w <= 1 {
                    0.0
                } else {
                    gw as f32 * (n - 1) as f32 / denom_w
                };
                let x_f = fx.floor() as i32;
                let x_c = (x_f + 1).min(n as i32 - 1);
                let dx = fx - x_f as f32;
                let x_f_u = x_f.clamp(0, n as i32 - 1) as usize;
                let x_c_u = x_c.clamp(0, n as i32 - 1) as usize;

                let w00 = (1.0 - dy) * (1.0 - dx);
                let w01 = (1.0 - dy) * dx;
                let w10 = dy * (1.0 - dx);
                let w11 = dy * dx;

                let i00 = (y_f_u * n + x_f_u) * h;
                let i01 = (y_f_u * n + x_c_u) * h;
                let i10 = (y_c_u * n + x_f_u) * h;
                let i11 = (y_c_u * n + x_c_u) * h;
                let out_off = (gh * grid_w + gw) * h;

                for k in 0..h {
                    let v = w00 * self.pos_embed_host_f32[i00 + k]
                        + w01 * self.pos_embed_host_f32[i01 + k]
                        + w10 * self.pos_embed_host_f32[i10 + k]
                        + w11 * self.pos_embed_host_f32[i11 + k];
                    out_bf16[out_off + k] = f32_to_bf16_bits(v);
                }
            }
        }
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(out_bf16.as_ptr() as *const u8, out_bf16.len() * 2)
        };
        gpu.copy_h2d_async(bytes, self.buf_pos_resampled, stream)
    }

    /// Build per-patch 2D rotary cos/sin in row-major patch order
    /// matching HF Qwen3-VL's `rot_pos_emb(grid_thw)`.  For each patch i
    /// at spatial (row, col) the cos/sin row is structured as
    /// `[row_freq; col_freq; row_freq; col_freq]` of length head_dim,
    /// where `row_freq[k] = cos/sin(row * inv_freq[k])` and
    /// `col_freq[k] = cos/sin(col * inv_freq[k])`. Upload BF16.
    pub(super) fn build_rope_cossin(
        &self,
        grid_h: usize,
        grid_w: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let p = grid_h * grid_w;
        let hd = self.head_dim;
        let half = hd / 2; // 36
        let inv_n = self.rope_inv_freq.len(); // 18 = hd/4
        debug_assert_eq!(inv_n * 2, half);

        let mut cos_bf16 = vec![0u16; p * hd];
        let mut sin_bf16 = vec![0u16; p * hd];

        // A/B toggle: when ATLAS_VISION_ROPE=0 we upload cos=1, sin=0 to
        // make the kernel behave as identity (pre-RoPE). Lets the sweep
        // test pos_embed interpolation and RoPE as two independent bugs.
        let rope_on = std::env::var("ATLAS_VISION_ROPE")
            .map(|v| v != "0")
            .unwrap_or(true);
        let one_bf16 = f32_to_bf16_bits(1.0);
        let zero_bf16 = f32_to_bf16_bits(0.0);
        for gh in 0..grid_h {
            for gw in 0..grid_w {
                let p_idx = gh * grid_w + gw;
                let row = gh as f32;
                let col = gw as f32;
                let off = p_idx * hd;
                for k in 0..inv_n {
                    if !rope_on {
                        for d in [k, inv_n + k, half + k, half + inv_n + k] {
                            cos_bf16[off + d] = one_bf16;
                            sin_bf16[off + d] = zero_bf16;
                        }
                        continue;
                    }
                    let rf = row * self.rope_inv_freq[k];
                    let cf = col * self.rope_inv_freq[k];
                    let (rs, rc) = (rf.sin(), rf.cos());
                    let (cs, cc) = (cf.sin(), cf.cos());
                    // first half [0..inv_n): row freq; [inv_n..2*inv_n): col freq
                    cos_bf16[off + k] = f32_to_bf16_bits(rc);
                    sin_bf16[off + k] = f32_to_bf16_bits(rs);
                    cos_bf16[off + inv_n + k] = f32_to_bf16_bits(cc);
                    sin_bf16[off + inv_n + k] = f32_to_bf16_bits(cs);
                    // second half duplicates first (HF cats emb with itself).
                    cos_bf16[off + half + k] = f32_to_bf16_bits(rc);
                    sin_bf16[off + half + k] = f32_to_bf16_bits(rs);
                    cos_bf16[off + half + inv_n + k] = f32_to_bf16_bits(cc);
                    sin_bf16[off + half + inv_n + k] = f32_to_bf16_bits(cs);
                }
            }
        }
        let cos_b: &[u8] = unsafe {
            std::slice::from_raw_parts(cos_bf16.as_ptr() as *const u8, cos_bf16.len() * 2)
        };
        let sin_b: &[u8] = unsafe {
            std::slice::from_raw_parts(sin_bf16.as_ptr() as *const u8, sin_bf16.len() * 2)
        };
        gpu.copy_h2d_async(cos_b, self.buf_rope_cos, stream)?;
        gpu.copy_h2d_async(sin_b, self.buf_rope_sin, stream)
    }
}
