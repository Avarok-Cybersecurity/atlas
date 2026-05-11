// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-side image preprocessing for Qwen3-VL vision inputs.
//!
//! Decodes base64 JPEG/PNG images, resizes to a grid snapped to
//! `patch_size × spatial_merge_size`, normalizes with ImageNet stats,
//! and produces a flat `f32` tensor ready for the GPU vision encoder.

use anyhow::{Context, Result};
use atlas_core::config::VisionConfig;
use image::{DynamicImage, ImageFormat};

/// SigLIP normalization — matches HF's Qwen2VLImageProcessor
/// (`image_mean = image_std = (0.5, 0.5, 0.5)` → pixels mapped to [-1, 1]).
const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Maximum allowed image dimension in pixels (longer side).
const MAX_DIM: u32 = 1280;

/// Decode a base64 data URI or raw base64 string into a `DynamicImage`.
fn decode_image(data_uri: &str) -> Result<DynamicImage> {
    // Strip optional "data:image/<fmt>;base64," prefix.
    let b64 = if let Some(pos) = data_uri.find(",base64,") {
        &data_uri[pos + 8..]
    } else if data_uri.starts_with("data:") {
        // "data:image/jpeg;base64,..."
        data_uri
            .find(',')
            .map(|p| &data_uri[p + 1..])
            .unwrap_or(data_uri)
    } else {
        data_uri
    };

    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64.trim())
        .context("base64 decode failed")?;

    // Probe format from magic bytes.
    let fmt = image::guess_format(&bytes).unwrap_or(ImageFormat::Jpeg);
    image::load_from_memory_with_format(&bytes, fmt).context("image decode failed")
}

/// Compute the target (H, W) so that:
/// - Neither side exceeds `MAX_DIM`.
/// - Both sides are multiples of `grid_unit = patch_size × spatial_merge_size`.
/// - Aspect ratio is preserved (rounded to nearest grid_unit).
fn target_size(orig_h: u32, orig_w: u32, grid_unit: u32) -> (u32, u32) {
    let scale = (MAX_DIM as f32) / (orig_h.max(orig_w) as f32);
    let scale = scale.min(1.0); // never upscale
    let target_h = ((orig_h as f32 * scale / grid_unit as f32).round() as u32).max(1) * grid_unit;
    let target_w = ((orig_w as f32 * scale / grid_unit as f32).round() as u32).max(1) * grid_unit;
    (target_h, target_w)
}

/// Preprocess a single base64-encoded image for the Qwen3-VL encoder.
///
/// Returns:
/// - `pixels`: flat `f32` tensor shaped `[P, C × T × H_p × W_p]` where:
///   - `P = (H/patch_size) × (W/patch_size)` — number of patches
///   - `C = 3` channels, `T = temporal_patch_size` (image duplicated), `H_p = W_p = patch_size`
/// - `grid_h`: number of patches along height
/// - `grid_w`: number of patches along width
pub fn preprocess_image(data_uri: &str, vcfg: &VisionConfig) -> Result<(Vec<f32>, usize, usize)> {
    let img = decode_image(data_uri)?;
    let img = img.to_rgb8();
    let (orig_w, orig_h) = (img.width(), img.height());

    let grid_unit = (vcfg.patch_size * vcfg.spatial_merge_size) as u32;
    let (th, tw) = target_size(orig_h, orig_w, grid_unit);

    // Resize with CatmullRom — closest BICUBIC match in the `image` crate,
    // matching HF's `Qwen2VLImageProcessor` which uses PIL resample=3 (BICUBIC).
    let img = image::imageops::resize(&img, tw, th, image::imageops::FilterType::CatmullRom);

    let ps = vcfg.patch_size;
    let tp = vcfg.temporal_patch_size;
    let grid_h = (th as usize) / ps;
    let grid_w = (tw as usize) / ps;
    let num_patches = grid_h * grid_w;
    // Flattened patch dim: C × temporal_patch_size × patch_size × patch_size
    let patch_dim = 3 * tp * ps * ps;
    let mut pixels = vec![0.0f32; num_patches * patch_dim];

    // Build patches. The temporal dimension is handled by duplicating the image `tp` times.
    // Layout: [P, C, T, Hp, Wp] → stored as [P, C*T*Hp*Wp] in row-major order.
    for ph in 0..grid_h {
        for pw in 0..grid_w {
            let patch_idx = ph * grid_w + pw;
            for c in 0..3usize {
                for t in 0..tp {
                    for py in 0..ps {
                        for px in 0..ps {
                            let pixel_y = ph * ps + py;
                            let pixel_x = pw * ps + px;
                            let raw =
                                img.get_pixel(pixel_x as u32, pixel_y as u32)[c] as f32 / 255.0;
                            let norm = (raw - MEAN[c]) / STD[c];
                            // Offset into patch_dim: c*(T*Hp*Wp) + t*(Hp*Wp) + py*Wp + px
                            let off = c * (tp * ps * ps) + t * (ps * ps) + py * ps + px;
                            pixels[patch_idx * patch_dim + off] = norm;
                        }
                    }
                }
            }
        }
    }

    Ok((pixels, grid_h, grid_w))
}

/// Number of image pad tokens produced per image after the vision
/// encoder's spatial merger. Qwen3-VL / Qwen3.6 fold a 2×2 patch block
/// into a single token, so the embedding stream has
/// `(grid_h / sms) * (grid_w / sms)` rows — not `grid_h * grid_w`.
pub fn image_pad_count(grid_h: usize, grid_w: usize, spatial_merge_size: usize) -> usize {
    let sms = spatial_merge_size.max(1);
    (grid_h / sms) * (grid_w / sms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_size_no_upscale() {
        // Small image: grid_unit=32, no upscale needed.
        let (h, w) = target_size(100, 150, 32);
        assert!(h <= 1280 && w <= 1280);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }

    #[test]
    fn test_target_size_downscale() {
        // Large image: should be downscaled.
        let (h, w) = target_size(2000, 3000, 32);
        assert!(h.max(w) <= 1280);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }

    #[test]
    fn test_image_pad_count_2x2_merge() {
        // Standard Qwen3-VL: 2×2 spatial merger folds a patch block
        // into one embedding token.
        assert_eq!(image_pad_count(64, 64, 2), 32 * 32);
        assert_eq!(image_pad_count(40, 80, 2), 20 * 40);
    }

    #[test]
    fn test_image_pad_count_no_merge() {
        // spatial_merge_size=1 → identity (each patch → one token).
        assert_eq!(image_pad_count(64, 64, 1), 64 * 64);
        assert_eq!(image_pad_count(8, 12, 1), 96);
    }

    #[test]
    fn test_image_pad_count_zero_sms_clamps_to_one() {
        // sms=0 is invalid; clamps to 1 so we never divide by zero.
        assert_eq!(image_pad_count(64, 64, 0), 64 * 64);
    }

    #[test]
    fn test_image_pad_count_non_divisible_floors() {
        // Integer division truncates: 65/2 = 32 (not 33).
        assert_eq!(image_pad_count(65, 64, 2), 32 * 32);
    }
}
