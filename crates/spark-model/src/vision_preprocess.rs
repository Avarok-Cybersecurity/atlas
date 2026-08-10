// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-side image preprocessing for Qwen3-VL vision inputs.
//!
//! Decodes base64 JPEG/PNG images, resizes to a grid snapped to
//! `patch_size × spatial_merge_size`, normalizes with ImageNet stats,
//! and produces a flat `f32` tensor ready for the GPU vision encoder.

use anyhow::{Context, Result, bail};
use atlas_core::config::VisionConfig;
use image::{DynamicImage, ImageFormat, ImageReader, Limits};

/// SigLIP normalization — matches HF's Qwen2VLImageProcessor
/// (`image_mean = image_std = (0.5, 0.5, 0.5)` → pixels mapped to [-1, 1]).
const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Maximum allowed image dimension in pixels (longer side), applied AFTER
/// decode by the resize below. See [`DECODE_MAX_SIDE`] for the pre-decode cap.
const MAX_DIM: u32 = 1280;

/// Decoder limit: reject a header declaring more than this on either side
/// before a single pixel is allocated. Everything is resized down to
/// [`MAX_DIM`] anyway, so this only has to be above any real camera image;
/// 16384 is ~4× the long side of a 50 MP photo.
const DECODE_MAX_SIDE: u32 = 16_384;

/// Decoder limit: bytes the decoder may hold at once for one image. The
/// `image` crate's own default is 512 MiB, which on GB10's UNIFIED 121 GB
/// CPU+GPU memory is a per-request budget competing directly with the KV
/// cache — and the request body arrives over HTTP from an unauthenticated
/// caller. 192 MiB still admits an 8000×8000 RGB image.
const DECODE_MAX_ALLOC: u64 = 192 * 1024 * 1024;

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
    // Decode through `ImageReader` rather than `load_from_memory_with_format`
    // so the limits are ours. (The free function is not unlimited — it applies
    // `Limits::default()`, i.e. 512 MiB alloc — but it sets NO dimension cap,
    // and the alloc cap is documented as non-strict.) A 40-byte PNG header can
    // declare 65535×65535; the dimension limit rejects that from the header,
    // before any buffer is reserved.
    let mut reader = ImageReader::new(std::io::Cursor::new(&bytes));
    reader.set_format(fmt);
    let mut limits = Limits::default();
    limits.max_image_width = Some(DECODE_MAX_SIDE);
    limits.max_image_height = Some(DECODE_MAX_SIDE);
    limits.max_alloc = Some(DECODE_MAX_ALLOC);
    reader.limits(limits);
    reader.decode().context("image decode failed")
}

/// Reject a vision config whose geometry cannot drive the preprocessor.
///
/// Every field here comes from a third-party `config.json` via
/// `parse_vision_config`, which reports a MISSING key as `0` — so an absent or
/// malformed `patch_size` reaches `preprocess_image` as a divisor of zero, and
/// `grid_unit = patch_size * spatial_merge_size` reaches the scale computation
/// as `0.0`, producing a 0×0 target and then a division by zero. Fail with a
/// named error instead. Deliberately no fallback default: silently assuming
/// `patch_size = 16` would let a mismatched checkpoint produce a wrongly-shaped
/// pixel buffer, which is the hazard the encoder's own length check exists for.
fn validate_geometry(vcfg: &VisionConfig) -> Result<()> {
    if vcfg.patch_size == 0 {
        bail!("vision_config.patch_size is 0 (missing or invalid in the checkpoint's config.json)");
    }
    if vcfg.spatial_merge_size == 0 {
        bail!("vision_config.spatial_merge_size is 0 (missing or invalid in config.json)");
    }
    if vcfg.temporal_patch_size == 0 {
        bail!("vision_config.temporal_patch_size is 0 (missing or invalid in config.json)");
    }
    Ok(())
}

/// Compute the target (H, W) so that:
/// - Neither side exceeds `MAX_DIM`.
/// - Both sides are multiples of `grid_unit = patch_size × spatial_merge_size`.
/// - Aspect ratio is preserved (rounded to nearest grid_unit).
fn target_size_with_max_pixels(
    orig_h: u32,
    orig_w: u32,
    grid_unit: u32,
    max_pixels: Option<usize>,
) -> (u32, u32) {
    let dim_scale = (MAX_DIM as f32) / (orig_h.max(orig_w) as f32);
    let pixel_scale = max_pixels
        .filter(|&p| p > 0)
        .map(|p| ((p as f32) / ((orig_h as f32) * (orig_w as f32))).sqrt())
        .unwrap_or(1.0);
    let scale = dim_scale.min(pixel_scale).min(1.0); // never upscale
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
    preprocess_image_with_max_pixels(data_uri, vcfg, None)
}

/// Preprocess an image with an optional max-pixels cap, matching vLLM-style
/// multimodal processor controls. `None` preserves Atlas' historical 1280px
/// long-side cap.
pub fn preprocess_image_with_max_pixels(
    data_uri: &str,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
) -> Result<(Vec<f32>, usize, usize)> {
    // Before anything divides by them.
    validate_geometry(vcfg)?;
    let img = decode_image(data_uri)?;
    let img = img.to_rgb8();
    let (orig_w, orig_h) = (img.width(), img.height());

    let grid_unit = (vcfg.patch_size * vcfg.spatial_merge_size) as u32;
    let (th, tw) = target_size_with_max_pixels(orig_h, orig_w, grid_unit, max_pixels);

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
        let (h, w) = target_size_with_max_pixels(100, 150, 32, None);
        assert!(h <= 1280 && w <= 1280);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }

    #[test]
    fn test_target_size_downscale() {
        // Large image: should be downscaled.
        let (h, w) = target_size_with_max_pixels(2000, 3000, 32, None);
        assert!(h.max(w) <= 1280);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }

    #[test]
    fn test_target_size_max_pixels() {
        let (h, w) = target_size_with_max_pixels(1254, 1254, 32, Some(512 * 512));
        assert_eq!((h, w), (512, 512));
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

    /// A `vision_config` for the shipped Qwen3-VL geometry.
    fn ok_cfg() -> VisionConfig {
        VisionConfig {
            depth: 27,
            hidden_size: 1152,
            num_heads: 16,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            intermediate_size: 4304,
            out_hidden_size: 2048,
            deepstack_visual_indexes: vec![8, 16, 24],
            image_pad_token_id: 151_655,
        }
    }

    /// A 1×1 PNG, the smallest decodable input.
    const TINY_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    /// `parse_vision_config` reports a MISSING key as 0, so a checkpoint whose
    /// `vision_config` omits `patch_size` reaches here as a zero divisor. The
    /// old code divided `th / ps` at line 98 and panicked — from an HTTP
    /// request body, i.e. a remote crash of the request task.
    #[test]
    fn zero_patch_size_is_an_error_not_a_divide_by_zero_panic() {
        let mut vcfg = ok_cfg();
        vcfg.patch_size = 0;
        let err = preprocess_image(TINY_PNG_B64, &vcfg)
            .unwrap_err()
            .to_string();
        assert!(err.contains("patch_size"), "{err}");
    }

    /// `grid_unit = patch_size * spatial_merge_size` is the other divisor, and
    /// a zero here made it 0.0 in the f32 scale computation → a 0×0 target.
    #[test]
    fn zero_spatial_merge_size_is_an_error() {
        let mut vcfg = ok_cfg();
        vcfg.spatial_merge_size = 0;
        let err = preprocess_image(TINY_PNG_B64, &vcfg)
            .unwrap_err()
            .to_string();
        assert!(err.contains("spatial_merge_size"), "{err}");
    }

    /// `temporal_patch_size = 0` collapses `patch_dim` to 0, yielding an empty
    /// pixel buffer that the encoder would then have to catch.
    #[test]
    fn zero_temporal_patch_size_is_an_error() {
        let mut vcfg = ok_cfg();
        vcfg.temporal_patch_size = 0;
        let err = preprocess_image(TINY_PNG_B64, &vcfg)
            .unwrap_err()
            .to_string();
        assert!(err.contains("temporal_patch_size"), "{err}");
    }

    /// The geometry check must not have broken the working path: a valid
    /// config still decodes and produces `num_patches × patch_dim` floats.
    #[test]
    fn valid_config_still_preprocesses() {
        let vcfg = ok_cfg();
        let (pixels, gh, gw) = preprocess_image(TINY_PNG_B64, &vcfg).unwrap();
        let patch_dim = 3 * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size;
        assert_eq!(pixels.len(), gh * gw * patch_dim);
        assert!(gh > 0 && gw > 0);
    }

    /// Garbage that is not an image must be a clean error, not a panic.
    #[test]
    fn undecodable_input_is_an_error() {
        assert!(preprocess_image("bm90LWFuLWltYWdl", &ok_cfg()).is_err());
        assert!(preprocess_image("!!! not base64 !!!", &ok_cfg()).is_err());
    }

    /// A PNG whose IHDR declares 65535×65535 (~12.9 GB of RGB) but whose file
    /// is 70 bytes. The dimension limit rejects it from the header, before any
    /// pixel buffer is reserved.
    #[test]
    fn decode_bomb_dimensions_are_rejected_before_allocation() {
        let mut png: Vec<u8> = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        let mut ihdr: Vec<u8> = b"IHDR".to_vec();
        ihdr.extend_from_slice(&65535u32.to_be_bytes()); // width
        ihdr.extend_from_slice(&65535u32.to_be_bytes()); // height
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB
        let crc = {
            // CRC-32 over the chunk type + data, as PNG requires.
            let mut c: u32 = 0xFFFF_FFFF;
            for &b in &ihdr {
                c ^= u32::from(b);
                for _ in 0..8 {
                    c = if c & 1 != 0 {
                        (c >> 1) ^ 0xEDB8_8320
                    } else {
                        c >> 1
                    };
                }
            }
            !c
        };
        png.extend_from_slice(&13u32.to_be_bytes());
        png.extend_from_slice(&ihdr);
        png.extend_from_slice(&crc.to_be_bytes());

        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &png);
        // The fixture must exceed the limit under test, or it proves nothing.
        const { assert!(65535 > DECODE_MAX_SIDE) };
        // Assert on the LIMIT error specifically, not just "some error": the
        // fixture is also a truncated PNG, so a decoder without limits would
        // still fail — just later, after reserving the buffer. Only this exact
        // message proves the header check fired first.
        let err = format!("{:#}", preprocess_image(&b64, &ok_cfg()).unwrap_err());
        assert!(err.contains("Image size exceeds limit"), "{err}");
    }

    #[test]
    fn test_image_pad_count_non_divisible_floors() {
        // Integer division truncates: 65/2 = 32 (not 33).
        assert_eq!(image_pad_count(65, 64, 2), 32 * 32);
    }
}
