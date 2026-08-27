// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-side image preprocessing for Qwen3-VL vision inputs.
//!
//! Decodes base64 JPEG/PNG images, resizes to a grid snapped to
//! `patch_size × spatial_merge_size`, normalizes with ImageNet stats,
//! and produces a flat `f32` tensor ready for the GPU vision encoder.

use anyhow::{Context, Result, bail};
use atlas_core::config::VisionConfig;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits};

/// SigLIP normalization — matches HF's Qwen2VLImageProcessor
/// (`image_mean = image_std = (0.5, 0.5, 0.5)` → pixels mapped to [-1, 1]).
/// `pub(crate)` because the video path normalizes with the identical stats —
/// a video frame is not a different kind of pixel, and two copies of these
/// numbers is two places for them to drift apart.
pub(crate) const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub(crate) const STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Long-side cap used ONLY when nothing else bounds the image — i.e. the
/// caller passed no `max_pixels` because the checkpoint shipped no
/// `preprocessor_config.json` and the operator set no `--vision-max-pixels`.
///
/// This was an UNCONDITIONAL ceiling until 2026-08-14, which silently threw
/// away most of the resolution such checkpoints allow. Qwen3.8-27B declares
/// `size = {longest_edge: 16777216, shortest_edge: 65536}` — pixel AREAS, so
/// up to 4096² — while this constant clamped every image to 1280 on the long
/// side, roughly a tenth of the permitted area. Measured before the change:
/// a 1344×1344 input came back as 1600 merged tokens (1280×1280), and
/// 1920×1080 as ~900 (1280×720). Detail-bearing inputs — documents, charts,
/// dense screenshots — paid for that directly, and nothing logged it.
const FALLBACK_MAX_DIM: u32 = 1280;

/// Absolute long-side ceiling that applies even when a `max_pixels` bound is
/// in force. `max_pixels` is an AREA, so on a pathological aspect ratio it
/// alone permits an unbounded long side (a 1×N strip). This is the safety
/// net [`FALLBACK_MAX_DIM`] was informally providing before it became a
/// fallback; it is deliberately far above any sane vision input.
const ABS_MAX_DIM: u32 = 4096;

/// Decoder limit: reject a header declaring more than this on either side
/// before a single pixel is allocated. Everything is resized down to at most
/// [`ABS_MAX_DIM`] anyway, so this only has to be above any real camera
/// image; 16384 is ~4× the long side of a 50 MP photo.
const DECODE_MAX_SIDE: u32 = 16_384;

/// Decoder limit: bytes the decoder may hold at once for one image. The
/// `image` crate's own default is 512 MiB, which on GB10's UNIFIED 121 GB
/// CPU+GPU memory is a per-request budget competing directly with the KV
/// cache — and the request body arrives over HTTP from an unauthenticated
/// caller. 192 MiB still admits an 8000×8000 RGB image.
const DECODE_MAX_ALLOC: u64 = 192 * 1024 * 1024;

/// Split a base64 `data:` URI (or a bare base64 string) into its declared
/// MIME type and decoded bytes.
///
/// Shared with the video path, which needs the SAME unwrapping but a
/// different decoder — and needs the MIME string, because "this is an mp4"
/// is worth saying by name rather than discovering as a parse failure.
/// The MIME is empty when the input carried no `data:` header.
pub(crate) fn decode_data_uri_bytes(data_uri: &str) -> Result<(String, Vec<u8>)> {
    // Strip optional "data:<mime>;base64," prefix.
    let (mime, b64) = if let Some(pos) = data_uri.find(",base64,") {
        (
            data_uri[..pos].trim_start_matches("data:").to_string(),
            &data_uri[pos + 8..],
        )
    } else if let Some(rest) = data_uri.strip_prefix("data:") {
        // "data:image/jpeg;base64,..." — the common, well-formed shape.
        match rest.find(',') {
            Some(p) => (
                rest[..p].trim_end_matches(";base64").to_string(),
                &rest[p + 1..],
            ),
            None => (String::new(), data_uri),
        }
    } else {
        (String::new(), data_uri)
    };

    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64.trim())
        .context("base64 decode failed")?;
    Ok((mime, bytes))
}

/// Decode a base64 data URI or raw base64 string into a `DynamicImage`.
fn decode_image(data_uri: &str) -> Result<DynamicImage> {
    let (_mime, bytes) = decode_data_uri_bytes(data_uri)?;

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

    // ★ EXIF ORIENTATION IS APPLIED. A camera writes the sensor's raw pixels
    // and records how to turn them upright in an EXIF tag rather than rotating
    // the data, so a phone photo is very often stored sideways with
    // `Orientation = 6` ("rotate 90° CW to display"). Decoding without that
    // tag hands the model an image rotated a quarter turn — and it does not
    // error or look broken, it simply answers about a sideways picture, which
    // is the failure mode this whole benchmark family exists to catch.
    //
    // Measured on 2026-08-14: Atlas ignored the tag entirely. Every viewer the
    // user compares against — their phone, their browser, their file manager —
    // honours it, so "what the model saw" and "what the user saw" silently
    // disagreed on a large fraction of real photographs.
    //
    // `into_decoder` rather than `decode`, because the tag lives on the
    // DECODER and is gone once the pixels are out. A format that carries no
    // orientation reports `NoTransforms`, so this is a no-op for PNG and for
    // any JPEG without the tag — the earlier behaviour, preserved exactly
    // where there is nothing to apply.
    let mut decoder = reader.into_decoder().context("image decode failed")?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder).context("image decode failed")?;
    if orientation != image::metadata::Orientation::NoTransforms {
        tracing::debug!("applying EXIF orientation {orientation:?}");
        img.apply_orientation(orientation);
    }
    Ok(img)
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
/// - The area bound is respected after grid snapping: `max_pixels` when the
///   caller supplies one, otherwise the long side is clamped to
///   [`FALLBACK_MAX_DIM`]. A bound below one grid cell uses that minimum cell.
/// - The long side never exceeds [`ABS_MAX_DIM`], bound or not.
/// - Both sides are multiples of `grid_unit = patch_size × spatial_merge_size`.
/// - Aspect ratio is preserved (rounded to nearest grid_unit).
/// - The continuous scale never upscales. Grid snapping may round a side up by
///   less than half a grid unit, and every target contains at least one cell.
///
/// `max_pixels` is an area, matching the `size.longest_edge` /
/// `shortest_edge` convention HF's Qwen2VL/Qwen3VL processors use (both are
/// pixel counts, not edge lengths, despite the names). It comes from the
/// checkpoint's `preprocessor_config.json` or the operator's
/// `--vision-max-pixels`; the operator's value wins.
///
/// ★ `max_pixels` REPLACES the long-side clamp rather than combining with it.
/// Combining was the bug: `dim_scale.min(pixel_scale)` meant a checkpoint
/// permitting 4096² could never exceed 1280 on the long side, so the model's
/// own declared bound could only ever lower the resolution, never raise it.
/// `pub(crate)` alias name used by the video path — see [`target_size_for`].
pub(crate) fn target_size_for(
    orig_h: u32,
    orig_w: u32,
    grid_unit: u32,
    max_pixels: Option<usize>,
) -> (u32, u32) {
    target_size_with_max_pixels(orig_h, orig_w, grid_unit, max_pixels)
}

fn target_size_with_max_pixels(
    orig_h: u32,
    orig_w: u32,
    grid_unit: u32,
    max_pixels: Option<usize>,
) -> (u32, u32) {
    let long_side = orig_h.max(orig_w) as f32;
    let area = (orig_h as f32) * (orig_w as f32);
    let bound_scale = match max_pixels.filter(|&p| p > 0) {
        // Model- or operator-declared AREA bound governs.
        Some(p) => ((p as f32) / area).sqrt(),
        // Nothing declared: fall back to the historical long-side clamp.
        None => (FALLBACK_MAX_DIM as f32) / long_side,
    };
    // Safety net, always applied.
    let abs_scale = (ABS_MAX_DIM as f32) / long_side;
    let scale = bound_scale.min(abs_scale).min(1.0); // never upscale
    let mut target_h =
        ((orig_h as f32 * scale / grid_unit as f32).round() as u32).max(1) * grid_unit;
    let mut target_w =
        ((orig_w as f32 * scale / grid_unit as f32).round() as u32).max(1) * grid_unit;

    // Nearest-grid rounding can raise BOTH axes past the continuous area
    // scale. A declared hard cap must survive that quantisation step. Shrink
    // one grid unit at a time, choosing the axis that leaves the closer source
    // aspect ratio. One grid cell is the smallest representable target, so a
    // smaller declared bound is normalized to that unavoidable minimum.
    if let Some(max_pixels) = max_pixels.filter(|&p| p > 0) {
        let grid_area = u64::from(grid_unit) * u64::from(grid_unit);
        let max_area = (max_pixels as u64).max(grid_area);
        let area = |h: u32, w: u32| u64::from(h) * u64::from(w);
        let aspect_error = |h: u32, w: u32| {
            if orig_h == 0 {
                0.0
            } else {
                ((w as f64 / h as f64) - (orig_w as f64 / orig_h as f64)).abs()
            }
        };

        while area(target_h, target_w) > max_area {
            let shorter_h = target_h.checked_sub(grid_unit).filter(|&h| h >= grid_unit);
            let shorter_w = target_w.checked_sub(grid_unit).filter(|&w| w >= grid_unit);
            match (shorter_h, shorter_w) {
                (Some(h), Some(w)) => {
                    let h_error = aspect_error(h, target_w);
                    let w_error = aspect_error(target_h, w);
                    if h_error < w_error
                        || (h_error == w_error && area(h, target_w) >= area(target_h, w))
                    {
                        target_h = h;
                    } else {
                        target_w = w;
                    }
                }
                (Some(h), None) => target_h = h,
                (None, Some(w)) => target_w = w,
                (None, None) => break,
            }
        }
    }
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

#[cfg(test)]
#[path = "vision_preprocess_tests.rs"]
mod tests;
