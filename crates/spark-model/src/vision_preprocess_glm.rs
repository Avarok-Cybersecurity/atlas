// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 (`Glm5NextImageProcessor`) image geometry: `smart_resize` to a
//! `patch × merge`-aligned canvas, aspect-preserving fit, zero pad.
//!
//! Split from `vision_preprocess.rs` for the file-size cap, and because the
//! resize POLICY differs from the Qwen family's in a way that is not a
//! parameter: Qwen resizes straight onto the snapped canvas (accepting a
//! sub-grid-unit aspect distortion), GLM fits the content inside it and pads
//! the remainder. Two policies, two functions.

use avarok_core::config::VisionConfig;
use image::RgbImage;

/// Qwen2-VL-family `smart_resize`, which GLM inherits: snap each side to the
/// nearest multiple of `factor`, then rescale by the area bound if the snapped
/// canvas violates it.
///
/// `factor = patch_size * merge_size` (28 for GLM), and the bounds are AREAS in
/// pixels. Returns `(h_bar, w_bar)`, both multiples of `factor` and at least
/// one factor each.
///
/// Note the asymmetry, which is the reference's and not a simplification: the
/// initial snap ROUNDS (so a 360-pixel side at factor 28 becomes 364, i.e. the
/// canvas may be slightly larger than the input), while the over-budget branch
/// FLOORS and the under-budget branch CEILS.
pub fn smart_resize(h: u32, w: u32, factor: u32, min_pixels: u64, max_pixels: u64) -> (u32, u32) {
    let f = factor.max(1) as f64;
    let (hf, wf) = (h.max(1) as f64, w.max(1) as f64);
    let snap = |v: f64| ((v / f).round() as u32).max(1) * factor;
    let (mut h_bar, mut w_bar) = (snap(hf), snap(wf));

    let area = |a: u32, b: u32| u64::from(a) * u64::from(b);
    if area(h_bar, w_bar) > max_pixels && max_pixels > 0 {
        let beta = (hf * wf / max_pixels as f64).sqrt();
        h_bar = (((hf / beta) / f).floor() as u32).max(1) * factor;
        w_bar = (((wf / beta) / f).floor() as u32).max(1) * factor;
    } else if area(h_bar, w_bar) < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        h_bar = (((hf * beta) / f).ceil() as u32).max(1) * factor;
        w_bar = (((wf * beta) / f).ceil() as u32).max(1) * factor;
    }
    (h_bar, w_bar)
}

/// The area bounds this checkpoint permits, in pixels.
///
/// GLM states its budget in MERGED TOKENS (`min_image_tokens = 16`,
/// `max_image_tokens = 8000`) where the Qwen family states a pixel area, so the
/// conversion is `tokens × factor²` — one merged token covers exactly one
/// `factor × factor` pixel block. An operator `--vision-max-pixels` still wins
/// when it is the tighter of the two: the flag exists to bound memory, and a
/// checkpoint may not raise its way past it.
///
/// ★ The ENCODER'S OWN CAPACITY is the third bound, and it is not optional.
/// `GlmVit` sizes every buffer from `derive_max_patches(vcfg.max_pixels,
/// patch_size)`, so a host-side budget derived only from the checkpoint's token
/// count would hand it images it cannot hold: GLM declares 8000 merged tokens,
/// which is 32000 unmerged patches, against an encoder that allocates 6400 rows
/// when nothing else bounds it. That produces a refused request, not a wrong
/// answer — but it is the same class of defect the `max_pixels` field exists to
/// remove, two sides of the pipeline deriving a bound independently and
/// agreeing only by luck. Both now call the SAME function.
pub fn pixel_bounds(vcfg: &VisionConfig, operator_max: Option<usize>) -> (u64, u64) {
    let factor = (vcfg.patch_size * vcfg.spatial_merge_size).max(1) as u64;
    let per_token = factor * factor;
    let min_pixels = vcfg.min_image_tokens as u64 * per_token;
    let model_max = vcfg.max_image_tokens as u64 * per_token;
    let (encoder_patches, _) = crate::layers::vision_encoder::enc_impl::init::derive_max_patches(
        vcfg.max_pixels.or(operator_max),
        vcfg.patch_size,
    );
    let encoder_max =
        encoder_patches as u64 * (vcfg.patch_size.max(1) * vcfg.patch_size.max(1)) as u64;
    let mut max_pixels = encoder_max;
    if model_max > 0 {
        max_pixels = max_pixels.min(model_max);
    }
    if let Some(p) = operator_max.filter(|&p| p > 0) {
        max_pixels = max_pixels.min(p as u64);
    }
    (min_pixels, max_pixels.max(per_token))
}

/// Fit `img` inside the `smart_resize` canvas preserving aspect, then zero-pad
/// right/bottom to the canvas exactly. Returns the padded canvas.
///
/// ★ VERIFIED against the reference's own `pixel_values` (2026-09-22), not
/// assumed. Two properties were read straight out of the golden for image c
/// (640x360 -> a 644x364 canvas), and both had a plausible alternative:
///
/// 1. **The pad is BLACK PIXELS, i.e. it precedes normalization.** Padded
///    cells in the golden hold exactly `(0 - mean)/std` =
///    `(-1.7926, -1.7520, -1.4801)`, not `0.0`. `REFERENCE.md` §5 recorded
///    `tvF.pad(..., fill=0)` without pinning it relative to
///    `rescale`/`normalize`; this settles it.
/// 2. **The fit NEVER UPSCALES.** The golden's content occupies rows 0..360
///    and columns 0..640 — the source dimensions, unresampled — with the
///    remaining 4 rows and 4 columns padded. An aspect-fit that scaled up to
///    fill the canvas (scale 1.00625 here) would have produced a 362x644
///    content box and only 2 pad rows. Every sampled pixel of the golden
///    matches the raw PNG normalized in place, to 4e-4 (float32 storage).
///
/// `tests/glm_vision_preprocess_pin.rs` pins both against the fixtures.
pub fn resize_and_pad(
    img: &RgbImage,
    vcfg: &VisionConfig,
    operator_max: Option<usize>,
) -> RgbImage {
    let factor = (vcfg.patch_size * vcfg.spatial_merge_size) as u32;
    let (min_pixels, max_pixels) = pixel_bounds(vcfg, operator_max);
    let (target_h, target_w) =
        smart_resize(img.height(), img.width(), factor, min_pixels, max_pixels);

    // Largest scale that keeps BOTH sides inside the canvas, and never above
    // 1.0 — `smart_resize` ROUNDS to the grid, so its canvas is routinely a
    // few pixels LARGER than the input, and filling it would resample an image
    // that needs no resampling at all.
    let scale = (target_h as f64 / img.height().max(1) as f64)
        .min(target_w as f64 / img.width().max(1) as f64)
        .min(1.0);
    let content_h = ((img.height() as f64 * scale).round() as u32).clamp(1, target_h);
    let content_w = ((img.width() as f64 * scale).round() as u32).clamp(1, target_w);

    // At scale 1.0 the content IS the source: return its pixels untouched
    // rather than running them through a filter that is only approximately an
    // identity. CatmullRom is the `image` crate's closest match to PIL BICUBIC,
    // which is what the HF processors resample with.
    let fitted = if (content_h, content_w) == (img.height(), img.width()) {
        img.clone()
    } else {
        image::imageops::resize(
            img,
            content_w,
            content_h,
            image::imageops::FilterType::CatmullRom,
        )
    };
    if content_h == target_h && content_w == target_w {
        return fitted;
    }
    let mut canvas = RgbImage::new(target_w, target_h);
    image::imageops::replace(&mut canvas, &fitted, 0, 0);
    canvas
}

/// Index of the patch at `(patch_row, patch_col)` in the 2×2-block-major
/// stream the GLM tower expects.
///
/// `((h_block * w_blocks + w_block) * merge + row_in_block) * merge +
/// col_in_block` — the flattening of `reshape(h/m, m, w/m, m).transpose(1, 2)`.
/// The 2×2 conv downsample later folds four CONSECUTIVE tokens into one output
/// row, which is correct only under exactly this order.
pub fn block_major_patch_index(
    patch_row: usize,
    patch_col: usize,
    grid_w: usize,
    merge: usize,
) -> usize {
    let m = merge.max(1);
    let w_blocks = grid_w / m;
    (((patch_row / m) * w_blocks + patch_col / m) * m + patch_row % m) * m + patch_col % m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The golden's third image, whose numbers came out of the reference run:
    /// 640×360 at factor 28 becomes a 644×364 canvas → a 46×26 patch grid →
    /// 23×13 = 299 merged tokens.
    #[test]
    fn smart_resize_reproduces_the_golden_non_square_canvas() {
        let (h, w) = smart_resize(360, 640, 28, 16 * 784, 8000 * 784);
        assert_eq!((h, w), (364, 644));
        assert_eq!((h / 14) * (w / 14), 26 * 46);
        assert_eq!((h / 28) * (w / 28), 299);
    }

    /// A square 448 input is already aligned: the canvas is itself, 32×32
    /// patches, 256 merged tokens — the other two golden images.
    #[test]
    fn an_aligned_square_is_left_alone() {
        assert_eq!(smart_resize(448, 448, 28, 16 * 784, 8000 * 784), (448, 448));
    }

    /// Over budget FLOORS onto the grid, so the result is inside the bound —
    /// a rounding step that went the other way would hand the encoder more
    /// patches than its buffers hold.
    #[test]
    fn an_over_budget_image_is_floored_inside_the_bound() {
        let max = 100u64 * 784; // 100 merged tokens
        let (h, w) = smart_resize(4000, 4000, 28, 16 * 784, max);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        assert!(
            u64::from(h) * u64::from(w) <= max,
            "{h}x{w} exceeds the {max}px budget"
        );
    }

    /// Under budget CEILS up to the minimum, and the minimum is expressed in
    /// tokens, so a 1×1 image still becomes a legal multi-patch grid.
    #[test]
    fn a_tiny_image_is_raised_to_the_minimum_token_count() {
        let min = 16u64 * 784;
        let (h, w) = smart_resize(1, 1, 28, min, 8000 * 784);
        assert!(u64::from(h) * u64::from(w) >= min, "{h}x{w} under {min}");
        assert_eq!((h % 28, w % 28), (0, 0));
    }

    fn glm_cfg() -> VisionConfig {
        VisionConfig {
            patch_size: 14,
            spatial_merge_size: 2,
            min_image_tokens: 16,
            max_image_tokens: 8000,
            ..VisionConfig::default()
        }
    }

    /// A tighter operator flag lowers the budget; a looser one raises it only
    /// as far as the ENCODER can follow, never to the checkpoint's declared
    /// 8000 merged tokens.
    ///
    /// 🪤 The flag moves the encoder too — `vcfg.max_pixels` is what
    /// `GlmVit::new` allocates from — so "looser flag, unchanged bound" would
    /// be the WRONG assertion here. What must hold is that the resolved bound
    /// never exceeds what the encoder built for that same flag can hold.
    #[test]
    fn the_operator_bound_moves_the_budget_only_as_far_as_the_encoder_follows() {
        let vcfg = glm_cfg();
        let (_, tight) = pixel_bounds(&vcfg, Some(512 * 512));
        assert!(
            tight <= 512 * 512,
            "a tighter flag must lower the bound: {tight}"
        );
        assert_eq!(tight / (14 * 14), 1337, "512x512 at patch 14 is 1337 rows");

        let (_, loose) = pixel_bounds(&vcfg, Some(64 * 1024 * 1024));
        let (_, none) = pixel_bounds(&vcfg, None);
        assert!(
            loose > none,
            "a looser flag may raise the bound: {loose} vs {none}"
        );
        // ...but only to the encoder's ceiling, not to the checkpoint's ask.
        assert_eq!(
            loose / (14 * 14),
            16_384,
            "the encoder's CEILING_MAX_PATCHES"
        );
        assert!(
            loose < 8000 * 784,
            "never as far as the declared token budget"
        );
    }

    /// The bound the checkpoint DECLARES is bigger than any encoder Atlas
    /// allocates, so the host budget has to come down to the encoder's — this
    /// is the one that would otherwise refuse large images at the GPU instead
    /// of resizing them on the host.
    #[test]
    fn the_encoder_capacity_caps_the_checkpoints_own_token_budget() {
        let vcfg = glm_cfg();
        let (_, resolved) = pixel_bounds(&vcfg, None);
        let declared = 8000u64 * 784;
        assert!(
            resolved < declared,
            "GLM declares {declared}px (8000 merged tokens); the encoder's default \
             allocation cannot hold that, so the host bound must be lower, got {resolved}"
        );
        // 6400 unmerged patches at patch 14 — exactly what `derive_max_patches`
        // allocates when nothing else bounds the image.
        assert_eq!(resolved / (14 * 14), 6400);
        // And the resolved area really does fit: a square image at that bound
        // produces no more patches than the encoder holds.
        let side = (resolved as f64).sqrt() as u64;
        assert!((side / 14) * (side / 14) <= 6400);
    }

    /// A checkpoint that declares no token budget still gets the encoder's.
    #[test]
    fn a_checkpoint_without_a_token_budget_still_gets_a_bound() {
        let vcfg = VisionConfig {
            max_image_tokens: 0,
            ..glm_cfg()
        };
        let (_, resolved) = pixel_bounds(&vcfg, None);
        assert_eq!(resolved / (14 * 14), 6400);
        let (_, with_flag) = pixel_bounds(&vcfg, Some(512 * 512));
        assert!(with_flag <= 512 * 512);
        assert_eq!(with_flag / (14 * 14), 1337);
    }

    /// The canvas may be LARGER than the input (smart_resize rounds up), and
    /// the content must not be stretched to fill it — the golden's image c has
    /// its 360x640 pixels unresampled inside a 364x644 canvas.
    #[test]
    fn a_canvas_larger_than_the_input_pads_rather_than_upscales() {
        let vcfg = glm_cfg();
        let img = RgbImage::new(640, 360);
        let out = resize_and_pad(&img, &vcfg, None);
        assert_eq!(
            (out.height(), out.width()),
            (364, 644),
            "the smart_resize canvas"
        );
        // Nothing to assert about pixel values on a blank image; the geometry
        // claim is pinned end-to-end against the reference fixtures in
        // `tests/glm_vision_preprocess_pin.rs`.
        assert_eq!((out.height() / 14) * (out.width() / 14), 26 * 46);
    }

    /// Block-major is a PERMUTATION of raster over the same grid, and the
    /// first four indices are one 2×2 block rather than one row of four.
    #[test]
    fn block_major_index_is_a_permutation_grouping_2x2_blocks() {
        let (gh, gw, m) = (26usize, 46usize, 2usize);
        let mut seen = vec![usize::MAX; gh * gw];
        for ph in 0..gh {
            for pw in 0..gw {
                let i = block_major_patch_index(ph, pw, gw, m);
                assert!(i < gh * gw, "index {i} out of range");
                assert_eq!(seen[i], usize::MAX, "index {i} used twice");
                seen[i] = ph * gw + pw;
            }
        }
        // Tokens 0..4 are the 2×2 block at the origin, in (row, col) order.
        assert_eq!(&seen[0..4], &[0, 1, gw, gw + 1]);
        // Token 4 starts the NEXT block along w, not the next column.
        assert_eq!(seen[4], 2);
    }

    /// The whole reason the order is declared: raster and block-major agree
    /// only on a grid ONE block wide. Any real image is wider, and there the
    /// two orders disagree from the third patch onward — which is why a port
    /// that picks the wrong one produces a plausible embedding, not an error.
    #[test]
    fn block_major_differs_from_raster_on_any_real_grid() {
        // One block wide (grid_w = 2): the two orders coincide.
        for (row, col) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            assert_eq!(
                block_major_patch_index(row, col, 2, 2),
                row * 2 + col,
                "1 block wide must equal raster at ({row}, {col})"
            );
        }
        // Two blocks wide: patch (0, 2) is raster index 2 but block-major 4,
        // because tokens 2 and 3 belong to the FIRST block's second row.
        assert_eq!(block_major_patch_index(0, 2, 4, 2), 4);
        assert_eq!(block_major_patch_index(1, 0, 4, 2), 2);
    }
}
