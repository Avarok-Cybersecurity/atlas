// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3's token-budget canvas.
//!
//! A port of `exllamav3/architecture/mm_processing/glm5_next.py::glm5_vision_canvas`
//! plus the content scale from `Glm5NextVisionModel.preprocess`. Every rounding
//! rule differs from the Qwen arm next door and each one is load-bearing:
//!
//! - **ceil** align to the 28px factor (Qwen ROUNDS to nearest, `:208-210`),
//! - **floor** integer candidate width inside the search,
//! - integer midpoint on the search,
//! - **ceil** after the under-budget float scale,
//! - the search's floor is `(factor, factor)`, not `(0, 0)`,
//! - and the content is then FLOORED and blitted at (0,0) into a zero canvas,
//!   rather than warped to fill it.
//!
//! Lives in its own file because `vision_preprocess.rs` is 324 lines and the
//! 500-line CI cap leaves no room for this plus its tests.

use atlas_core::config::VisionConfig;

use crate::layers::vision_encoder::enc_impl::init::CEILING_MAX_PATCHES;

/// GLM-5.3's declared image-token budget when the processor config was not
/// read (`processor_config.json[image_processor].max_image_tokens`).
const DEFAULT_MAX_IMAGE_TOKENS: usize = 8000;
/// `min_image_tokens` from the same place.
const DEFAULT_MIN_IMAGE_TOKENS: usize = 16;

/// The aligned canvas and the content rectangle placed at its top-left.
///
/// `target` is what gets patchified — padding included. Padded patches are real
/// patches: they attend normally and consume merged tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Canvas {
    pub target_h: u32,
    pub target_w: u32,
    pub content_h: u32,
    pub content_w: u32,
}

fn align(v: u64, f: u64) -> u64 {
    v.div_ceil(f) * f
}

/// Resolve the image-token budget, clamping it to what the encoder can hold.
///
/// GLM declares 8000 tokens = 32000 pre-merge patches, which is 2× the
/// encoder's `CEILING_MAX_PATCHES`. The full declared budget is NOT serveable,
/// so the clamp here degrades a missing `--vision-max-pixels` into a smaller
/// image rather than a loud encoder guard failure deep in the scheduler.
fn max_tokens_for(vcfg: &VisionConfig, max_pixels: Option<usize>, factor: usize) -> usize {
    let merge2 = (vcfg.spatial_merge_size * vcfg.spatial_merge_size).max(1);
    vcfg.max_image_tokens
        .unwrap_or(DEFAULT_MAX_IMAGE_TOKENS)
        .min(
            max_pixels
                .filter(|&a| a > 0)
                .map(|a| a / (factor * factor))
                .unwrap_or(usize::MAX),
        )
        .min(CEILING_MAX_PATCHES / merge2)
        .max(1)
}

/// Aligned canvas + content rectangle for one still image.
pub(crate) fn glm5_canvas_for(
    orig_h: u32,
    orig_w: u32,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
) -> Canvas {
    let factor = (vcfg.patch_size * vcfg.spatial_merge_size).max(1) as u64;
    let tps = vcfg.temporal_patch_size.max(1) as u64;
    let (h, w) = (orig_h.max(1) as u64, orig_w.max(1) as u64);

    let pixels_per_token = tps * factor * factor;
    let min_tokens = vcfg.min_image_tokens.unwrap_or(DEFAULT_MIN_IMAGE_TOKENS) as u64;
    let max_tokens = max_tokens_for(vcfg, max_pixels, factor as usize) as u64;
    let min_pixels = min_tokens * pixels_per_token;
    let max_pixels_budget = max_tokens * pixels_per_token;

    // A still image is resized as `temporal_patch_size` duplicated frames, so
    // the frame count in the budget is `tps`, not 1.
    let aligned_frames = tps;
    let mut ah = align(h, factor);
    let mut aw = align(w, factor);
    let mut budget = aligned_frames * ah * aw;

    if budget < min_pixels {
        // The one path where GLM UPSCALES; Qwen's `.min(1.0)` never can.
        let scale = ((min_pixels as f64) / ((tps * h * w) as f64)).sqrt();
        ah = align((((h as f64) * scale).ceil() as u64).max(1), factor);
        aw = align((((w as f64) * scale).ceil() as u64).max(1), factor);
        budget = aligned_frames * ah * aw;
    }

    if budget > max_pixels_budget {
        // Bisect on the ORIGINAL height. `align(ch)*align(cw)` is monotone
        // non-decreasing in `ch` (both factors are non-decreasing step
        // functions of it), which is what makes the search valid.
        let (mut lo, mut hi) = (1u64, h);
        let (mut best_h, mut best_w) = (factor, factor);
        while lo <= hi {
            let ch = (lo + hi) / 2;
            let cw = ((w * ch) / h).max(1);
            let (cand_h, cand_w) = (align(ch, factor), align(cw, factor));
            if aligned_frames * cand_h * cand_w <= max_pixels_budget {
                best_h = cand_h;
                best_w = cand_w;
                lo = ch + 1;
            } else {
                // `lo >= 1` for the life of the loop, so `ch >= 1` and this
                // cannot underflow.
                hi = ch - 1;
            }
        }
        ah = best_h;
        aw = best_w;
    }

    // Content scale: preserve aspect, never upscale ONCE the raw pixels already
    // meet the minimum budget. Below that threshold an upscale is exactly what
    // the canvas step just asked for, so the `min(1.0)` must not fire.
    let mut scale = ((ah as f64) / (h as f64)).min((aw as f64) / (w as f64));
    if h * w >= factor * factor * min_tokens {
        scale = scale.min(1.0);
    }
    let content_h = (((h as f64) * scale).floor() as u64).clamp(1, ah);
    let content_w = (((w as f64) * scale).floor() as u64).clamp(1, aw);

    Canvas {
        target_h: ah as u32,
        target_w: aw as u32,
        content_h: content_h as u32,
        content_w: content_w as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GLM-5.3's real vision geometry.
    fn glm() -> VisionConfig {
        VisionConfig {
            patch_size: 14,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            min_image_tokens: Some(16),
            max_image_tokens: Some(8000),
            model_type: "glm5_next_vision".to_string(),
            ..VisionConfig::default()
        }
    }

    fn merged_tokens(c: &Canvas) -> u32 {
        (c.target_h / 14) * (c.target_w / 14) / 4
    }

    /// Already factor-aligned and inside budget: nothing moves, nothing pads.
    #[test]
    fn an_aligned_image_is_left_alone() {
        let c = glm5_canvas_for(448, 448, &glm(), None);
        assert_eq!(
            c,
            Canvas {
                target_h: 448,
                target_w: 448,
                content_h: 448,
                content_w: 448
            }
        );
        assert_eq!(merged_tokens(&c), 256);
    }

    /// The common case: ceil-align up, keep the content at its original size,
    /// and pad the remainder. This is where Atlas's Qwen arm would instead have
    /// WARPED the image to 476x308.
    #[test]
    fn an_unaligned_image_is_padded_not_warped() {
        let c = glm5_canvas_for(450, 300, &glm(), None);
        assert_eq!(c.target_h, 476, "ceil(450/28)*28");
        assert_eq!(c.target_w, 308, "ceil(300/28)*28");
        assert_eq!((c.content_h, c.content_w), (450, 300), "content unscaled");
        assert_eq!(merged_tokens(&c), 187);
    }

    /// A tiny image sits exactly ON the minimum budget after alignment
    /// (2*112*112 == 16*1568), so the upscale branch does NOT fire — `budget <
    /// min_pixels` is strict. The scale guard also does not fire
    /// (100*100 = 10_000 < 28^2*16 = 12_544), which is the one case where GLM
    /// is ALLOWED to upscale content; here it simply scales 100 → 112.
    #[test]
    fn a_tiny_image_reaches_the_minimum_budget_by_alignment_alone() {
        let c = glm5_canvas_for(100, 100, &glm(), None);
        assert_eq!((c.target_h, c.target_w), (112, 112));
        assert_eq!(
            (c.content_h, c.content_w),
            (112, 112),
            "the min(1.0) guard must NOT fire below 12544 raw pixels"
        );
        assert_eq!(merged_tokens(&c), 16);
    }

    /// Strictly below the minimum budget: the upscale branch fires and the
    /// canvas grows past the aligned original.
    #[test]
    fn an_image_under_the_minimum_budget_is_upscaled() {
        let c = glm5_canvas_for(28, 28, &glm(), None);
        assert!(
            c.target_h > 28 && c.target_w > 28,
            "expected an upscale, got {c:?}"
        );
        assert!(2 * (c.target_h as u64) * (c.target_w as u64) >= 16 * 1568);
    }

    /// A big image against an operator bound: the result must FIT the bound and
    /// be the largest aligned canvas that does.
    #[test]
    fn a_large_image_is_bisected_down_to_the_bound() {
        // --vision-max-pixels 3211264 == CEILING_MAX_PATCHES * 14^2.
        let c = glm5_canvas_for(4000, 3000, &glm(), Some(3_211_264));
        assert_eq!(c.target_h % 28, 0);
        assert_eq!(c.target_w % 28, 0);
        assert!(merged_tokens(&c) <= 4096, "{c:?}");
        assert!(c.content_h <= c.target_h && c.content_w <= c.target_w);
        // The aspect ratio survives the downscale to within one factor.
        let want = 3000.0 / 4000.0;
        let got = c.content_w as f64 / c.content_h as f64;
        assert!((got - want).abs() < 0.02, "{c:?}");
    }

    /// The encoder ceiling is applied even with no operator flag and no
    /// checkpoint bound: GLM's declared 8000 tokens is 2x what the tower holds.
    #[test]
    fn the_declared_budget_is_clamped_to_the_encoder_ceiling() {
        let v = glm();
        assert_eq!(max_tokens_for(&v, None, 28), CEILING_MAX_PATCHES / 4);
        let c = glm5_canvas_for(6000, 6000, &v, None);
        assert!(
            merged_tokens(&c) <= (CEILING_MAX_PATCHES / 4) as u32,
            "{c:?}"
        );
    }

    /// Degenerate inputs must not divide by zero or produce a 0-patch canvas.
    #[test]
    fn degenerate_sizes_produce_a_usable_canvas() {
        for (h, w) in [(1u32, 1u32), (1, 4000), (4000, 1)] {
            let c = glm5_canvas_for(h, w, &glm(), Some(3_211_264));
            assert!(c.target_h >= 28 && c.target_w >= 28, "{h}x{w} -> {c:?}");
            assert!(c.content_h >= 1 && c.content_w >= 1);
            assert!(c.content_h <= c.target_h && c.content_w <= c.target_w);
        }
    }
}
