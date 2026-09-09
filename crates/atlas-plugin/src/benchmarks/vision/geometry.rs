// SPDX-License-Identifier: AGPL-3.0-only

//! How many vision tokens a fixture should become, and why that is the leg
//! worth having.
//!
//! A capability probe ("what colour is this?") answers whether the model saw
//! *an* image. It cannot tell you whether it saw the RIGHT one. The
//! 2026-08-14 resolution-cap defect is the proof: every image above 1280px on
//! the long side was silently downscaled to about a tenth of the area its
//! checkpoint permitted, and every capability probe still passed — a
//! downscaled picture is still recognisably red.
//!
//! Vision-token count is the observable that moves when preprocessing
//! changes, and the server reports it for free in `usage.prompt_tokens`. This
//! module is the model those assertions are checked against.

/// Snap a side to the patch grid, the way the preprocessor does when no
/// downscale is required: `round(side / grid) * grid`.
///
/// `f32::round` is half-away-from-zero, which is what makes 336 → 352 rather
/// than 320. Pinned by `the_rounding_mode_is_pinned`; if
/// the engine ever switches to half-even this model must move with it or every
/// expectation silently drifts by one grid unit.
pub fn snap(side: u32, grid: u32) -> u32 {
    (((side as f32) / (grid as f32)).round() as u32).max(1) * grid
}

/// Ceil-align a side to the grid: `ceil(side / grid) * grid`.
///
/// Not every preprocessor rounds. GLM-5.3-Flash lays the image on a ceil-aligned
/// CANVAS and keeps the content at its original size — see
/// `spark-model`'s `vision_preprocess/glm5_canvas.rs`, whose own tests pin
/// `ceil(450/28)*28` and `ceil(300/28)*28`. Rounding a 512-wide image to 504
/// where the engine pads it to 532 is a whole grid unit of drift on one axis,
/// and two rungs of the ladder differ by less than that.
pub fn ceil_align(side: u32, grid: u32) -> u32 {
    side.div_ceil(grid).max(1) * grid
}

/// How a model family's preprocessor turns pixels into merged vision tokens.
///
/// This used to be two bare `u32`s (`patch`, `merge`) hardcoded to `16, 2` at
/// every call site — Qwen3-VL's geometry, and the only geometry the ladder had
/// ever been pointed at.
///
/// 🪤 That silently mis-scored an ENTIRE OTHER FAMILY. GLM-5.3-Flash is patch
/// 14 / merge 2 with a ceil canvas and a `min_image_tokens` floor; against the
/// Qwen model it scored 5/14 with the engine perfectly correct on all fourteen
/// fixtures. Worse, the miss is self-concealing: the driver calibrates template
/// overhead as `total(224) - predicted(224)`, so a wrong prediction at the
/// anchor produces a compensating wrong overhead (31 instead of 16) and every
/// 224-sized rung passes BY CONSTRUCTION. A reader sees "5/14, and the small
/// ones are fine" and concludes the engine mishandles large images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionGeometry {
    /// Side of one square patch, in pixels.
    pub patch: u32,
    /// Patches per side in one merge block.
    pub merge: u32,
    /// True when sides are ceil-aligned to the grid rather than round-snapped.
    pub ceil_aligned: bool,
    /// Floor from the checkpoint's `min_image_tokens`. 0 = no floor.
    pub min_tokens: u32,
}

impl VisionGeometry {
    /// Pixels per side of the block that becomes ONE token.
    pub const fn grid(&self) -> u32 {
        self.patch * self.merge
    }

    fn align(&self, side: u32) -> u32 {
        if self.ceil_aligned {
            ceil_align(side, self.grid())
        } else {
            snap(side, self.grid())
        }
    }
}

/// Qwen3-VL: patch 16, merge 2, round-snapped, no floor. A 448×448 image is
/// 784 patches and 196 tokens — measured against a live server 2026-08-14, the
/// figure this whole model is anchored to. THE DEFAULT, so every variant that
/// existed before geometry became selectable scores byte-identically.
pub const QWEN3_VL: VisionGeometry = VisionGeometry {
    patch: 16,
    merge: 2,
    ceil_aligned: false,
    min_tokens: 0,
};

/// GLM-5.3-Flash: patch 14, merge 2, ceil canvas, `min_image_tokens` 16.
/// Every field is read off the checkpoint's own
/// `processor_config.json [image_processor]`; the ceil comes from
/// `vision_preprocess/glm5_canvas.rs`. Reproduces the engine on 14/14 fixtures.
pub const GLM5: VisionGeometry = VisionGeometry {
    patch: 14,
    merge: 2,
    ceil_aligned: true,
    min_tokens: 16,
};

impl Default for VisionGeometry {
    /// Qwen3-VL. The driver derives `Default`, and this is the geometry every
    /// call site hardcoded before the profile existed — so a driver built and
    /// never configured behaves exactly as it did.
    fn default() -> Self {
        QWEN3_VL
    }
}

/// Resolve a `vision_geometry` param value. The set is closed and the param is
/// a `Choice`, so an unknown name is a bug rather than user error — but it
/// returns an error instead of panicking because it is reachable from a
/// hand-built `ParamValues` in a test.
pub fn geometry_by_name(name: &str) -> anyhow::Result<VisionGeometry> {
    match name {
        "qwen3_vl" => Ok(QWEN3_VL),
        "glm5" => Ok(GLM5),
        other => anyhow::bail!("unknown vision_geometry {other:?} (expected qwen3_vl or glm5)"),
    }
}

/// Merged vision tokens for a `w × h` image at the given geometry.
///
/// Both sides snap to `patch × merge` first, then each `patch × patch` square
/// is one patch and each `merge × merge` block of patches becomes one token.
/// For Qwen3-VL geometry (patch 16, merge 2) a 448×448 image is 784 patches
/// and **196 tokens** — the figure measured against a live server on
/// 2026-08-14, which anchors this whole model.
pub fn expected_vision_tokens(w: u32, h: u32, g: VisionGeometry) -> u32 {
    let sw = g.align(w);
    let sh = g.align(h);
    let t = (sw / g.patch) * (sh / g.patch) / (g.merge * g.merge);
    t.max(g.min_tokens)
}

// A `within_bound(w, h, max_pixels)` helper used to live here, documented as
// letting the geometry leg report UNMEASURED when the served bound could not
// be determined. It was removed on the reasoning that "the ladder is
// deliberately built so that every fixture is inside any plausible bound".
//
// ★ That premise turned out to be false, and this is what replaced it. A
// serve started with `--vision-max-pixels 262144` — a perfectly ordinary
// deployment setting — puts FIVE of the fourteen fixtures outside the bound,
// and the run reported 9/14 geometry cells FAILING against a model whose
// vision path was entirely healthy (2026-08-21). The engine does not REFUSE an
// image above the bound; `preprocess_image_with_max_pixels` silently
// downscales it and answers normally, so the reply is a success carrying fewer
// vision tokens than a native-resolution prediction expects.
//
// The fix is NOT to skip those rungs. Skipping is what the removal comment
// rightly warned against: the rung that straddles the retired 1280px clamp is
// the entire point of the ladder (see `provision::FIXTURES`), and reporting it
// "unmeasured" would let the regression it exists to catch pass silently.
// Instead the bound became an INPUT to the prediction: tell the benchmark what
// the serve declares, and it predicts the downscaled geometry and asserts THAT.
// Every rung stays asserted at every bound, and an engine that downscales
// differently from its own declared bound still fails — which is the property
// the ladder was built for.

/// The engine's historical long-side clamp, used only when NOTHING declares an
/// area bound. Mirrors `FALLBACK_MAX_DIM` in `spark-model`'s
/// `vision_preprocess`.
pub const FALLBACK_MAX_DIM: u32 = 1280;

/// The absolute long-side ceiling, applied bound or not. Mirrors `ABS_MAX_DIM`.
/// This is what contains a strip like 64×2048, whose AREA is small but whose
/// long side is not.
pub const ABS_MAX_DIM: u32 = 4096;

/// The `(w, h)` the server actually encodes for a `w × h` source under a
/// declared area `max_pixels`.
///
/// Mirror of `spark_model::vision_preprocess::target_size_for`. It is a mirror
/// rather than a call because `atlas-plugin` drives a server over HTTP and does
/// not link the model crate; `the_mirror_matches_the_engines_anchors` pins it
/// to the two figures the engine's own documentation records, so a drift in
/// either direction fails a test rather than a benchmark run.
///
/// `max_pixels` is an AREA, matching the `--vision-max-pixels` flag.
pub fn served_size(w: u32, h: u32, grid: u32, max_pixels: Option<u64>) -> (u32, u32) {
    let long_side = w.max(h) as f32;
    let area = (w as f32) * (h as f32);
    let bound_scale = match max_pixels.filter(|&p| p > 0) {
        // A declared AREA bound governs, and REPLACES the long-side clamp.
        Some(p) => ((p as f32) / area).sqrt(),
        None => (FALLBACK_MAX_DIM as f32) / long_side,
    };
    let abs_scale = (ABS_MAX_DIM as f32) / long_side;
    let scale = bound_scale.min(abs_scale).min(1.0); // never upscale
    let tw = ((w as f32 * scale / grid as f32).round() as u32).max(1) * grid;
    let th = ((h as f32 * scale / grid as f32).round() as u32).max(1) * grid;
    (tw, th)
}

/// Merged vision tokens for `w × h` when the serve declares an area bound.
///
/// `max_pixels == 0` means "nothing declared", and is NOT the same as passing
/// `None` to [`served_size`]: an undeclared bound on a real checkpoint means
/// the checkpoint's own (large) bound governs and nothing downscales, which is
/// exactly [`expected_vision_tokens`]. The `None` arm of [`served_size`] models
/// the historical 1280px fallback, which is a DEFECT mode here, not a default.
pub fn expected_vision_tokens_bounded(w: u32, h: u32, g: VisionGeometry, max_pixels: u64) -> u32 {
    if max_pixels == 0 {
        return expected_vision_tokens(w, h, g);
    }
    let (tw, th) = served_size(w, h, g.grid(), Some(max_pixels));
    let t = (tw / g.patch) * (th / g.patch) / (g.merge * g.merge);
    t.max(g.min_tokens)
}

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod geometry_tests;
