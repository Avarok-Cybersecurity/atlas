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

/// Merged vision tokens for a `w × h` image at the given geometry.
///
/// Both sides snap to `patch × merge` first, then each `patch × patch` square
/// is one patch and each `merge × merge` block of patches becomes one token.
/// For Qwen3-VL geometry (patch 16, merge 2) a 448×448 image is 784 patches
/// and **196 tokens** — the figure measured against a live server on
/// 2026-08-14, which anchors this whole model.
pub fn expected_vision_tokens(w: u32, h: u32, patch: u32, merge: u32) -> u32 {
    let grid = patch * merge;
    let sw = snap(w, grid);
    let sh = snap(h, grid);
    (sw / patch) * (sh / patch) / (merge * merge)
}

// A `within_bound(w, h, max_pixels)` helper used to live here, documented as
// letting the geometry leg report UNMEASURED when the served bound could not
// be determined. Nothing ever called it — the driver asserts every rung
// unconditionally — so the doc described behaviour the benchmark did not have.
// Removed rather than wired up: the ladder is deliberately built so that every
// fixture is inside any plausible bound, which makes the prediction independent
// of the bound and the unconditional assertion correct. The one rung that
// straddles the retired 1280px clamp is the point of the exercise (see
// `provision::FIXTURES`), and it must be ASSERTED rather than skipped, or the
// regression it exists to catch would be reported as "unmeasured" and pass.

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod geometry_tests;
