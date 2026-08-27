// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn the_measured_anchor_holds() {
    // 448x448 -> 196 merged tokens, measured against a live server 2026-08-14
    // (215 prompt_tokens minus 19 of chat template). Every other expectation
    // in this module is the same arithmetic; if this one is wrong they all
    // are, so it is asserted on its own.
    assert_eq!(expected_vision_tokens(448, 448, 16, 2), 196);
}

#[test]
fn token_count_is_quadratic_in_the_side() {
    // Catches an off-by-one in the merge divisor that a single hard-coded
    // expectation would not: doubling the side must quadruple the tokens.
    let a = expected_vision_tokens(224, 224, 16, 2);
    let b = expected_vision_tokens(448, 448, 16, 2);
    let c = expected_vision_tokens(896, 896, 16, 2);
    assert_eq!(a, 49);
    assert_eq!(b, a * 4, "{a} -> {b} is not quadratic");
    assert_eq!(c, b * 4, "{b} -> {c} is not quadratic");
}

#[test]
fn snapping_is_applied_before_counting() {
    // 336 is NOT a multiple of 32; it snaps to 352 (11 grid units). Counting
    // from the raw 336 gives 110 tokens, from the snapped 352 gives 121. The
    // ladder contains such sizes deliberately, so this distinction is the
    // difference between a correct expectation and a spurious failure.
    assert_eq!(snap(336, 32), 352);
    assert_eq!(expected_vision_tokens(336, 336, 16, 2), 121);
    assert_ne!(
        expected_vision_tokens(336, 336, 16, 2),
        (336 / 16) * (336 / 16) / 4,
        "expectation must come from the SNAPPED size, not the raw one"
    );
}

#[test]
fn every_ladder_size_has_a_defined_expectation() {
    // Non-square and portrait included — a transposed grid_h/grid_w would
    // survive a square-only suite.
    for (w, h, want) in [
        (224u32, 224u32, 49u32),
        (336, 336, 121),
        (512, 384, 192),
        (640, 360, 220),
        (768, 768, 576),
        (1024, 576, 576),
        // 1280x720 -> 920 corroborates the live 2026-08-14 measurement of a
        // 1920x1080 image at ~920 vision tokens: the old unconditional 1280px
        // clamp downscaled it to exactly this.
        (1280, 720, 920),
        (480, 854, 405),
        // The only rung above the old clamp. See
        // `the_ladder_can_actually_detect_a_regression_to_the_old_clamp`.
        (1600, 900, 1400),
    ] {
        assert_eq!(
            expected_vision_tokens(w, h, 16, 2),
            want,
            "{w}x{h} expectation drifted"
        );
    }
}

#[test]
fn portrait_and_landscape_of_the_same_shape_agree() {
    // Transposing must not change the count. A grid_h/grid_w swap in the
    // preprocessor is otherwise invisible on square fixtures.
    assert_eq!(
        expected_vision_tokens(512, 384, 16, 2),
        expected_vision_tokens(384, 512, 16, 2)
    );
}

#[test]
fn snap_never_returns_zero() {
    // A sub-grid image must still produce one grid unit, not a 0x0 target and
    // a division by zero downstream.
    assert_eq!(snap(1, 32), 32);
    assert_eq!(snap(15, 32), 32);
    assert!(expected_vision_tokens(1, 1, 16, 2) >= 1);
}

/// ★ The test that justifies the ladder's shape.
///
/// A gate that cannot fail on the defect it was written for is decoration.
/// This asserts the discriminating property directly: at least one fixture
/// must produce a DIFFERENT token count under the old 1280px long-side clamp
/// than under the checkpoint's declared area bound. Without the 1600x900 rung
/// this test fails, which is exactly the guard wanted — someone trimming the
/// ladder for runtime has to break this test to do it.
#[test]
fn the_ladder_can_actually_detect_a_regression_to_the_old_clamp() {
    /// What the retired unconditional clamp did: scale so the LONG side is
    /// 1280, never upscaling.
    fn under_old_clamp(w: u32, h: u32) -> u32 {
        let long = w.max(h) as f32;
        let s = (1280.0 / long).min(1.0);
        expected_vision_tokens(
            ((w as f32) * s).round() as u32,
            ((h as f32) * s).round() as u32,
            16,
            2,
        )
    }

    let ladder: Vec<(u32, u32)> = crate::benchmarks::vision::provision::FIXTURES
        .iter()
        .map(|&(_, _, w, h)| (w, h))
        .collect();
    let discriminating: Vec<(u32, u32)> = ladder
        .iter()
        .copied()
        .filter(|&(w, h)| under_old_clamp(w, h) != expected_vision_tokens(w, h, 16, 2))
        .collect();

    assert!(
        !discriminating.is_empty(),
        "every fixture in the ladder sits at or under the 1280px clamp, so a \
         regression to it would change no expectation and the geometry leg \
         would pass on a broken engine. Add a fixture above 1280 on the long \
         side."
    );

    // And name the numbers, so a future change to the fixture set that
    // weakens the margin is visible rather than silent.
    assert_eq!(under_old_clamp(1600, 900), 920);
    assert_eq!(expected_vision_tokens(1600, 900, 16, 2), 1400);
}

#[test]
fn the_rounding_mode_is_pinned() {
    // `f32::round` is half-AWAY-FROM-ZERO. If the engine ever switches to
    // half-even, 336 and 720 flip a grid unit and every expectation above
    // drifts. Pinned here so that lands as a named failure rather than a
    // mysterious token-count mismatch on a GPU box.
    assert_eq!(snap(224, 32), 224, "already exact");
    assert_eq!(snap(336, 32), 352, "10.5 rounds away from zero");
    assert_eq!(snap(360, 32), 352, "11.25 rounds down");
    assert_eq!(snap(720, 32), 736, "22.5 rounds away from zero");
    assert_eq!(snap(854, 32), 864, "26.69 rounds up");
}

// ── the declared area bound ──────────────────────────────────────────────
//
// These pin the mirror of the engine's `target_size_for`. A benchmark that
// predicts geometry from a COPY of the engine's arithmetic is only as good as
// the copy, so the copy is asserted against figures the engine's own source
// documents rather than against itself.

/// Every fixture in the committed ladder, as `(w, h)`.
const LADDER: [(u32, u32); 14] = [
    (224, 224),
    (336, 336),
    (512, 384),
    (640, 360),
    (768, 768),
    (1024, 576),
    (1280, 720),
    (480, 854),
    (1600, 900),
    (8, 8),
    (64, 2048),
    (224, 224),
    (224, 224),
    (224, 224),
];

#[test]
fn the_mirror_matches_the_engines_anchors() {
    // Both figures are quoted in `provision::FIXTURES` for the 1600x900 rung,
    // which exists precisely to tell these two apart:
    //   * a correct engine honours the checkpoint's large declared bound and
    //     leaves it alone                                   -> 1400 tokens
    //   * the retired long-side clamp scales it to 1280x720 ->  920 tokens
    assert_eq!(
        expected_vision_tokens(1600, 900, 16, 2),
        1400,
        "unbounded: the checkpoint's own bound is far above 1.44M px"
    );
    let (tw, th) = served_size(1600, 900, 32, None);
    assert_eq!((tw, th), (1280, 736), "the 1280px fallback clamp");
    assert_eq!(
        (tw / 16) * (th / 16) / 4,
        920,
        "the figure the fallback clamp produces, per provision::FIXTURES"
    );
}

#[test]
fn zero_means_nothing_was_declared() {
    // The param default. It must be EXACTLY the historical behaviour, or
    // adding the parameter would silently re-baseline every existing record.
    for (w, h) in LADDER {
        assert_eq!(
            expected_vision_tokens_bounded(w, h, 16, 2, 0),
            expected_vision_tokens(w, h, 16, 2),
            "{w}x{h} moved when no bound was declared"
        );
    }
}

#[test]
fn a_declared_bound_moves_exactly_the_fixtures_above_it() {
    // The 2026-08-21 case: a serve started with `--vision-max-pixels 262144`
    // scored 9/14 because five fixtures exceed that area and were silently
    // downscaled. Predicting under the bound must move those five and ONLY
    // those five — if it moved a sixth, the mirror would be manufacturing
    // failures of its own.
    const CAP: u64 = 262_144;
    let moved: Vec<(u32, u32)> = LADDER
        .iter()
        .copied()
        .filter(|&(w, h)| {
            expected_vision_tokens_bounded(w, h, 16, 2, CAP) != expected_vision_tokens(w, h, 16, 2)
        })
        .collect();
    assert_eq!(
        moved,
        vec![
            (768, 768),
            (1024, 576),
            (1280, 720),
            (480, 854),
            (1600, 900)
        ],
        "exactly the five fixtures whose area exceeds {CAP}"
    );
    for &(w, h) in &moved {
        assert!(
            (w as u64) * (h as u64) > CAP,
            "{w}x{h} moved but is inside the bound"
        );
    }
}

#[test]
fn a_declared_bound_never_upscales() {
    // A bound is a CEILING. The 8x8 and 64x2048 rungs are far inside 262144,
    // and a `sqrt(bound/area)` scale factor is greater than 1 for both — so
    // this is the arm where a missing `.min(1.0)` would inflate a fixture
    // instead of leaving it alone.
    assert_eq!(expected_vision_tokens_bounded(8, 8, 16, 2, 262_144), 1);
    assert_eq!(
        expected_vision_tokens_bounded(64, 2048, 16, 2, 262_144),
        128
    );
}

#[test]
fn the_discriminating_rung_stays_discriminating_under_a_bound() {
    // The reason the fix predicts rather than skips. Declaring a bound must
    // not blunt the one rung the ladder exists for: under a 262144 bound the
    // correct answer is 252, and an engine that ignored the declared bound and
    // fell back to the 1280px clamp would still answer 920 and still FAIL.
    let honoured = expected_vision_tokens_bounded(1600, 900, 16, 2, 262_144);
    assert_eq!(honoured, 252);
    let (tw, th) = served_size(1600, 900, 32, None);
    assert_ne!(
        honoured,
        (tw / 16) * (th / 16) / 4,
        "a declared bound must not make the fallback-clamp defect indistinguishable"
    );
}

#[test]
fn the_absolute_long_side_ceiling_still_applies_under_a_bound() {
    // A generous AREA bound cannot license an unbounded long side: 64x8192 is
    // only 512K px, but 8192 is past the 4096 ceiling, so the strip is scaled
    // by the ceiling rather than by the area.
    let (_, th) = served_size(64, 8192, 32, Some(16_777_216));
    assert!(th <= ABS_MAX_DIM, "{th} exceeds the absolute ceiling");
}
