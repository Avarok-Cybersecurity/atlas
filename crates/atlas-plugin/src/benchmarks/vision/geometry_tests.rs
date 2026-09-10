// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn token_count_is_quadratic_in_the_side() {
    // Catches an off-by-one in the merge divisor that a single hard-coded
    // expectation would not: doubling the side must quadruple the tokens.
    let a = expected_vision_tokens(224, 224, QWEN3_VL);
    let b = expected_vision_tokens(448, 448, QWEN3_VL);
    let c = expected_vision_tokens(896, 896, QWEN3_VL);
    assert_eq!(a, 49);
    assert_eq!(b, a * 4, "{a} -> {b} is not quadratic");
    assert_eq!(c, b * 4, "{b} -> {c} is not quadratic");
}

#[test]
fn every_ladder_size_has_a_defined_expectation() {
    let got: Vec<_> = crate::benchmarks::vision::provision::FIXTURES
        .iter()
        .map(|&(name, _, w, h)| (name, w, h, expected_vision_tokens(w, h, QWEN3_VL)))
        .collect();
    assert_eq!(
        got,
        vec![
            ("01_square_224.png", 224, 224, 49),
            ("02_square_336.png", 336, 336, 121),
            ("03_landscape_512x384.png", 512, 384, 192),
            ("04_wide_640x360.png", 640, 360, 220),
            ("05_square_768.png", 768, 768, 576),
            ("06_wide_1024x576.png", 1024, 576, 576),
            ("07_hd_1280x720.png", 1280, 720, 920),
            ("08_portrait_480x854.png", 480, 854, 405),
            ("09_over_clamp_1600x900.png", 1600, 900, 1400),
            ("10_tiny_8x8.png", 8, 8, 1),
            ("11_strip_64x2048.png", 64, 2048, 128),
            ("12_rgba_224.png", 224, 224, 49),
            ("13_gray_224.jpg", 224, 224, 49),
            ("14_png16_224.png", 224, 224, 49),
        ]
    );
}

#[test]
fn portrait_and_landscape_of_the_same_shape_agree() {
    // Transposing must not change the count. This rejects an expectation that
    // accidentally uses one side twice, which square fixtures cannot expose.
    assert_eq!(
        expected_vision_tokens(512, 384, QWEN3_VL),
        expected_vision_tokens(384, 512, QWEN3_VL)
    );
}

#[test]
fn snap_never_returns_zero() {
    // A sub-grid image must still produce one grid unit, not a 0x0 target and
    // a division by zero downstream.
    assert_eq!(snap(1, 32), 32);
    assert_eq!(snap(15, 32), 32);
    assert_eq!(expected_vision_tokens(1, 1, QWEN3_VL), 1);
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
            QWEN3_VL,
        )
    }

    let ladder: Vec<(u32, u32)> = crate::benchmarks::vision::provision::FIXTURES
        .iter()
        .map(|&(_, _, w, h)| (w, h))
        .collect();
    let discriminating: Vec<(u32, u32)> = ladder
        .iter()
        .copied()
        .filter(|&(w, h)| under_old_clamp(w, h) != expected_vision_tokens(w, h, QWEN3_VL))
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
    assert_eq!(expected_vision_tokens(1600, 900, QWEN3_VL), 1400);
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
        expected_vision_tokens(1600, 900, QWEN3_VL),
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
            expected_vision_tokens_bounded(w, h, QWEN3_VL, 0),
            expected_vision_tokens(w, h, QWEN3_VL),
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
            expected_vision_tokens_bounded(w, h, QWEN3_VL, CAP)
                != expected_vision_tokens(w, h, QWEN3_VL)
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
    assert_eq!(expected_vision_tokens_bounded(8, 8, QWEN3_VL, 262_144), 1);
    assert_eq!(
        expected_vision_tokens_bounded(64, 2048, QWEN3_VL, 262_144),
        128
    );
}

#[test]
fn the_discriminating_rung_stays_discriminating_under_a_bound() {
    // The reason the fix predicts rather than skips. Declaring a bound must
    // not blunt the one rung the ladder exists for: under a 262144 bound the
    // correct answer is 252, and an engine that ignored the declared bound and
    // fell back to the 1280px clamp would still answer 920 and still FAIL.
    let honoured = expected_vision_tokens_bounded(1600, 900, QWEN3_VL, 262_144);
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

/// GLM-5.3-Flash geometry, pinned against a LIVE serve.
///
/// Every triple below is the engine's own answer, measured 2026-09-09 on
/// GLM-5.3-Flash-EXL3 K2 at TP=2/EP=2 (binary fe8a53a1ededa20d): the run's
/// reported `usage.prompt_tokens` minus the template overhead of 16 that this
/// geometry solves for. Under the Qwen profile the same fourteen fixtures score
/// 5/14 with the engine correct on all of them, which is the regression this
/// test exists to stop recurring for the next family that is not Qwen.
#[test]
fn glm5_geometry_reproduces_the_engine_on_every_fixture() {
    // (w, h, vision tokens the engine actually produced)
    const MEASURED: &[(u32, u32, u32)] = &[
        (224, 224, 64),    // 01_square_224, and 12/13/14 which share its size
        (336, 336, 144),   // 02_square_336
        (512, 384, 266),   // 03_landscape_512x384
        (640, 360, 299),   // 04_wide_640x360
        (768, 768, 784),   // 05_square_768
        (1024, 576, 777),  // 06_wide_1024x576
        (1280, 720, 1196), // 07_hd_1280x720
        (480, 854, 558),   // 08_portrait_480x854
        (1600, 900, 1914), // 09_over_clamp_1600x900
        (8, 8, 16),        // 10_tiny_8x8 - the min_image_tokens floor, not 1
        (64, 2048, 222),   // 11_strip_64x2048
    ];
    for &(w, h, want) in MEASURED {
        assert_eq!(
            expected_vision_tokens(w, h, GLM5),
            want,
            "GLM5 geometry drifted at {w}x{h}"
        );
    }
}

/// The floor is what makes an 8x8 image 16 tokens rather than 1, and it is the
/// one field a reader is most likely to drop when copying the profile.
#[test]
fn the_glm5_token_floor_binds_only_below_it() {
    assert_eq!(expected_vision_tokens(8, 8, GLM5), 16, "floor must bind");
    assert_eq!(
        expected_vision_tokens(224, 224, GLM5),
        64,
        "floor must not bind"
    );
    assert_eq!(
        expected_vision_tokens(8, 8, QWEN3_VL),
        1,
        "Qwen has no floor and must be unaffected"
    );
}

/// Ceil vs round is a whole grid unit on one axis, and 512 is where they split:
/// round(512/28)=18 -> 504, ceil -> 532.
#[test]
fn glm5_ceil_aligns_where_qwen_would_round_down() {
    assert_eq!(ceil_align(512, 28), 532);
    assert_eq!(snap(512, 28), 504);
    // Same patch/merge, alignment the only difference - isolates the mode.
    let rounding = VisionGeometry {
        ceil_aligned: false,
        ..GLM5
    };
    assert_ne!(
        expected_vision_tokens(512, 384, GLM5),
        expected_vision_tokens(512, 384, rounding),
        "if these agree the alignment mode is not being consulted"
    );
    assert_eq!(expected_vision_tokens(512, 384, GLM5), 266);
}

/// The default must stay Qwen3-VL, or every variant that predates the profile
/// silently rescores.
#[test]
fn the_default_geometry_is_still_qwen() {
    assert_eq!(VisionGeometry::default(), QWEN3_VL);
    assert_eq!(geometry_by_name("qwen3_vl").unwrap(), QWEN3_VL);
    assert_eq!(geometry_by_name("glm5").unwrap(), GLM5);
    assert!(geometry_by_name("nope").is_err());
}
