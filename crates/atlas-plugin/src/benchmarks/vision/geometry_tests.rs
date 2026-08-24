// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

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
fn every_ladder_size_has_a_defined_expectation() {
    let got: Vec<_> = crate::benchmarks::vision::provision::FIXTURES
        .iter()
        .map(|&(name, _, w, h)| (name, w, h, expected_vision_tokens(w, h, 16, 2)))
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
    assert_eq!(expected_vision_tokens(1, 1, 16, 2), 1);
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
