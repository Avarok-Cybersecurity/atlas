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
