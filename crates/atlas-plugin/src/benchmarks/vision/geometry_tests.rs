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

#[test]
fn the_bound_check_is_conservative_when_unknown() {
    // Unknown bound must read as "cannot assert", never as "fits" — an
    // expectation computed against a guessed bound would fail a correct
    // engine, which is worse than reporting UNMEASURED.
    assert!(within_bound(448, 448, Some(16_777_216)));
    assert!(!within_bound(8192, 8192, Some(16_777_216)));
    assert!(!within_bound(64, 64, None), "unknown must not read as fits");
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
