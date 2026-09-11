// SPDX-License-Identifier: AGPL-3.0-only

//! Band, lever and handle-presence edges of the N-column-blocked decode tier.
//!
//! The rule under test is `ncol_plan`, deliberately pure: the per-row NUMERIC
//! parity claim is the GPU oracle's
//! (`examples/native_fp8_attn_decode_batch_microtest`), and the dispatch wiring
//! is pinned on the mock backend in the two call sites' own test files.

use super::{NcolWidth, ncol_plan};

/// Both handles present, lever on — the default state an operator who sets
/// `ATLAS_ATTN_NCOL_GEMV` gets.
fn plan(m: usize, width: NcolWidth) -> Option<NcolWidth> {
    ncol_plan(m, width, true, true, true)
}

#[test]
fn selects_across_the_batch16_band() {
    for m in [5, 6, 8, 12, 15, 16] {
        assert_eq!(
            plan(m, NcolWidth::Two),
            Some(NcolWidth::Two),
            "m={m} is inside the 5..=16 band"
        );
    }
}

#[test]
fn width_four_is_selected_when_asked_for() {
    assert_eq!(plan(16, NcolWidth::Four), Some(NcolWidth::Four));
    assert_eq!(NcolWidth::Four.cols(), 4);
    assert_eq!(NcolWidth::Two.cols(), 2);
}

/// Single-row decode keeps the scalar `w8a16_gemv`, and 2..=4 keeps
/// `w8a16_gemv_batch4`: the ALU wall this tier attacks is proportional to M,
/// and at m <= 4 that kernel pays ~10 ops per weight byte already.
#[test]
fn declines_below_the_band() {
    for m in [0, 1, 2, 3, 4] {
        assert_eq!(plan(m, NcolWidth::Two), None, "m={m} is below the band");
    }
}

/// Above the kernel's MAX_M it would compute rows 0..15 and leave the rest as
/// stale memory rather than failing, so the upper edge is the template bound.
#[test]
fn declines_above_the_kernel_max_m() {
    for m in [17, 20, 32] {
        assert_eq!(plan(m, NcolWidth::Two), None, "m={m} is above MAX_M=16");
    }
}

#[test]
fn lever_off_declines_every_width() {
    for m in [5, 8, 16] {
        assert_eq!(ncol_plan(m, NcolWidth::Two, true, true, false), None);
        assert_eq!(ncol_plan(m, NcolWidth::Four, true, true, false), None);
    }
}

/// A shadow that carries only one instantiation must decline the OTHER width
/// rather than substitute — a route line and an A/B have to name the kernel
/// that actually ran.
#[test]
fn a_missing_entry_point_declines_that_width_only() {
    assert_eq!(ncol_plan(16, NcolWidth::Four, true, false, true), None);
    assert_eq!(
        ncol_plan(16, NcolWidth::Two, true, false, true),
        Some(NcolWidth::Two)
    );
    assert_eq!(ncol_plan(16, NcolWidth::Two, false, true, true), None);
    assert_eq!(
        ncol_plan(16, NcolWidth::Four, false, true, true),
        Some(NcolWidth::Four)
    );
}
