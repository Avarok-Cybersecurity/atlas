// SPDX-License-Identifier: AGPL-3.0-only

//! CPU tests for the Hopper GDN decode twins' launch geometry.
//!
//! ORACLE: the kernel header of `kernels/hopper/common/gdn_decode_hopper.cu`
//! plus the nsys shape it was written for (48 value heads, k_dim = v_dim =
//! 128, 132 SMs on an H100 SXM5). These run without a GPU, which is the only
//! layer of this change CI can see — `ATLAS_SKIP_BUILD=1` means no PTX exists
//! in a normal `cargo test` run.

use super::*;

/// POSITIVE. The defect the twin exists for: at C=1 the parent's grid is
/// `num_v_heads` CTAs = 48, and an H100 has 132 SMs, so 84 are idle. The
/// chooser must narrow until the grid covers the device.
#[test]
fn c1_on_hopper_narrows_until_the_grid_covers_every_sm() {
    let cols = gdn_hopper_cols_per_cta(128, 48, 132);
    assert_eq!(cols, 32, "48 CTAs on 132 SMs must narrow to one warp");
    assert!(
        48 * 128u32.div_ceil(cols) >= 132,
        "the narrowed grid must actually reach 132 SMs"
    );
}

/// POSITIVE. 64 columns is enough at two rows (96 CTAs -> 192), so the
/// chooser must stop there instead of always going to the narrowest tile: a
/// wider tile keeps four warps' worth of work per CTA.
#[test]
fn the_chooser_stops_at_the_widest_tile_that_fills_the_device() {
    assert_eq!(gdn_hopper_cols_per_cta(128, 96, 132), 64);
}

/// NEGATIVE. Where the natural grid already fills the device the tile must
/// stay at the parent's width. GB10 is the measured instance — it has exactly
/// 48 SMs, one per value head, and narrowing there measured 1.42x against the
/// parent where the parent's own shape measured 1.97x (dgx2, 2026-09-11).
#[test]
fn a_grid_that_already_fills_the_device_keeps_the_parents_tile() {
    assert_eq!(gdn_hopper_cols_per_cta(128, 48, 48), 128, "GB10, C=1");
    assert_eq!(gdn_hopper_cols_per_cta(128, 768, 132), 128, "H100, n=16");
}

/// NEGATIVE. A v_dim that is not a multiple of 32 would produce a ragged grid
/// for no benefit, and a v_dim that is already one warp wide has nothing to
/// split. Neither may be narrowed however empty the device is.
#[test]
fn ragged_and_single_warp_head_dims_are_left_alone() {
    assert_eq!(gdn_hopper_cols_per_cta(100, 4, 132), 100, "ragged");
    assert_eq!(gdn_hopper_cols_per_cta(32, 4, 132), 32, "already one warp");
}

/// POSITIVE. When even the narrowest tile cannot fill the device — few heads,
/// one row — the answer is still the narrowest: the SMs that a wider tile
/// would leave empty are not doing anything else. This is the shape the
/// chooser reaches by falling off the end of its search, so it is pinned
/// rather than left implicit.
#[test]
fn a_grid_that_cannot_fill_the_device_still_takes_the_narrowest_tile() {
    assert_eq!(gdn_hopper_cols_per_cta(96, 4, 132), 32);
    assert_eq!(gdn_hopper_cols_per_cta(128, 1, 132), 32);
}

/// The chooser must never hand the launcher a tile that changes the work:
/// the grid always covers v_dim exactly once.
#[test]
fn every_tile_choice_covers_v_dim_exactly_once() {
    for v_dim in [32u32, 64, 96, 100, 128] {
        for natural in [1u32, 4, 48, 96, 768] {
            for sms in [48u32, 132] {
                let cols = gdn_hopper_cols_per_cta(v_dim, natural, sms);
                assert!(
                    cols > 0 && cols <= v_dim.max(1),
                    "v_dim={v_dim} cols={cols}"
                );
                let tiles = v_dim.div_ceil(cols);
                assert!(
                    tiles * cols >= v_dim && (tiles - 1) * cols < v_dim,
                    "v_dim={v_dim} cols={cols} tiles={tiles} does not tile v_dim once"
                );
            }
        }
    }
}

/// The strided twin's extra precondition is not cosmetic: its state-norm
/// clamp reduces over `norm_sums[4]` and `col / 32`, i.e. exactly the four
/// warps of a 128-thread block covering one whole head.
#[test]
fn the_strided_twin_refuses_any_head_shape_but_128() {
    assert!(gdn_hopper_strided_dims_ok(128, 128));
    assert!(!gdn_hopper_strided_dims_ok(128, 64));
    assert!(!gdn_hopper_strided_dims_ok(128, 256));
    // ...while the contiguous twin, whose parent carries no clamp, does not
    // care how wide the head is.
    assert!(gdn_hopper_dims_ok(128, 64));
    assert!(gdn_hopper_dims_ok(64, 128));
    // Both refuse a k_dim past the statically sized staging buffers, and one
    // that the `j += 4` loop would walk off the end of.
    assert!(!gdn_hopper_dims_ok(256, 128));
    assert!(!gdn_hopper_dims_ok(126, 128));
}
