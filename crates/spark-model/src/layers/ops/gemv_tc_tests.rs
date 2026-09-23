// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

// Real 27B projection shapes (N, K): every one must route, or the energy
// fix silently does not apply to that projection.
const SHAPES_27B: [(u32, u32); 7] = [
    (16384, 5120),  // GDN in_proj qkvz
    (12288, 5120),  // attention q + gate
    (1024, 5120),   // attention k / v
    (5120, 6144),   // attention o / GDN out_proj
    (34816, 5120),  // FFN gate+up
    (5120, 17408),  // FFN down
    (248320, 5120), // lm_head
];

#[test]
fn every_27b_shape_routes_at_every_row_count() {
    for (n, k) in SHAPES_27B {
        for m in 1..=8 {
            assert_eq!(tc_route(m, n, k, true, true, true), Some(TcKind::M8), "m={m} n={n} k={k}");
        }
        for m in 9..=16 {
            assert_eq!(tc_route(m, n, k, true, true, true), Some(TcKind::M16), "m={m} n={n} k={k}");
        }
    }
}

#[test]
fn above_sixteen_rows_declines() {
    // tc16's A fragment holds 16 rows; row 16 would never be written.
    assert_eq!(tc_route(17, 5120, 5120, true, true, true), None);
}

#[test]
fn unaligned_shapes_decline_instead_of_dropping_a_tail() {
    // K: each quad reads 128 contiguous k per step.
    assert_eq!(tc_route(4, 5120, 5120 + 64, true, true, true), None);
    // N: whole 8-column tiles only (GDN in_proj_a/b are N=48 — aligned — but
    // a 44-row head would leave 4 columns unwritten).
    assert_eq!(tc_route(4, 44, 5120, true, true, true), None);
    assert_eq!(tc_route(4, 48, 5120, true, true, true), Some(TcKind::M8));
}

#[test]
fn degenerate_launches_decline() {
    assert_eq!(tc_route(0, 5120, 5120, true, true, true), None);
    assert_eq!(tc_route(4, 0, 5120, true, true, true), None);
    assert_eq!(tc_route(4, 5120, 0, true, true, true), None);
}

#[test]
fn kill_switch_declines() {
    assert_eq!(tc_route(4, 5120, 5120, false, true, true), None);
}

#[test]
fn missing_entries_fall_back_correctly() {
    // A target without tc8 still serves narrow rows on tc16 (it covers M<=16).
    assert_eq!(tc_route(4, 5120, 5120, true, false, true), Some(TcKind::M16));
    // Without tc16 the wide rows keep the CUDA-core tier.
    assert_eq!(tc_route(12, 5120, 5120, true, true, false), None);
    // Nothing loaded: CUDA-core tier.
    assert_eq!(tc_route(4, 5120, 5120, true, false, false), None);
}

#[test]
fn grid_covers_every_column() {
    for (n, _) in SHAPES_27B {
        for kind in [TcKind::M8, TcKind::M16] {
            let ctas = n.div_ceil(kind.cols_per_cta());
            assert!(ctas * kind.cols_per_cta() >= n);
            assert!((ctas - 1) * kind.cols_per_cta() < n, "no empty CTA");
        }
    }
}
