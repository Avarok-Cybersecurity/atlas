// SPDX-License-Identifier: AGPL-3.0-only

//! The Hopper FP8-activation-quantizer's GROUP AND ELEMENT MAPPING, on the
//! host.
//!
//! `native_fp8_act_quant_hopper_microtest` proves the bytes agree on a device;
//! these prove the thing a device cannot easily show — that the launcher's grid
//! and the kernel's self-derived span between them touch every K-group of every
//! token EXACTLY ONCE, with no gap and no double-write, at every shape #928's
//! attribution names. A gap is a group whose scale is never written (stale
//! scratch straight into a GEMM); a double-write is two CTAs racing on one
//! scale. Neither shows up as a crash and both are invisible to a rel_rms
//! check at small M.
//!
//! The oracle is `kernels/hopper/common/fp8_act_quant_hopper.cu`, re-stated in
//! [`fp8_quant_hopper_span`] and [`fp8_quant_hopper_lane`]. That makes this a
//! test of the SSOT pair, not of a second implementation: if the `.cu`'s
//! arithmetic ever changes, these functions change with it and this file is
//! what says the launcher was updated too.

use super::*;

/// Every (M, K) the round-13 attribution prices, plus the ragged shapes the
/// serve actually hands the quantizer (the 17- and 25-token tail chunks, the
/// n=16 decode step).
const K_DIMS: [u32; 3] = [5120, 6144, 17408];
const M_DIMS: [u32; 5] = [16, 17, 25, 1168, 4576];

/// The Hopper grid plus the kernel's own span is a PARTITION of `0..K/128`.
#[test]
fn the_hopper_grid_covers_every_k_group_exactly_once() {
    for k in K_DIMS {
        let groups = k / 128;
        let [_, grid_y, _] = fp8_quant_grid(true, 1, k);
        let mut seen = vec![0u32; groups as usize];
        for by in 0..grid_y {
            let (g0, g1) = fp8_quant_hopper_span(groups, grid_y, by);
            for g in g0..g1 {
                seen[g as usize] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "K={k}: grid_y={grid_y} does not partition {groups} groups: {seen:?}"
        );
    }
}

/// …and the same holds for a Y extent the launcher does NOT pick. The kernel
/// derives its span from `gridDim.y`, so the launcher is free to change the
/// 8-groups-per-CTA choice for performance without a correctness review; this
/// is what makes that claim true rather than aspirational.
#[test]
fn any_grid_y_still_partitions_the_groups() {
    for k in K_DIMS {
        let groups = k / 128;
        for grid_y in [1, 2, 3, 5, 7, 8, 17, groups - 1, groups] {
            let mut seen = vec![0u32; groups as usize];
            for by in 0..grid_y {
                let (g0, g1) = fp8_quant_hopper_span(groups, grid_y, by);
                for g in g0..g1 {
                    seen[g as usize] += 1;
                }
            }
            assert!(
                seen.iter().all(|&c| c == 1),
                "K={k} grid_y={grid_y}: not a partition: {seen:?}"
            );
        }
    }
}

/// The 128 threads of one CTA cover the 8 groups' 1024 elements exactly once:
/// 16 lanes x 8 elements per group. This is the amax domain — a lane that
/// covered a element twice would fold it into the max twice (harmless) but
/// would also STORE it twice, and a lane that covered none would leave an FP8
/// byte unwritten.
#[test]
fn the_hopper_lane_map_covers_a_full_tile_exactly_once() {
    let span = FP8_QUANT_HOPPER_GROUPS_PER_CTA;
    let mut seen = vec![0u32; (span * 128) as usize];
    for tid in 0..128u32 {
        let (sub, lo, hi) = fp8_quant_hopper_lane(tid, span).expect("full tile: every tid is live");
        assert_eq!(hi - lo, 8, "tid {tid} must own 8 elements (one uint4)");
        for e in lo..hi {
            seen[(sub * 128 + e) as usize] += 1;
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "the 16x8 lane map is not a partition of the 8x128 tile"
    );
}

/// A PARTIAL tile — the CTA whose span is shorter than 8 groups, which is what
/// a K whose group count is not a multiple of 8 produces. Threads past the span
/// must be inert, and every element of the live groups must still be covered.
#[test]
fn a_partial_tile_leaves_the_spare_threads_inert() {
    for span in 1..FP8_QUANT_HOPPER_GROUPS_PER_CTA {
        let mut seen = vec![0u32; (span * 128) as usize];
        let mut inert = 0;
        for tid in 0..128u32 {
            match fp8_quant_hopper_lane(tid, span) {
                None => inert += 1,
                Some((sub, lo, hi)) => {
                    for e in lo..hi {
                        seen[(sub * 128 + e) as usize] += 1;
                    }
                }
            }
        }
        assert_eq!(
            inert,
            (128 - span * 16) as usize,
            "span {span}: wrong number of inert threads"
        );
        assert!(
            seen.iter().all(|&c| c == 1),
            "span {span}: live groups not covered exactly once"
        );
    }
}

/// The shared arm's grid is unchanged — one CTA per group, M on X. The Hopper
/// arm launches 8x fewer CTAs for the same work, which is the whole lever.
#[test]
fn the_shared_grid_is_untouched_and_the_hopper_grid_is_an_eighth_of_it() {
    for k in K_DIMS {
        for m in M_DIMS {
            let shared = fp8_quant_grid(false, m, k);
            let hopper = fp8_quant_grid(true, m, k);
            assert_eq!(
                shared,
                [m, k / 128, 1],
                "shared grid changed for M={m} K={k}"
            );
            assert_eq!(hopper[0], m);
            assert_eq!(hopper[2], 1);
            assert_eq!(hopper[1], (k / 128).div_ceil(8));
            assert!(hopper[1] < shared[1], "M={m} K={k}: no CTA reduction");
        }
    }
}

/// `Fp8ActQuant` picks the entry point and the grid from the SAME bit. A pair
/// that had a twin handle but a shared grid would quantize a fraction of each
/// row and leave the rest of the scratch stale — the failure this type exists
/// to make unrepresentable.
#[test]
fn the_pair_never_mixes_one_kernels_handle_with_the_others_grid() {
    let shared = Fp8ActQuant::shared_only(KernelHandle(0xA1));
    assert!(shared.available() && !shared.is_hopper());
    assert_eq!(shared.kernel().0, 0xA1);
    assert_eq!(shared.grid(1168, 5120), fp8_quant_grid(false, 1168, 5120));

    let twin = Fp8ActQuant {
        shared: KernelHandle(0xA1),
        hopper: KernelHandle(0xB2),
    };
    assert!(twin.available() && twin.is_hopper());
    assert_eq!(twin.kernel().0, 0xB2);
    assert_eq!(twin.grid(1168, 5120), fp8_quant_grid(true, 1168, 5120));

    assert!(!Fp8ActQuant::default().available());
    assert!(!Fp8ActQuant::shared_only(KernelHandle(0)).available());
}
