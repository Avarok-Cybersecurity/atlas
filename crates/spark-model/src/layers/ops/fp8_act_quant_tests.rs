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
// The floor, the reject grammar and the log slots live in the sibling module
// `fp8_act_quant_floor.rs`, so `super` (= `fp8_act_quant`) does not carry
// them; they are reached through `layers::ops`, which re-exports both halves.
use crate::layers::ops::{
    FP8_QUANT_LOG_SLOTS, FP8_QUANT_MIN_CTAS_PER_SM, FP8_QUANT_REJECTS, FP8_QUANT_TOO_FEW_CTAS,
    Fp8QuantLogSlot, fp8_quant_hopper_ctas, fp8_quant_hopper_min_m, fp8_quant_log_slot,
    fp8_quant_min_ctas,
};

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
    assert!(shared.available() && !shared.twin_present());
    assert_eq!(shared.kernel(1168, 5120).0, 0xA1);
    assert_eq!(shared.grid(1168, 5120), fp8_quant_grid(false, 1168, 5120));

    let twin = PAIR;
    assert!(twin.available() && twin.twin_present());
    for (m, k) in [(1168, 5120), (16, 5120)] {
        let pick = twin.pick_with(true, m, k, SM);
        assert_eq!(pick.kernel.0, if pick.twin { 0xB2 } else { 0xA1 });
        assert_eq!(pick.grid, fp8_quant_grid(pick.twin, m, k));
    }

    assert!(!Fp8ActQuant::default().available());
    assert!(!Fp8ActQuant::shared_only(KernelHandle(0)).available());
}

// ── The width floor (#928, round-16 receipt §2.1, Recommendation 2) ────────
//
// Round 16 selected the twin by kernel PRESENCE and measured it 0.76×–0.95× at
// six of the fifteen arms below — every decode width at K ∈ {5120, 6144}. The
// floor is what turns that measurement into a route, so these tests grade the
// route AT THE MEASURED POINTS rather than at invented ones.

/// The compiled target's SM count, as the H100 the arms were measured on.
const SM: u32 = 132;

/// A pair with both handles, which is what a Hopper image resolves.
const PAIR: Fp8ActQuant = Fp8ActQuant {
    shared: KernelHandle(0xA1),
    hopper: KernelHandle(0xB2),
};

/// The microtest's fifteen (M, K) arms and the arm each one MEASURED faster,
/// verbatim from the round-16 receipt §2.1. `true` = the Hopper twin won.
///
/// This table is the test's oracle. It is a transcription of a measurement, so
/// a change to the floor that moves any row has to argue with a number.
const MEASURED: [(u32, u32, bool); 15] = [
    // K = 5120: twin 0.84×, 0.95×, 0.76× at the decode widths.
    (16, 5120, false),
    (17, 5120, false),
    (25, 5120, false),
    (1168, 5120, true), // 3.37×
    (4576, 5120, true), // 3.41×
    // K = 6144: twin 0.82×, 0.80×, 0.87×.
    (16, 6144, false),
    (17, 6144, false),
    (25, 6144, false),
    (1168, 6144, true), // 3.53×
    (4576, 6144, true), // 3.44×
    // K = 17408: the twin is ALREADY ahead at M=16 (1.02×, 1.02×, 1.08×),
    // because 136 groups is a 17-wide grid Y and 16 × 17 = 272 ≥ 264 CTAs.
    (16, 17408, true),
    (17, 17408, true),
    (25, 17408, true),
    (1168, 17408, true), // 3.30×
    (4576, 17408, true), // 3.59×
];

/// The floor routes every measured arm to the kernel that measured faster.
///
/// Not "the floor is 264" — that is arithmetic. This is the claim the lever
/// makes: at each of the fifteen points the receipt priced, the engine now
/// launches the arm the receipt says is quicker.
#[test]
fn the_floor_routes_every_measured_arm_to_the_faster_kernel() {
    for (m, k, twin_won) in MEASURED {
        let pick = PAIR.pick_with(true, m, k, SM);
        assert_eq!(
            pick.twin,
            twin_won,
            "M={m} K={k}: routed to the {} but round 16 measured the {} faster \
             ({} CTAs against a floor of {})",
            if pick.twin { "twin" } else { "parent" },
            if twin_won { "twin" } else { "parent" },
            fp8_quant_hopper_ctas(m, k),
            fp8_quant_min_ctas(SM),
        );
        // Whichever arm won, the handle and the grid came from one decision.
        assert_eq!(pick.kernel.0, if twin_won { 0xB2 } else { 0xA1 });
        assert_eq!(pick.grid, fp8_quant_grid(twin_won, m, k));
        assert_eq!(pick.reject.is_none(), twin_won);
    }
}

/// The per-K thresholds the rule produces on a 132-SM part, and the fact that
/// they BRACKET the measurement: one M below each is a parent arm, the
/// threshold itself is a twin arm.
///
/// K=17408's threshold is 16, which is why the three K=17408 decode arms stay
/// on the twin — they measured 1.02×–1.08×, so routing them to the parent
/// would give back a win for nothing. A flat token floor could not express
/// that: the same M is a different amount of machine at a different K.
#[test]
fn the_per_k_thresholds_are_the_documented_ones() {
    for (k, min_m) in [(5120_u32, 53_u32), (6144, 44), (17408, 16)] {
        assert_eq!(
            fp8_quant_hopper_min_m(k, SM),
            min_m,
            "K={k}: grid Y is {}, floor {} CTAs",
            fp8_quant_grid(true, 1, k)[1],
            fp8_quant_min_ctas(SM),
        );
        assert!(PAIR.pick_with(true, min_m, k, SM).twin, "K={k} M={min_m}");
        assert!(
            !PAIR.pick_with(true, min_m - 1, k, SM).twin,
            "K={k} M={}: the threshold must be the smallest accepted M",
            min_m - 1
        );
    }
}

/// Each guard, once, and the reason string it returns — so a serve log's
/// refusal names a guard this table knows about.
#[test]
fn every_reject_reason_has_its_own_log_slot() {
    let cases = [
        ("not requested", PAIR.pick_with(false, 4576, 5120, SM)),
        (
            "kernel absent from this image (kernels/hopper only)",
            Fp8ActQuant::shared_only(KernelHandle(0xA1)).pick_with(true, 4576, 5120, SM),
        ),
        (FP8_QUANT_TOO_FEW_CTAS, PAIR.pick_with(true, 16, 5120, SM)),
    ];
    for (why, pick) in cases {
        assert!(!pick.twin);
        assert_eq!(pick.reject, Some(why));
        assert!(
            FP8_QUANT_REJECTS.contains(&why),
            "{why:?} has no slot in FP8_QUANT_REJECTS"
        );
        let slot = fp8_quant_log_slot(&pick);
        // A refusal is only worth a line when the lever ASKED for the twin.
        if pick.requested {
            assert_eq!(
                slot,
                Some(Fp8QuantLogSlot::Reject(
                    FP8_QUANT_REJECTS.iter().position(|r| *r == why).unwrap()
                ))
            );
        } else {
            assert_eq!(slot, None, "the lever is off: the parent is the answer");
        }
    }
    assert_eq!(
        fp8_quant_log_slot(&PAIR.pick_with(true, 4576, 5120, SM)),
        Some(Fp8QuantLogSlot::Twin)
    );
}

/// A serve says BOTH lines, each at the width it was reached at.
///
/// Round 15 shipped the BA-gates twin with ONE once-flag shared by both
/// branches; a 27-token smoke request arrived first and the positive line
/// could never print, so five serve logs claimed the twin was off while nsys
/// showed it running 48× per prefill. This lever's verdict changes with M for
/// the same reason, so it gets the slot table from the start: replaying a real
/// call order (smoke → decode → prefill) must land in two distinct slots.
#[test]
fn a_decode_then_prefill_serve_fills_two_distinct_slots() {
    let slots: Vec<_> = [(27, 5120), (16, 5120), (4576, 5120), (1168, 6144)]
        .into_iter()
        .filter_map(|(m, k)| fp8_quant_log_slot(&PAIR.pick_with(true, m, k, SM)))
        .collect();
    let reject = Fp8QuantLogSlot::Reject(
        FP8_QUANT_REJECTS
            .iter()
            .position(|r| *r == FP8_QUANT_TOO_FEW_CTAS)
            .unwrap(),
    );
    assert_eq!(
        slots,
        vec![reject, reject, Fp8QuantLogSlot::Twin, Fp8QuantLogSlot::Twin],
        "the negative and the positive must not share a once-flag"
    );
    assert_eq!(FP8_QUANT_LOG_SLOTS, FP8_QUANT_REJECTS.len() + 1);
}

/// With no parent in the pair there is nothing to decline TO, so the twin
/// takes every width.
///
/// This is the shape `native_fp8_act_quant_hopper_microtest` builds to force
/// each arm (`shared: KernelHandle(0)`), and it is why the microtest still
/// measures the twin at M ∈ {16, 17, 25} — the six arms whose numbers opened
/// this lever — after the floor lands. Without it the harness would launch
/// handle 0 at exactly the points it exists to price.
#[test]
fn a_pair_with_no_parent_keeps_the_twin_at_every_width() {
    let only_twin = Fp8ActQuant {
        shared: KernelHandle(0),
        hopper: KernelHandle(0xB2),
    };
    for (m, k, _) in MEASURED {
        let pick = only_twin.pick_with(true, m, k, SM);
        assert!(pick.twin && pick.reject.is_none(), "M={m} K={k}");
        assert_eq!(pick.kernel.0, 0xB2);
    }
    // …and the LEVER does not change that: `ATLAS_FP8_ACT_QUANT_HOPPER=0` on a
    // pair with no parent would otherwise launch `KernelHandle(0)`. The escape
    // outranks the lever for the same reason it outranks the floor — an
    // operator's "prefer the parent" cannot mean "launch nothing".
    assert!(only_twin.pick_with(false, 4576, 5120, SM).twin);
    // Which is a property of the PAIR, not of the lever: give it a parent and
    // the same lever setting declines the twin at every width.
    assert!(!PAIR.pick_with(false, 4576, 5120, SM).twin);
}

/// The floor scales with the device rather than pinning H100's numbers, and a
/// nonsense SM count cannot make it divide by zero.
#[test]
fn the_floor_follows_the_sm_count() {
    assert_eq!(fp8_quant_min_ctas(132), 264);
    assert_eq!(fp8_quant_min_ctas(148), 296);
    assert_eq!(fp8_quant_min_ctas(0), FP8_QUANT_MIN_CTAS_PER_SM);
    // 148 SMs (B200's count) lifts every threshold, which is one reason that
    // target declares the row false rather than inheriting Hopper's.
    assert!(fp8_quant_hopper_min_m(5120, 148) > fp8_quant_hopper_min_m(5120, 132));
    // K below one group: grid Y is clamped to 1, so the CTA count is M and the
    // arithmetic stays defined.
    assert_eq!(fp8_quant_hopper_ctas(64, 64), 64);
    assert_eq!(fp8_quant_hopper_min_m(64, 132), 264);
}
