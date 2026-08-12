// SPDX-License-Identifier: AGPL-3.0-only

//! The grouped-GEMM MoE decode arm is a WIDTH trade, and the gate that
//! decides it is the whole fix (#415 follow-up).
//!
//! Before this, the arm was gated on `ATLAS_MOE_GROUPED_DECODE=1` alone —
//! width-blind. That is wrong in both directions and each direction has a
//! measurement behind it:
//!
//! - env set, n=4: ran, and LOSES badly (31 vs 56 tok/s on Holo)
//! - env unset, n=32: did NOT run, forgoing a measured +25% (SSM-side C=32
//!   172.7 -> 216.2) and #415's +7.9%/+9.7%
//!
//! and since no recipe ever set the variable, the shipped behaviour was the
//! second one everywhere: the win existed and was never realised.
//!
//! Tests target the PURE decider so neither polarity depends on process env
//! or on `OnceLock` latch order.

use super::{moe_grouped_decode_decide, moe_grouped_decode_min_rows};

/// POSITIVE: at and above the measured-winning width the arm engages.
///
/// PROVEN BY: reverting the gate to the width-blind `enabled` test does NOT
/// turn this red (it engages there too) — which is exactly why the negative
/// below is the load-bearing one, and why this test alone would have passed
/// over the old behaviour.
#[test]
fn the_grouped_arm_engages_at_and_above_the_measured_width() {
    let w = moe_grouped_decode_min_rows();
    assert!(
        moe_grouped_decode_decide(w, true, false),
        "n={w} is the smallest width measured on the winning side (+25%); \
         the arm must engage there"
    );
    assert!(
        moe_grouped_decode_decide(w * 4, true, false),
        "wider is more favourable, not less — the amortisation only improves"
    );
}

/// NEGATIVE: below the measured width the arm must NOT engage.
///
/// This is the one that pins the fix. `n=4` is measured at 31 vs 56 tok/s —
/// a ~45% loss — so an arm that engages there is a regression shipped as a
/// feature.
///
/// PROVEN BY: restoring the old width-blind gate (return `enabled` alone)
/// turns this RED at n=4 and n=15, while every other assertion in this file
/// stays green.
#[test]
fn the_grouped_arm_stays_off_below_the_measured_width() {
    for n in [1usize, 2, 4, 8, moe_grouped_decode_min_rows() - 1] {
        assert!(
            !moe_grouped_decode_decide(n, true, false),
            "n={n} is below the measured-winning width; at n=4 this arm is a \
             ~45% LOSS (31 vs 56 tok/s), and n=5..15 is unmeasured — the gate \
             must fall to the per-token loop"
        );
    }
}

/// NEGATIVE: the kill switch beats the width, at every width.
///
/// Without this, "disabled" could be quietly ignored above the threshold —
/// the failure mode where a kill switch exists but does not kill.
#[test]
fn the_kill_switch_wins_over_any_width() {
    for n in [1usize, 4, 16, 64, 1024] {
        assert!(
            !moe_grouped_decode_decide(n, false, false),
            "kill switch set: the arm must not engage at n={n} regardless of width"
        );
        assert!(
            !moe_grouped_decode_decide(n, false, true),
            "kill switch must also beat the diagnostic force at n={n}, or the \
             two knobs contradict each other"
        );
    }
}

/// POSITIVE: the diagnostic force reaches BELOW the threshold — that is its
/// only purpose (measuring the unmeasured n=5..15 gap without a rebuild).
///
/// PROVEN BY: dropping `|| forced` from the decider turns this red, which is
/// what would silently strand #415's original measurement recipe.
#[test]
fn the_force_override_reaches_below_the_threshold() {
    assert!(
        moe_grouped_decode_decide(4, true, true),
        "ATLAS_MOE_GROUPED_DECODE=1 exists so a below-threshold width can be \
         measured; if it cannot reach n=4 it cannot measure the loss it was \
         used to find"
    );
}

/// The threshold is the smallest MEASURED winning width, not a round number.
///
/// Pinned because the comment justifying 16 cites a specific measurement
/// (SSM-side C=32 +25%, #415 attention-side +7.9%/+9.7%); moving the constant
/// without a new measurement should require editing this assertion and
/// noticing why.
#[test]
fn the_threshold_is_the_measured_width() {
    assert_eq!(
        moe_grouped_decode_min_rows(),
        16,
        "16 is the smallest width measured on the winning side; changing it \
         needs a measurement, not a preference"
    );
}
