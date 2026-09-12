// SPDX-License-Identifier: AGPL-3.0-only

//! The `fp8_act_quant_hopper` row of the target table (#928, round-16 receipt
//! §2.1, Recommendation 2).
//!
//! Split from `target_defaults_tests.rs` for the reason
//! `target_defaults_gateup_tests.rs` was: the parent is at the 500-line cap,
//! and a per-lever seam keeps each row's declaration, override and reported
//! spelling in one place.
//!
//! A child of `tests`, not a sibling, so `HOPPER`, `GB10`, `with`, `empty` and
//! `format_levers` come from the parent. The FLOOR this row arms is graded
//! separately, in `fp8_act_quant_tests.rs`, against the microtest's fifteen
//! (M, K) points — this file grades only the row.

use super::*;

/// The declaration, both ways round.
///
/// Hopper ships it ON with no accuracy receipt because there is no accuracy
/// question: the twin's output is bit-identical to its parent's. The claim is
/// GB/s — 63.7–68.4 % of HBM at M ∈ {1168, 4576} against the parent's
/// 18.6–19.1 %, 3.30–3.59× — and it is a claim the row alone does not qualify,
/// which is why the arm also carries a CTA floor.
///
/// GB10 declares it OFF: `fp8_act_quant_hopper.cu` lives under
/// `kernels/hopper` only, so the lookup returns 0 there and the row is inert.
/// It is declared anyway, so the lever list is one list.
#[test]
fn hopper_arms_the_fp8_act_quant_twin_and_gb10_does_not() {
    assert!(empty(&HOPPER).fp8_act_quant_hopper.value);
    assert!(!empty(&GB10).fp8_act_quant_hopper.value);
    // An empty environment sourced nothing: "reproducible from the binary".
    assert!(!empty(&HOPPER).fp8_act_quant_hopper.from_env());
    assert!(!empty(&GB10).fp8_act_quant_hopper.from_env());
}

/// The override, in both directions and under the whole 2026-09-11 grammar.
///
/// `=0` means OFF, and it is the kill switch round 16 did not have: the twin
/// was selected by kernel PRESENCE, so an operator watching a decode step take
/// the 0.76× arm had no way to decline it short of rebuilding without the
/// file. There is no `ATLAS_NO_FP8_ACT_QUANT_HOPPER` — the lever is new, so no
/// script predates the grammar and none can be surprised by it.
#[test]
fn the_environment_overrides_the_row_in_both_directions() {
    for off in ["0", "false", "off", "no", "OFF", " 0 "] {
        assert_eq!(
            with(&HOPPER, &[("ATLAS_FP8_ACT_QUANT_HOPPER", off)]).fp8_act_quant_hopper,
            Resolved::env(false),
            "ATLAS_FP8_ACT_QUANT_HOPPER={off:?} must kill the twin",
        );
    }
    for on in ["1", "true", "on", "yes", ""] {
        assert_eq!(
            with(&GB10, &[("ATLAS_FP8_ACT_QUANT_HOPPER", on)]).fp8_act_quant_hopper,
            Resolved::env(true),
            "ATLAS_FP8_ACT_QUANT_HOPPER={on:?} must arm the A/B on a target \
             that declares it off",
        );
    }
}

/// The serve line reports the row, in both polarities and with the `(env)` tag
/// when an operator moved it.
///
/// Load-bearing for this lever in particular. Round 16's binary had NO boot
/// line naming the quantizer at all — selection was by presence — so the only
/// way to know which arm a serve ran was to attach nsys. The line is half the
/// observability fix; the once-per-branch route line in
/// `fp8_act_quant_floor.rs` is the other half.
#[test]
fn the_serve_line_names_the_row_and_marks_an_override() {
    let line = format_levers(&empty(&HOPPER));
    assert!(line.contains("fp8_act_quant_hopper=on"), "{line}");
    assert!(!line.contains("fp8_act_quant_hopper=on (env)"), "{line}");
    let line = format_levers(&empty(&GB10));
    assert!(line.contains("fp8_act_quant_hopper=off"), "{line}");
    let line = format_levers(&with(&HOPPER, &[("ATLAS_FP8_ACT_QUANT_HOPPER", "0")]));
    assert!(line.contains("fp8_act_quant_hopper=off (env)"), "{line}");
}

/// The umbrella `ATLAS_M16_TC` arms the two M16 tensor-core tiers and NOTHING
/// else. Pinned here for the same reason it is pinned for the row beside it.
#[test]
fn the_m16_umbrella_does_not_reach_this_row() {
    assert!(
        !with(&GB10, &[("ATLAS_M16_TC", "1")])
            .fp8_act_quant_hopper
            .value
    );
}
