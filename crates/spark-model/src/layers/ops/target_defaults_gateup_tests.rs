// SPDX-License-Identifier: AGPL-3.0-only

//! The `ffn_gateup_fused` row of the target table (#927).
//!
//! Split from `target_defaults_tests.rs` for the reason
//! the parent is near the 500-line cap,
//! and a per-lever seam keeps each row's declaration, override and reported
//! spelling in one place instead of scattered through three whole-table tests.
//!
//! A child of `tests`, not a sibling, so `HOPPER`, `GB10`, `with`, `empty` and
//! `format_levers` come from the parent. A second copy of those fixtures is how
//! two files come to disagree about what Hopper declares.

use super::*;

/// The declaration, both ways round.
///
/// Hopper ships it ON without an accuracy receipt, and for a stronger reason
/// than the `ssm_ba_gates_hopper` row beside it: splitting `N` gives
/// INDEPENDENT output columns over the same `K` with the same block scales, so
/// the fused GEMM cannot change a bit. What it attacks is nsys round 13's
/// 5 730.5 µs of gate+up GEMM per n=16 step at 59.4 % of HBM — beside the same
/// arm's `down`, which moves the same 89.1 MB per layer in ONE launch at
/// 71.4 %.
///
/// GB10 declares it OFF: the arm is a call-shape change on the cuBLASLt W8A8
/// path and GB10 declares `cublas_gemm_scope = "off"`, so the arm it changes is
/// not even armed there. The row is declared anyway, so the lever list is one
/// list.
#[test]
fn hopper_arms_the_fused_gate_up_gemm_and_gb10_does_not() {
    assert!(empty(&HOPPER).ffn_gateup_fused.value);
    assert!(!empty(&GB10).ffn_gateup_fused.value);
    // An empty environment sourced nothing: that is what "reproducible from
    // the binary" means for this row too.
    assert!(!empty(&HOPPER).ffn_gateup_fused.from_env());
    assert!(!empty(&GB10).ffn_gateup_fused.from_env());
}

/// The override, in both directions and under the whole 2026-09-11 grammar.
///
/// `=0` means OFF. There is no `ATLAS_NO_FFN_GATEUP_FUSED`: the lever is new,
/// so no script predates the grammar and none can be surprised by it — which
/// is exactly why `0`, `false`, `off` and `no` all have to work, and why
/// anything else has to arm it.
#[test]
fn the_environment_overrides_the_row_in_both_directions() {
    for off in ["0", "false", "off", "no", "OFF", " 0 "] {
        assert_eq!(
            with(&HOPPER, &[("ATLAS_FFN_GATEUP_FUSED", off)]).ffn_gateup_fused,
            Resolved::env(false),
            "ATLAS_FFN_GATEUP_FUSED={off:?} must kill the arm",
        );
    }
    for on in ["1", "true", "on", "yes", ""] {
        assert_eq!(
            with(&GB10, &[("ATLAS_FFN_GATEUP_FUSED", on)]).ffn_gateup_fused,
            Resolved::env(true),
            "ATLAS_FFN_GATEUP_FUSED={on:?} must arm the A/B on a target that \
             declares it off",
        );
    }
}

/// The serve line reports the row, in both polarities and with the `(env)` tag
/// when an operator moved it. A lever the boot line does not name is a lever
/// that can be off for a whole campaign without anyone noticing.
#[test]
fn the_serve_line_names_the_row_and_marks_an_override() {
    let line = format_levers(&empty(&HOPPER));
    assert!(line.contains("ffn_gateup_fused=on"), "{line}");
    assert!(!line.contains("ffn_gateup_fused=on (env)"), "{line}");
    let line = format_levers(&empty(&GB10));
    assert!(line.contains("ffn_gateup_fused=off"), "{line}");
    let line = format_levers(&with(&HOPPER, &[("ATLAS_FFN_GATEUP_FUSED", "0")]));
    assert!(line.contains("ffn_gateup_fused=off (env)"), "{line}");
}
