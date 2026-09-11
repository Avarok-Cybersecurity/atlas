// SPDX-License-Identifier: AGPL-3.0-only

//! The `attn_qkv_fused` row of the target table (#927).
//!
//! Split from `target_defaults_tests.rs` for the reason
//! `target_defaults_gateup_tests.rs` was: the parent is at the 500-line cap,
//! and a per-lever seam keeps each row's declaration, override and reported
//! spelling in one place instead of scattered through three whole-table tests.
//!
//! A child of `tests`, not a sibling, so `HOPPER`, `GB10`, `with`, `empty` and
//! `format_levers` come from the parent. A second copy of those fixtures is how
//! two files come to disagree about what Hopper declares.

use super::*;

/// The declaration, both ways round.
///
/// Hopper ships it ON without an accuracy receipt, and for the same reason the
/// gate+up row beside it does: splitting `N` gives INDEPENDENT output columns
/// over the same `K` with the same block scales, so the fused GEMM cannot
/// change a bit. What it attacks is nsys round 13's `k_proj` + `v_proj` — 32
/// graph nodes, 15.97 µs each, **328 GB/s = 9.8 % of HBM** — beside `q_proj`
/// at 65.3 % on the same arm in the same step.
///
/// GB10 declares it OFF: the arm is a call-shape change on the cuBLASLt W8A8
/// path and GB10 declares `cublas_gemm_scope = "off"`, so the arm it changes is
/// not even armed there. The row is declared anyway, so the lever list is one
/// list.
#[test]
fn hopper_arms_the_fused_qkv_gemm_and_gb10_does_not() {
    assert!(empty(&HOPPER).attn_qkv_fused.value);
    assert!(!empty(&GB10).attn_qkv_fused.value);
    // An empty environment sourced nothing: that is what "reproducible from
    // the binary" means for this row too.
    assert!(!empty(&HOPPER).attn_qkv_fused.from_env());
    assert!(!empty(&GB10).attn_qkv_fused.from_env());
}

/// The override, in both directions and under the whole 2026-09-11 grammar.
///
/// `=0` means OFF. There is no `ATLAS_NO_ATTN_QKV_FUSED`: the lever is new, so
/// no script predates the grammar and none can be surprised by it — which is
/// exactly why `0`, `false`, `off` and `no` all have to work, and why anything
/// else has to arm it.
#[test]
fn the_environment_overrides_the_row_in_both_directions() {
    for off in ["0", "false", "off", "no", "OFF", " 0 "] {
        assert_eq!(
            with(&HOPPER, &[("ATLAS_ATTN_QKV_FUSED", off)]).attn_qkv_fused,
            Resolved::env(false),
            "ATLAS_ATTN_QKV_FUSED={off:?} must kill the arm",
        );
    }
    for on in ["1", "true", "on", "yes", ""] {
        assert_eq!(
            with(&GB10, &[("ATLAS_ATTN_QKV_FUSED", on)]).attn_qkv_fused,
            Resolved::env(true),
            "ATLAS_ATTN_QKV_FUSED={on:?} must arm the A/B on a target that \
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
    assert!(line.contains("attn_qkv_fused=on"), "{line}");
    assert!(!line.contains("attn_qkv_fused=on (env)"), "{line}");
    let line = format_levers(&empty(&GB10));
    assert!(line.contains("attn_qkv_fused=off"), "{line}");
    let line = format_levers(&with(&HOPPER, &[("ATLAS_ATTN_QKV_FUSED", "0")]));
    assert!(line.contains("attn_qkv_fused=off (env)"), "{line}");
}

/// The two fusion levers are SEPARATE rows and neither spelling reaches the
/// other. They attack different arms with different receipts — 1 476 µs for
/// gate+up against 428 µs here — and an operator A/Bing one must not silently
/// move the other.
#[test]
fn the_two_fusion_levers_are_independent() {
    let l = with(&HOPPER, &[("ATLAS_ATTN_QKV_FUSED", "0")]);
    assert!(!l.attn_qkv_fused.value);
    assert!(l.ffn_gateup_fused.value, "gate+up is untouched");
    let l = with(&HOPPER, &[("ATLAS_FFN_GATEUP_FUSED", "0")]);
    assert!(!l.ffn_gateup_fused.value);
    assert!(l.attn_qkv_fused.value, "q/k/v is untouched");
}

/// The umbrella `ATLAS_M16_TC` arms the two M16 tensor-core tiers and NOTHING
/// else. Pinned here because this row sits beside them in the table and an
/// umbrella that quietly grew would move an arm nobody asked it to.
#[test]
fn the_m16_umbrella_does_not_reach_this_row() {
    assert!(!with(&GB10, &[("ATLAS_M16_TC", "1")]).attn_qkv_fused.value);
}
