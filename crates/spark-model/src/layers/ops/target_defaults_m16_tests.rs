// SPDX-License-Identifier: AGPL-3.0-only

//! The M16 tensor-core family of the target table (#927): `ffn_m16_tc`,
//! `attn_m16_tc`, `lm_head_m16_tc`, `attn_ncol_gemv` and the `ATLAS_M16_TC`
//! umbrella.
//!
//! Split from `target_defaults_tests.rs` because the parent crossed the house
//! 500-line cap when these rows landed, and a per-family seam keeps each row's
//! declaration, override and reported spelling in one place.
//!
//! A child of `tests`, not a sibling, so `HOPPER`, `GB10`, `with`, `empty` and
//! `format_levers` come from the parent. A second copy of those fixtures is how
//! two files come to disagree about what Hopper declares.

use super::*;

/// Round 6 measured the dense-FFN arm of `w8a16_gemm_m16` as a LOSS (C=16
/// aggregate −5.2%), so Hopper declares it OFF and an H100 serve with an empty
/// environment does not run it.
#[test]
fn hopper_leaves_the_ffn_tensor_core_arm_off_by_declaration() {
    assert!(!empty(&HOPPER).ffn_m16_tc.value);
    assert!(!empty(&HOPPER).ffn_m16_tc.from_env());
    assert!(!empty(&GB10).ffn_m16_tc.value);
}

/// `ATLAS_FFN_M16_TC` is the A/B that re-runs it, in BOTH directions, and says
/// it came from the environment.
#[test]
fn the_ffn_tensor_core_arm_is_overridable_in_both_directions() {
    let on = with(&HOPPER, &[("ATLAS_FFN_M16_TC", "1")]);
    assert!(on.ffn_m16_tc.value && on.ffn_m16_tc.from_env());
    let armed = TargetDefaults {
        ffn_m16_tc: true,
        ..HOPPER
    };
    let off = with(&armed, &[("ATLAS_FFN_M16_TC", "0")]);
    assert!(!off.ffn_m16_tc.value && off.ffn_m16_tc.from_env());
}

/// `ATLAS_M16_TC` is round 6's umbrella and still arms this arm — folded in by
/// the RESOLVER, so it composes with the narrow variable rather than racing it.
#[test]
fn the_m16_umbrella_arms_the_ffn_arm_too() {
    let on = with(&HOPPER, &[("ATLAS_M16_TC", "1")]);
    assert!(on.ffn_m16_tc.value && on.ffn_m16_tc.from_env());
    // The narrow variable WINS when both are set, so `ATLAS_FFN_M16_TC=0
    // ATLAS_M16_TC=1` means what it reads as rather than depending on export
    // order.
    let narrow_off = with(&HOPPER, &[("ATLAS_FFN_M16_TC", "0"), ("ATLAS_M16_TC", "1")]);
    assert!(!narrow_off.ffn_m16_tc.value);
}

/// The serve log names the row, so an operator can tell a target default from
/// an `(env)` override without reading the recipe.
#[test]
fn the_serve_line_names_the_ffn_tensor_core_row() {
    assert!(format_levers(&empty(&HOPPER)).contains("ffn_m16_tc=off"));
    assert!(
        format_levers(&with(&HOPPER, &[("ATLAS_FFN_M16_TC", "1")])).contains("ffn_m16_tc=on (env)")
    );
}

/// The attention half of the SAME kernel family goes the other way: round 9
/// cell W measured +5.3% C=16 aggregate, so Hopper declares it ON and an H100
/// serve with an empty environment runs it.
#[test]
fn hopper_arms_the_attention_tensor_core_tiers_by_declaration() {
    let h = empty(&HOPPER);
    assert!(h.attn_m16_tc.value && !h.attn_m16_tc.from_env());
    assert!(!h.ffn_m16_tc.value, "the two rows are independent");
    assert!(!empty(&GB10).attn_m16_tc.value);
}

/// `ATLAS_ATTN_M16_TC=0` is the one-variable A/B that pins the parent tiers,
/// and it reports that it came from the environment.
#[test]
fn the_attention_tiers_are_disarmable_from_the_environment() {
    let off = with(&HOPPER, &[("ATLAS_ATTN_M16_TC", "0")]);
    assert!(!off.attn_m16_tc.value && off.attn_m16_tc.from_env());
    assert!(format_levers(&off).contains("attn_m16_tc=off (env)"));
}

/// The umbrella arms this half too — and cannot disarm a declaration, which is
/// why it is folded in here and not at the consumer.
#[test]
fn the_m16_umbrella_arms_both_halves() {
    let both = with(&GB10, &[("ATLAS_M16_TC", "1")]);
    assert!(both.ffn_m16_tc.value && both.attn_m16_tc.value);
    assert!(both.ffn_m16_tc.from_env() && both.attn_m16_tc.from_env());
}

/// The BF16 decode head's tensor-core arm: ON for Hopper on round 9 cell Y
/// (+4.09% C=16 aggregate), and NOT reachable through the round-6 umbrella,
/// which predates the arm and never measured it.
#[test]
fn hopper_arms_the_tensor_core_head_and_the_umbrella_does_not() {
    assert!(empty(&HOPPER).lm_head_m16_tc.value);
    assert!(!empty(&GB10).lm_head_m16_tc.value);
    let umbrella = with(&GB10, &[("ATLAS_M16_TC", "1")]);
    assert!(
        !umbrella.lm_head_m16_tc.value,
        "ATLAS_M16_TC is round 6's, and round 6 did not measure the head"
    );
    let off = with(&HOPPER, &[("ATLAS_LM_HEAD_M16_TC", "0")]);
    assert!(!off.lm_head_m16_tc.value && off.lm_head_m16_tc.from_env());
}

/// The N-column GEMV row is OFF on every target, and the row says why: no
/// serving A/B exists for it anywhere. `ATLAS_ATTN_NCOL_GEMV` runs that A/B.
#[test]
fn the_ncol_gemv_row_is_off_everywhere_and_armable() {
    assert!(!empty(&HOPPER).attn_ncol_gemv.value);
    assert!(!empty(&GB10).attn_ncol_gemv.value);
    let on = with(&HOPPER, &[("ATLAS_ATTN_NCOL_GEMV", "1")]);
    assert!(on.attn_ncol_gemv.value && on.attn_ncol_gemv.from_env());
}

/// The pre-existing family kill switch OUTRANKS both the declaration and the
/// positive variable — a switch that turns a family off must not be silently
/// narrowed by a new row underneath it.
#[test]
fn the_attention_decode_batch_kill_switch_outranks_the_row() {
    let armed = TargetDefaults {
        attn_ncol_gemv: true,
        ..HOPPER
    };
    for env in [
        vec![("ATLAS_NO_ATTN_DECODE_BATCH", "1")],
        vec![
            ("ATLAS_NO_ATTN_DECODE_BATCH", "1"),
            ("ATLAS_ATTN_NCOL_GEMV", "1"),
        ],
    ] {
        let l = with(&armed, &env);
        assert!(!l.attn_ncol_gemv.value, "{env:?}");
        assert!(l.attn_ncol_gemv.from_env(), "{env:?}");
    }
}
