// SPDX-License-Identifier: AGPL-3.0-only

//! Lever resolution tests: the polarity of every switch.
//!
//! Split out of `model_levers.rs` to keep it under the repository's
//! 500-LoC cap, the same pattern `mtp_carry_tests.rs` uses. The source-level
//! guards that keep these reads off hot paths live in
//! `hot_path_env_guards.rs`, because they guard other modules too.

use super::*;
// The production resolver, driven directly rather than copied. `resolve` is
// private to `model_levers`; this module is its child, so it can reach in.
use super::resolve::from_values;
use std::collections::HashMap;

/// Resolve against a fixed map instead of the process environment.
///
/// `set_var` is unsafe and process-global, so a test that mutated the
/// environment would race every other test in this binary. Driving
/// `from_values` directly exercises the PRODUCTION resolution rather than a
/// copy of it.
fn resolve(values: &[(&str, &str)]) -> ModelLevers {
    let values: HashMap<_, _> = values.iter().copied().collect();
    from_values(
        |name| values.get(name).map(|value| (*value).to_owned()),
        |name| values.contains_key(name),
        0,
        crate::model::drafter_context::DrafterContext::BOTH,
        0.0,
    )
}

/// The levers this branch lifted off hot paths. A child module so it shares
/// `resolve` above instead of copying it.
#[path = "model_levers_hot_path_tests.rs"]
mod hot_path_levers;

#[test]
fn the_opt_out_lever_is_on_by_default_and_every_opt_in_is_off() {
    let d = ModelLevers::defaults();
    assert_eq!(
        resolve(&[]),
        d,
        "absent environment uses the public default"
    );
    assert_eq!(
        d,
        ModelLevers {
            gdn_regresident: true,
            gdn_wy17: true,
            gdn_wyn: true,
            gemv_sw: true,
            ffn_small_m: true,
            // The SSM batch-4 GEMV tier is the sixth opt-out lever. This
            // literal is spelled out rather than derived so that adding a
            // lever forces an author to state its polarity HERE, in the
            // test, instead of inheriting whatever `Default` gives.
            ssm_gemv_batch4: true,
            // The dense-FFN opt-outs. Each ships ON and is disabled by the
            // PRESENCE of its variable — spelled out here so that adding a
            // lever forces an author to state its polarity in the test
            // rather than inherit whatever `Default` gives.
            decode_split_silu: true,
            ffn_nvfp4_mmq: true,
            ffn_nvfp4_mmq_down: true,
            prefill_v2: true,
            max_decode_seqs: 1,
            drafter: crate::model::drafter_context::DrafterContext::BOTH,
            ..ModelLevers::default()
        }
    );
}

#[test]
fn exact_one_opt_ins_map_to_their_own_fields() {
    let cases = [
        ("ATLAS_KV_POISON", [true, false, false, false, false, false]),
        (
            "ATLAS_GDN_BATCHED_FLA",
            [false, true, false, false, false, false],
        ),
        (
            "ATLAS_DECODE_FFN_VIA_GEMM",
            [false, false, true, false, false, false],
        ),
        (
            "ATLAS_MOE_UNION_STATS",
            [false, false, false, true, false, false],
        ),
        (
            "ATLAS_DFLASH_CONTIG_ATTN",
            [false, false, false, false, true, false],
        ),
        ("ATLAS_K4_DIAG", [false, false, false, false, false, true]),
    ];
    for (name, expected) in cases {
        let d = resolve(&[(name, "1")]);
        assert_eq!(
            [
                d.kv_poison,
                d.gdn_batched_fla,
                d.decode_ffn_via_gemm,
                d.moe_union_stats,
                d.dflash_contig_attn,
                d.k4_diag
            ],
            expected,
            "{name}"
        );
    }
    assert!(!resolve(&[("ATLAS_K4_DIAG", "true")]).k4_diag);
}

#[test]
fn truthy_opt_ins_map_independently_and_presence_is_distinct() {
    let cases = [
        (
            "ATLAS_HOLO_MOE_DOWN_FP4",
            [true, false, false, false, false],
        ),
        (
            "ATLAS_HOLO_MOE_GATEUP_FP4",
            [false, true, false, false, false],
        ),
        ("ATLAS_LORA_EAGER", [false, false, true, false, false]),
        ("ATLAS_LORA_ROTATE", [false, false, false, true, false]),
        ("ATLAS_DIAG_GEMMA4", [false, false, false, false, true]),
    ];
    for (name, expected) in cases {
        let d = resolve(&[(name, "TrUe")]);
        assert_eq!(
            [
                d.holo_moe_down_fp4,
                d.holo_moe_gateup_fp4,
                d.lora_eager,
                d.lora_rotate,
                d.gemma4_diag
            ],
            expected,
            "{name}"
        );
    }
    assert!(resolve(&[("ATLAS_BF16_TC_PROJ", "0")]).bf16_tc_proj);
    // `TQ_PLUS_WEIGHT_ROTATION` is VALUE-gated, not presence-gated — the
    // opposite of the line above. All five former implementations agreed
    // on `=1`-or-`true`, and this pins that the consolidation kept it.
    assert!(!resolve(&[]).weight_pre_rotated);
    assert!(resolve(&[("TQ_PLUS_WEIGHT_ROTATION", "1")]).weight_pre_rotated);
    assert!(resolve(&[("TQ_PLUS_WEIGHT_ROTATION", "TRUE")]).weight_pre_rotated);
    assert!(!resolve(&[("TQ_PLUS_WEIGHT_ROTATION", "0")]).weight_pre_rotated);

    // ★ THE SSM DECODE FIVE, AND THEIR POLARITIES DIFFER. Three are opt-in
    // diagnostics, one ships ON and opts out with `=0`, and the fifth
    // stores the POSITIVE of a variable whose call site reads the negative.
    // Getting any of these backwards silently changes which kernel runs on
    // the decode path, so the defaults are pinned explicitly.
    let d = resolve(&[]);
    assert!(!d.ssm_ms_profile, "profiling is off unless asked for");
    assert!(!d.ssm_detail_profile);
    assert!(!d.gdn_fused_conv);
    assert!(!d.moe_legacy_pertoken_decode, "default is token-major MoE");
    assert!(d.ssm_gemv_batch4, "batch-4 GEMV ships ON");

    assert!(resolve(&[("ATLAS_SSM_MS_PROFILE", "1")]).ssm_ms_profile);
    assert!(resolve(&[("ATLAS_SSM_DETAIL_PROFILE", "1")]).ssm_detail_profile);
    assert!(resolve(&[("ATLAS_GDN_FUSED_CONV", "1")]).gdn_fused_conv);
    assert!(resolve(&[("ATLAS_MOE_LEGACY_PERTOKEN_DECODE", "1")]).moe_legacy_pertoken_decode);
    assert!(!resolve(&[("ATLAS_SSM_GEMV_BATCH4", "0")]).ssm_gemv_batch4);
    // `=0` on an opt-in is NOT enabling — the trap `ATLAS_BF16_TC_PROJ`
    // falls into by being presence-gated.
    assert!(!resolve(&[("ATLAS_GDN_FUSED_CONV", "0")]).gdn_fused_conv);
}

#[test]
fn kill_switches_and_zero_opt_outs_keep_their_distinct_polarities() {
    let d = resolve(&[
        ("ATLAS_NO_GDN_REGRESIDENT", "1"),
        ("ATLAS_NO_GEMV_SW", "1"),
        ("ATLAS_GDN_WY17", "0"),
        ("ATLAS_GDN_WYN", "0"),
        ("ATLAS_FFN_SMALLM", "0"),
    ]);
    assert!(!d.gdn_regresident);
    assert!(!d.gemv_sw);
    assert!(!d.gdn_wy17);
    assert!(!d.gdn_wyn);
    assert!(!d.ffn_small_m);
    assert!(resolve(&[("ATLAS_NO_GDN_REGRESIDENT", "0")]).gdn_regresident);
    assert!(resolve(&[("ATLAS_GDN_WY17", "1")]).gdn_wy17);
}

#[test]
fn externally_resolved_shadow_and_drafter_values_are_carried() {
    let d = from_values(
        |_| None,
        |_| false,
        7,
        crate::model::drafter_context::DrafterContext::OFF,
        0.42,
    );
    assert_eq!(d.shadow_topk, 7);
    assert_eq!(
        d.draft_conf_tau, 0.42,
        "the confidence clamp is resolved OUTSIDE `from_values` and carried \
         in, like shadow_topk and drafter — reading it inside broke the \
         function's purity and made a sibling test fail under parallelism"
    );
    assert_eq!(
        d.drafter,
        crate::model::drafter_context::DrafterContext::OFF
    );
}
