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
    )
}

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

/// ★ THE DENSE-FFN ELEVEN ARE ALL PRESENCE-GATED, AND SIX OF THEM ARE `NO_`
/// OR `DISABLE_` VARIABLES WHOSE FIELD STORES THE OPPOSITE OF THEIR NAME.
///
/// Presence, not value: `=0` neither enables an opt-in nor re-enables an
/// opt-out. That is the shipped behaviour of every one of these (they were
/// `std::env::var_os(..).is_some()` / `.is_none()`), and it is the trap
/// `ATLAS_BF16_TC_PROJ` already falls into two tests above. Getting one
/// backwards silently changes which GEMM every dense FFN layer launches.
#[test]
fn the_dense_ffn_levers_are_presence_gated_and_their_polarities_hold() {
    let d = resolve(&[]);
    assert!(d.decode_split_silu, "split SiLU+down ships ON");
    assert!(d.ffn_nvfp4_mmq, "gate/up NVFP4 MMQ ships ON");
    assert!(d.ffn_nvfp4_mmq_down, "down NVFP4 MMQ ships ON");
    assert!(d.prefill_v2, "the v2 BF16 prefill kernel ships ON");
    assert!(!d.bf16_tc_prefill);
    assert!(!d.fp8_m64_prefill);
    assert!(!d.int8_prefill);
    assert!(!d.int8_faith5);
    assert!(!d.ffn_mmq);
    assert!(
        !d.ffn_mmq_down_q4k,
        "down stays on the NVFP4 hybrid by default"
    );
    assert!(!d.fp4_prefill);

    // Every opt-in arms on presence alone, including `=0`.
    let armed: [(&str, fn(&ModelLevers) -> bool); 7] = [
        ("ATLAS_BF16_TC_PREFILL", |l| l.bf16_tc_prefill),
        ("ATLAS_FP8_M64_PREFILL", |l| l.fp8_m64_prefill),
        ("ATLAS_INT8_PREFILL", |l| l.int8_prefill),
        ("ATLAS_INT8_FAITH5", |l| l.int8_faith5),
        ("ATLAS_FFN_MMQ", |l| l.ffn_mmq),
        ("ATLAS_FFN_MMQ_DOWN_Q4K", |l| l.ffn_mmq_down_q4k),
        ("ATLAS_FP4_PREFILL", |l| l.fp4_prefill),
    ];
    for (name, read) in armed {
        assert!(read(&resolve(&[(name, "1")])), "{name} did not arm");
        assert!(
            read(&resolve(&[(name, "0")])),
            "{name} is presence-gated: `=0` still arms it"
        );
    }

    // Every kill switch disables on presence alone, including `=0`.
    let killed: [(&str, fn(&ModelLevers) -> bool); 4] = [
        ("ATLAS_NO_DECODE_SPLIT_SILU", |l| l.decode_split_silu),
        ("ATLAS_NO_FFN_NVFP4_MMQ", |l| l.ffn_nvfp4_mmq),
        ("ATLAS_NO_FFN_NVFP4_MMQ_DOWN", |l| l.ffn_nvfp4_mmq_down),
        ("ATLAS_DISABLE_PREFILL_V2", |l| l.prefill_v2),
    ];
    for (name, read) in killed {
        assert!(!read(&resolve(&[(name, "1")])), "{name} did not kill");
        assert!(
            !read(&resolve(&[(name, "0")])),
            "{name} is presence-gated: `=0` does NOT re-enable"
        );
    }

    // The two down-projection gates are independent of their gate/up
    // siblings — down is the heavy-tailed projection and has its own arm.
    assert!(resolve(&[("ATLAS_NO_FFN_NVFP4_MMQ_DOWN", "1")]).ffn_nvfp4_mmq);
    assert!(resolve(&[("ATLAS_NO_FFN_NVFP4_MMQ", "1")]).ffn_nvfp4_mmq_down);
}

/// The MoE routed-prefill levers. Four opt-ins, one tri-state, one numeric —
/// and the tri-state is the interesting one: its DEFAULT is model-dependent
/// (NVFP4 checkpoints only), so `None` must stay distinguishable from
/// `Some(false)` or the call site cannot apply that default.
/// The decode-step levers. `ssm_save_dump` is PRESENCE-gated (it was
/// `std::env::var(..).is_ok()`) while the two graph levers are TRUTHY-gated
/// (`is_ok_and(|v| v == "1" || v == "true")`) — three variables read on the
/// same line of the same function with two different spellings, which is
/// exactly the kind of thing a consolidation quietly unifies by accident.
#[test]
fn the_decode_step_levers_keep_their_two_different_spellings() {
    let d = resolve(&[]);
    assert!(!d.ssm_save_dump);
    assert!(!d.ep_graphs);
    assert!(!d.gdn_decode_graph);

    // Presence: any value arms it, `0` included.
    assert!(resolve(&[("ATLAS_SSM_SAVE_DUMP", "1")]).ssm_save_dump);
    assert!(resolve(&[("ATLAS_SSM_SAVE_DUMP", "0")]).ssm_save_dump);
    assert!(resolve(&[("ATLAS_SSM_SAVE_DUMP", "")]).ssm_save_dump);

    // Truthy: `1` or `true`, nothing else.
    for (name, read) in [
        (
            "ATLAS_EP_GRAPHS",
            (|l: &ModelLevers| l.ep_graphs) as fn(&ModelLevers) -> bool,
        ),
        ("ATLAS_GDN_DECODE_GRAPH", |l: &ModelLevers| {
            l.gdn_decode_graph
        }),
    ] {
        assert!(read(&resolve(&[(name, "1")])), "{name} at =1");
        assert!(read(&resolve(&[(name, "true")])), "{name} at =true");
        assert!(!read(&resolve(&[(name, "0")])), "{name} armed at =0");
        assert!(
            !read(&resolve(&[(name, "")])),
            "{name} is truthy-gated, not presence-gated"
        );
        // ★ CASE-SENSITIVE, unlike every other truthy lever in this struct.
        // The originals spelled it `v == "1" || v == "true"`. Accepting
        // `TRUE` would arm an experimental CUDA-graph capture on a spelling
        // that previously did nothing — the direction that turns capture ON
        // unexpectedly, which is the one that must not widen by accident.
        assert!(
            !read(&resolve(&[(name, "TRUE")])),
            "{name} must stay case-SENSITIVE: `TRUE` did not arm it before"
        );
    }
    // The contrast, in the same test so the difference is visible: the
    // sibling truthy levers ARE case-insensitive and must stay that way.
    assert!(resolve(&[("ATLAS_LORA_EAGER", "TRUE")]).lora_eager);
}

/// The MoE-forward and MTP-drafter levers. All five are strict `=1` opt-ins
/// read on a per-layer-per-decode-token or per-drafted-token path.
///
/// `fp32_routing` is the one to watch: it is the LAST term of a five-way
/// conjunction in `MoeFfnLayer::fp32_routing_active`, whose other four terms
/// are weight/kernel preconditions. Defaulting it ON would change which norm
/// kernel every MoE decode launches on any model that happens to satisfy
/// those four.
#[test]
fn the_moe_forward_and_mtp_levers_are_strict_opt_ins() {
    let d = resolve(&[]);
    assert!(!d.fp32_routing);
    assert!(!d.fp32_gate);
    assert!(!d.frankenstein_decode_via_prefill);
    assert!(!d.k2_diag);
    assert!(!d.mtp_debug_norms);

    let cases: [(&str, fn(&ModelLevers) -> bool); 5] = [
        ("ATLAS_FP32_ROUTING", |l| l.fp32_routing),
        ("ATLAS_FP32_GATE", |l| l.fp32_gate),
        ("ATLAS_FRANKENSTEIN_DECODE_VIA_PREFILL", |l| {
            l.frankenstein_decode_via_prefill
        }),
        ("ATLAS_K2_DIAG", |l| l.k2_diag),
        ("ATLAS_MTP_DEBUG_NORMS", |l| l.mtp_debug_norms),
    ];
    for (name, read) in cases {
        assert!(read(&resolve(&[(name, "1")])), "{name} did not arm at =1");
        assert!(!read(&resolve(&[(name, "0")])), "{name} armed at =0");
        assert!(
            !read(&resolve(&[(name, "true")])),
            "{name} is strict `1`, not truthy — that is how it was spelled"
        );
    }
    // The two FP32 levers are independent: the gate one is the batched-path
    // sibling, not an alias.
    assert!(!resolve(&[("ATLAS_FP32_ROUTING", "1")]).fp32_gate);
    assert!(!resolve(&[("ATLAS_FP32_GATE", "1")]).fp32_routing);
}

#[test]
fn the_moe_prefill_levers_keep_the_tri_state_distinguishable() {
    let d = resolve(&[]);
    assert!(!d.moe_grouped_cutlass);
    assert!(!d.moe_grouped_down);
    assert!(!d.moe_prefill_zero);
    assert!(!d.moe_prefill_fp8_down);
    assert_eq!(
        d.moe_prefill_exact_tiles, None,
        "unset must defer to the checkpoint, not decide"
    );
    assert_eq!(d.moe_prefill_max_load_factor, None);

    assert!(resolve(&[("ATLAS_HOLO_MOE_GROUPED_CUTLASS", "1")]).moe_grouped_cutlass);
    assert!(resolve(&[("ATLAS_HOLO_MOE_GROUPED_DOWN", "1")]).moe_grouped_down);
    assert!(resolve(&[("ATLAS_MOE_PREFILL_ZERO", "1")]).moe_prefill_zero);
    assert!(resolve(&[("ATLAS_MOE_PREFILL_FP8_DOWN", "1")]).moe_prefill_fp8_down);
    // These four are value-gated, not presence-gated.
    assert!(!resolve(&[("ATLAS_MOE_PREFILL_ZERO", "0")]).moe_prefill_zero);

    assert_eq!(
        resolve(&[("ATLAS_MOE_PREFILL_EXACT_TILES", "1")]).moe_prefill_exact_tiles,
        Some(true)
    );
    assert_eq!(
        resolve(&[("ATLAS_MOE_PREFILL_EXACT_TILES", "0")]).moe_prefill_exact_tiles,
        Some(false),
        "`0` is an explicit OFF, not an absent lever — the p90 measured -5.0% \
         there and +4.9% at ON, so both directions must stay reachable"
    );
    assert_eq!(
        resolve(&[("ATLAS_MOE_PREFILL_EXACT_TILES", "yes")]).moe_prefill_exact_tiles,
        None
    );

    assert_eq!(
        resolve(&[("ATLAS_MOE_PREFILL_MAX_LOAD_FACTOR", "4")]).moe_prefill_max_load_factor,
        Some(4)
    );
    // `0` means "no cap", which is `None` — not a cap of zero, which would
    // size every expert's tile bound to one tile and drop rows.
    assert_eq!(
        resolve(&[("ATLAS_MOE_PREFILL_MAX_LOAD_FACTOR", "0")]).moe_prefill_max_load_factor,
        None
    );
    assert_eq!(
        resolve(&[("ATLAS_MOE_PREFILL_MAX_LOAD_FACTOR", "x")]).moe_prefill_max_load_factor,
        None
    );
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
    );
    assert_eq!(d.shadow_topk, 7);
    assert_eq!(
        d.drafter,
        crate::model::drafter_context::DrafterContext::OFF
    );
}
