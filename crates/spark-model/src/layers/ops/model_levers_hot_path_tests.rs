// SPDX-License-Identifier: AGPL-3.0-only

//! Polarity tests for the levers moved OFF hot paths.
//!
//! Split from `model_levers_tests.rs` at 477 lines, before it became the trap
//! `levers.rs` was at 497. The seam is by provenance: the parent file holds
//! the levers that predate this work, this file holds the ones lifted out of
//! per-token, per-layer and per-forward `std::env::var` calls. Each carries
//! the SPELLING it had at the site it came from — presence vs value vs truthy
//! vs case-sensitive-truthy — because a consolidation's characteristic
//! failure is quietly unifying four spellings into one.
//!
//! A child of `tests`, not a sibling, so `resolve` and the imports come from
//! the parent rather than being copied.

use super::*;

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

/// The confidence clamp, exercised through the PURE resolver.
///
/// No `set_var` anywhere: this file's parent says why — `set_var` is
/// process-global and `cargo test` runs this binary's tests in parallel, so
/// an env-mutating test races every other one. An earlier version of this
/// test did mutate the environment and duly broke
/// `the_opt_out_lever_is_on_by_default_and_every_opt_in_is_off`, which read
/// `draft_conf_tau: 0.99` out of a `resolve(&[])` that should have been
/// hermetic.
#[test]
fn the_draft_confidence_clamp_holds_at_both_ends() {
    use crate::speculative::parse_draft_conf_tau as tau;
    assert_eq!(
        tau(None),
        0.0,
        "unset means OFF — three sites gate on `> 0.0`"
    );
    assert_eq!(tau(Some("")), 0.0);
    assert_eq!(tau(Some("junk")), 0.0);
    assert_eq!(tau(Some("0.7")), 0.7);
    assert_eq!(
        tau(Some("5.0")),
        0.99,
        "the upper clamp is load-bearing: an unclamped 5.0 puts the floor \
         above any achievable confidence and discards EVERY draft, turning \
         speculation off with nothing logged"
    );
    assert_eq!(
        tau(Some("-1")),
        0.0,
        "and the lower clamp cannot go negative"
    );
    // And the carried lever is that value, not a second spelling of it.
    assert_eq!(
        resolve(&[]).draft_conf_tau,
        0.0,
        "`from_values` takes the resolved tau as an INPUT; if it read the \
         environment itself this would depend on the ambient process"
    );
}

/// One flag, two structs — and they must name the same variable.
///
/// `ATLAS_DFLASH_DEBUG_DUMP_FULL` arms both halves of the DFlash reference
/// dump: the token sequence from `TransformerModel` and the tensors from the
/// drafter head. `TransformerModel` cannot reach the head's `DFlashLevers`
/// (`proposer` is a `dyn DraftProposer`), so each carries its own resolved
/// copy — the same shape as `ATLAS_DSPARK_ANCHOR_BIAS`, which had two
/// implementations that nothing compared.
///
/// Checked at the SOURCE rather than at runtime, because the runtime version
/// needed `set_var` to say anything, and because the property that actually
/// matters is that the two resolvers name one string. A typo in either
/// spelling is the whole failure mode.
#[test]
fn the_two_halves_of_the_dflash_dump_name_the_same_flag() {
    const FLAG: &str = "ATLAS_DFLASH_DEBUG_DUMP_FULL";
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    for (rel, field) in [
        (
            "layers/ops/model_levers_resolve.rs",
            "dflash_debug_dump_full:",
        ),
        ("layers/dflash_head/levers.rs", "debug_dump_full:"),
    ] {
        let text = std::fs::read_to_string(src.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
        let line = text
            .lines()
            .find(|l| l.trim_start().starts_with(field))
            .unwrap_or_else(|| panic!("{rel} no longer resolves `{field}`"));
        // ★ EXACT, not `contains`. A first draft of this used
        // `line.contains(FLAG)` and its control passed: the typo injected to
        // break it was `ATLAS_DFLASH_DEBUG_DUMP_FULL_TYPO`, which contains
        // the correct name as a prefix. A guard that cannot fail is worse
        // than no guard, so the quoted string is extracted and compared.
        let named = line.split('"').nth(1).unwrap_or_else(|| {
            panic!("{rel}: `{field}` names no quoted variable: {}", line.trim())
        });
        assert_eq!(
            named, FLAG,
            "{rel} resolves `{field}` from `{named}`, not `{FLAG}` — the two \
             halves of the dump would no longer arm together"
        );
    }
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
