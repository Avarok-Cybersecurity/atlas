// SPDX-License-Identifier: AGPL-3.0-only

//! The per-target serving-default resolution table, pinned.
//!
//! Three properties, and the middle one is the whole point of the change:
//!
//! 1. **Hopper resolves to the round-9 recipe with an EMPTY environment.** The
//!    configuration that used to live in a launch script outside this repo is
//!    now reproducible from the binary alone.
//! 2. **GB10 with an empty environment is byte-for-byte today's behaviour.**
//!    Asserted against the literals every resolver hardcoded before this
//!    module existed, so "GB10 is unchanged" is a test rather than a claim.
//! 3. **The environment still wins, and says so.** Every lever is overridable
//!    in BOTH directions and reports [`Source::Env`] when it was.
//!
//! The tables are constructed here rather than read from `kernels/*/
//! HARDWARE.toml`: this file grades the RESOLVER. That the checked-in TOML
//! actually holds these values is `atlas-kernels/tests/target_defaults.rs`,
//! which parses the real files with the real build-script parser.

use super::*;
use atlas_kernels::TargetDefaults;

/// `kernels/gb10/HARDWARE.toml` `[defaults]`, and also
/// `atlas-kernels/build_defaults.rs::baseline` — the two agree on purpose.
const GB10: TargetDefaults = TargetDefaults {
    hw: "gb10",
    cublas_gemm_scope: "off",
    ffn_batch16_tier: false,
    ffn_m16_tc: false,
    attn_m16_tc: false,
    attn_ncol_gemv: false,
    lm_head_m16_tc: false,
    lm_head_batchm_max: 8,
    ssm_batched_recurrent: false,
    decode_split_silu: true,
    ssm_decode_ring_slots: "auto",
};

/// `kernels/hopper/HARDWARE.toml` `[defaults]` — the round-9 recipe.
const HOPPER: TargetDefaults = TargetDefaults {
    hw: "hopper",
    cublas_gemm_scope: "ffn,ssm,attn",
    ffn_batch16_tier: false,
    ffn_m16_tc: false,
    attn_m16_tc: true,
    attn_ncol_gemv: false,
    lm_head_m16_tc: true,
    lm_head_batchm_max: 16,
    ssm_batched_recurrent: true,
    decode_split_silu: true,
    ssm_decode_ring_slots: "auto",
};

fn with(defaults: &TargetDefaults, env: &[(&str, &str)]) -> TargetLevers {
    let env: Vec<(String, String)> = env
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    resolve(defaults, |name| {
        env.iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.to_owned())
    })
}

fn empty(defaults: &TargetDefaults) -> TargetLevers {
    with(defaults, &[])
}

// ── (1) Hopper reproduces the recipe with nothing in the environment ──

/// POSITIVE, and the deliverable: an H100 serve with NO `ATLAS_*` set
/// resolves to round 9's configuration — scoped cuBLASLt on all three
/// projection families, the attention M16 tensor-core tiers, the M16 BF16
/// head, a 16-row head band and the batched GDN recurrence.
#[test]
fn hopper_resolves_the_round_nine_recipe_from_an_empty_environment() {
    let l = empty(&HOPPER);
    assert_eq!(
        l.cublas.value,
        CublasScope {
            ffn: true,
            attn: true,
            ssm: true,
            head: false,
        },
        "cuBLASLt must arm exactly ffn+ssm+attn — `head` has no consumer and \
         arming it would be a claim nothing measured"
    );
    assert!(
        l.attn_m16_tc.value,
        "round 6: -21.7% on the attention phase"
    );
    assert!(l.lm_head_m16_tc.value, "+4% on the serve");
    assert_eq!(l.lm_head_batchm_max.value, 16);
    assert!(l.ssm_batched_recurrent.value, "+6%, md5-identical output");
    assert!(l.decode_split_silu.value);
    // The two arms round 6 measured as LOSSES stay off, and so does the tier
    // whose receipt does not exist.
    assert!(!l.ffn_m16_tc.value, "round 6: +13.7% on the SSM-layer FFN");
    assert!(
        !l.ffn_batch16_tier.value,
        "the cuBLASLt FFN arm owns these widths on H100"
    );
    assert!(!l.attn_ncol_gemv.value, "no H100 serving receipt");
    assert_eq!(l.ssm_decode_ring_slots.value, None, "auto-fit at preflight");
    // Nothing came from the environment: that is what "reproducible from the
    // binary" means.
    for from_env in [
        l.cublas.from_env(),
        l.ffn_batch16_tier.from_env(),
        l.attn_m16_tc.from_env(),
        l.lm_head_m16_tc.from_env(),
        l.lm_head_batchm_max.from_env(),
        l.ssm_batched_recurrent.from_env(),
    ] {
        assert!(!from_env, "an empty environment sourced nothing from it");
    }
}

// ── (2) GB10 is unchanged ──

/// THE REGRESSION GATE. Every value here is the literal the corresponding
/// resolver hardcoded before this module existed:
///
/// | lever | pre-change literal |
/// |---|---|
/// | `cublas` | `CublasScope::OFF` (`ATLAS_CUBLAS_GEMM` absent) |
/// | `ffn_batch16_tier` | on, but the kernel is not in GB10's set → inert |
/// | `ffn_m16_tc` / `attn_m16_tc` | `var_os(..).is_some()` → false |
/// | `attn_ncol_gemv` | `var_os(..).is_some()` → false |
/// | `lm_head_m16_tc` | `var_os(..).is_some()` → false |
/// | `lm_head_batchm_max` | `DENSE_GEMV_BATCHM_DECODE_MAX_M` = 8 |
/// | `ssm_batched_recurrent` | `ATLAS_SSM_BATCHED_RECURRENT == "1"` → false |
/// | `decode_split_silu` | on unless `ATLAS_NO_DECODE_SPLIT_SILU` |
/// | `ssm_decode_ring_slots` | `auto` |
///
/// `ffn_batch16_tier` is the one row that reads as a change and is not: the
/// `w8a16_gemv_batch16` entry point does not exist in GB10's kernel set (it is
/// Hopper-tuned, `kernels/hopper/common`), so the tier's handle was 0 and
/// `batch16_plan` declined at every width. Declaring it off states in the
/// target file what the kernel set already enforced.
#[test]
fn gb10_with_an_empty_environment_is_todays_behaviour() {
    let l = empty(&GB10);
    assert_eq!(l.cublas.value, CublasScope::OFF);
    assert!(!l.ffn_batch16_tier.value);
    assert!(!l.ffn_m16_tc.value);
    assert!(!l.attn_m16_tc.value);
    assert!(!l.attn_ncol_gemv.value);
    assert!(!l.lm_head_m16_tc.value);
    assert_eq!(
        l.lm_head_batchm_max.value, DENSE_GEMV_BATCHM_DECODE_MAX_M,
        "the FROZEN band: the A/B behind it measured the GEMV negative above 8 \
         on GB10 (-14.4% at C=16)"
    );
    assert!(!l.ssm_batched_recurrent.value);
    assert!(l.decode_split_silu.value);
    assert_eq!(l.ssm_decode_ring_slots.value, None);
}

/// The baseline band in `atlas-kernels/build_defaults.rs` is a LITERAL `8`,
/// because atlas-kernels sits below spark-model and cannot name the constant.
/// This is the join that keeps the duplicate honest.
#[test]
fn the_baseline_band_is_the_frozen_one() {
    assert_eq!(BASELINE_BATCHM_MAX, DENSE_GEMV_BATCHM_DECODE_MAX_M);
    assert_eq!(GB10.lm_head_batchm_max, DENSE_GEMV_BATCHM_DECODE_MAX_M);
}

// ── (3) the environment overrides, in both directions, and says so ──

/// Every toggle is overridable from EITHER default, and the resolution
/// reports [`Source::Env`] so the serve log can mark it. The `=0` arm is the
/// behaviour change this module documents: these were presence-gated, where
/// `VAR=0` meant ON.
#[test]
fn the_environment_overrides_every_toggle_in_both_directions() {
    // Hopper's ON levers, turned off.
    let off = with(
        &HOPPER,
        &[
            ("ATLAS_ATTN_M16_TC", "0"),
            ("ATLAS_LM_HEAD_M16_TC", "false"),
            ("ATLAS_SSM_BATCHED_RECURRENT", "off"),
            ("ATLAS_CUBLAS_GEMM", "off"),
            ("ATLAS_LM_HEAD_BATCHM_MAX", "8"),
        ],
    );
    assert_eq!(off.attn_m16_tc, Resolved::env(false));
    assert_eq!(off.lm_head_m16_tc, Resolved::env(false));
    assert_eq!(off.ssm_batched_recurrent, Resolved::env(false));
    assert_eq!(off.cublas, Resolved::env(CublasScope::OFF));
    assert_eq!(off.lm_head_batchm_max, Resolved::env(8));

    // GB10's OFF levers, turned on — the A/B an operator runs before a row of
    // `kernels/gb10/HARDWARE.toml` is allowed to change.
    let on = with(
        &GB10,
        &[
            ("ATLAS_ATTN_M16_TC", "1"),
            ("ATLAS_LM_HEAD_M16_TC", "1"),
            ("ATLAS_SSM_BATCHED_RECURRENT", "1"),
            ("ATLAS_ATTN_NCOL_GEMV", "1"),
            ("ATLAS_FFN_BATCH16", "1"),
            ("ATLAS_CUBLAS_GEMM", "ffn"),
            ("ATLAS_LM_HEAD_BATCHM_MAX", "16"),
        ],
    );
    assert_eq!(on.attn_m16_tc, Resolved::env(true));
    assert_eq!(on.lm_head_m16_tc, Resolved::env(true));
    assert_eq!(on.ssm_batched_recurrent, Resolved::env(true));
    assert_eq!(on.attn_ncol_gemv, Resolved::env(true));
    assert_eq!(on.ffn_batch16_tier, Resolved::env(true));
    assert_eq!(
        on.cublas,
        Resolved::env(CublasScope {
            ffn: true,
            ..CublasScope::OFF
        })
    );
    assert_eq!(on.lm_head_batchm_max, Resolved::env(16));
}

/// The ring depth is a DECLARATION here and nothing else.
/// `ATLAS_SSM_DECODE_RING` means `1` = the full depth and `0` = no ring, a
/// grammar that lives — with its own precedence rung, below the published
/// depth — in `ssm_reserve::decode_rollback_ring_slots_with`. Reading it here
/// as a plain integer would give `=1` two contradictory meanings in one binary.
#[test]
fn the_ring_slot_lever_never_reads_its_environment_variable() {
    let l = with(&GB10, &[("ATLAS_SSM_DECODE_RING", "1")]);
    assert_eq!(
        l.ssm_decode_ring_slots,
        Resolved::target(None),
        "the declaration is `auto`, and the variable's own grammar is applied \
         elsewhere"
    );
}

/// The toggle grammar, spelling by spelling. `VAR=` (present but empty) is ON,
/// matching the presence rule every A/B recipe was written against.
#[test]
fn the_toggle_grammar_maps_each_spelling() {
    for (raw, expected) in [
        ("0", false),
        ("false", false),
        ("FALSE", false),
        ("off", false),
        ("no", false),
        (" 0 ", false),
        ("1", true),
        ("true", true),
        ("", true),
        ("yes", true),
    ] {
        assert_eq!(
            resolve_toggle(true, Some(raw), false).value,
            expected,
            "VAR={raw:?} from an ON default"
        );
        assert_eq!(
            resolve_toggle(false, Some(raw), false).value,
            expected,
            "VAR={raw:?} from an OFF default"
        );
    }
    // Absent takes the target's declaration, both ways, and is NOT `Env`.
    assert_eq!(resolve_toggle(true, None, false), Resolved::target(true));
    assert_eq!(resolve_toggle(false, None, false), Resolved::target(false));
}

/// The legacy `ATLAS_NO_*` kill switches stay PRESENCE-gated and outrank the
/// positive variable. They are the hatch an operator reaches for while a serve
/// misbehaves; one a stale `ATLAS_FFN_BATCH16=1` in the same shell could veto
/// would not be a hatch.
#[test]
fn the_legacy_kill_switches_win_over_everything() {
    assert_eq!(resolve_toggle(true, None, true), Resolved::env(false));
    assert_eq!(resolve_toggle(true, Some("1"), true), Resolved::env(false));
    let l = with(
        &HOPPER,
        &[("ATLAS_FFN_BATCH16", "1"), ("ATLAS_FFN_NO_BATCH16", "")],
    );
    assert_eq!(l.ffn_batch16_tier, Resolved::env(false));
    let silu = with(&GB10, &[("ATLAS_NO_DECODE_SPLIT_SILU", "")]);
    assert_eq!(silu.decode_split_silu, Resolved::env(false));
}

/// `ATLAS_M16_TC` is round 6's umbrella: it ARMS both M16 tiers and can never
/// disarm one the target declares on. A lever that an umbrella could silently
/// clear would make the recipe depend on which variable was exported last.
#[test]
fn the_m16_umbrella_only_arms() {
    let l = with(&GB10, &[("ATLAS_M16_TC", "1")]);
    assert_eq!(l.ffn_m16_tc, Resolved::env(true));
    assert_eq!(l.attn_m16_tc, Resolved::env(true));
    // `=0` disarms the umbrella itself; it does not touch Hopper's declaration.
    let h = with(&HOPPER, &[("ATLAS_M16_TC", "0")]);
    assert_eq!(h.attn_m16_tc, Resolved::target(true));
    assert_eq!(h.ffn_m16_tc, Resolved::target(false));
}

/// The band clamps at the kernel's compile-time row bound rather than
/// erroring at every decode step, and an unusable value keeps the target's.
#[test]
fn the_band_clamps_and_ignores_unusable_values() {
    assert_eq!(
        resolve_batchm_max(8, Some("999")).value,
        DENSE_GEMV_BATCHM_MAX_M
    );
    assert_eq!(resolve_batchm_max(8, Some("0")), Resolved::target(8));
    assert_eq!(resolve_batchm_max(8, Some("junk")), Resolved::target(8));
    assert_eq!(resolve_batchm_max(8, None), Resolved::target(8));
    // A target declaring more than the kernel can serve is clamped too — the
    // declaration is data, and data is not exempt from the row bound.
    assert_eq!(
        resolve_batchm_max(64, None),
        Resolved::target(DENSE_GEMV_BATCHM_MAX_M)
    );
}

/// `auto` is not a depth, a depth above the ring's capacity is not a depth,
/// and neither may become one by accident.
#[test]
fn the_ring_declaration_parses_auto_and_bounded_depths_only() {
    assert_eq!(resolve_ring_slots("auto"), None);
    assert_eq!(resolve_ring_slots(" auto "), None);
    assert_eq!(resolve_ring_slots("0"), Some(0));
    assert_eq!(resolve_ring_slots("8"), Some(8));
    assert_eq!(
        resolve_ring_slots("9"),
        None,
        "above DECODE_ROLLBACK_RING_SLOTS resolves to auto, not to a depth the \
         ring was never sized for"
    );
    assert_eq!(resolve_ring_slots("junk"), None);
}

// ── (4) the serve log line ──

/// THE OPERATOR-FACING DELIVERABLE. A serve prints ONE line naming every
/// resolved value and marking the ones the environment supplied, so a number
/// reported from a run can be attributed to a configuration without also
/// having the launch script that produced it — which was the whole failure the
/// 2026-09-11 review named.
///
/// Graded through `format_levers` rather than `summary_line()`: the latter
/// reads (and seals) the process-wide `OnceLock`, which would make this test
/// order-dependent and would grade the environment of whoever ran it.
#[test]
fn the_serve_line_names_every_lever_and_marks_the_environment_ones() {
    let line = format_levers(&empty(&HOPPER));
    assert!(line.starts_with("target defaults (hopper): "), "{line}");
    for field in [
        "cublas_gemm_scope=ffn,attn,ssm",
        "ffn_batch16_tier=off",
        "ffn_m16_tc=off",
        "attn_m16_tc=on",
        "attn_ncol_gemv=off",
        "lm_head_m16_tc=on",
        "lm_head_batchm_max=16",
        "ssm_batched_recurrent=on",
        "decode_split_silu=on",
        "ssm_decode_ring_slots=auto",
    ] {
        assert!(line.contains(field), "missing `{field}` in:\n{line}");
    }
    assert!(
        !line.contains("(env)"),
        "an empty environment must mark nothing as an override:\n{line}"
    );

    // …and the override is VISIBLE. A line that reported the value without its
    // provenance would let a run be labelled with the target's recipe while it
    // actually ran the operator's.
    let overridden = with(
        &GB10,
        &[
            ("ATLAS_ATTN_M16_TC", "1"),
            ("ATLAS_LM_HEAD_BATCHM_MAX", "16"),
        ],
    );
    let line = format_levers(&overridden);
    assert!(line.contains("attn_m16_tc=on (env)"), "{line}");
    assert!(line.contains("lm_head_batchm_max=16 (env)"), "{line}");
    assert!(line.contains("lm_head_m16_tc=off"), "{line}");
    assert!(!line.contains("lm_head_m16_tc=off (env)"), "{line}");
}

/// A target that names no hardware (a build that read no HARDWARE.toml) still
/// produces a readable line rather than `target defaults (): …`.
#[test]
fn an_unnamed_target_still_prints_a_readable_line() {
    let anon = TargetDefaults { hw: "", ..GB10 };
    assert!(
        format_levers(&empty(&anon)).starts_with("target defaults (unknown): "),
        "{}",
        format_levers(&empty(&anon))
    );
}
