// SPDX-License-Identifier: AGPL-3.0-only

//! The per-target serving-default resolution table, pinned.
//!
//! Three properties, and the middle one is the whole point of the change:
//!
//! 1. **Hopper resolves to its measured configuration with an EMPTY
//!    environment.** What used to live in a launch script outside this repo is
//!    now reproducible from the binary alone.
//! 2. **GB10 with an empty environment is byte-for-byte today's behaviour.**
//!    Asserted against the literals every resolver hardcoded before this
//!    module existed, so "GB10 is unchanged" is a test rather than a claim.
//! 3. **The environment still wins, and says so.** Every lever is overridable
//!    in BOTH directions and reports [`Source::Env`] when it was.
//!
//! The tables are constructed here rather than read from
//! `kernels/*/HARDWARE.toml`: this file grades the RESOLVER. That the
//! checked-in TOML actually holds these values is
//! `atlas-kernels/tests/target_defaults.rs`, which parses the real files with
//! the real build-script parser.

use super::*;
use atlas_kernels::TargetDefaults;

/// `kernels/gb10/HARDWARE.toml` `[defaults]` — field for field
/// `build_defaults::baseline`, which is what makes GB10's "unchanged" claim
/// checkable rather than argued.
const GB10: TargetDefaults = TargetDefaults {
    hw: "gb10",
    lm_head_batchm_max: 8,
    ssm_batched_recurrent: false,
    decode_split_silu: true,
    attn_decode_splitk: "legacy",
};

/// `kernels/hopper/HARDWARE.toml` `[defaults]`.
///
/// One row differs from GB10's: the batched GDN recurrence, ON, on a Hopper
/// receipt (+6% on the serve, md5-identical output to the per-sequence
/// launches). The head band deliberately holds at the frozen 8 — see
/// `atlas-kernels/tests/target_defaults.rs`.
const HOPPER: TargetDefaults = TargetDefaults {
    hw: "hopper",
    lm_head_batchm_max: 8,
    ssm_batched_recurrent: true,
    decode_split_silu: true,
    attn_decode_splitk: "auto",
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

// ── (1) Hopper reproduces its recipe with nothing in the environment ──

/// POSITIVE, and the deliverable: an H100 serve with NO `ATLAS_*` set
/// resolves to the batched GDN recurrence, which is the line that used to be
/// `ATLAS_SSM_BATCHED_RECURRENT=1` in an external launch script.
#[test]
fn hopper_resolves_its_recipe_from_an_empty_environment() {
    let l = empty(&HOPPER);
    assert!(
        l.ssm_batched_recurrent.value,
        "+6% on the serve, md5-identical output"
    );
    assert_eq!(
        l.ssm_batched_recurrent.source,
        Source::Target,
        "with an empty environment every value must be attributed to the \
         TARGET — an ` (env)` tag here would mean the log credits a prefix \
         nobody typed"
    );
    assert!(l.decode_split_silu.value);
    assert_eq!(l.lm_head_batchm_max.value, 8);
    assert_eq!(l.hw, "hopper");
}

// ── (2) GB10 is unchanged ──

/// THE REGRESSION GATE. Every value here is the literal the corresponding
/// resolver hardcoded before this module existed, so a GB10 serve with an
/// empty environment behaves exactly as it did.
#[test]
fn gb10_with_an_empty_environment_is_todays_behaviour() {
    let l = empty(&GB10);
    assert_eq!(l.lm_head_batchm_max.value, DENSE_GEMV_BATCHM_DECODE_MAX_M);
    assert!(!l.ssm_batched_recurrent.value);
    assert!(l.decode_split_silu.value);
    for source in [
        l.lm_head_batchm_max.source,
        l.ssm_batched_recurrent.source,
        l.decode_split_silu.source,
    ] {
        assert_eq!(source, Source::Target);
    }
}

/// The band constant this module publishes IS the frozen one in
/// `gemm_quant.rs`, and `atlas-kernels/build_defaults.rs::baseline` repeats it
/// as a literal because atlas-kernels sits below spark-model and cannot name
/// it. This is the join that stops the two from drifting.
#[test]
fn the_baseline_band_is_the_frozen_one() {
    assert_eq!(BASELINE_BATCHM_MAX, DENSE_GEMV_BATCHM_DECODE_MAX_M);
    assert_eq!(
        BASELINE_BATCHM_MAX, GB10.lm_head_batchm_max,
        "kernels/gb10 declares the frozen band; if this fails one of the two \
         moved without the other"
    );
}

// ── (3) the environment still wins, in both directions ──

/// A target that declares a lever ON can be turned OFF from the environment,
/// which PRESENCE gating could not express — and is the reason the grammar
/// changed at all.
#[test]
fn a_declared_on_lever_can_be_turned_off_by_the_environment() {
    for off in ["0", "false", "off", "no", "OFF", " 0 "] {
        let l = with(&HOPPER, &[("ATLAS_SSM_BATCHED_RECURRENT", off)]);
        assert!(!l.ssm_batched_recurrent.value, "`{off}` must read as off");
        assert!(l.ssm_batched_recurrent.from_env());
    }
}

/// …and a target that declares it OFF is still armed by the bare `=1` every
/// existing A/B recipe uses. `VAR=1` means what it always meant.
#[test]
fn a_declared_off_lever_is_still_armed_by_the_bare_one() {
    let l = with(&GB10, &[("ATLAS_SSM_BATCHED_RECURRENT", "1")]);
    assert!(l.ssm_batched_recurrent.value);
    assert!(l.ssm_batched_recurrent.from_env());
}

/// The legacy PRESENCE kill switch is unchanged and outranks the declaration.
/// It is the hatch an operator reaches for while a serve misbehaves; a hatch a
/// stale positive variable could veto is not one.
#[test]
fn the_legacy_kill_switch_still_forces_the_lever_off() {
    for value in ["1", "0", ""] {
        let l = with(&HOPPER, &[("ATLAS_NO_DECODE_SPLIT_SILU", value)]);
        assert!(
            !l.decode_split_silu.value,
            "ATLAS_NO_DECODE_SPLIT_SILU={value:?} is PRESENCE-gated and must \
             force the lever off whatever it is set to"
        );
        assert!(l.decode_split_silu.from_env());
    }
}

/// The band is a BAND, not a switch: `0` and garbage keep the target's
/// declaration rather than disabling the tier, and any request is clamped to
/// the kernel's compile-time row bound — `dense_gemv_batchm` refuses above it,
/// so an unclamped lever would `Err` at every decode step.
#[test]
fn the_head_band_clamps_and_has_no_off() {
    assert_eq!(
        resolve_batchm_max(8, Some("64")).value,
        DENSE_GEMV_BATCHM_MAX_M,
        "clamped to the kernel's row bound, not passed through"
    );
    assert_eq!(resolve_batchm_max(8, Some("12")).value, 12);
    for keep in [Some("0"), Some("banana"), Some(""), None] {
        assert_eq!(
            resolve_batchm_max(8, keep).value,
            8,
            "{keep:?} must keep the target's declaration"
        );
        assert_eq!(resolve_batchm_max(8, keep).source, Source::Target);
    }
    // A DECLARATION above the kernel bound is clamped too — a target cannot
    // ask for rows the kernel will not write.
    assert_eq!(
        resolve_batchm_max(64, None).value,
        DENSE_GEMV_BATCHM_MAX_M,
        "the clamp is on the resolved value, whichever rung it came from"
    );
}

// ── the log line ──

/// The line NAMES every lever, its resolved value, and which came from the
/// environment. Graded through [`format_levers`] rather than [`summary_line`]:
/// the latter seals a process-wide `OnceLock` against the real environment,
/// which would make this test order-dependent inside the binary.
#[test]
fn the_summary_line_names_every_lever_and_flags_the_environment() {
    let line = format_levers(&with(&HOPPER, &[("ATLAS_LM_HEAD_BATCHM_MAX", "12")]));
    assert!(line.starts_with("target defaults (hopper): "), "{line}");
    for field in [
        "sm_count=",
        "lm_head_batchm_max=12 (env)",
        "ssm_batched_recurrent=on",
        "decode_split_silu=on",
        "attn_decode_splitk=auto",
    ] {
        assert!(line.contains(field), "missing `{field}` in:\n{line}");
    }
    // …and a target-sourced value carries NO tag, so ` (env)` in a serve log
    // always means a prefix was typed.
    let clean = format_levers(&empty(&HOPPER));
    assert!(!clean.contains("(env)"), "{clean}");
    // GB10 keeps the pre-#928 split rule and says so.
    assert!(
        format_levers(&empty(&GB10)).contains("attn_decode_splitk=legacy"),
        "{}",
        format_levers(&empty(&GB10))
    );
}

/// The split-K policy row (#928): declaration first, environment second, and
/// the resolved value printed in the spelling that reproduces it.
///
/// The `=0` rendering is deliberate and is the one an A/B log carries:
/// `ATLAS_ATTN_DECODE_SPLITK=0` resolves to `Pinned(1)` — one split IS no
/// split-K — so the line reads `attn_decode_splitk=1 (env)`. The label is the
/// RESOLVED policy, not the string that was typed, because the number the
/// grid used is the thing a reader needs.
#[test]
fn the_split_k_policy_resolves_and_reports_like_every_other_lever() {
    assert_eq!(
        empty(&HOPPER).attn_decode_splitk.value,
        SplitkPolicy::Auto,
        "an H100 serve with an empty environment must reach the split count \
         that fills 132 SMs — the whole content of #928"
    );
    assert_eq!(
        empty(&GB10).attn_decode_splitk.value,
        SplitkPolicy::Legacy,
        "GB10 is unchanged"
    );
    assert_eq!(empty(&HOPPER).attn_decode_splitk.source, Source::Target);

    // The A/B an H100 round runs against the new default.
    let off = with(&HOPPER, &[("ATLAS_ATTN_DECODE_SPLITK", "0")]);
    assert_eq!(off.attn_decode_splitk.value, SplitkPolicy::Pinned(1));
    assert_eq!(off.attn_decode_splitk.source, Source::Env);
    assert!(
        format_levers(&off).contains("attn_decode_splitk=1 (env)"),
        "{}",
        format_levers(&off)
    );

    // …and the one that arms it on a target that declares `legacy`.
    let on = with(&GB10, &[("ATLAS_ATTN_DECODE_SPLITK", "auto")]);
    assert_eq!(on.attn_decode_splitk.value, SplitkPolicy::Auto);
    assert!(
        format_levers(&on).contains("attn_decode_splitk=auto (env)"),
        "{}",
        format_levers(&on)
    );

    // A typo keeps the DECLARATION rather than guessing: silently resolving a
    // misspelling to `auto` would arm a geometry change on a card with no
    // receipt for it.
    let typo = with(&HOPPER, &[("ATLAS_ATTN_DECODE_SPLITK", "atuo")]);
    assert_eq!(typo.attn_decode_splitk.value, SplitkPolicy::Auto);
    assert_eq!(typo.attn_decode_splitk.source, Source::Target);
}

/// A build that read no HARDWARE.toml at all has an empty `hw`, and the line
/// must still be readable rather than `target defaults (): …`.
#[test]
fn an_anonymous_build_still_prints_a_readable_line() {
    let anon = TargetDefaults { hw: "", ..GB10 };
    assert!(
        format_levers(&empty(&anon)).starts_with("target defaults (unknown): "),
        "{}",
        format_levers(&empty(&anon))
    );
}

/// The process-wide resolution is this binary's own declaration — the join
/// between the baked constant and the resolver. Without it a `resolved()` that
/// read some other table would pass every test above.
#[test]
fn the_process_resolution_reads_this_binarys_declaration() {
    assert_eq!(resolved().hw, declared().hw);
    assert_eq!(
        resolved().hw,
        atlas_kernels::TARGET_DEFAULTS.hw,
        "one table, one resolution"
    );
}
