// SPDX-License-Identifier: AGPL-3.0-only

//! The `[defaults]` and `[hardware] sm_count` declarations that are actually
//! CHECKED IN, parsed with the real build-script parser.
//!
//! Companion to `spark-model`'s `target_defaults_tests`, and deliberately a
//! different question. That file grades the RESOLVER against tables spelled
//! out in Rust. This one grades the DATA: that `kernels/hopper/HARDWARE.toml`
//! really declares what an H100 serve runs with, that
//! `kernels/gb10/HARDWARE.toml` really declares today's behaviour, and that
//! the file the build reads is the file a reviewer read.
//!
//! An integration test rather than a `#[cfg(test)]` module inside the build
//! script, because cargo never runs a build script's own unit tests — the same
//! reason `tests/kernel_build_flags.rs` and `tests/kernel_target_arch.rs`
//! exist. It compiles `build_defaults.rs` directly, so there is no second
//! parser to drift.

#[path = "../build_defaults.rs"]
mod build_defaults;

use build_defaults::{
    BASELINE_SM_COUNT, Defaults, baseline, literal, parse_defaults, read_defaults, read_sm_count,
    sm_count_literal,
};

use std::path::PathBuf;

/// Every NVIDIA target that carries a `[defaults]` table. Named once so a new
/// hardware tree makes someone decide rather than inherit silently.
const DECLARING: &[&str] = &["gb10", "hopper", "b200"];

fn kernels_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/atlas-kernels is two levels below the workspace root")
        .join("kernels")
}

fn declared(hw: &str) -> Defaults {
    read_defaults(&kernels_root(), hw)
}

// ── the data ──

/// THE DELIVERABLE, as data. `ssm_batched_recurrent` was a line in an H100
/// launch script outside this repository before the 2026-09-11 maintainer
/// review ("every Hopper/GB10 divergence is expressed as an env lever set by
/// an H100 recipe living outside this repo"). An H100 serve with an empty
/// environment now resolves to it.
#[test]
fn hopper_declares_what_an_h100_serve_runs_with() {
    let d = declared("hopper");
    assert_eq!(d.hw, "hopper");
    assert!(
        d.ssm_batched_recurrent,
        "+6% on the serve, md5-identical output to the per-sequence launches"
    );
    assert!(d.decode_split_silu);
    // ★ NOT 16. The 16 an H100 recipe exported was measured with the
    // tensor-core head arm (`dense_gemm_m16_bf16`, #927) also on, where that
    // arm serves 5..=16 and this band is very nearly inert. The arm is not in
    // this kernel set, so 16 here would be an unmeasured configuration; the
    // row moves with the commit that lands the arm. A default is a claim
    // about a measurement.
    assert_eq!(
        d.lm_head_batchm_max,
        baseline("hopper").lm_head_batchm_max,
        "the band holds at the frozen 8 until the arm it was measured beside \
         lands"
    );
    // The one row round 13 ADDED to the recipe, and the largest measured win of
    // the campaign: cell T1 against cell A on the same binary, C=1 TTFT
    // 269.1 -> 162.4 ms and 889.3 -> 491.5 ms, C=16 aggregate +21.5%/+31.4%,
    // coherency 4/4, determinism 8/8 x 3.
    assert!(
        d.gdn_prefill_tc,
        "round 13: the tensor-core GDN prefill family is Hopper's default — \
         -39.6%/-44.7% on C=1 TTFT, +21.5%/+31.4% on C=16 aggregate"
    );
    // The row round 14 adds. BIT-IDENTICAL to its parent by construction, so
    // it is on without an accuracy receipt and its worst case is a null; the
    // cost it attacks is nsys round 13's 26 881.8 us = 5.85% of the 4593-token
    // prefill at 96 reads of every token's activation row, one per BA output.
    assert!(
        d.ssm_ba_gates_hopper,
        "the BA-gates twin is bit-identical to its parent and Hopper-only; it \
         is on because it cannot change output and off it re-reads every \
         activation row 96 times (`SSM-BA-GATES-ATTRIBUTION.md`)"
    );
}

/// THE REGRESSION GATE, as data: `kernels/gb10` declares EXACTLY the baseline,
/// i.e. the literals every resolver hardcoded before the table existed. A GB10
/// serve with an empty environment is unchanged by this whole change, and the
/// way to keep it that way is for this assertion to be an equality against
/// [`baseline`] rather than a list somebody has to remember to update.
#[test]
fn gb10_declares_the_baseline_and_nothing_else() {
    assert_eq!(
        declared("gb10"),
        baseline("gb10"),
        "kernels/gb10/HARDWARE.toml [defaults] must restate the pre-existing \
         hardcoded defaults and nothing else — it exists to SAY what GB10 \
         serves with"
    );
}

/// B200 has no serving receipt of any kind, so it declares the conservative
/// table and NOT Hopper's. Copying a recipe across because both cards are
/// datacentre parts is the reasoning this whole mechanism replaces.
#[test]
fn b200_declares_the_conservative_table_not_hoppers() {
    let d = declared("b200");
    assert_eq!(d, baseline("b200"));
    assert!(
        !d.ssm_batched_recurrent && declared("hopper").ssm_batched_recurrent,
        "the batched GDN recurrence is ON for Hopper on a Hopper receipt and \
         OFF here for want of one — B200 must not inherit a measured recipe by \
         resemblance"
    );
    assert!(
        !d.gdn_prefill_tc && declared("hopper").gdn_prefill_tc,
        "the GDN prefill family is ON for Hopper on a Hopper receipt (round 13) \
         and OFF here for want of one — the same rule, stated on the row that \
         most recently moved"
    );
    assert!(
        !d.ssm_ba_gates_hopper && declared("hopper").ssm_ba_gates_hopper,
        "the BA-gates twin is Hopper-only source; B200's common/ does not link \
         it, so the row is inert here and must read false"
    );
}

/// The targets that declare NO `[defaults]` table are unaffected: they resolve
/// to the baseline, which is what their resolvers did before. Named
/// explicitly so adding a hardware tree makes someone decide.
#[test]
fn the_silent_targets_resolve_to_the_baseline() {
    for hw in ["metal", "strix", "strix-hip"] {
        assert_eq!(
            declared(hw),
            baseline(hw),
            "kernels/{hw}/HARDWARE.toml declares no [defaults] and must be \
             byte-for-byte unaffected"
        );
    }
}

/// Every declaring target states EVERY lever explicitly, even where the value
/// agrees with the baseline.
///
/// An absent key falls through to [`baseline`], which is correct behaviour and
/// terrible documentation: a reader of `kernels/hopper/HARDWARE.toml` would
/// have to know the baseline to know what H100 serves with, which is the
/// "recipe lives somewhere else" problem this table replaces. A row present in
/// one target's table and missing from another's is also how a lever comes to
/// mean two things in one repository.
#[test]
fn every_declaring_target_states_every_lever() {
    for hw in DECLARING {
        let raw = std::fs::read_to_string(kernels_root().join(hw).join("HARDWARE.toml"))
            .unwrap_or_else(|e| panic!("kernels/{hw}/HARDWARE.toml: {e}"));
        for lever in [
            "lm_head_batchm_max",
            "ssm_batched_recurrent",
            "gdn_prefill_tc",
            "ssm_ba_gates_hopper",
            "decode_split_silu",
        ] {
            assert!(
                raw.contains(&format!("\n{lever} = ")),
                "kernels/{hw}/HARDWARE.toml [defaults] must declare `{lever}` \
                 explicitly, not inherit it from the baseline"
            );
        }
    }
}

// ── the SM count ──

/// The declared SM count per target, as data. The defect it closes was
/// `atlas_core::device::sm121::NUM_SMS = 48` — a constant named after one card
/// — reaching grid sizing on another, so this file asserting 132 is the fix's
/// oracle.
#[test]
fn every_target_declares_the_sm_count_of_its_own_part() {
    let root = kernels_root();
    assert_eq!(
        read_sm_count(&root, "hopper"),
        132,
        "H100 SXM5/PCIe/NVL and H200 SXM5 are all GH100 with 132 SMs"
    );
    assert_eq!(read_sm_count(&root, "gb10"), 48, "DGX Spark GB10");
    assert_eq!(read_sm_count(&root, "b200"), 148, "GB100, 148 SMs enabled");
    // The non-CUDA trees say nothing and must therefore keep the frozen GB10
    // value — a target that declares nothing is a target nothing changed for.
    for hw in ["metal", "strix", "strix-hip"] {
        assert_eq!(read_sm_count(&root, hw), BASELINE_SM_COUNT, "{hw}");
    }
    // A target the tree does not have at all falls back rather than panicking,
    // because this runs on the ATLAS_SKIP_BUILD path too.
    assert_eq!(read_sm_count(&root, "no-such-hw"), BASELINE_SM_COUNT);
}

/// A zero or non-integer `sm_count` fails the BUILD rather than resolving to
/// a constant named after another card — the failure this row exists to end.
#[test]
#[should_panic(expected = "sm_count = 0 is not a positive u32")]
fn a_zero_sm_count_fails_the_build() {
    let toml: toml::Value = "[hardware]\nsm_count = 0\n".parse().unwrap();
    let _ = build_defaults::parse_sm_count("fictional", &toml);
}

/// The emitted constant is a `u32` const with the parsed value — the generated
/// half of the same statement.
#[test]
fn the_sm_count_literal_is_a_compilable_const() {
    let line = sm_count_literal(132);
    assert!(
        line.contains("pub const TARGET_SM_COUNT: u32 = 132;"),
        "{line}"
    );
    assert!(line.contains("Auto-generated by build.rs"), "{line}");
}

// ── the parser ──

/// A target declares only what it DIFFERS on; every absent key falls through
/// to the baseline. Without this a new lever would silently change every
/// target that had not been updated yet.
#[test]
fn absent_keys_fall_through_to_the_baseline() {
    let toml: toml::Value = "[defaults]\nlm_head_batchm_max = 16\n".parse().unwrap();
    let d = parse_defaults("fictional", &toml);
    assert_eq!(d.lm_head_batchm_max, 16);
    assert_eq!(
        Defaults {
            lm_head_batchm_max: baseline("fictional").lm_head_batchm_max,
            ..d
        },
        baseline("fictional"),
        "one declared key must move one field"
    );
}

/// A MISTYPED lever name must fail the build, not read as agreement with the
/// baseline. This is the failure mode a per-target default table cannot have:
/// it would re-create, inside the fix, exactly the silent divergence the
/// review objected to. It is also what makes the four-places-one-commit
/// contract enforceable rather than merely asked for.
#[test]
#[should_panic(expected = "has no key `ssm_batched_recurrent_misspelt`")]
fn an_unknown_lever_name_fails_the_build() {
    let toml: toml::Value = "[defaults]\nssm_batched_recurrent_misspelt = true\n"
        .parse()
        .unwrap();
    let _ = parse_defaults("fictional", &toml);
}

/// …and so must a value of the wrong TYPE, naming the key.
#[test]
#[should_panic(expected = "[defaults] ssm_batched_recurrent must be a bool")]
fn a_mistyped_value_fails_the_build_naming_the_key() {
    let toml: toml::Value = "[defaults]\nssm_batched_recurrent = \"yes\"\n"
        .parse()
        .unwrap();
    let _ = parse_defaults("fictional", &toml);
}

/// A tree with no HARDWARE.toml at all resolves to the baseline rather than
/// panicking: the generator runs on the `ATLAS_SKIP_BUILD` path, where
/// `kernels/` may not be present (a vendored crate, a docs build). The normal
/// build still panics on a bad HARDWARE.toml, in `resolve_targets`.
#[test]
fn a_missing_hardware_toml_resolves_to_the_baseline() {
    let nowhere = kernels_root().join("no-such-hardware-tree-for-tests");
    assert_eq!(
        read_defaults(&nowhere, "gb10"),
        baseline("gb10"),
        "a missing tree must not fail a skip build"
    );
}

// ── the generated constant ──

/// The emitted `const` must be the initialiser `lib.rs` `include!`s — a
/// `TargetDefaults` literal naming every field. Checked as text because the
/// generator's output is compiled by a LATER rustc invocation, so a missing
/// field would surface as an unrelated error in `atlas-kernels` rather than
/// here.
#[test]
fn the_generated_constant_names_every_field() {
    let generated = literal(&declared("hopper"));
    assert!(generated.contains("pub const TARGET_DEFAULTS: TargetDefaults = TargetDefaults {"));
    for field in [
        "hw: \"hopper\"",
        "lm_head_batchm_max: 8",
        "ssm_batched_recurrent: true",
        "gdn_prefill_tc: true",
        "ssm_ba_gates_hopper: true",
        "decode_split_silu: true",
    ] {
        assert!(
            generated.contains(field),
            "generated constant is missing `{field}`:\n{generated}"
        );
    }
}

/// The constant this BINARY was built with is the one its own hardware tree
/// declares. The join between the generator and the runtime: without it, a
/// build script that wrote the wrong tree's table would pass every test above.
#[test]
fn the_baked_constant_matches_its_own_hardware_tree() {
    let baked = atlas_kernels::TARGET_DEFAULTS;
    // `ATLAS_SKIP_BUILD` (every CPU gate) with no `ATLAS_TARGET_HW` bakes the
    // default tree. Whatever tree it is, its declaration must round-trip.
    let declared = read_defaults(&kernels_root(), baked.hw);
    assert_eq!(baked.hw, declared.hw);
    assert_eq!(baked.lm_head_batchm_max, declared.lm_head_batchm_max);
    assert_eq!(baked.ssm_batched_recurrent, declared.ssm_batched_recurrent);
    assert_eq!(baked.gdn_prefill_tc, declared.gdn_prefill_tc);
    assert_eq!(baked.ssm_ba_gates_hopper, declared.ssm_ba_gates_hopper);
    assert_eq!(baked.decode_split_silu, declared.decode_split_silu);
    assert_eq!(
        atlas_kernels::TARGET_SM_COUNT,
        read_sm_count(&kernels_root(), baked.hw),
        "the baked SM count and the baked defaults must come from ONE tree"
    );
}
