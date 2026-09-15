// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the crate root. Split out of `lib.rs` to keep it under the
//! repo's 500-LoC cap; `lib.rs` re-attaches this file with `#[path]`, the
//! same idiom used by `atlas-closure` and `atlas-plugin::gate`.

use super::*;

#[test]
#[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
fn all_ptx_modules_non_empty() {
    for (name, blob) in ptx_modules() {
        assert!(
            !blob.is_empty(),
            "PTX module '{name}' is empty — nvcc compilation may have failed"
        );
        // Blobs are `&[u8]` (uniform across backends). For the NVIDIA
        // build under test the bytes are ASCII PTX, so decode and check
        // the `.version` directive; on a non-text backend this lossily
        // decodes to "" and the assert would (correctly) not apply.
        let ptx = std::str::from_utf8(blob).unwrap_or("");
        assert!(
            ptx.contains(".version"),
            "PTX module '{name}' doesn't contain .version directive"
        );
    }
}

// These tests assert that PTX modules were actually compiled into the
// crate at build time. They require nvcc + a real CUDA toolchain — the
// CI host runs with `ATLAS_SKIP_BUILD=1`, which emits an empty stub
// registry by design (so `cargo check` / `cargo clippy` / `cargo test`
// can run on hosts without a GPU). Mark them `#[ignore]` so default
// `cargo test` is green; they're still exercised on a developer
// machine via `cargo test -p atlas-kernels -- --ignored` after a
// real PTX build.

#[test]
#[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
fn available_targets_non_empty() {
    let targets = available_targets();
    assert!(!targets.is_empty(), "No kernel targets available");
    assert!(
        targets.iter().any(|t| t.target.quant == "nvfp4"),
        "Expected at least one NVFP4 target"
    );
}

#[test]
#[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
fn all_targets_have_modules() {
    for t in available_targets() {
        assert!(
            t.modules.len() >= 31,
            "Target {} has only {} modules (expected >= 31)",
            t.target,
            t.modules.len()
        );
    }
}

/// #438: the exact-verify `_snap` twins (#435) ship ONLY in
/// qwen3.6-27b/nvfp4's shadow set, but `qwen3_ssm::init` issues their three
/// lookups on EVERY GDN model. The boot gate fails CLOSED on an unresolved
/// lookup that is not declared `[expected_absent]`, so every GDN target that
/// does not compile these modules MUST declare them — qwen3.6-35b-a3b was
/// unservable without this (3 required-unresolved at boot).
///
/// Issuer proxy: a target constructs `qwen3_ssm::init` iff it either ships
/// `gated_delta_rule_wy17` or declares it expected-absent — a GDN target with
/// NEITHER would already fail its own boot gate on the wy17 lookup, so a
/// green fleet cannot contain one.
#[test]
#[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
fn exact_verify_snap_lookups_resolve_or_are_declared_on_every_gdn_target() {
    const PAIRS: [(&str, &str); 3] = [
        (
            "gated_delta_rule_snap",
            "gated_delta_rule_decode_f32_norm_snap",
        ),
        (
            "gated_delta_rule_snap",
            "gated_delta_rule_decode_f32_strided_norm_snap",
        ),
        (
            "gdn_verify_fused_conv_kn_f32",
            "gdn_verify_fused_conv_kn_f32",
        ),
    ];
    let ships = |t: &TargetPtxSet, m: &str| t.modules.iter().any(|(name, _)| *name == m);
    let declares = |t: &TargetPtxSet, m: &str, f: &str| {
        t.expected_absent
            .iter()
            .any(|(em, ef)| *em == m && *ef == f)
    };

    let mut gdn_targets = 0usize;
    let mut by_presence = 0usize; // pair resolves because the module is compiled (qwen3.6-27b)
    let mut by_declaration = 0usize; // pair declared expected-absent (the #438 fix)
    let mut violations: Vec<String> = Vec::new();
    for t in available_targets() {
        let issues_gdn = ships(&t, "gated_delta_rule_wy17")
            || declares(&t, "gated_delta_rule_wy17", "gated_delta_rule_wy17");
        if !issues_gdn {
            continue;
        }
        gdn_targets += 1;
        for (m, f) in PAIRS {
            if ships(&t, m) {
                by_presence += 1;
            } else if declares(&t, m, f) {
                by_declaration += 1;
            } else {
                violations.push(format!(
                    "{} misses {m}::{f} UNDECLARED — its boot gate will refuse to serve",
                    t.target
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "GDN targets with undeclared unresolvable snap lookups:\n{}",
        violations.join("\n")
    );
    // Non-vacuity guards: the invariant must have been exercised from BOTH
    // sides, or a build/staging regression could pass this test silently.
    assert!(
        gdn_targets >= 2,
        "expected at least the 27B and 35B GDN targets, saw {gdn_targets}"
    );
    assert!(
        by_presence >= 3,
        "qwen3.6-27b must still SHIP all three snap modules — fixing the 35B \
         by unshipping the 27B is not a fix (pairs resolved by presence: {by_presence})"
    );
    assert!(
        by_declaration >= 3,
        "at least the 35B must cover all three pairs by declaration \
         (pairs covered: {by_declaration})"
    );
}

#[test]
#[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
fn ptx_for_model_lookup() {
    let found = ptx_for_model("qwen3-next-80b").expect("compiled qwen3-next target");
    assert_eq!(
        found.target.model, "qwen3-next-80b-a3b",
        "lookup returned a different compiled target"
    );
}

/// End-to-end resolution against the COMPILED registry (multi-target build):
/// the config-identical dense-27B checkpoints must land on their own targets,
/// the identity-free reference must hard-error, and the pin must break it.
/// The same routes are pinned against the raw MODEL.tomls in
/// `tests/target_resolution.rs`, which runs on the skip-build CI host; this
/// leg proves build.rs carried the declarations into the binary intact.
#[test]
#[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset (multi-target build)"]
fn ptx_for_config_breaks_the_dense_27b_tie_in_the_compiled_registry() {
    let name = |r: Result<Option<TargetPtxSet>, TargetResolveError>| {
        r.expect("resolves").expect("some target").target.model
    };
    assert_eq!(
        name(ptx_for_config(
            "qwen3_5",
            5120,
            &["unsloth/Qwen3.8-27B-NVFP4"],
            None
        )),
        "qwen3.8-27b"
    );
    assert_eq!(
        name(ptx_for_config(
            "qwen3_5",
            5120,
            &["unsloth/Qwen3.6-27B-NVFP4"],
            None
        )),
        "qwen3.6-27b"
    );
    assert_eq!(
        name(ptx_for_config(
            "qwen3_5",
            5120,
            &["Kbenkhaled/Qwen3.5-27B-NVFP4"],
            None
        )),
        "qwen3.6-27b"
    );
    assert!(matches!(
        ptx_for_config("qwen3_5", 5120, &["/model"], None),
        Err(TargetResolveError::Ambiguous { .. })
    ));
    assert_eq!(
        name(ptx_for_config(
            "qwen3_5",
            5120,
            &["/model"],
            Some("qwen3.8-27b")
        )),
        "qwen3.8-27b"
    );
    // The redirected target embeds the SAME kernel set as its source.
    let q36 = ptx_for_exact_target("qwen3.6-27b", "nvfp4").expect("compiled");
    let q38 = ptx_for_exact_target("qwen3.8-27b", "nvfp4").expect("compiled");
    assert_eq!(q36.modules.len(), q38.modules.len());
    let names36: Vec<&str> = q36.modules.iter().map(|(n, _)| *n).collect();
    let names38: Vec<&str> = q38.modules.iter().map(|(n, _)| *n).collect();
    assert_eq!(names36, names38, "kernel_source must mirror the module set");
}

/// Every `[[serve_presets]]` entry build.rs carried into the binary resolves
/// by name — case-insensitively — to the target that declares it, and to no
/// other. Vacuous on the skip-build stub (empty registry); on a real build it
/// proves the codegen emitted the presets intact. `build.rs` refuses duplicate
/// preset names and target-shaped names at compile time, so uniqueness is
/// asserted here only as the runtime consequence.
#[test]
fn compiled_serve_presets_resolve_to_their_declaring_target() {
    for t in all_ptx_sets() {
        for p in t.serve_presets {
            assert!(
                !p.name.is_empty() && !p.hf_id.is_empty(),
                "{}: {p:?}",
                t.target.model
            );
            let (owner, found) = preset_named(p.name).expect("declared preset resolves");
            assert_eq!(owner, t.target.model, "{}", p.name);
            assert_eq!(found, p);
            let upper = p.name.to_ascii_uppercase();
            assert_eq!(preset_named(&upper).map(|(o, _)| o), Some(t.target.model));
            assert_ne!(
                ptx_for_model(p.name).map(|s| s.target.model),
                Some(p.name),
                "a preset name must not also be a kernel-target name"
            );
        }
    }
    assert!(preset_named("no-such-preset").is_none());
}

#[test]
fn behavior_default_prose_budget_matches_shared_constant() {
    // #328: `ModelBehavior::default()` sat at 384 for a month after P2-1
    // raised the intended default to 3072 in spark-server — production
    // resolves the budget from THIS struct, so every model without an
    // explicit MODEL.toml pin kept truncating agent narration at 384
    // tokens. The default must come from the shared constant (also
    // `include!`d by the build script) and stay plan-sized.
    let b = ModelBehavior::default();
    assert_eq!(b.max_inter_tool_prose, DEFAULT_MAX_INTER_TOOL_PROSE);
    assert!(
        b.max_inter_tool_prose >= 2048,
        "inter-tool prose budget default must fit a plan/analysis turn"
    );
}
