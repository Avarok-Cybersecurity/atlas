// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the crate root. Split out of `lib.rs` to keep it under the
//! repo's 500-LoC cap; `lib.rs` re-attaches this file with `#[path]`, the
//! same idiom used by `atlas-closure` and `atlas-plugin::gate`.

use super::*;

#[test]
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
fn module_count_matches_cu_files() {
    let count = ptx_modules().len();
    assert!(count >= 31, "Expected at least 31 PTX modules, got {count}");
}

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
    let found = ptx_for_model("qwen3-next-80b");
    assert!(
        found.is_some(),
        "ptx_for_model('qwen3-next-80b') should find the default target"
    );
}

// ── Model-shape selection ─────────────────────────────────────────
// `ptx_for_shape` picks the MOST SPECIFIC declaring entry. These are
// build-dependent (they need the real compiled registry), hence the
// same `#[ignore]` gate as `ptx_for_model_lookup` above.

/// Specificity ordering is a pure function of the declarations, so it
/// is testable without a compiled registry.
#[test]
fn specificity_prefers_the_entry_that_pins_mtp_depth() {
    let shape = ModelShape {
        model_type: "nemotron_h",
        hidden_size: 2688,
        mtp_layers: 1,
    };
    let wildcard = ModelTypeMatch {
        model_type: "nemotron_h",
        hidden_size: None,
        mtp_layers: None,
    };
    let hidden_only = ModelTypeMatch {
        model_type: "nemotron_h",
        hidden_size: Some(2688),
        mtp_layers: None,
    };
    let both = ModelTypeMatch {
        model_type: "nemotron_h",
        hidden_size: Some(2688),
        mtp_layers: Some(1),
    };
    assert!(both.specificity(&shape) > hidden_only.specificity(&shape));
    assert!(hidden_only.specificity(&shape) > wildcard.specificity(&shape));
}

#[test]
fn a_pinned_mtp_depth_excludes_the_other_variant() {
    // The split itself: Nano declares 0, Lightning declares 1, and
    // neither may absorb the other's checkpoint. Before the
    // discriminator existed BOTH shapes matched Nano's single
    // (nemotron_h, 2688) entry.
    let nano_entry = ModelTypeMatch {
        model_type: "nemotron_h",
        hidden_size: Some(2688),
        mtp_layers: Some(0),
    };
    let lightning_entry = ModelTypeMatch {
        model_type: "nemotron_h",
        hidden_size: Some(2688),
        mtp_layers: Some(1),
    };
    let nano_shape = ModelShape {
        model_type: "nemotron_h",
        hidden_size: 2688,
        mtp_layers: 0,
    };
    let lightning_shape = ModelShape {
        model_type: "nemotron_h",
        hidden_size: 2688,
        mtp_layers: 1,
    };
    assert!(nano_entry.specificity(&nano_shape).is_some());
    assert!(nano_entry.specificity(&lightning_shape).is_none());
    assert!(lightning_entry.specificity(&lightning_shape).is_some());
    assert!(lightning_entry.specificity(&nano_shape).is_none());
}

#[test]
fn mismatched_model_type_or_hidden_size_never_matches() {
    let entry = ModelTypeMatch {
        model_type: "nemotron_h",
        hidden_size: Some(2688),
        mtp_layers: Some(1),
    };
    for shape in [
        ModelShape {
            model_type: "qwen3_6_moe",
            hidden_size: 2688,
            mtp_layers: 1,
        },
        ModelShape {
            model_type: "nemotron_h",
            hidden_size: 4096,
            mtp_layers: 1,
        },
    ] {
        assert!(entry.specificity(&shape).is_none());
    }
}

// The registry-level counterpart of the two tests above — resolving a
// (nemotron_h, 2688, mtp_layers) shape to two DISTINCT compiled targets —
// lands with the PR that creates the second target
// (`nemotron-3.5-lightning-30b-a3b`). Until then the only nemotron_h/2688
// target declares `mtp_layers = None`, so both depths correctly resolve to
// it and there is nothing for such a test to discriminate.
