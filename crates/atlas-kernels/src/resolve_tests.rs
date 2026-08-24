// SPDX-License-Identifier: AGPL-3.0-only

//! Resolution-rule tests. Pure — no compiled PTX needed, so these run on
//! the `ATLAS_SKIP_BUILD=1` CI host, unlike the `#[ignore]`d target tests
//! in `lib_tests.rs`.
//!
//! The synthetic fixture mirrors the real gb10 tree's one load-bearing
//! collision: `qwen3.6-27b` and `qwen3.8-27b` both declare exact
//! `(qwen3_5, 5120)` because the checkpoints are architecturally
//! indistinguishable. `tests/target_resolution.rs` drives the same rules
//! with the REAL MODEL.toml declarations.

use super::*;

const fn m(model_type: &'static str, hidden_size: Option<usize>) -> ModelTypeMatch {
    ModelTypeMatch {
        model_type,
        hidden_size,
    }
}

// Declaration slices live in consts so the borrowed fixtures are 'static.
const Q35_MATCHES: &[ModelTypeMatch] = &[m("qwen3_5", None), m("qwen3_6_moe", Some(5120))];
const Q36_MATCHES: &[ModelTypeMatch] = &[m("qwen3_5", Some(5120)), m("qwen3_6_moe", Some(5120))];
const Q38_MATCHES: &[ModelTypeMatch] = &[m("qwen3_5", Some(5120))];

/// The dense-27B corner of the real tree: two exact-colliding targets plus
/// the wildcard sibling, in build order (sorted by name — `.find()` order
/// would have picked qwen3.5-27b's wildcard tier ordering wrongly before).
fn dense_27b_fixture() -> Vec<ResolveCandidate<'static>> {
    vec![
        ResolveCandidate {
            name: "qwen3.5-27b",
            type_matches: Q35_MATCHES,
            match_names: &["qwen3.5-27b"],
        },
        ResolveCandidate {
            name: "qwen3.6-27b",
            type_matches: Q36_MATCHES,
            match_names: &["qwen3.6-27b", "qwen3.5-27b"],
        },
        ResolveCandidate {
            name: "qwen3.8-27b",
            type_matches: Q38_MATCHES,
            match_names: &["qwen3.8-27b"],
        },
    ]
}

fn resolve_name(
    cands: &[ResolveCandidate<'_>],
    model_type: &str,
    hidden: usize,
    refs: &[&str],
) -> Result<Option<&'static str>, TargetResolveError> {
    resolve_target(cands, model_type, hidden, refs)
        .map(|o| o.map(|i| ["qwen3.5-27b", "qwen3.6-27b", "qwen3.8-27b"][i]))
}

#[test]
fn exact_collision_broken_by_checkpoint_reference() {
    let c = dense_27b_fixture();
    assert_eq!(
        resolve_name(&c, "qwen3_5", 5120, &["unsloth/Qwen3.6-27B-NVFP4"]),
        Ok(Some("qwen3.6-27b"))
    );
    assert_eq!(
        resolve_name(&c, "qwen3_5", 5120, &["unsloth/Qwen3.8-27B-NVFP4"]),
        Ok(Some("qwen3.8-27b"))
    );
    // centml W4A4 re-upload still lands on the 3.6 target (quant gate
    // arbitrates quant, not resolution).
    assert_eq!(
        resolve_name(&c, "qwen3_5", 5120, &["centml/Qwen3.6-27B-W4A4-mlpinf"]),
        Ok(Some("qwen3.6-27b"))
    );
}

/// The documented (HANDOFF §9a) routing of the Kbenkhaled 3.5 checkpoint to
/// the 3.6 target survives: qwen3.6-27b explicitly claims the
/// "qwen3.5-27b" needle, qwen3.8-27b does not.
#[test]
fn qwen35_checkpoint_still_routes_to_qwen36_target() {
    let c = dense_27b_fixture();
    assert_eq!(
        resolve_name(&c, "qwen3_5", 5120, &["Kbenkhaled/Qwen3.5-27B-NVFP4"]),
        Ok(Some("qwen3.6-27b"))
    );
}

#[test]
fn reference_matching_is_case_insensitive_and_scans_all_refs() {
    let c = dense_27b_fixture();
    // Uppercased id in the SECOND reference (e.g. --model-name); the first
    // (a bare path) carries no identity.
    assert_eq!(
        resolve_name(&c, "qwen3_5", 5120, &["/model", "QWEN3.8-27B"]),
        Ok(Some("qwen3.8-27b"))
    );
    // HF-cache directory mangling still contains the needle.
    assert_eq!(
        resolve_name(
            &c,
            "qwen3_5",
            5120,
            &["/root/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/abc"]
        ),
        Ok(Some("qwen3.8-27b"))
    );
}

#[test]
fn exact_collision_with_no_matching_reference_is_a_hard_error() {
    let c = dense_27b_fixture();
    let err = resolve_name(&c, "qwen3_5", 5120, &["/model"]).unwrap_err();
    match &err {
        TargetResolveError::Ambiguous {
            tier,
            candidates,
            matched,
            ..
        } => {
            assert_eq!(*tier, "exact");
            assert!(
                matched.is_empty(),
                "nothing should have matched: {matched:?}"
            );
            let names: Vec<&str> = candidates.iter().map(|(n, _)| n.as_str()).collect();
            assert_eq!(names, ["qwen3.6-27b", "qwen3.8-27b"]);
        }
        other => panic!("expected Ambiguous, got {other:?}"),
    }
    // The operator-facing text must carry the remedies.
    let msg = err.to_string();
    assert!(msg.contains("--kernel-target"), "no pin remedy in: {msg}");
    assert!(
        msg.contains("ATLAS_TARGET_MODEL"),
        "no build remedy in: {msg}"
    );
}

#[test]
fn reference_matching_multiple_candidates_is_a_hard_error() {
    let c = dense_27b_fixture();
    let err = resolve_name(
        &c,
        "qwen3_5",
        5120,
        &["myorg/Qwen3.6-27B-to-Qwen3.8-27B-distill"],
    )
    .unwrap_err();
    match err {
        TargetResolveError::Ambiguous { matched, .. } => {
            assert_eq!(matched, ["qwen3.6-27b", "qwen3.8-27b"]);
        }
        other => panic!("expected Ambiguous, got {other:?}"),
    }
}

/// An unresolved exact collision must NOT quietly downgrade to the wildcard
/// tier (qwen3.5-27b's `(qwen3_5, None)` is right there and must not catch
/// the fall-through).
#[test]
fn ambiguous_exact_tier_never_falls_through_to_wildcard() {
    let c = dense_27b_fixture();
    assert!(matches!(
        resolve_name(&c, "qwen3_5", 5120, &["/model"]),
        Err(TargetResolveError::Ambiguous { tier: "exact", .. })
    ));
}

#[test]
fn single_exact_match_needs_no_reference() {
    // The common case (every non-colliding model): refs are never consulted.
    const MATCHES: &[ModelTypeMatch] = &[m("qwen3_6_moe", Some(2048))];
    let c = vec![ResolveCandidate {
        name: "qwen3.6-35b-a3b",
        type_matches: MATCHES,
        match_names: &[],
    }];
    assert_eq!(resolve_target(&c, "qwen3_6_moe", 2048, &[]), Ok(Some(0)));
}

#[test]
fn wildcard_fallback_when_no_exact_match() {
    let c = dense_27b_fixture();
    // hidden_size 9999 exact-matches nothing; qwen3_5 wildcard catches it.
    assert_eq!(
        resolve_name(&c, "qwen3_5", 9999, &[]),
        Ok(Some("qwen3.5-27b"))
    );
}

#[test]
fn no_declaration_resolves_to_none() {
    let c = dense_27b_fixture();
    assert_eq!(resolve_name(&c, "deepseek_v4", 5120, &[]), Ok(None));
}

/// Same-name multi-quant candidates are one target, not a collision — the
/// downstream quant-compat gate arbitrates quant, as it always has.
#[test]
fn multi_quant_variants_of_one_target_are_not_ambiguous() {
    let c = vec![
        ResolveCandidate {
            name: "qwen3.6-27b",
            type_matches: Q38_MATCHES,
            match_names: &[],
        },
        ResolveCandidate {
            name: "qwen3.6-27b",
            type_matches: Q38_MATCHES,
            match_names: &[],
        },
    ];
    assert_eq!(resolve_target(&c, "qwen3_5", 5120, &[]), Ok(Some(0)));
}

#[test]
fn wildcard_tier_collisions_error_too() {
    const WILD: &[ModelTypeMatch] = &[m("qwen3_5", None)];
    let c = vec![
        ResolveCandidate {
            name: "a",
            type_matches: WILD,
            match_names: &["a"],
        },
        ResolveCandidate {
            name: "b",
            type_matches: WILD,
            match_names: &["b"],
        },
    ];
    assert!(matches!(
        resolve_target(&c, "qwen3_5", 1234, &["/model"]),
        Err(TargetResolveError::Ambiguous {
            tier: "wildcard",
            ..
        })
    ));
    assert_eq!(
        resolve_target(&c, "qwen3_5", 1234, &["org/b-7b"]),
        Ok(Some(1))
    );
}

/// An empty match_names list can never match — a colliding target that
/// declares nothing (which build.rs rejects anyway) fails loudly, it does
/// not match everything.
#[test]
fn empty_match_names_never_matches() {
    const T1: &[ModelTypeMatch] = &[m("t", Some(1))];
    let c = vec![
        ResolveCandidate {
            name: "a",
            type_matches: T1,
            match_names: &[],
        },
        ResolveCandidate {
            name: "b",
            type_matches: T1,
            match_names: &["b"],
        },
    ];
    // "a" appears in the ref but "a" declared no needles → only "b" can win,
    // and only via its own needle.
    assert_eq!(resolve_target(&c, "t", 1, &["org/a-and-b"]), Ok(Some(1)));
    assert!(matches!(
        resolve_target(&c, "t", 1, &["org/a-only"]),
        Err(TargetResolveError::Ambiguous { .. })
    ));
}

// ── pinning ──

#[test]
fn pin_overrides_the_tie_break() {
    let c = dense_27b_fixture();
    assert_eq!(resolve_pinned(&c, "qwen3.8-27b", "qwen3_5", 5120), Ok(2));
    assert_eq!(resolve_pinned(&c, "QWEN3.8-27B", "qwen3_5", 5120), Ok(2));
    // Pin selects even a target whose needles would NOT have matched.
    assert_eq!(resolve_pinned(&c, "qwen3.6-27b", "qwen3_5", 5120), Ok(1));
    // Wildcard declarations satisfy a pin.
    assert_eq!(resolve_pinned(&c, "qwen3.5-27b", "qwen3_5", 7777), Ok(0));
}

#[test]
fn pin_to_unknown_target_errors_with_the_available_list() {
    let c = dense_27b_fixture();
    match resolve_pinned(&c, "qwen3.9-27b", "qwen3_5", 5120) {
        Err(TargetResolveError::PinNotFound { available, .. }) => {
            assert_eq!(available, ["qwen3.5-27b", "qwen3.6-27b", "qwen3.8-27b"]);
        }
        other => panic!("expected PinNotFound, got {other:?}"),
    }
}

#[test]
fn pin_to_incompatible_target_errors() {
    let c = dense_27b_fixture();
    // qwen3.8-27b declares only (qwen3_5, 5120); pinning it for a MoE
    // config must refuse rather than serve the wrong kernels.
    assert!(matches!(
        resolve_pinned(&c, "qwen3.8-27b", "qwen3_6_moe", 2048),
        Err(TargetResolveError::PinIncompatible { .. })
    ));
}
