// SPDX-License-Identifier: AGPL-3.0-only

//! Target-resolution pins against the REAL `kernels/gb10/*/MODEL.toml`
//! declarations.
//!
//! `src/resolve_tests.rs` proves the RULES on synthetic fixtures; this file
//! proves the DATA: that the declarations actually checked into the tree
//! route the fleet's checkpoints where the fleet expects. Same pattern as
//! `kernel_shadow_detector.rs` — the logic under test is compiled from
//! source and driven with the real tree, so `cargo test` covers it on the
//! GPU-free `ATLAS_SKIP_BUILD=1` runner where `all_ptx_sets()` is an empty
//! stub.
//!
//! Background: Qwen3.8-27B's config.json is bit-identical to Qwen3.6-27B's
//! (same `model_type` "qwen3_5", same `hidden_size` 5120 — verified
//! tensor-by-tensor 2026-08-14), so both targets declare the same exact
//! pair and resolution must break the tie on the checkpoint reference. A
//! mis-route here would silently serve the MLPerf-edge flagship with the
//! wrong sampling presets and behavior flags, which is exactly the failure
//! these pins exist to make impossible.

use atlas_kernels::resolve::{ResolveCandidate, TargetResolveError, resolve_target};
use atlas_kernels::{ModelTypeMatch, resolve::resolve_pinned};
use std::path::{Path, PathBuf};

fn gb10_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/atlas-kernels is two levels below the workspace root")
        .join("kernels")
        .join("gb10")
}

/// One parsed target: (name, [[model_types]], [model].match_names,
/// [model].kernel_source).
struct ParsedTarget {
    name: &'static str,
    type_matches: Vec<ModelTypeMatch>,
    match_names: Vec<&'static str>,
    kernel_source: Option<String>,
}

/// `ModelTypeMatch` carries `&'static str` (it is normally baked by
/// build.rs); leaking the parsed strings is the test-only bridge.
fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

fn parse_targets() -> Vec<ParsedTarget> {
    let mut out = Vec::new();
    let mut names: Vec<String> = std::fs::read_dir(gb10_dir())
        .expect("kernels/gb10 exists")
        .filter_map(|e| {
            let e = e.ok()?;
            let name = e.file_name().to_string_lossy().to_string();
            e.path().join("MODEL.toml").exists().then_some(name)
        })
        .collect();
    // Same deterministic order build.rs uses (sort by model name).
    names.sort();
    for name in names {
        let path = gb10_dir().join(&name).join("MODEL.toml");
        let text = std::fs::read_to_string(&path).expect("readable MODEL.toml");
        let toml: toml::Value =
            toml::from_str(&text).unwrap_or_else(|e| panic!("bad TOML in {}: {e}", path.display()));
        let type_matches = toml
            .get("model_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|entry| ModelTypeMatch {
                        model_type: leak(
                            entry
                                .get("model_type")
                                .and_then(|v| v.as_str())
                                .expect("model_type is a string"),
                        ),
                        hidden_size: entry
                            .get("hidden_size")
                            .and_then(|v| v.as_integer())
                            .map(|v| v as usize),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let model = toml.get("model");
        let match_names = model
            .and_then(|m| m.get("match_names"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| leak(v.as_str().expect("match_names entries are strings")))
                    .collect()
            })
            .unwrap_or_default();
        let kernel_source = model
            .and_then(|m| m.get("kernel_source"))
            .and_then(|v| v.as_str())
            .map(String::from);
        out.push(ParsedTarget {
            name: leak(&name),
            type_matches,
            match_names,
            kernel_source,
        });
    }
    out
}

fn candidates(parsed: &[ParsedTarget]) -> Vec<ResolveCandidate<'_>> {
    parsed
        .iter()
        .map(|t| ResolveCandidate {
            name: t.name,
            type_matches: &t.type_matches,
            match_names: &t.match_names,
        })
        .collect()
}

fn resolve_name(model_type: &str, hidden: usize, refs: &[&str]) -> Option<&'static str> {
    let parsed = parse_targets();
    let cands = candidates(&parsed);
    resolve_target(&cands, model_type, hidden, refs)
        .unwrap_or_else(|e| {
            panic!("resolution errored for ({model_type}, {hidden}, {refs:?}): {e}")
        })
        .map(|i| parsed[i].name)
}

// ── The dense-27B family ──

#[test]
fn qwen38_checkpoint_resolves_to_its_own_target() {
    assert_eq!(
        resolve_name("qwen3_5", 5120, &["unsloth/Qwen3.8-27B-NVFP4"]),
        Some("qwen3.8-27b")
    );
    // Base repo id and HF-cache path spellings too.
    assert_eq!(
        resolve_name("qwen3_5", 5120, &["Qwen/Qwen3.8-27B"]),
        Some("qwen3.8-27b")
    );
    assert_eq!(
        resolve_name(
            "qwen3_5",
            5120,
            &["/root/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/ab12"]
        ),
        Some("qwen3.8-27b")
    );
}

/// The MLPerf-edge flagship keeps its target. Load-bearing: its Gate D
/// score is defined against qwen3.6-27b's sampling presets and behavior.
#[test]
fn qwen36_flagship_checkpoints_keep_their_target() {
    for id in [
        "unsloth/Qwen3.6-27B-NVFP4",
        "nvidia/Qwen3.6-27B-NVFP4",
        "centml/Qwen3.6-27B-W4A4-mlpinf",
        "Qwen/Qwen3.6-27B",
    ] {
        assert_eq!(
            resolve_name("qwen3_5", 5120, &[id]),
            Some("qwen3.6-27b"),
            "{id}"
        );
    }
}

/// HANDOFF §9a: the standard multi-target build routes the Kbenkhaled 3.5
/// checkpoint to qwen3.6-27b (checked rc=0 2026-08-03). The tie-break must
/// preserve that routing, which is why qwen3.6-27b claims the
/// "qwen3.5-27b" needle.
#[test]
fn kbenkhaled_qwen35_checkpoint_still_routes_to_qwen36() {
    assert_eq!(
        resolve_name("qwen3_5", 5120, &["Kbenkhaled/Qwen3.5-27B-NVFP4"]),
        Some("qwen3.6-27b")
    );
}

/// An identity-free reference (`--model-from-path /model`) cannot break the
/// (qwen3_5, 5120) tie: startup must refuse loudly, never pick by build
/// order — and the error must name both candidates and the remedies.
#[test]
fn identity_free_reference_is_a_hard_error_for_the_dense_27b_pair() {
    let parsed = parse_targets();
    let cands = candidates(&parsed);
    let err = resolve_target(&cands, "qwen3_5", 5120, &["/model"])
        .expect_err("bare path must not resolve the 3.6/3.8 collision");
    match &err {
        TargetResolveError::Ambiguous {
            tier,
            candidates,
            matched,
            ..
        } => {
            assert_eq!(*tier, "exact");
            assert!(matched.is_empty());
            let names: Vec<&str> = candidates.iter().map(|(n, _)| n.as_str()).collect();
            assert_eq!(names, ["qwen3.6-27b", "qwen3.8-27b"]);
        }
        other => panic!("expected Ambiguous, got {other:?}"),
    }
    let msg = err.to_string();
    assert!(msg.contains("--kernel-target"), "missing pin remedy: {msg}");
    // …and the pin actually works for that case.
    let idx = resolve_pinned(&cands, "qwen3.8-27b", "qwen3_5", 5120).expect("pin resolves");
    assert_eq!(parsed[idx].name, "qwen3.8-27b");
}

// ── The pre-existing collisions keep their historical routing ──

#[test]
fn nemotron_puzzle_checkpoint_resolves_to_the_puzzle_target() {
    assert_eq!(
        resolve_name(
            "nemotron_h_puzzle",
            4096,
            &["nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4"]
        ),
        Some("nemotron-labs-3-puzzle-75b-a9b")
    );
    // Super checkpoints don't collide (nemotron_h) and resolve untied.
    assert_eq!(
        resolve_name(
            "nemotron_h",
            4096,
            &["nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"]
        ),
        Some("nemotron-super-120b-a12b")
    );
}

// ── Tree-wide invariants ──

/// Mirror of build.rs's `validate_collision_match_names`, run here because
/// the CI host skips the build script's kernel path entirely
/// (`ATLAS_SKIP_BUILD=1`): every set of differently-named targets sharing a
/// `(model_type, hidden_size)` declaration must declare explicit needles.
#[test]
fn every_colliding_target_declares_match_names() {
    let parsed = parse_targets();
    let mut by_pair: std::collections::HashMap<(&str, Option<usize>), Vec<&ParsedTarget>> =
        std::collections::HashMap::new();
    for t in &parsed {
        for m in &t.type_matches {
            by_pair
                .entry((m.model_type, m.hidden_size))
                .or_default()
                .push(t);
        }
    }
    let mut colliding_groups = 0usize;
    for ((model_type, hidden), group) in by_pair {
        if group.len() < 2 {
            continue;
        }
        colliding_groups += 1;
        for t in group {
            assert!(
                !t.match_names.is_empty(),
                "target '{}' collides on ({model_type}, {hidden:?}) but declares no \
                 [model] match_names — resolution could never select it unambiguously",
                t.name
            );
        }
    }
    assert!(
        colliding_groups >= 2,
        "expected the Nemotron and dense-27B collision groups, saw {colliding_groups}"
    );
}

/// Mirror of build.rs's dominated-needle check: within a colliding
/// `(model_type, hidden_size)` group, every target must own at least one
/// needle that no sibling needle is a substring of — otherwise any
/// reference matching it also matches the sibling and the tier is a
/// guaranteed `Ambiguous` startup error for exactly the checkpoints it
/// exists to route. This shipped once: qwen3.5-27b's `["qwen3.5-27b"]` was
/// a strict subset of qwen3.6-27b's needles on their shared
/// (qwen3_6_moe, 5120) entry, so the "belt-and-braces" tier could only
/// ever hard-error. The entry is gone; this keeps the shape out.
#[test]
fn no_colliding_target_is_needle_dominated() {
    let parsed = parse_targets();
    let mut by_pair: std::collections::HashMap<(&str, Option<usize>), Vec<&ParsedTarget>> =
        std::collections::HashMap::new();
    for t in &parsed {
        for m in &t.type_matches {
            by_pair
                .entry((m.model_type, m.hidden_size))
                .or_default()
                .push(t);
        }
    }
    let mut colliding_groups = 0usize;
    for ((model_type, hidden), group) in by_pair {
        if group.len() >= 2 {
            colliding_groups += 1;
        }
        for a in &group {
            for b in &group {
                if a.name == b.name {
                    continue;
                }
                let winnable = a.match_names.iter().any(|na| {
                    let na = na.to_lowercase();
                    !b.match_names
                        .iter()
                        .any(|nb| na.contains(&nb.to_lowercase()))
                });
                assert!(
                    winnable,
                    "target '{}' can never win the ({model_type}, {hidden:?}) tie \
                     against '{}': every needle in {:?} contains one of {:?}",
                    a.name, b.name, a.match_names, b.match_names
                );
            }
        }
    }
    assert!(
        colliding_groups >= 2,
        "expected the Nemotron and dense-27B collision groups, saw {colliding_groups}"
    );
}

/// The (qwen3_6_moe, 5120) belt-and-braces tier — the hypothetical MoE
/// rewrite of a dense-27B re-upload — resolves UNCONTESTED to qwen3.6-27b
/// now that qwen3.5-27b's dead duplicate entry is gone. Before the removal
/// this exact lookup was a hard `Ambiguous` error (both targets declared
/// the pair and qwen3.5-27b's needles were a subset of qwen3.6-27b's).
#[test]
fn qwen36_moe_beltandbraces_tier_resolves_uncontested() {
    assert_eq!(
        resolve_name("qwen3_6_moe", 5120, &["Kbenkhaled/Qwen3.5-27B-NVFP4"]),
        Some("qwen3.6-27b")
    );
    // Identity-free too: with one declarer there is no tie to break.
    assert_eq!(
        resolve_name("qwen3_6_moe", 5120, &["/model"]),
        Some("qwen3.6-27b")
    );
}

/// Every `kernel_source` redirect points at a real sibling target that owns
/// its sources (no chains), and the redirected quant tree exists — the
/// build-time contract, pinned here for the skip-build runner.
#[test]
fn kernel_source_redirects_are_wellformed() {
    let parsed = parse_targets();
    let names: Vec<&str> = parsed.iter().map(|t| t.name).collect();
    let mut redirects = 0usize;
    for t in &parsed {
        let Some(src) = &t.kernel_source else {
            continue;
        };
        redirects += 1;
        assert!(
            names.contains(&src.as_str()),
            "{}: kernel_source '{src}' names no gb10 target",
            t.name
        );
        let src_parsed = parsed.iter().find(|p| p.name == src.as_str()).unwrap();
        assert!(
            src_parsed.kernel_source.is_none(),
            "{}: kernel_source '{src}' itself redirects — chains are not allowed",
            t.name
        );
        assert!(
            gb10_dir().join(src).join("nvfp4").is_dir(),
            "{}: kernel_source '{src}' has no nvfp4/ kernel tree to compile",
            t.name
        );
    }
    assert!(redirects > 0, "no kernel_source redirects were checked");
}

/// qwen3.8-27b exists, reuses qwen3.6-27b's kernels, and keeps the
/// tool-path penalty-free — the qwen3.6-27b 2026-07-04 lesson (penalties
/// shift the argmax among grammar-legal JSON tokens; BFCL non_live
/// ~85 -> ~76) must not be reintroduced by the new target.
#[test]
fn qwen38_target_declarations_are_sound() {
    let path = gb10_dir().join("qwen3.8-27b").join("MODEL.toml");
    let text = std::fs::read_to_string(&path).expect("qwen3.8-27b/MODEL.toml exists");
    let toml: toml::Value = toml::from_str(&text).expect("valid TOML");

    let model = toml.get("model").expect("[model]");
    assert_eq!(
        model.get("kernel_source").and_then(|v| v.as_str()),
        Some("qwen3.6-27b"),
        "3.8 must reuse the 3.6 kernel tree (no 3.8-specific kernel work exists)"
    );

    let tools = toml
        .get("sampling")
        .and_then(|s| s.get("tools"))
        .expect("[sampling.tools]");
    assert_eq!(
        tools.get("presence_penalty").and_then(|v| v.as_float()),
        Some(0.0),
        "tools presence_penalty must stay 0.0"
    );
    assert_eq!(
        tools.get("frequency_penalty").and_then(|v| v.as_float()),
        Some(0.0),
        "tools frequency_penalty must stay 0.0"
    );
    assert_eq!(
        tools.get("repetition_penalty").and_then(|v| v.as_float()),
        Some(1.0),
        "tools repetition_penalty must stay 1.0"
    );
    assert_eq!(
        tools.get("dry_multiplier").and_then(|v| v.as_float()),
        Some(0.0),
        "tools dry_multiplier must stay 0.0"
    );

    // The 3.6-trained DFlash drafter must NOT be paired with 3.8 weights.
    assert!(
        toml.get("dflash").is_none(),
        "no [dflash] on qwen3.8-27b: z-lab/Qwen3.6-27B-DFlash was trained on \
         3.6 hidden states and would be out-of-distribution on 3.8"
    );
}
