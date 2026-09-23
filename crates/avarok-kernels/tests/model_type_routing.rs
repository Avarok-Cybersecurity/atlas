// SPDX-License-Identifier: AGPL-3.0-only

//! `[[model_types]]` routing pins.
//!
//! `atlas_kernels::ptx_for_config(model_type, hidden_size)` picks a compiled
//! kernel target by matching the `[[model_types]]` claims declared in each
//! `kernels/{hw}/{model}/MODEL.toml`. An exact `(model_type, hidden_size)`
//! claim wins; a claim without `hidden_size` is the wildcard fallback; nothing
//! matching returns `None` and the model cannot boot at all.
//!
//! That makes the claim table load-bearing in a way nothing else checks.
//! Poolside ships Laguna at two hidden sizes, and `laguna-s-2.1` claims
//! `(laguna, 3072)` only — so before `laguna-xs-2.1` existed a 2048-hidden XS
//! checkpoint matched neither the exact rule nor a wildcard and
//! `ptx_for_config` returned `None`. The variants also disagree on
//! `thinking_default` and on their sampling presets, which is expressible only
//! as two MODEL.toml files.
//!
//! These run against the MODEL.toml files in the tree rather than the compiled
//! registry, so they hold on the GPU-free CI runner (`ATLAS_SKIP_BUILD=1`),
//! where `available_targets()` is an empty stub.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/atlas-kernels is two levels below the workspace root")
        .join("kernels")
}

/// One `[[model_types]]` entry: `(model_type, hidden_size)` -> declaring targets.
type Claims = BTreeMap<(String, Option<u64>), Vec<String>>;

/// Every `[[model_types]]` claim under `kernels/{hw}/*/MODEL.toml`, keyed by
/// the pair claimed and valued by the `{hw}/{model}` targets claiming it.
fn claims() -> Claims {
    let mut out: Claims = BTreeMap::new();
    for (target, manifest) in manifests() {
        let text = std::fs::read_to_string(&manifest).expect("manifest is readable");
        let parsed: toml::Value = toml::from_str(&text)
            .unwrap_or_else(|e| panic!("{} is not valid TOML: {e}", manifest.display()));
        let Some(entries) = parsed.get("model_types").and_then(|v| v.as_array()) else {
            continue;
        };
        for entry in entries {
            let model_type = entry
                .get("model_type")
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| {
                    panic!(
                        "{} has a [[model_types]] entry with no model_type",
                        manifest.display()
                    )
                })
                .to_string();
            let hidden = entry.get("hidden_size").and_then(|v| v.as_integer());
            out.entry((model_type, hidden.map(|h| h as u64)))
                .or_default()
                .push(target.clone());
        }
    }
    assert!(
        !out.is_empty(),
        "no [[model_types]] claims found under {} — the walk is broken, not the tree",
        kernels_root().display()
    );
    out
}

/// `({hw}/{model}, path to its MODEL.toml)` for every kernel target in the tree.
fn manifests() -> Vec<(String, PathBuf)> {
    let root = kernels_root();
    let mut out = Vec::new();
    let hw_dirs = std::fs::read_dir(&root).expect("kernels/ is readable");
    for hw in hw_dirs.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
        let Ok(models) = std::fs::read_dir(&hw) else {
            continue;
        };
        for model in models.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
            let manifest = model.join("MODEL.toml");
            if !manifest.is_file() {
                continue;
            }
            out.push((
                format!(
                    "{}/{}",
                    hw.file_name().unwrap().to_string_lossy(),
                    model.file_name().unwrap().to_string_lossy()
                ),
                manifest,
            ));
        }
    }
    out.sort();
    out
}

/// Both Laguna hidden sizes must route, each to its own target and to nothing
/// else. The `assert_eq` on the full claimant list is deliberate: a second
/// target claiming the same pair would make `ptx_for_config` resolve by target
/// enumeration order.
#[test]
fn both_laguna_hidden_sizes_route_to_their_own_target() {
    let claims = claims();
    let lookup = |hidden: u64| -> Vec<String> {
        claims
            .get(&("laguna".to_string(), Some(hidden)))
            .cloned()
            .unwrap_or_default()
    };

    assert_eq!(
        lookup(2048),
        vec!["gb10/laguna-xs-2.1".to_string()],
        "Laguna-XS-2.1 (hidden_size 2048) must be claimed by gb10/laguna-xs-2.1 \
         and only by it; unclaimed means ptx_for_config returns None and XS \
         cannot boot at all"
    );
    assert_eq!(
        lookup(3072),
        vec!["gb10/laguna-s-2.1".to_string()],
        "Laguna-S-2.1 (hidden_size 3072) must stay on gb10/laguna-s-2.1"
    );
    // Non-vacuity: a wildcard `laguna` claim would satisfy both lookups at
    // runtime for the wrong reason, and would silently capture the next
    // variant onto whichever target declared it.
    assert!(
        !claims.contains_key(&("laguna".to_string(), None)),
        "a wildcard `laguna` claim would swallow every future hidden size; \
         each variant gets an explicit claim"
    );
}

/// A target's `[[model_types]]` hidden_size claims must INCLUDE the
/// `[model].hidden_dim` it documents. The two are written independently, and a
/// target whose claims match nothing it documents routes a checkpoint onto a
/// target tuned for another shape.
///
/// ★ WHY "INCLUDE" AND NOT "EQUAL". The first form of this test required EVERY
/// `[[model_types]]` row to equal `[model].hidden_dim`. That premise is false
/// for any target that deliberately matches more than one shape, and
/// `kernels/{b200,b300,gb10}/kimi-k3/MODEL.toml` is exactly that — it says so
/// in its own header:
///
///     🔴 THE MATCHER is [[model_types]]. Dimension keys below are documentation.
///     Two rows so production (7168) never swallows the 0.40B twin (1024).
///
/// so it carries `hidden_size = 7168` (production moonshotai/Kimi-K3) beside
/// `hidden_size = 1024` (the inference-optimization/Kimi-K3-0.40B twin) under a
/// single `[model] hidden_dim = 7168`. The strict form failed all three K3
/// targets. The manifest is right and the premise was wrong: `[model]`
/// documents the production shape, while `[[model_types]]` is the match set.
///
/// The Laguna risk this test was written for survives intact — a target whose
/// claims are ALL foreign to its documented dim still fails, and that is the
/// mis-routing case. What is now permitted is an ADDITIONAL row for a twin.
#[test]
fn claimed_hidden_size_matches_documented_hidden_dim() {
    let mut checked = 0usize;
    let mut violations: Vec<String> = Vec::new();
    for (_, manifest) in manifests() {
        let text = std::fs::read_to_string(&manifest).expect("manifest is readable");
        let parsed: toml::Value = toml::from_str(&text).expect("MODEL.toml parses");
        let Some(hidden_dim) = parsed
            .get("model")
            .and_then(|m| m.get("hidden_dim"))
            .and_then(|v| v.as_integer())
        else {
            continue;
        };
        let Some(entries) = parsed.get("model_types").and_then(|v| v.as_array()) else {
            continue;
        };
        let claims: Vec<i64> = entries
            .iter()
            .filter_map(|e| e.get("hidden_size").and_then(|v| v.as_integer()))
            .collect();
        if claims.is_empty() {
            continue;
        }
        checked += claims.len();
        if !claims.contains(&hidden_dim) {
            violations.push(format!(
                "{}: no [[model_types]] hidden_size claim ({}) matches [model] hidden_dim {hidden_dim}",
                manifest.display(),
                claims
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    assert!(violations.is_empty(), "{}", violations.join("\n"));
    assert!(
        checked >= 2,
        "expected at least the two Laguna targets to carry an explicit \
         hidden_size claim, checked {checked}"
    );
}
