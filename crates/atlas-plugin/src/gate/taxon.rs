// SPDX-License-Identifier: AGPL-3.0-only

//! The kernel taxonomy: `kernels/<hardware>/<model>/<quant>/`.
//!
//! This walk must agree with `atlas-kernels/build.rs::resolve_targets()`, which
//! is the authority on what gets compiled. It is duplicated rather than shared
//! because that code lives in a build script — not linkable from a normal crate
//! — and a gate that cannot enumerate targets without a CUDA toolchain is a
//! gate that cannot run in CI. `taxon_tests.rs` cross-checks the two against the
//! real tree; disagreement is a lie, not a discrepancy.
//!
//! # Fail-closed
//!
//! Every fallible step here resolves to "affected" rather than "unaffected". A
//! target whose sources cannot be resolved must never read as unchanged: the
//! hash of an empty source set is a constant, so a silent resolution failure
//! would make every target look identical to every other. [`sources`] returns
//! `None` rather than an empty vector for exactly that reason.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// One compiled unit: `(hardware, model, quant)`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Target {
    pub hardware: String,
    pub model: String,
    pub quant: String,
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/{}", self.hardware, self.model, self.quant)
    }
}

/// Kernel-source extension for a vendor.
///
/// `build_target.rs` owns the real mapping but lives in a build script. An
/// unknown vendor returns `None` and its targets resolve as affected — a new
/// backend must be taught this table before the gate will skip anything for it,
/// which is the safe direction.
fn source_ext(vendor: &str) -> Option<&'static str> {
    match vendor {
        "nvidia" | "amd" | "hip" => Some("cu"),
        "apple" => Some("metal"),
        _ => None,
    }
}

fn subdirs(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .collect();
    out.sort();
    out
}

/// The vendor string from `kernels/<hw>/HARDWARE.toml`.
///
/// Parsed by line rather than with a TOML crate: this is one flat key, and the
/// gate crate has no TOML dependency. A malformed file yields `None`, and the
/// hardware's targets then resolve as affected.
fn vendor(root: &Path, hardware: &str) -> Option<String> {
    let text =
        std::fs::read_to_string(root.join("kernels").join(hardware).join("HARDWARE.toml")).ok()?;
    text.lines()
        .map(str::trim)
        .find_map(|l| l.strip_prefix("vendor"))
        .and_then(|rest| rest.trim_start().strip_prefix('='))
        .map(|v| v.trim().trim_matches('"').to_string())
}

/// Directory that owns a model's per-quant kernel sources.
///
/// Most models own their sources. A `[model] kernel_source` entry redirects
/// only the quant directory; the target's own MODEL.toml remains a build
/// input. Malformed TOML, an invalid referent, and redirect chains all fail
/// closed as an unresolved target.
fn kernel_source_dir(root: &Path, hardware: &str, model: &str) -> Option<PathBuf> {
    let hw_dir = root.join("kernels").join(hardware);
    let model_dir = hw_dir.join(model);
    let text = std::fs::read_to_string(model_dir.join("MODEL.toml")).ok()?;
    let parsed: toml::Value = toml::from_str(&text).ok()?;
    let Some(source) = parsed
        .get("model")
        .and_then(|table| table.get("kernel_source"))
    else {
        return Some(model_dir);
    };
    let source = source.as_str()?.trim();
    if source.is_empty() {
        return None;
    }
    let source_dir = hw_dir.join(source);
    let source_text = std::fs::read_to_string(source_dir.join("MODEL.toml")).ok()?;
    let source_toml: toml::Value = toml::from_str(&source_text).ok()?;
    if source_toml
        .get("model")
        .and_then(|table| table.get("kernel_source"))
        .is_some()
    {
        return None;
    }
    Some(source_dir)
}

/// Every target in the tree.
///
/// Mirrors `resolve_targets()` with both wildcards expanded: a hardware dir is
/// one holding `HARDWARE.toml`, a model dir one holding `MODEL.toml` (which is
/// what excludes `common/`), and every subdir of the model's resolved kernel
/// source directory is a quant.
pub fn walk(root: &Path) -> Vec<Target> {
    let kernels = root.join("kernels");
    let mut targets = Vec::new();
    for hardware in subdirs(&kernels) {
        let hw_dir = kernels.join(&hardware);
        if !hw_dir.join("HARDWARE.toml").exists() {
            continue;
        }
        for model in subdirs(&hw_dir) {
            let model_dir = hw_dir.join(&model);
            if !model_dir.join("MODEL.toml").exists() {
                continue;
            }
            let Some(source_dir) = kernel_source_dir(root, &hardware, &model) else {
                continue;
            };
            for quant in subdirs(&source_dir) {
                targets.push(Target {
                    hardware: hardware.clone(),
                    model: model.clone(),
                    quant: quant.clone(),
                });
            }
        }
    }
    targets
}

/// Kernel sources for `target`, after shadowing.
///
/// Mirrors `collect_cu_files`: `common/` is the base layer and the model's
/// quant dir overrides it, keyed by file STEM — so the model dir holds
/// overrides, not the model's kernels. Sorted, so callers get a stable order.
///
/// `None` means resolution failed (unknown vendor, or no source resolved at
/// all). Callers must treat that as affected; see the fail-closed note above.
pub fn sources(root: &Path, target: &Target) -> Option<Vec<PathBuf>> {
    let hw_dir = root.join("kernels").join(&target.hardware);
    let ext = source_ext(&vendor(root, &target.hardware)?)?;
    let source_dir = kernel_source_dir(root, &target.hardware, &target.model)?;

    let mut by_stem: BTreeMap<String, PathBuf> = BTreeMap::new();
    for dir in [hw_dir.join("common"), source_dir.join(&target.quant)] {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for path in entries.flatten().map(|e| e.path()) {
            if path.extension().and_then(|e| e.to_str()) != Some(ext) {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            by_stem.insert(stem.to_string(), path);
        }
    }

    // An empty set is a resolution failure, never a target with no kernels:
    // hashing it would produce one constant shared by every broken target.
    if by_stem.is_empty() {
        return None;
    }
    Some(by_stem.into_values().collect())
}

/// Config files that steer a target's compile without being sources.
///
/// Their contents are inside the closure hash, so editing a tuned constant in
/// `MODEL.toml` invalidates the target that reads it — and only that target.
pub fn configs(root: &Path, target: &Target) -> Vec<PathBuf> {
    let hw_dir = root.join("kernels").join(&target.hardware);
    let source_dir = kernel_source_dir(root, &target.hardware, &target.model)
        .unwrap_or_else(|| hw_dir.join(&target.model));
    [
        hw_dir.join("HARDWARE.toml"),
        hw_dir.join("common").join("KERNEL.toml"),
        hw_dir.join(&target.model).join("MODEL.toml"),
        source_dir.join(&target.quant).join("KERNEL.toml"),
    ]
    .into_iter()
    .filter(|p| p.exists())
    .collect()
}

/// The hardware node a repo-relative path sits under, if any.
///
/// `kernels/gb10/common/x.cu` and `kernels/gb10/qwen3.6-27b/nvfp4/x.cu` both
/// answer `gb10`.
pub fn hardware_of(path: &str) -> Option<&str> {
    path.strip_prefix("kernels/")?.split('/').next()
}

/// The model node a path sits under, if it is under one.
///
/// `common/` is deliberately not a model: a shared kernel belongs to no single
/// model, which is the whole reason the closure hash exists.
pub fn model_of(path: &str) -> Option<(&str, &str)> {
    let rest = path.strip_prefix("kernels/")?;
    let mut parts = rest.split('/');
    let hw = parts.next()?;
    let model = parts.next()?;
    // A file directly in the hardware dir (HARDWARE.toml) has no model, and
    // `common` is a shared dir rather than a model node.
    if model == "common" || parts.next().is_none() {
        return None;
    }
    Some((hw, model))
}

/// Targets a changed path set can affect.
///
/// A path under `common/` affects every target on that hardware — it is shared,
/// and whether a given target actually compiles it is the closure hash's
/// question, not this one. A path under a model's quant dir affects that target
/// alone. A path outside `kernels/` affects nothing *here*; the caller's path
/// boundary already invalidates everything for those.
pub fn affected(root: &Path, changed: &[String]) -> BTreeSet<Target> {
    let all = walk(root);
    let mut out = BTreeSet::new();
    for path in changed {
        let Some(hw) = hardware_of(path) else {
            continue;
        };
        match model_of(path) {
            Some((_, model)) => out.extend(
                all.iter()
                    .filter(|t| {
                        t.hardware == hw
                            && (t.model == model
                                || kernel_source_dir(root, &t.hardware, &t.model)
                                    .and_then(|p| p.file_name()?.to_str().map(str::to_string))
                                    .as_deref()
                                    == Some(model))
                    })
                    .cloned(),
            ),
            None => out.extend(all.iter().filter(|t| t.hardware == hw).cloned()),
        }
    }
    out
}

/// Hardware nodes a change spans.
///
/// Two model nodes under one hardware is provable from paths and can honestly
/// fail a one-node-per-PR check. Two HARDWARE nodes is the documented exemption
/// — kernels are shared, so an AMD port must pass both hardwares' benches in
/// one PR rather than be split into two that are each green alone.
pub fn hardware_span(changed: &[String]) -> BTreeSet<String> {
    changed
        .iter()
        .filter_map(|p| hardware_of(p))
        .map(str::to_string)
        .collect()
}

/// Model nodes a change spans, per hardware.
pub fn model_span(changed: &[String]) -> BTreeSet<(String, String)> {
    changed
        .iter()
        .filter_map(|p| model_of(p))
        .map(|(h, m)| (h.to_string(), m.to_string()))
        .collect()
}

#[cfg(test)]
#[path = "taxon_tests.rs"]
mod taxon_tests;
