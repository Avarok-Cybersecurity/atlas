// SPDX-License-Identifier: AGPL-3.0-only

//! Rung 0: excuse a kernel-only diff when the affected targets compile to the
//! same device code.
//!
//! The path boundary in [`super::coverage`] is the floor and stays exactly as
//! it was — every `kernels/` path still invalidates every gate. This module
//! sits on top and can only ever *narrow* that, never widen it, and only for
//! paths inside `kernels/`.
//!
//! # What "the same" means here
//!
//! A record stores, per target, the closure hash of its sources together with
//! the non-source inputs it was computed under (arch, compiler, nvcc flags).
//! The check recomputes using **the record's own** stored inputs, so the only
//! thing that can move the hash is a change to the sources themselves. That is
//! deliberate and is the exact question rung 0 asks: *did the code change?*
//!
//! Whether the record was measured on a binary built from those sources at all
//! is a different question, answered by the two-sided check at record-write
//! time — not here. Conflating the two would make this check silently depend on
//! the toolchain of whatever machine runs CI.
//!
//! # Fail-closed
//!
//! Every uncertainty resolves to "not excused":
//!
//! - a record written before this existed carries no attestation;
//! - a target the record does not mention;
//! - sources that will not resolve ([`super::taxon::sources`] returning `None`);
//! - an unresolvable `#include`, or any I/O error;
//! - an empty affected set for a non-empty path list.
//!
//! The cost of a false "not excused" is a re-run. The cost of a false "excused"
//! is a shipped regression with a green gate, so the asymmetry is the design.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::taxon::{self, Target};

/// What one target compiled to, and under what.
///
/// The non-source inputs are stored per target rather than per record because
/// `KERNEL.toml` can add per-target nvcc flags; a single record-level copy
/// would be wrong for exactly the targets that tune themselves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetClosure {
    /// Hex sha256 from [`atlas_closure::hash`].
    pub hash: String,
    pub arch: String,
    pub compiler: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flags: Vec<String>,
}

impl TargetClosure {
    fn inputs(&self, root: &Path, target: &Target) -> Option<atlas_closure::ClosureInputs> {
        Some(atlas_closure::ClosureInputs {
            sources: taxon::sources(root, target)?,
            configs: taxon::configs(root, target),
            flags: self.flags.clone(),
            arch: self.arch.clone(),
            compiler: self.compiler.clone(),
        })
    }
}

/// Per-target attestations carried by a record, keyed by `hw/model/quant`.
pub type Attestation = BTreeMap<String, TargetClosure>;

/// Compute an attestation for every target from the TREE.
///
/// ★ Not what a record stores. A record must carry the attestation baked into
/// the measuring binary (`atlas_kernels::TARGET_CLOSURES`, via
/// `GateRecord::with_closure`), because the tree and the binary differ exactly
/// when it matters — a stale `target/`, a dirty tree, an image carried between
/// boxes. Attesting from the tree at record-write time would paper over all
/// three.
///
/// This is the tree-side reference implementation: it builds fixtures in tests
/// and is what `spark-server/tests/closure_attestation.rs` checks the baked
/// values against. A target whose sources will not resolve is omitted, which
/// costs a future re-run rather than excusing it wrongly.
pub fn attest(
    root: &Path,
    arch: &str,
    compiler: &str,
    flags: &BTreeMap<String, Vec<String>>,
) -> Attestation {
    let mut out = BTreeMap::new();
    for target in taxon::walk(root) {
        let key = target.to_string();
        let entry = TargetClosure {
            hash: String::new(),
            arch: arch.to_string(),
            compiler: compiler.to_string(),
            flags: flags.get(&key).cloned().unwrap_or_default(),
        };
        let Some(inputs) = entry.inputs(root, &target) else {
            continue;
        };
        let Ok(hash) = atlas_closure::hash(root, &inputs) else {
            continue;
        };
        out.insert(key, TargetClosure { hash, ..entry });
    }
    out
}

/// Whether an unchanged closure excuses every one of `paths`.
///
/// `paths` are the ones that survived the path boundary. Returns `false` unless
/// all of them are inside `kernels/` *and* every target they can affect still
/// hashes to what the record measured.
pub fn excuses(root: &Path, paths: &[String], attestation: &Attestation) -> bool {
    if paths.is_empty() || attestation.is_empty() {
        return false;
    }
    // Only kernel paths are in scope. Anything else — host code, Cargo.lock,
    // a patch under `3rdparty_patches/` — is outside what a device-code hash
    // can speak about, so its presence ends the question.
    if !paths.iter().all(|p| taxon::hardware_of(p).is_some()) {
        return false;
    }
    let affected = taxon::affected(root, paths);
    if affected.is_empty() {
        // Kernel paths that map to no target: a new hardware dir, a renamed
        // model, a file at a level the walk does not model. Unknown, so no.
        return false;
    }
    affected.iter().all(|target| {
        attestation
            .get(&target.to_string())
            .and_then(|recorded| {
                let inputs = recorded.inputs(root, target)?;
                let current = atlas_closure::hash(root, &inputs).ok()?;
                Some(current == recorded.hash)
            })
            .unwrap_or(false)
    })
}

/// The targets `paths` affects whose code actually changed.
///
/// The honest complement of [`excuses`], for reporting: it names *which*
/// targets re-opened the gate rather than just saying it is open. A target that
/// cannot be resolved or attested is listed — unknown is reported as changed,
/// never omitted.
pub fn changed_targets(root: &Path, paths: &[String], attestation: &Attestation) -> Vec<String> {
    taxon::affected(root, paths)
        .into_iter()
        .filter(|target| {
            !attestation
                .get(&target.to_string())
                .and_then(|recorded| {
                    let inputs = recorded.inputs(root, target)?;
                    let current = atlas_closure::hash(root, &inputs).ok()?;
                    Some(current == recorded.hash)
                })
                .unwrap_or(false)
        })
        .map(|t| t.to_string())
        .collect()
}

#[cfg(test)]
#[path = "closure_tests.rs"]
mod closure_tests;
