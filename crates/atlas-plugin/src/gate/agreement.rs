// SPDX-License-Identifier: AGPL-3.0-only
//! Do the records a PR ADDS agree with each other?
//!
//! Each record is already bound to a commit by its signature. This binds them
//! to *each other*: without it a PR can present a favourable record measured at
//! one commit beside another measured at a different commit, each individually
//! valid and signed.
//!
//! # Why signer agreement is per metric class
//!
//! The rule used to be blanket — "One campaign, one box, one identity" — and it
//! cost a night. A campaign split across three boxes to save wall-clock produced
//! a record set spanning three signing keys (the key is
//! `<ATLAS_HOME>/identity/ed25519.pk8`, so it is per ATLAS_HOME, not per
//! machine; one box had two). CI rejected the lot and seven gates were
//! re-measured on one box.
//!
//! The blanket rule is right for half the suite and wrong for the other half,
//! and [`Sensitivity`] already draws exactly that line:
//!
//! * [`Sensitivity::Speed`] — "thermally corruptible". Boxes genuinely differ by
//!   far more than their own run-to-run noise. Measured 2026-09-06 on one gate
//!   at one commit: dgx2 mean 22.78 tok/s (sigma 0.063, n=10), dgx3 mean 23.44
//!   (sigma 0.070, n=10). A 0.66 tok/s gap, ten times either box's sigma. A
//!   floor drawn on one box does not describe another, so these records must
//!   come from ONE box.
//! * [`Sensitivity::Correctness`] — "accuracy, fidelity, state integrity". The
//!   same BFCL gate measured independently on dgx1 and dgx2 at sha 82552fe34d
//!   returned byte-identical scores: overall 86.55, normalized 86.95, n=1004.
//!   Nothing about the box enters the number, so spanning boxes is sound — and
//!   it is what lets a sharded gate run four ways at once.
//!
//! Every signer must still be committed in `.github/record-signers/`; this
//! relaxes WHICH keys may appear together, never whether a key is vouched for.

use super::coverage;
use crate::hardware::policy::Sensitivity;
use crate::registry;

/// One record as the agreement rule sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedRecord {
    /// Path, for the operator — never parsed.
    pub path: String,
    /// The benchmark this record is for.
    pub benchmark_id: String,
    /// The commit the record was measured at.
    pub git_sha: String,
    /// The signing key fingerprint from the `.sig` sidecar.
    pub signer: String,
}

/// Why a set of added records does not hang together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Disagreement {
    /// Records measured at more than one commit. Always fatal, every class.
    Commits(Vec<String>),
    /// Speed-class records signed by more than one identity.
    SpeedSigners {
        /// The gates that forced the rule, so the operator knows which to redo.
        gates: Vec<String>,
        signers: Vec<String>,
    },
    /// A record naming a benchmark the registry does not have. Fails closed:
    /// an unknown id must not default to the permissive class.
    UnknownBenchmark(String),
}

impl std::fmt::Display for Disagreement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Commits(shas) => write!(
                f,
                "records measured at {} different commits ({}). A certification \
                 is ONE campaign at ONE commit; re-measure the stragglers at the \
                 head you intend to merge.",
                shas.len(),
                shas.join(", ")
            ),
            Self::SpeedSigners { gates, signers } => write!(
                f,
                "{} speed-class gate(s) ({}) carry {} different signing keys ({}). \
                 Throughput and latency are box-dependent — measured 0.66 tok/s \
                 between two boxes against a within-box sigma of 0.07 — so these \
                 must all come from ONE box. Correctness-class gates may span \
                 boxes; these may not.",
                gates.len(),
                gates.join(", "),
                signers.len(),
                signers.join(", ")
            ),
            Self::UnknownBenchmark(id) => write!(
                f,
                "record names benchmark {id:?}, which is not in the registry — \
                 refusing to classify it. An unrecognised gate must not inherit \
                 the permissive rule."
            ),
        }
    }
}

/// The class a gate's records belong to.
///
/// Reads the registry, never the record: a record that asserted its own class
/// could choose the permissive one.
pub fn sensitivity_of(benchmark_id: &str) -> Option<Sensitivity> {
    registry::find(benchmark_id).map(|d| d.sensitivity)
}

/// Check that the records a PR adds agree with one another.
///
/// Returns every disagreement found rather than the first, so one CI run tells
/// the operator everything that needs re-measuring.
pub fn check(added: &[AddedRecord]) -> Vec<Disagreement> {
    let mut out = Vec::new();
    if added.is_empty() {
        return out;
    }

    let mut shas: Vec<String> = added.iter().map(|r| r.git_sha.clone()).collect();
    shas.sort();
    shas.dedup();
    if shas.len() > 1 {
        out.push(Disagreement::Commits(shas));
    }

    let mut speed_gates: Vec<String> = Vec::new();
    let mut speed_signers: Vec<String> = Vec::new();
    for r in added {
        match sensitivity_of(&r.benchmark_id) {
            None => out.push(Disagreement::UnknownBenchmark(r.benchmark_id.clone())),
            Some(Sensitivity::Speed) => {
                speed_gates.push(r.benchmark_id.clone());
                speed_signers.push(r.signer.clone());
            }
            Some(Sensitivity::Correctness) => {}
        }
    }
    speed_gates.sort();
    speed_gates.dedup();
    speed_signers.sort();
    speed_signers.dedup();
    if speed_signers.len() > 1 {
        out.push(Disagreement::SpeedSigners {
            gates: speed_gates,
            signers: speed_signers,
        });
    }
    out
}

/// Every required gate, split by the class that decides its signer rule.
///
/// Exists so the split is inspectable rather than something a reader has to
/// reconstruct by grepping descriptors.
pub fn required_by_class() -> (Vec<&'static str>, Vec<&'static str>) {
    let mut speed = Vec::new();
    let mut correctness = Vec::new();
    for g in coverage::REQUIRED.iter() {
        match sensitivity_of(g.id) {
            Some(Sensitivity::Speed) => speed.push(g.id),
            Some(Sensitivity::Correctness) => correctness.push(g.id),
            None => {}
        }
    }
    (speed, correctness)
}
