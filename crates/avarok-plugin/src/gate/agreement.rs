// SPDX-License-Identifier: AGPL-3.0-only
//! Do the records a PR ADDS agree with each other?
//!
//! Each record is already bound to a commit by its signature. This binds them
//! to *each other* and to the head: without it a PR can present a favourable
//! record measured at one commit beside another measured at a different
//! commit, each individually valid and signed.
//!
//! # Why "one commit" became "each record stands at head"
//!
//! The rule used to demand one `git_sha` across every added record — "ONE
//! campaign at ONE commit". On 2026-09-14 (stack 1089308) it refused sixteen
//! records at one commit beside two sweep records re-earned at the next,
//! where the diff between the two commits was a single file every other gate
//! excludes: every record STOOD at head by the gate's own rule
//! (`check::record_still_stands`), and the set was rejected anyway, at the
//! price of a full fleet campaign that measured nothing new (issue #1086).
//!
//! What the commit rule protects is real — a record from an unrelated tree,
//! or from a commit whose successors changed what the gate measures, must
//! not ride in — and `check::record_standing` answers exactly that, per
//! record, per gate, by CONTENT: the diff from the record's commit to the
//! head must invalidate nothing for that gate (never ancestry — this
//! repository squash-merges, see `coverage_squash_tests`). So the rule is
//! now the owner's formulation: a commit K that is certified stays certified
//! for every K+n that does not touch a perf path. One commit remains the
//! normal OUTCOME of a campaign; it is no longer a requirement.
//!
//! # Why signer agreement is per metric class
//!
//! The rule used to be blanket — "One campaign, one box, one identity" — and it
//! cost a night. A campaign split across three boxes to save wall-clock produced
//! a record set spanning three signing keys (the key is
//! `<AVAROK_HOME>/identity/ed25519.pk8`, so it is per AVAROK_HOME, not per
//! machine; one box had two). CI rejected the lot and seven gates were
//! re-measured on one box.
//!
//! The blanket rule is right for half the suite and wrong for the other half,
//! and `Sensitivity` already draws exactly that line:
//!
//! * `Sensitivity::Speed` — "thermally corruptible". Boxes genuinely differ by
//!   far more than their own run-to-run noise. Measured 2026-09-06 on one gate
//!   at one commit: dgx2 mean 22.78 tok/s (sigma 0.063, n=10), dgx3 mean 23.44
//!   (sigma 0.070, n=10). A 0.66 tok/s gap, ten times either box's sigma. A
//!   floor drawn on one box does not describe another, so these records must
//!   come from ONE box.
//! * `Sensitivity::Correctness` — "accuracy, fidelity, state integrity". The
//!   same BFCL gate measured independently on dgx1 and dgx2 at sha 82552fe34d
//!   returned byte-identical scores: overall 86.55, normalized 86.95, n=1004.
//!   Nothing about the box enters the number, so spanning boxes is sound — and
//!   it is what lets a sharded gate run four ways at once.
//!
//! # When two signers may share the Speed half
//!
//! Since `spark bench certify --with-nodes`, a Speed-class set MAY span boxes —
//! but only boxes that [`crate::hardware::equivalence`] calls one box: same
//! GPU, same driver line, clock ceiling and memory within tolerance, no thermal
//! reason asserted, chassis within 10 °C, and a valid post-run check on every
//! record. The decision is made from the RECORDS' own `hardware` and
//! `hardware_state` captures, so CI judges what was measured rather than what
//! a scheduler believed at planning time. A record that cannot say (no state
//! capture) is not equivalent to anything.
//!
//! Every signer must still be committed in `.github/record-signers/`; this
//! relaxes WHICH keys may appear together, never whether a key is vouched for.

use super::check::Standing;
use super::coverage;
use crate::hardware::equivalence::{EquivalencePolicy, HardwareFingerprint, equivalent};
use crate::hardware::policy::Sensitivity;
use crate::registry;

/// One record as the agreement rule sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct AddedRecord {
    /// Path, for the operator — never parsed.
    pub path: String,
    /// The benchmark this record is for.
    pub benchmark_id: String,
    /// The commit the record was measured at.
    pub git_sha: String,
    /// The signing key fingerprint from the `.sig` sidecar.
    pub signer: String,
    /// What the record says about the box it was measured on. `None` when
    /// the record could not be parsed that far — which makes it equivalent
    /// to nothing.
    pub hardware: Option<HardwareFingerprint>,
    /// The box class the record names (`Hardware::gate_key`), which decides
    /// whose equivalence policy judges it.
    pub hardware_class: String,
    /// Where the record stands at the head being certified, by
    /// [`super::check::record_standing`] — computed by the caller, which is
    /// the only party that knows the head and has the repository.
    pub standing: Standing,
}

/// Why a set of added records does not hang together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Disagreement {
    /// A record that does not stand at the head: measured off this history,
    /// or at a commit whose successors changed what its gate measures.
    /// Always fatal, every class.
    Straggler {
        path: String,
        git_sha: String,
        why: String,
    },
    /// Speed-class records signed by more than one identity, on boxes the
    /// records themselves do not show to be equivalent.
    SpeedSigners {
        /// The gates that forced the rule, so the operator knows which to redo.
        gates: Vec<String>,
        signers: Vec<String>,
        /// Why the boxes are not one box, per offending pair.
        mismatches: Vec<String>,
    },
    /// A record naming a benchmark the registry does not have. Fails closed:
    /// an unknown id must not default to the permissive class.
    UnknownBenchmark(String),
}

impl std::fmt::Display for Disagreement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Straggler { path, git_sha, why } => write!(
                f,
                "{path} was measured at {git_sha} and does not stand at the head: {why}. \
                 Re-measure it at the head you intend to merge."
            ),
            Self::SpeedSigners {
                gates,
                signers,
                mismatches,
            } => write!(
                f,
                "{} speed-class gate(s) ({}) carry {} different signing keys ({}) \
                 and the records do not show the boxes to be equivalent: {}. \
                 Throughput and latency are box-dependent — measured 0.66 tok/s \
                 between two boxes against a within-box sigma of 0.07 — so these \
                 must come from ONE box, or from boxes whose captures agree on \
                 GPU, driver line, clock ceiling, memory and thermal state. \
                 Correctness-class gates may span boxes freely.",
                gates.len(),
                gates.join(", "),
                signers.len(),
                signers.join(", "),
                mismatches.join("; ")
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

/// A record's [`Standing`] at `head`, with the coverage its benchmark reads
/// (a shard is a run of its group and reads the group's entry). A benchmark
/// with no coverage entry cannot be judged and is reported as unknown — the
/// fail-closed side.
pub fn standing_at(root: &std::path::Path, head: &str, record: &super::GateRecord) -> Standing {
    match coverage::find(&record.benchmark_id) {
        Some(gate) => super::check::record_standing(root, head, record, gate),
        None => Standing::Unknown,
    }
}

fn policy_for(root: &std::path::Path, class: &str) -> anyhow::Result<Option<EquivalencePolicy>> {
    EquivalencePolicy::speed_for(root, class)
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
/// the operator everything that needs re-measuring. `root` is where the
/// hardware class's equivalence policy is read from
/// (`kernels/<hw>/HARDWARE.toml`); a class that declares none makes every
/// cross-signer Speed pair a mismatch by name — nothing is borrowed from
/// another card.
pub fn check(root: &std::path::Path, added: &[AddedRecord]) -> Vec<Disagreement> {
    let mut out = Vec::new();
    if added.is_empty() {
        return out;
    }

    for r in added {
        let why = match &r.standing {
            Standing::Stands => continue,
            Standing::Unknown => {
                "its commit cannot be diffed against the head (unknown to this repository, \
                 or git failed)"
                    .to_string()
            }
            Standing::Invalidated(paths) => format!(
                "commits since it touched what its gate measures ({})",
                paths.join(", ")
            ),
        };
        out.push(Disagreement::Straggler {
            path: r.path.clone(),
            git_sha: r.git_sha.clone(),
            why,
        });
    }

    let speed: Vec<&AddedRecord> = added
        .iter()
        .filter(|r| match sensitivity_of(&r.benchmark_id) {
            None => {
                out.push(Disagreement::UnknownBenchmark(r.benchmark_id.clone()));
                false
            }
            Some(Sensitivity::Speed) => true,
            Some(Sensitivity::Correctness) => false,
        })
        .collect();
    let mut speed_signers: Vec<String> = speed.iter().map(|r| r.signer.clone()).collect();
    speed_signers.sort();
    speed_signers.dedup();
    if speed_signers.len() > 1 {
        // Every cross-signer pair must be one box by the records' own word.
        let mut mismatches = Vec::new();
        let mut gates = Vec::new();
        for (i, a) in speed.iter().enumerate() {
            for b in &speed[i + 1..] {
                if a.signer == b.signer {
                    continue;
                }
                let why = match (&a.hardware, &b.hardware) {
                    (Some(x), Some(y)) => match policy_for(root, &a.hardware_class) {
                        Ok(Some(policy)) => match equivalent(x, y, &policy) {
                            Ok(()) => continue,
                            Err(m) => m
                                .iter()
                                .map(ToString::to_string)
                                .collect::<Vec<_>>()
                                .join(", "),
                        },
                        Ok(None) => format!(
                            "kernels/{}/HARDWARE.toml declares no [benchmarks.limits.thermal] \
                             envelope, so two boxes of that class are never one box",
                            a.hardware_class
                        ),
                        Err(e) => format!("{e:#}"),
                    },
                    _ => "a record carries no hardware capture".to_owned(),
                };
                mismatches.push(format!(
                    "{} ({}) vs {} ({}): {why}",
                    a.benchmark_id,
                    short(&a.signer),
                    b.benchmark_id,
                    short(&b.signer)
                ));
                gates.push(a.benchmark_id.clone());
                gates.push(b.benchmark_id.clone());
            }
        }
        if !mismatches.is_empty() {
            gates.sort();
            gates.dedup();
            out.push(Disagreement::SpeedSigners {
                gates,
                signers: speed_signers,
                mismatches,
            });
        }
    }
    out
}

fn short(signer: &str) -> &str {
    &signer[..signer.len().min(12)]
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
