// SPDX-License-Identifier: AGPL-3.0-only

//! The campaign's last word: the same gate table `--pull-request-gate-check`
//! prints, then the agreement check over every record the campaign added.
//!
//! Both are required for "certified". A campaign whose every unit passed can
//! still have written records that the coverage check rejects (a shard set
//! with a duplicated index, a record that lost its signature) or that
//! disagree with each other (two commits, two Speed-class signers) — and the
//! CI step that refuses those runs after the GPU hours are spent. Running the
//! same two checks here is what makes exit 0 mean what it says.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use atlas_plugin::gate::agreement::{self, AddedRecord, Disagreement};
use atlas_plugin::gate::{self, GateStatus};

/// Every `.benchmarks/**/*.json` that git does not track — the records a
/// commit would add. Includes what this campaign wrote and what a previous,
/// interrupted one left behind, because the commit will carry both.
pub fn untracked_records(root: &Path) -> Result<Vec<PathBuf>> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            ".benchmarks",
        ])
        .stdin(std::process::Stdio::null())
        .output()
        .context("listing untracked records")?;
    if !out.status.success() {
        anyhow::bail!(
            "git ls-files failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.ends_with(".json"))
        .map(|l| root.join(l))
        .collect())
}

/// The signer fingerprint a record's sidecar names, or `NO_SIDECAR`.
pub fn signer_of(record: &Path) -> String {
    std::fs::read_to_string(gate::signing::sig_path(record))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v.get("key")?.as_str().map(str::to_owned))
        .unwrap_or_else(|| "NO_SIDECAR".into())
}

/// Read the added records into the shape the agreement rule judges, each
/// with its standing at `anchor`.
pub fn added_records(root: &Path, anchor: &str, paths: &[PathBuf]) -> Vec<AddedRecord> {
    paths
        .iter()
        .filter_map(|p| {
            let r = gate::read_record(p).ok()?;
            Some(AddedRecord {
                path: p.display().to_string(),
                hardware: Some(
                    atlas_plugin::hardware::equivalence::HardwareFingerprint::from_record(&r),
                ),
                hardware_class: r.hardware.gate_key(),
                standing: agreement::standing_at(root, anchor, &r),
                benchmark_id: r.benchmark_id,
                git_sha: r.git_sha,
                signer: signer_of(p),
            })
        })
        .collect()
}

pub struct Final {
    pub statuses: BTreeMap<String, GateStatus>,
    pub open: Vec<&'static str>,
    pub added: Vec<AddedRecord>,
    pub disagreements: Vec<Disagreement>,
}

impl Final {
    pub fn certified(&self) -> bool {
        self.open.is_empty() && self.disagreements.is_empty()
    }
}

/// Evaluate without printing — the driver prints in its own format.
pub fn evaluate(root: &Path, anchor: &str) -> Result<Final> {
    let statuses = gate::check_gates(root, anchor);
    let open: Vec<&'static str> = gate::REQUIRED_GATES
        .iter()
        .copied()
        .filter(|id| !matches!(statuses.get(*id), Some(GateStatus::Pass)))
        .collect();
    let added = added_records(root, anchor, &untracked_records(root)?);
    let disagreements = agreement::check(root, &added);
    Ok(Final {
        statuses,
        open,
        added,
        disagreements,
    })
}

/// The human rendering of the verdict.
pub fn print(f: &Final, anchor: &str, root: &Path) {
    println!();
    println!("gate check for {anchor} ({})", root.display());
    let open = super::super::bench_gate_check::print_statuses(&f.statuses);
    debug_assert_eq!(open, f.open);
    println!();
    println!("{} record(s) would be added by a commit:", f.added.len());
    for a in &f.added {
        println!(
            "  {:<60} gate={} sha={} signer={}",
            a.path, a.benchmark_id, a.git_sha, a.signer
        );
    }
    for d in &f.disagreements {
        println!("  DISAGREE  {d}");
    }
    println!();
    if f.certified() {
        println!(
            "CERTIFIED at {anchor}: all {} required gates pass and the added records agree.",
            gate::REQUIRED_GATES.len()
        );
    } else {
        println!(
            "NOT CERTIFIED at {anchor}: {} gate(s) open{}",
            f.open.len(),
            if f.disagreements.is_empty() {
                String::new()
            } else {
                format!(", {} disagreement(s)", f.disagreements.len())
            }
        );
    }
}
