// SPDX-License-Identifier: AGPL-3.0-only

//! `--pull-request-gate-check`: does this commit have a passing record for
//! every required gate? Pure reads over `.benchmarks/` plus git ancestry —
//! no endpoint, no GPU, fast enough for every PR in CI.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::record::{GateBaseline, GateRecord, read_baseline, read_record};
use super::{REQUIRED_GATES, gate_dir};

/// One metric's comparison, or why it cannot be judged.
pub enum Comparison {
    Pass,
    Fail(String),
    Skip(String),
}

/// Compare one recorded metric against its bound.
pub fn compare(name: &str, value: f64, bound: &super::record::Bound) -> Comparison {
    let noise = bound.noise.unwrap_or(0.0);
    match (bound.min, bound.max) {
        (Some(min), None) if value + noise >= min => Comparison::Pass,
        (Some(min), None) => Comparison::Fail(format!(
            "{name} {value:.2} is below the floor {min:.2} (noise {noise:.2})"
        )),
        (None, Some(max)) if value - noise <= max => Comparison::Pass,
        (None, Some(max)) => Comparison::Fail(format!(
            "{name} {value:.2} is above the ceiling {max:.2} (noise {noise:.2})"
        )),
        _ => Comparison::Skip(format!("{name} has a malformed bound")),
    }
}

/// Check one record against its baseline. `None` means every checkable metric
/// passed; `Some` carries the list of failures. A record whose model does not
/// match the baseline's is a hard failure — comparing gate numbers across
/// checkpoints manufactures results.
pub fn check_record(record: &GateRecord, baseline: &GateBaseline) -> Option<Vec<String>> {
    // The record names both axes: which box served it, and which checkpoint.
    // Score it against THAT pair's thresholds or not at all — a TTFT ceiling
    // from another box, or a BFCL floor from another checkpoint, is not a
    // lenient comparison, it is a meaningless one.
    let hardware = record.hardware.gate_key();
    let entry = match baseline.resolve(&hardware, Some(&record.target_model)) {
        Ok((_, entry)) => entry,
        Err(e) => return Some(vec![format!("{e:#}")]),
    };
    let mut problems = Vec::new();
    for (name, bound) in &entry.metrics {
        let Some(value) = record.metrics.get(name) else {
            problems.push(format!("{name}: missing from the record"));
            continue;
        };
        match compare(name, *value, bound) {
            Comparison::Pass => {}
            Comparison::Fail(reason) => problems.push(reason),
            Comparison::Skip(reason) => problems.push(reason),
        }
    }
    if problems.is_empty() {
        None
    } else {
        Some(problems)
    }
}

/// One required bench's standing in the committed tree.
#[derive(Debug)]
pub enum GateStatus {
    /// The newest covering record passes the baseline.
    Pass,
    /// The newest covering record exists but fails: the record's own verdict
    /// and each baseline breach.
    Fail(Vec<String>),
    /// No covering record exists, or the newest one is unreadable or never
    /// completed.
    Missing(String),
}

/// The newest-first list of record files in one benchmark's directory. The
/// `YYYY-MM-DD-<sha>` prefix keeps lexical order chronological, so a sort is
/// a time sort; `BASELINE.json` is not a record.
pub fn records_newest_first(root: &Path, benchmark_id: &str) -> Vec<PathBuf> {
    let dir = gate_dir(root, benchmark_id);
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|entries| entries.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    candidates.retain(|p| {
        p.extension().is_some_and(|e| e == "json")
            && p.file_name()
                .is_some_and(|n| n.to_string_lossy() != "BASELINE.json")
    });
    candidates.sort();
    candidates.reverse();
    candidates
}

/// Whether a record measured at `record_sha` still stands for `head`.
///
/// Same commit always covers itself. An ancestor covers `head` while nothing
/// the binary measures changed in between — a diff touching `crates/`,
/// `kernels/`, `Cargo.toml`, `Cargo.lock` or `vendor/` invalidates every
/// earlier record, because the measured binary is no longer the recorded
/// one. A record can never be written AT `head` (committing it moves head),
/// so this ancestry rule is what makes "gated at the current commit"
/// achievable at all.
pub fn record_covers(root: &Path, head: &str, record_sha: &str) -> bool {
    if head == record_sha {
        return true;
    }
    let is_ancestor = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["merge-base", "--is-ancestor", record_sha, head])
        .stdin(std::process::Stdio::null())
        .output()
        .is_ok_and(|o| o.status.success());
    if !is_ancestor {
        return false;
    }
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "diff",
            "--name-only",
            record_sha,
            head,
            "--",
            "crates",
            "kernels",
            "Cargo.toml",
            "Cargo.lock",
            "vendor",
        ])
        .stdin(std::process::Stdio::null())
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().is_empty(),
        _ => false,
    }
}

/// The full gate verdict for `sha`: every required bench, in order.
pub fn check_gates(root: &Path, sha: &str) -> BTreeMap<String, GateStatus> {
    let mut out = BTreeMap::new();
    for id in REQUIRED_GATES {
        out.insert((*id).to_string(), check_one(root, id, sha));
    }
    out
}

fn check_one(root: &Path, benchmark_id: &str, sha: &str) -> GateStatus {
    let paths = records_newest_first(root, benchmark_id);
    if paths.is_empty() {
        return GateStatus::Missing("no gate records committed".into());
    }
    // The newest record that still stands for `sha`. A record measured at an
    // ancestor covers head while no perf-path file changed in between; a
    // record whose commit is unrelated, or was invalidated since, is skipped
    // rather than failed — the branch's current word is the newest one still
    // valid, and an old clean record is better than none.
    let mut covered: Option<GateRecord> = None;
    for path in &paths {
        if let Ok(record) = read_record(path)
            && record_covers(root, sha, &record.git_sha)
        {
            covered = Some(record);
            break;
        }
    }
    let Some(record) = covered else {
        let newest_sha = read_record(&paths[0]).ok().map(|r| r.git_sha);
        return GateStatus::Missing(match newest_sha {
            Some(newest) => format!(
                "latest record is for {newest} ({}), which does not cover this commit",
                paths[0]
                    .file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or_default()
            ),
            None => "latest record is unreadable".to_string(),
        });
    };
    if record.frame_status_failed() {
        return GateStatus::Fail(vec![format!(
            "the run itself failed: {}",
            record.verdict_reason
        )]);
    }
    let baseline = match read_baseline(root, benchmark_id) {
        Ok(b) => b,
        Err(e) => return GateStatus::Missing(format!("baseline unreadable: {e:#}")),
    };
    let mut problems = Vec::new();
    if !record.verdict_passes() {
        problems.push(format!(
            "run verdict is not PASS: {}",
            record.verdict_reason
        ));
    }
    if let Some(breaches) = check_record(&record, &baseline) {
        problems.extend(breaches);
    }
    if problems.is_empty() {
        GateStatus::Pass
    } else {
        GateStatus::Fail(problems)
    }
}
