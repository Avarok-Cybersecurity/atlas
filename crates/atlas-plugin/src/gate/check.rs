// SPDX-License-Identifier: AGPL-3.0-only

//! `--pull-request-gate-check`: does this commit have a passing record for
//! every required gate? Pure reads over `.benchmarks/` plus git ancestry —
//! no endpoint, no GPU, fast enough for every PR in CI.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::record::{GateBaseline, GateRecord, read_baseline, read_record};
use super::{PERF_PATHS, REQUIRED_GATES, gate_dir};

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
        // BOTH bounds: a range, or — when they are equal — an EXACT pin.
        //
        // ★ This arm was missing, and a two-sided bound fell through to
        // "malformed". That is fail-closed, so nothing was scored leniently,
        // but it made an exact pin unusable: the gate failed every time and
        // blamed the baseline's syntax rather than the measurement. The BFCL
        // draw size is pinned this way (n=995 / n=1004), because a draw that
        // silently changes size produces a plausible score against thresholds
        // that no longer apply.
        (Some(min), Some(max)) if value + noise >= min && value - noise <= max => Comparison::Pass,
        (Some(min), Some(max)) if (min - max).abs() < f64::EPSILON => Comparison::Fail(format!(
            "{name} is {value:.0}, but this gate is pinned to exactly {min:.0} — \
             the run measured something other than what the baseline describes"
        )),
        (Some(min), Some(max)) => Comparison::Fail(format!(
            "{name} {value:.2} is outside [{min:.2}, {max:.2}] (noise {noise:.2})"
        )),
        (None, None) => Comparison::Skip(format!("{name} has no bound")),
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
    // ★ An entry with no thresholds must not read as "everything passed". The
    // loop below is a no-op over an empty map, so without this the strictest
    // possible verdict — Pass, unconditionally, whatever the run measured —
    // would be produced by the WEAKEST possible baseline. A gate with nothing
    // to enforce has not been passed; it has not been defined.
    if entry.metrics.is_empty() {
        return Some(vec![format!(
            "the baseline entry for {} on {hardware} declares no thresholds — \
             there is nothing here for this run to have passed",
            record.target_model
        )]);
    }
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

/// The newest-first list of record files in one benchmark's directory, ordered
/// by each record's own `recorded_at`. `BASELINE.json` is not a record.
///
/// ★ **The filename is not a clock.** A record is named
/// `YYYY-MM-DD-<sha>.json`, so a lexical sort orders by DATE and then by SHA —
/// and a sha is random. Two records cut on the same UTC day therefore sorted by
/// which hex digit happened to come first, which is exactly the situation a
/// re-run produces: measure, commit a fix, measure again, both records dated
/// today. The gate takes the first covering record as the branch's current
/// word, so under the old order a FAIL measured after a PASS was silently
/// discarded whenever its sha sorted lower — the gate passing on a superseded
/// result. It fails the other way just as easily, and neither is detectable
/// after the fact.
///
/// `recorded_at` is written by [`super::record::GateRecord::from_run`] from the
/// run itself, and it is the same number the filename's date is derived from,
/// so the two agree by construction and only the within-day tie changes. An
/// unreadable record sorts last (it can never be selected anyway) but stays in
/// the list, so a directory of nothing but corrupt records still reports
/// "unreadable" rather than "no records committed".
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
    let mut keyed: Vec<(u64, PathBuf)> = candidates
        .into_iter()
        .map(|p| (read_record(&p).map(|r| r.recorded_at).unwrap_or(0), p))
        .collect();
    // Newest first, and the filename breaks a tie so the order is total and
    // reproducible rather than dependent on readdir.
    keyed.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    keyed.into_iter().map(|(_, p)| p).collect()
}

/// Whether a record measured at `record_sha` still stands for `head`.
///
/// Same commit always covers itself. An ancestor covers `head` while nothing
/// the run measures changed in between — a diff touching any of
/// [`PERF_PATHS`] invalidates every earlier record, because the binary and the
/// prompts it renders are no longer the recorded ones. A record can never be
/// written AT `head` (committing it moves head), so this ancestry rule is what
/// makes "gated at the current commit" achievable at all.
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
        .args(["diff", "--name-only", record_sha, head, "--"])
        .args(PERF_PATHS)
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
    // ★ A record measured from a dirty tree does not describe its own sha.
    //
    // `record_covers` above proved that nothing in the invalidation set changed
    // between the record's commit and head. That proof is worthless if the
    // binary already differed from the record's commit when the run started —
    // the diff was never committed, so no ancestry walk can ever see it. Fail
    // rather than skip: the record's numbers are real, but they belong to no
    // commit, and the only thing that makes the file true again is a re-run on
    // a clean tree. Records written before this field existed carry an empty
    // vector and are unaffected.
    if !record.dirty_paths.is_empty() {
        problems.push(format!(
            "measured from a dirty tree — {} uncommitted invalidation-set \
             file(s) when the run started ({}), so the binary was not {}",
            record.dirty_paths.len(),
            record.dirty_paths.join(", "),
            record.git_sha
        ));
    }
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
