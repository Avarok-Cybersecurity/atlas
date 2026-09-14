// SPDX-License-Identifier: AGPL-3.0-only

//! `check_one`'s group path: aggregate a benchmark group's member records
//! into one verdict.
//!
//! Split out of `check.rs` to keep that file under the repository's 500-LoC
//! cap. The rules it enforces — all members present, one commit, a real
//! shard partition, no transport-degraded member, every member judged by the
//! same per-record rules as a plain gate — are documented on
//! [`super::group`], which owns them.
//!
//! ★ Since 2026-09-13 this is the ONLY path a group has. A whole-draw record
//! under the group's own id no longer satisfies it, so every per-record check
//! `check_one` applies to a plain gate is applied HERE, per member: subject,
//! frame status, dirty tree, signature. Before that date the whole-draw arm
//! ran first and masked the fact that a shard was never asked those
//! questions; a forged or dirty-tree shard would have passed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::check::{
    GateStatus, record_is_for, record_is_required_subject, record_still_stands,
    records_newest_first,
};
use super::record::{GateBaseline, GateRecord, read_baseline, read_record};

/// A benchmark group: every member must have a covering record at this commit,
/// and their combined tallies must clear the GROUP's thresholds.
///
/// Reuses `records_newest_first`, `record_still_stands` and `check_record`
/// unchanged — a group changes where the metrics come FROM, not how a gate is
/// judged. What it adds is the all-or-nothing rule: three members of four is
/// not 75% measured, it is an aggregate over a sample set the group's
/// thresholds were never drawn against.
pub(super) fn check_group(
    root: &Path,
    group: &'static super::group::BenchmarkGroup,
    sha: &str,
) -> GateStatus {
    use crate::benchmarks::bfcl::aggregate;

    let baseline = match read_baseline(root, group.id) {
        Ok(b) => b,
        Err(e) => return GateStatus::Missing(format!("baseline unreadable: {e:#}")),
    };

    let mut shards: Vec<BTreeMap<String, aggregate::Tally>> = Vec::new();
    let mut members: Vec<super::group::MemberRecord> = Vec::new();
    let mut newest: Option<GateRecord> = None;
    let mut missing: Vec<&str> = Vec::new();
    let mut why_missing: Vec<String> = Vec::new();
    let mut declared: Vec<(usize, usize)> = Vec::new();
    let mut problems: Vec<String> = Vec::new();

    for member in group.members {
        let Some(member_gate) =
            super::coverage::find(member).or_else(|| super::coverage::find(group.id))
        else {
            return GateStatus::Missing(format!("{member} has no coverage entry"));
        };
        let Some((record, path)) = covering_member(root, &baseline, member, sha, member_gate)
        else {
            missing.push(member);
            why_missing.push(why_member_missing(root, member, sha, member_gate));
            continue;
        };
        problems.extend(member_problems(root, member, &record, &path));
        // A member that ran but carries no per-subset tallies cannot be folded
        // in. Counting it as an empty contribution would shrink the union and
        // score the group over fewer samples than the draw.
        let Some(t) = aggregate::tallies_from_metrics(&record.metrics) else {
            return GateStatus::Missing(format!(
                "{member} has a covering record but no per-subset tallies — it was \
                 measured by a binary older than the shard split, so the group \
                 cannot be aggregated. Re-run {member} at this commit."
            ));
        };
        // A member that lost samples to transport failures scored them as
        // "no call" — the correct answer for most irrelevance rows — so a
        // degraded shard can score BETTER while measuring less. Refuse it
        // rather than fold it in.
        let errs = record
            .metrics
            .get("transport_errors")
            .copied()
            .unwrap_or(0.0);
        if errs > 0.0 {
            return GateStatus::Fail(vec![format!(
                "{member} recorded {errs:.0} transport failures. Each was scored as \
                 \"made no call\", which is the CORRECT answer on the irrelevance \
                 subsets, so a degraded shard can raise the aggregate while \
                 measuring less of the draw. Re-run {member}."
            )]);
        }
        // What this record says it ran. Absent on a pre-shard binary, which
        // cannot be folded in for the same reason a missing tally cannot.
        let (Some(idx), Some(cnt)) = (
            record.metrics.get("shard.index").copied(),
            record.metrics.get("shard.count").copied(),
        ) else {
            return GateStatus::Missing(format!(
                "{member} has a covering record that does not say which shard it \
                 ran — it was measured by a binary older than the shard identity \
                 metric. Re-run {member} at this commit."
            ));
        };
        declared.push((idx as usize, cnt as usize));
        members.push(super::group::MemberRecord {
            id: (*member).to_string(),
            git_sha: record.git_sha.clone(),
        });
        shards.push(t);
        newest = Some(record);
    }

    if !missing.is_empty() {
        let fault = super::group::GroupFault::Missing {
            group: group.id,
            missing,
        };
        if let Some(note) = whole_draw_note(root, group) {
            why_missing.push(note);
        }
        return GateStatus::Missing(format!("{fault} {}", why_missing.join(" ")));
    }
    if !problems.is_empty() {
        return GateStatus::Fail(problems);
    }
    if let Err(fault) = super::group::composition_ok(group, &members) {
        return GateStatus::Fail(vec![fault.to_string()]);
    }
    // The members exist and agree on the commit; do they actually cover the
    // draw exactly once between them?
    if let Err(fault) = super::group::partition_ok(group.id, &declared) {
        return GateStatus::Fail(vec![fault.to_string()]);
    }

    let agg = aggregate::aggregate(&aggregate::union(&shards));
    let Some(mut record) = newest else {
        return GateStatus::Missing(format!("{} has no members", group.id));
    };
    // Judge the AGGREGATE, carrying one member's provenance (checkpoint, serve
    // overrides, hardware) — composition_ok has already established they agree
    // on the commit, and `check_record` needs a record shape, not a new one.
    record.benchmark_id = group.id.to_string();
    record
        .metrics
        .insert("overall_accuracy".into(), agg.overall_accuracy);
    record.metrics.insert(
        "normalized_single_turn_score".into(),
        agg.normalized_single_turn_score,
    );
    record
        .metrics
        .insert("samples".into(), agg.total_samples as f64);

    match super::scoring::check_record(&record, &baseline) {
        None => GateStatus::Pass,
        Some(breaches) => GateStatus::Fail(breaches),
    }
}

/// The newest record for `member` that is FOR it, is the group's required
/// subject, and still stands at `sha` — the same three predicates `check_one`
/// uses to pick a plain gate's record. The PATH travels with the record
/// because the signature sidecar lives beside it.
fn covering_member(
    root: &Path,
    baseline: &GateBaseline,
    member: &str,
    sha: &str,
    gate: &super::coverage::GateCoverage,
) -> Option<(GateRecord, PathBuf)> {
    records_newest_first(root, member)
        .into_iter()
        .find_map(|path| {
            let r = read_record(&path).ok()?;
            (record_is_for(&r, member, &path)
                && record_is_required_subject(baseline, &r, member, &path)
                && record_still_stands(root, sha, &r, gate))
            .then_some((r, path))
        })
}

/// The per-record rules `check_one` applies to a plain gate, asked of one
/// member. A group member is a gate record like any other; being one quarter
/// of the measurement does not exempt it from any of them. Empty means the
/// record counts.
fn member_problems(root: &Path, member: &str, record: &GateRecord, path: &Path) -> Vec<String> {
    let mut problems = Vec::new();
    if record.frame_status_failed() {
        problems.push(format!(
            "{member}: the run itself failed: {}",
            record.verdict_reason
        ));
    }
    if !record.dirty_paths.is_empty() {
        problems.push(format!(
            "{member}: measured from a dirty tree — {} uncommitted invalidation-set \
             file(s) when the run started ({}), so the binary was not {}",
            record.dirty_paths.len(),
            record.dirty_paths.join(", "),
            record.git_sha
        ));
    }
    if let Err(why) = super::signing::verify_record(root, path, &record.git_sha, record.recorded_at)
    {
        problems.push(format!("{member}: {why}"));
    }
    problems
}

/// The members of `group` a certification at `sha` still OWES: those with no
/// covering record, and those whose covering record would not count (failed
/// frame, dirty tree, bad signature). The same two questions `check_group`
/// asks, so a planner that skips the members not listed here skips exactly
/// the shards the verdict will accept — a shard is re-measured only when the
/// gate would refuse it.
///
/// A group with no readable baseline owes every member: nothing can be
/// judged, so nothing is banked.
pub fn members_owed(
    root: &Path,
    group: &'static super::group::BenchmarkGroup,
    sha: &str,
) -> Vec<&'static str> {
    let Ok(baseline) = read_baseline(root, group.id) else {
        return group.members.to_vec();
    };
    group
        .members
        .iter()
        .copied()
        .filter(|member| {
            let Some(gate) =
                super::coverage::find(member).or_else(|| super::coverage::find(group.id))
            else {
                return true;
            };
            match covering_member(root, &baseline, member, sha, gate) {
                Some((record, path)) => !member_problems(root, member, &record, &path).is_empty(),
                None => true,
            }
        })
        .collect()
}

/// Why does `member` have no covering record? Names the perf-path files that
/// re-opened its newest record, so that "your shards are missing" carries the
/// same 20-second-fix property a plain gate's "invalidated by …" does. A
/// member that has never run says so; a group whose directory holds only
/// whole-draw records says that those stopped counting.
fn why_member_missing(
    root: &Path,
    member: &str,
    sha: &str,
    gate: &super::coverage::GateCoverage,
) -> String {
    let paths = records_newest_first(root, member);
    let Some(path) = paths.first() else {
        return format!("{member}: no record has ever been committed.");
    };
    let Ok(newest) = read_record(path) else {
        return format!("{member}: its newest record is unreadable.");
    };
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let Some(why) = super::check_paths::invalidating_paths(root, sha, &newest.git_sha, gate) else {
        return format!(
            "{member}: newest record is for {} ({name}) — git cannot diff that commit \
             against this one; is it in this clone?",
            newest.git_sha
        );
    };
    if why.is_empty() {
        return format!(
            "{member}: newest record is for {} ({name}) — its recorded build inputs \
             do not match this commit.",
            newest.git_sha
        );
    }
    format!(
        "{member}: newest record is for {} ({name}) — invalidated by {}.",
        newest.git_sha,
        super::check_fmt::summarize_paths(&why)
    )
}

/// The one-line explanation appended when a group directory still holds
/// whole-draw records but no member has one: since 2026-09-13 those are
/// history, not evidence.
fn whole_draw_note(root: &Path, group: &super::group::BenchmarkGroup) -> Option<String> {
    let whole = records_newest_first(root, group.id);
    let newest = whole.first()?;
    Some(format!(
        "The {} whole-draw record(s) under {} (newest {}) no longer satisfy the gate: since \
         2026-09-13 a group is certified only by its {} shards — see gate::group.",
        whole.len(),
        group.id,
        newest.file_name().unwrap_or_default().to_string_lossy(),
        group.members.len()
    ))
}
