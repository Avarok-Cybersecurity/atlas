// SPDX-License-Identifier: AGPL-3.0-only

//! `check_one`'s group path: aggregate a benchmark group's shard records
//! into one verdict.
//!
//! Split out of `check.rs` to keep that file under the repository's 500-LoC
//! cap. The rules it enforces — a complete partition at one commit, a real
//! shard partition, no transport-degraded shard, every shard judged by the
//! same per-record rules as a plain gate — are documented on
//! [`super::group`], which owns them.
//!
//! ★ Since 2026-09-13 this is the ONLY path a group has. A whole-draw record
//! under the group's own id no longer satisfies it, so every per-record check
//! `check_one` applies to a plain gate is applied HERE, per shard: subject,
//! frame status, dirty tree, signature. Before that date the whole-draw arm
//! ran first and masked the fact that a shard was never asked those
//! questions; a forged or dirty-tree shard would have passed.
//!
//! ★ Since 2026-09-15 the shard COUNT is whatever the campaign chose: shards
//! are the group's own benchmark run with `--param shard=i/n`, filed under
//! the group's directory with `-s<i>of<n>` in the name, and the verdict
//! takes the newest complete partition the standing records form at one
//! commit ([`super::group::select_partition`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::check::{
    GateStatus, record_is_for, record_is_required_subject, record_still_stands,
    records_newest_first,
};
use super::group::ShardRecord;
use super::record::{GateBaseline, GateRecord, read_baseline, read_record};

/// A group's shard records at `sha` that pass every predicate `check_one`
/// applies to a plain gate's record: for this benchmark, the required
/// subject, still standing — and carrying a shard identity. Whole-draw
/// records are skipped here and mentioned by [`whole_draw_note`].
fn standing_shards(
    root: &Path,
    baseline: &GateBaseline,
    group: &'static str,
    sha: &str,
    gate: &super::coverage::GateCoverage,
) -> Vec<(GateRecord, PathBuf)> {
    records_newest_first(root, group)
        .into_iter()
        .filter_map(|path| {
            let r = read_record(&path).ok()?;
            (r.shard().is_some()
                && record_is_for(&r, group, &path)
                && record_is_required_subject(baseline, &r, group, &path)
                && record_still_stands(root, sha, &r, gate))
            .then_some((r, path))
        })
        .collect()
}

/// The per-record rules `check_one` applies to a plain gate, asked of one
/// shard. A shard is a gate record like any other; being one slice of the
/// measurement does not exempt it from any of them. Empty means the record
/// counts.
fn shard_problems(root: &Path, label: &str, record: &GateRecord, path: &Path) -> Vec<String> {
    let mut problems = Vec::new();
    if record.frame_status_failed() {
        problems.push(format!(
            "{label}: the run itself failed: {}",
            record.verdict_reason
        ));
    }
    if !record.dirty_paths.is_empty() {
        problems.push(format!(
            "{label}: measured from a dirty tree — {} uncommitted invalidation-set \
             file(s) when the run started ({}), so the binary was not {}",
            record.dirty_paths.len(),
            record.dirty_paths.join(", "),
            record.git_sha
        ));
    }
    if let Err(why) = super::signing::verify_record(root, path, &record.git_sha, record.recorded_at)
    {
        problems.push(format!("{label}: {why}"));
    }
    problems
}

fn label(group: &str, shard: (usize, usize)) -> String {
    format!("{group}[{}/{}]", shard.0, shard.1)
}

/// A group: the newest complete partition of shard records at `sha`, every
/// shard clean, aggregated over COUNTS and judged against the group's
/// thresholds.
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
    let Some(gate) = super::coverage::find(group.id) else {
        return GateStatus::Missing(format!("{} has no coverage entry", group.id));
    };

    let candidates = standing_shards(root, &baseline, group.id, sha, gate);
    let shards: Vec<ShardRecord> = candidates
        .iter()
        .enumerate()
        .filter_map(|(handle, (r, _))| {
            let (index, count) = r.shard()?;
            Some(ShardRecord {
                index,
                count,
                git_sha: r.git_sha.clone(),
                recorded_at: r.recorded_at,
                handle,
            })
        })
        .collect();
    let partition = match super::group::select_partition(group.id, &shards) {
        Ok(p) => p,
        Err(fault) => {
            let mut why = vec![fault.to_string()];
            if let Some(note) = whole_draw_note(root, group) {
                why.push(note);
            }
            if let Some(note) = why_stale(root, group.id, sha, gate) {
                why.push(note);
            }
            return GateStatus::Missing(why.join(" "));
        }
    };

    let count = partition.count;
    let mut tallies: Vec<BTreeMap<String, aggregate::Tally>> = Vec::new();
    let mut problems: Vec<String> = Vec::new();
    let mut newest: Option<GateRecord> = None;
    for handle in partition.handles {
        let (record, path) = &candidates[handle];
        let lbl = label(group.id, record.shard().unwrap_or((0, count)));
        problems.extend(shard_problems(root, &lbl, record, path));
        // A shard that ran but carries no per-subset tallies cannot be folded
        // in. Counting it as an empty contribution would shrink the union and
        // score the group over fewer samples than the draw.
        let Some(t) = aggregate::tallies_from_metrics(&record.metrics) else {
            return GateStatus::Missing(format!(
                "{lbl} has a covering record but no per-subset tallies — it was \
                 measured by a binary older than the shard split, so the group \
                 cannot be aggregated. Re-run it at this commit."
            ));
        };
        // A shard that lost samples to transport failures scored them as
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
                "{lbl} recorded {errs:.0} transport failures. Each was scored as \
                 \"made no call\", which is the CORRECT answer on the irrelevance \
                 subsets, so a degraded shard can raise the aggregate while \
                 measuring less of the draw. Re-run it."
            )]);
        }
        tallies.push(t);
        if newest
            .as_ref()
            .is_none_or(|n| record.recorded_at > n.recorded_at)
        {
            newest = Some(record.clone());
        }
    }
    if !problems.is_empty() {
        return GateStatus::Fail(problems);
    }

    let agg = aggregate::aggregate(&aggregate::union(&tallies));
    let Some(mut record) = newest else {
        return GateStatus::Missing(format!("{} has no shards", group.id));
    };
    // Judge the AGGREGATE, carrying one shard's provenance (checkpoint, serve
    // overrides, hardware) — the shards agree on the commit, and
    // `check_record` needs a record shape, not a new one.
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
    record.metrics.insert("shard.count".into(), count as f64);

    match super::scoring::check_record(&record, &baseline) {
        None => GateStatus::Pass,
        Some(breaches) => GateStatus::Fail(breaches),
    }
}

/// The shards of `group` a certification at `sha` still OWES, for a planner
/// that wants `wanted` shards: nothing when a complete partition already
/// stands at one commit; otherwise the indices missing from the most
/// complete partition begun AT `sha` (finishing what was started, whatever
/// its count — a partition is never assembled across commits, so one begun
/// at another commit cannot be finished here); or `0..wanted` when nothing
/// counts yet. A present shard whose record would be refused (failed frame,
/// dirty tree, bad signature) is owed too. The same questions
/// `check_group` asks, so a planner that skips what is not listed skips
/// exactly what the verdict will accept.
///
/// A group with no readable baseline or coverage owes every shard: nothing
/// can be judged, so nothing is banked.
pub fn shards_owed(
    root: &Path,
    group: &'static super::group::BenchmarkGroup,
    sha: &str,
    wanted: usize,
) -> Vec<(usize, usize)> {
    let fresh = |n: usize| (0..n).map(|i| (i, n)).collect::<Vec<_>>();
    let (Ok(baseline), Some(gate)) = (
        read_baseline(root, group.id),
        super::coverage::find(group.id),
    ) else {
        return fresh(wanted);
    };
    let candidates = standing_shards(root, &baseline, group.id, sha, gate);
    // The shards that count: present and clean.
    let clean: Vec<ShardRecord> = candidates
        .iter()
        .enumerate()
        .filter_map(|(handle, (record, path))| {
            let (index, count) = record.shard()?;
            shard_problems(root, &label(group.id, (index, count)), record, path)
                .is_empty()
                .then(|| ShardRecord {
                    index,
                    count,
                    git_sha: record.git_sha.clone(),
                    recorded_at: record.recorded_at,
                    handle,
                })
        })
        .collect();
    if super::group::select_partition(group.id, &clean).is_ok() {
        return Vec::new();
    }
    // Finish the partition at THIS commit that is closest to done.
    let held = super::group::held_by(&clean);
    match held
        .iter()
        .filter(|((_, commit), _)| commit.starts_with(sha) || sha.starts_with(commit.as_str()))
        .max_by_key(|((count, _), idx)| (idx.len() * 1000 / *count, *count))
    {
        Some(((count, _), idx)) => (0..*count)
            .filter(|i| !idx.contains_key(i))
            .map(|i| (i, *count))
            .collect(),
        None => fresh(wanted),
    }
}

/// Why is the group's newest record not counting? Names the perf-path files
/// that re-opened it, so that "your shards are missing" carries the same
/// 20-second-fix property a plain gate's "invalidated by …" does.
fn why_stale(
    root: &Path,
    group: &str,
    sha: &str,
    gate: &super::coverage::GateCoverage,
) -> Option<String> {
    let newest = records_newest_first(root, group)
        .into_iter()
        .find_map(|p| {
            read_record(&p)
                .ok()
                .filter(|r| r.shard().is_some())
                .map(|r| (r, p))
        })?;
    let (record, path) = newest;
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    if record_still_stands(root, sha, &record, gate) {
        return None;
    }
    let why = super::check_paths::invalidating_paths(root, sha, &record.git_sha, gate)?;
    Some(if why.is_empty() {
        format!(
            "Newest shard record is for {} ({name}) — its recorded build inputs do not \
             match this commit.",
            record.git_sha
        )
    } else {
        format!(
            "Newest shard record is for {} ({name}) — invalidated by {}.",
            record.git_sha,
            super::check_fmt::summarize_paths(&why)
        )
    })
}

/// The one-line explanation appended when a group directory holds
/// whole-draw records but no complete partition: since 2026-09-13 those are
/// history, not evidence.
fn whole_draw_note(root: &Path, group: &super::group::BenchmarkGroup) -> Option<String> {
    let whole: Vec<PathBuf> = records_newest_first(root, group.id)
        .into_iter()
        .filter(|p| read_record(p).is_ok_and(|r| r.shard().is_none()))
        .collect();
    let newest = whole.first()?;
    Some(format!(
        "The {} whole-draw record(s) under {} (newest {}) no longer satisfy the gate: since \
         2026-09-13 a group is certified only by a complete partition of shards — see \
         gate::group.",
        whole.len(),
        group.id,
        newest.file_name().unwrap_or_default().to_string_lossy(),
    ))
}
