// SPDX-License-Identifier: AGPL-3.0-only

//! `check_one`'s group path: aggregate a benchmark group's member records
//! into one verdict.
//!
//! Split out of `check.rs` to keep that file under the repository's 500-LoC
//! cap. The rules it enforces — all members present, one commit, a real
//! shard partition, no transport-degraded member — are documented on
//! [`super::group`], which owns them.

use std::collections::BTreeMap;
use std::path::Path;

use super::check::{GateStatus, record_is_for, record_still_stands, records_newest_first};
use super::record::{GateRecord, read_baseline, read_record};

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
    let mut declared: Vec<(usize, usize)> = Vec::new();

    for member in group.members {
        // Same selection rule as a plain gate: the newest record that is FOR
        // this benchmark, is the required subject, and still stands at `sha`.
        let Some(member_gate) =
            super::coverage::find(member).or_else(|| super::coverage::find(group.id))
        else {
            return GateStatus::Missing(format!("{member} has no coverage entry"));
        };
        let mut found: Option<GateRecord> = None;
        for path in &records_newest_first(root, member) {
            if let Ok(r) = read_record(path)
                && record_is_for(&r, member, path)
                && record_still_stands(root, sha, &r, member_gate)
            {
                found = Some(r);
                break;
            }
        }
        let Some(record) = found else {
            missing.push(member);
            continue;
        };
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
        return GateStatus::Missing(
            super::group::GroupFault::Missing {
                group: group.id,
                missing,
            }
            .to_string(),
        );
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
