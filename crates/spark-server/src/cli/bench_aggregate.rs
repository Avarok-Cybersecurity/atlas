// SPDX-License-Identifier: AGPL-3.0-only
//! `spark benchmark aggregate <group>` — the group's number, from what is
//! already committed.
//!
//! Pure and GPU-free. It exists because a sharded gate has a failure mode a
//! single gate does not: three of the four members present, which is not 75%
//! measured but a DIFFERENT measurement. Waiting for CI to discover that after
//! a campaign is the expensive way to find out; this says it in a second.

use anyhow::{Result, bail};
use atlas_plugin::benchmarks::bfcl::aggregate;
use atlas_plugin::gate;

use super::bench_args::{AggregateArgs, OutputFormat};

pub fn aggregate_cmd(args: AggregateArgs) -> Result<i32> {
    let root = super::bench_run::repo_root()?;
    let Some(group) = gate::group::find(&args.id) else {
        // Say what it IS, when it is a member — "unknown group" would be wrong
        // and would send the reader looking for a typo.
        if let Some(g) = gate::group::member_of(&args.id) {
            bail!(
                "{} is a MEMBER of the group {}, not a group. A shard has no \
                 aggregate of its own; aggregate {} instead.",
                args.id,
                g.id,
                g.id
            );
        }
        let known: Vec<&str> = gate::group::GROUPS.iter().map(|g| g.id).collect();
        bail!(
            "{} is not a benchmark group — the groups are: {}",
            args.id,
            known.join(", ")
        );
    };
    let sha = match &args.sha {
        Some(s) => s.clone(),
        None => gate::git_sha(&root)?,
    };

    let mut shards = Vec::new();
    let mut missing = Vec::new();
    let mut rows = Vec::new();
    for m in group.members {
        match newest_tallies(&root, m) {
            Some((path, t)) => {
                let n: u64 = t.values().map(|x| x.n).sum();
                let hits: u64 = t.values().map(|x| x.hits).sum();
                rows.push((m.to_string(), path, hits, n));
                shards.push(t);
            }
            None => missing.push(*m),
        }
    }

    if args.format == OutputFormat::Json {
        let agg = aggregate::aggregate(&aggregate::union(&shards));
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "group": group.id,
                "sha": sha,
                "complete": missing.is_empty(),
                "missing": missing,
                "members": rows.iter().map(|(id, p, h, n)| serde_json::json!({
                    "id": id, "record": p, "hits": h, "n": n
                })).collect::<Vec<_>>(),
                "overall_accuracy": agg.overall_accuracy,
                "normalized_single_turn_score": agg.normalized_single_turn_score,
                "samples": agg.total_samples,
            }))?
        );
        return Ok(i32::from(!missing.is_empty()));
    }

    println!("group {} at {sha}", group.id);
    for (id, path, hits, n) in &rows {
        println!("  {id:<24} {hits:>5} / {n:<5}  {path}");
    }
    for m in &missing {
        println!("  {m:<24}     —   NO COVERING RECORD");
    }
    if !missing.is_empty() {
        println!();
        println!(
            "INCOMPLETE — {} of {} members have no record. A group is satisfied only \
             when EVERY member has run: an aggregate over a subset is computed on a \
             sample set the thresholds were never drawn against, which is a different \
             measurement, not a partial one.",
            missing.len(),
            group.members.len()
        );
        return Ok(1);
    }
    let agg = aggregate::aggregate(&aggregate::union(&shards));
    println!();
    println!(
        "  overall_accuracy              {:.2}",
        agg.overall_accuracy
    );
    println!(
        "  normalized_single_turn_score  {:.2}",
        agg.normalized_single_turn_score
    );
    println!("  samples                       {}", agg.total_samples);
    Ok(0)
}

/// The newest record for `member` that carries per-subset tallies.
fn newest_tallies(
    root: &std::path::Path,
    member: &str,
) -> Option<(String, std::collections::BTreeMap<String, aggregate::Tally>)> {
    for path in gate::check::records_newest_first(root, member) {
        let Ok(record) = gate::read_record(&path) else {
            continue;
        };
        if record.benchmark_id != member {
            continue;
        }
        if let Some(t) = aggregate::tallies_from_metrics(&record.metrics) {
            return Some((path.file_name()?.to_string_lossy().into_owned(), t));
        }
    }
    None
}
