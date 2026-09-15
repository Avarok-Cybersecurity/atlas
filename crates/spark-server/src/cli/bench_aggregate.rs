// SPDX-License-Identifier: AGPL-3.0-only
//! `spark benchmark aggregate <group>` — the group's number, from what is
//! already committed.
//!
//! Pure and GPU-free. It exists because a sharded gate has a failure mode a
//! single gate does not: an incomplete partition — three of four shards, or
//! a re-run shard at a newer commit than its siblings — which is not a
//! partial measurement but a DIFFERENT one. Waiting for CI to discover that
//! after a campaign is the expensive way to find out; this says it in a
//! second, applying the same partition rule the gate does
//! (`gate::group::select_partition`).

use anyhow::{Result, bail};
use atlas_plugin::benchmarks::bfcl::aggregate;
use atlas_plugin::gate;

use super::bench_args::{AggregateArgs, OutputFormat};

pub fn aggregate_cmd(args: AggregateArgs) -> Result<i32> {
    let root = super::bench_run::repo_root()?;
    let Some(group) = gate::group::find(&args.id) else {
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

    // Every shard record at the commit that carries tallies, newest first.
    let mut rows: Vec<Row> = Vec::new();
    for path in gate::check::records_newest_first(&root, group.id) {
        let Ok(record) = gate::read_record(&path) else {
            continue;
        };
        let Some(shard) = record.shard() else {
            continue;
        };
        if record.benchmark_id != group.id
            || !(record.git_sha.starts_with(&sha) || sha.starts_with(&record.git_sha))
        {
            continue;
        }
        let Some(t) = aggregate::tallies_from_metrics(&record.metrics) else {
            continue;
        };
        rows.push(Row {
            shard,
            git_sha: record.git_sha.clone(),
            recorded_at: record.recorded_at,
            file: path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            tallies: t,
        });
    }
    let shards: Vec<gate::group::ShardRecord> = rows
        .iter()
        .enumerate()
        .map(|(handle, r)| gate::group::ShardRecord {
            index: r.shard.0,
            count: r.shard.1,
            git_sha: r.git_sha.clone(),
            recorded_at: r.recorded_at,
            handle,
        })
        .collect();
    let partition = gate::group::select_partition(group.id, &shards);

    if args.format == OutputFormat::Json {
        let (complete, count, chosen) = match &partition {
            Ok(p) => (true, Some(p.count), p.handles.clone()),
            Err(_) => (false, None, Vec::new()),
        };
        let tallies: Vec<_> = chosen.iter().map(|h| rows[*h].tallies.clone()).collect();
        let agg = aggregate::aggregate(&aggregate::union(&tallies));
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "group": group.id,
                "sha": sha,
                "complete": complete,
                "shard_count": count,
                "fault": partition.as_ref().err().map(ToString::to_string),
                "shards": chosen.iter().map(|h| {
                    let r = &rows[*h];
                    let n: u64 = r.tallies.values().map(|x| x.n).sum();
                    let hits: u64 = r.tallies.values().map(|x| x.hits).sum();
                    serde_json::json!({
                        "index": r.shard.0, "count": r.shard.1, "record": r.file,
                        "hits": hits, "n": n
                    })
                }).collect::<Vec<_>>(),
                "overall_accuracy": agg.overall_accuracy,
                "normalized_single_turn_score": agg.normalized_single_turn_score,
                "samples": agg.total_samples,
            }))?
        );
        return Ok(i32::from(!complete));
    }

    println!("group {} at {sha}", group.id);
    let partition = match partition {
        Ok(p) => p,
        Err(fault) => {
            for r in &rows {
                println!("  [{}/{}]  {}", r.shard.0, r.shard.1, r.file);
            }
            println!();
            println!("INCOMPLETE — {fault}");
            return Ok(1);
        }
    };
    let mut tallies = Vec::new();
    for h in &partition.handles {
        let r = &rows[*h];
        let n: u64 = r.tallies.values().map(|x| x.n).sum();
        let hits: u64 = r.tallies.values().map(|x| x.hits).sum();
        println!(
            "  [{}/{}]  {hits:>5} / {n:<5}  {}",
            r.shard.0, r.shard.1, r.file
        );
        tallies.push(r.tallies.clone());
    }
    let agg = aggregate::aggregate(&aggregate::union(&tallies));
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

/// One shard record with its tallies, as read from the group's directory.
struct Row {
    shard: (usize, usize),
    git_sha: String,
    recorded_at: u64,
    file: String,
    tallies: std::collections::BTreeMap<String, aggregate::Tally>,
}
