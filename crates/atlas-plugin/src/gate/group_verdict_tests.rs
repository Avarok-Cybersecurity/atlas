// SPDX-License-Identifier: AGPL-3.0-only
//! `check_one`'s group path: four shards produce one verdict, and a partial set
//! produces none.
//!
//! The transition rule is tested as carefully as the aggregation, because it is
//! the part that can take `main` down: `bfcl-subset` is a REQUIRED context and
//! every record committed today is a whole-draw one.

use super::check::check_one;
use super::tests::{bfcl_baseline, tempdir};
use super::*;
use crate::result::Verdict;
use std::collections::BTreeMap;

const SHA: &str = "1111111111";

/// Plant a shard record carrying per-subset tallies, the way `report.rs` writes
/// them.
fn plant_shard(root: &std::path::Path, id: &str, sha: &str, secs: u64, hits: u64, n: u64) {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 90.0);
    metrics.insert("subset.simple_python.hits".to_string(), hits as f64);
    metrics.insert("subset.simple_python.n".to_string(), n as f64);
    // What the real binary records: which shard ran, and whether the transport
    // dropped any sample. Derived from the id here exactly as the registry
    // binds index to id, so these fixtures keep modelling real records.
    let index = match id.chars().last().expect("shard id ends in a letter") {
        'a' => 0.0,
        'b' => 1.0,
        'c' => 2.0,
        'd' => 3.0,
        other => panic!("{id} does not end in a shard letter: {other}"),
    };
    metrics.insert("shard.index".to_string(), index);
    metrics.insert("shard.count".to_string(), 4.0);
    metrics.insert("transport_errors".to_string(), 0.0);
    let record = super::tests::run_record(metrics, Verdict::pass("ok"));
    let mut gate = GateRecord::from_run(
        &record,
        super::tests::hw(),
        sha.to_string(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    gate.benchmark_id = id.to_string();
    gate.verdict = Some("PASS".to_string());
    gate.recorded_at = secs;
    write_record(root, &gate).unwrap();
}

/// Plant a shard record with the metrics overridden after the fact, for the
/// degraded/mislabelled cases the id-derived defaults cannot express.
fn plant_shard_with(
    root: &std::path::Path,
    id: &str,
    sha: &str,
    secs: u64,
    overrides: &[(&str, f64)],
) {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 90.0);
    metrics.insert("subset.simple_python.hits".to_string(), 95.0);
    metrics.insert("subset.simple_python.n".to_string(), 100.0);
    metrics.insert("shard.count".to_string(), 4.0);
    metrics.insert("transport_errors".to_string(), 0.0);
    let index = match id.chars().last().expect("shard id ends in a letter") {
        'a' => 0.0,
        'b' => 1.0,
        'c' => 2.0,
        'd' => 3.0,
        other => panic!("{id} does not end in a shard letter: {other}"),
    };
    metrics.insert("shard.index".to_string(), index);
    for (k, v) in overrides {
        metrics.insert((*k).to_string(), *v);
    }
    let record = super::tests::run_record(metrics, Verdict::pass("ok"));
    let mut gate = GateRecord::from_run(
        &record,
        super::tests::hw(),
        sha.to_string(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    gate.benchmark_id = id.to_string();
    gate.verdict = Some("PASS".to_string());
    gate.recorded_at = secs;
    write_record(root, &gate).unwrap();
}

fn scaffold() -> tempdir::Dir {
    let dir = tempdir::Dir::new();
    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(dir.path(), id)).unwrap();
        super::fixture_baseline::write_baseline(dir.path(), id, &bfcl_baseline());
    }
    dir
}

/// Four shards, one commit, aggregate clears the group's bar -> Pass. This is
/// the whole feature: no whole-draw record exists anywhere.
#[test]
fn four_shards_satisfy_the_group() {
    let dir = scaffold();
    let root = dir.path();
    for (i, m) in [
        "bfcl-subset-a",
        "bfcl-subset-b",
        "bfcl-subset-c",
        "bfcl-subset-d",
    ]
    .iter()
    .enumerate()
    {
        std::fs::create_dir_all(gate_dir(root, m)).unwrap();
        plant_shard(root, m, SHA, 1_785_891_000 + i as u64, 95, 100);
    }
    assert!(
        matches!(check_one(root, "bfcl-subset", SHA), GateStatus::Pass),
        "{:?}",
        check_one(root, "bfcl-subset", SHA)
    );
}

/// THREE shards is not 75% measured. It must not pass, and the reason must name
/// the shard.
#[test]
fn three_shards_do_not_satisfy_the_group() {
    let dir = scaffold();
    let root = dir.path();
    for (i, m) in ["bfcl-subset-a", "bfcl-subset-b", "bfcl-subset-c"]
        .iter()
        .enumerate()
    {
        std::fs::create_dir_all(gate_dir(root, m)).unwrap();
        plant_shard(root, m, SHA, 1_785_891_000 + i as u64, 95, 100);
    }
    match check_one(root, "bfcl-subset", SHA) {
        GateStatus::Missing(why) => {
            assert!(why.contains("bfcl-subset-d"), "{why}");
            assert!(why.contains("different measurement"), "{why}");
        }
        other => panic!("three shards must not satisfy the group, got {other:?}"),
    }
}

/// THE TRANSITION RULE. A whole-draw record under the group's own id still
/// satisfies it, with no shards present at all. Without this, becoming a group
/// would turn every record already on main into "4 members missing" and red
/// every open PR the moment it landed.
#[test]
fn a_whole_draw_record_still_satisfies_the_group() {
    let dir = scaffold();
    let root = dir.path();
    super::tests::plant(root, "bfcl-subset", SHA, 1_785_891_382, "PASS");
    assert!(
        matches!(check_one(root, "bfcl-subset", SHA), GateStatus::Pass),
        "{:?}",
        check_one(root, "bfcl-subset", SHA)
    );
}

/// And with NO records at all, the group must report like an ordinary gate —
/// "no gate records committed" — not "your four shards are missing". An author
/// whose whole-draw record was invalidated needs to know WHAT invalidated it.
#[test]
fn an_unsharded_group_with_no_records_reports_like_a_plain_gate() {
    let dir = scaffold();
    match check_one(dir.path(), "bfcl-subset", SHA) {
        GateStatus::Missing(why) => assert!(
            !why.contains("members have no record"),
            "should read as a plain gate, got: {why}"
        ),
        other => panic!("expected Missing, got {other:?}"),
    }
}

/// A shard measured by a binary older than the split carries no tallies. It
/// cannot be folded in, and counting it as empty would score the group over
/// fewer samples than the draw.
#[test]
fn a_shard_without_tallies_is_refused_rather_than_counted_as_empty() {
    let dir = scaffold();
    let root = dir.path();
    for m in ["bfcl-subset-a", "bfcl-subset-b", "bfcl-subset-c"] {
        std::fs::create_dir_all(gate_dir(root, m)).unwrap();
        plant_shard(root, m, SHA, 1_785_891_000, 95, 100);
    }
    std::fs::create_dir_all(gate_dir(root, "bfcl-subset-d")).unwrap();
    super::tests::plant(root, "bfcl-subset-d", SHA, 1_785_891_001, "PASS");
    match check_one(root, "bfcl-subset", SHA) {
        GateStatus::Missing(why) => assert!(why.contains("no per-subset tallies"), "{why}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// ★ END TO END: two members ran shard C, nobody ran shard D. Every member has
/// a record, all at one commit, and the row count is exactly what the draw
/// expects — so `samples`, `composition_ok` and the missing-member rule all
/// pass. Only `partition_ok` sees it. This exercises the CALL SITE, not just
/// the predicate: without the wiring in `check_group` this returns Pass.
#[test]
fn two_members_running_the_same_shard_fail_the_group() {
    let dir = scaffold();
    let root = dir.path();
    for (i, m) in [
        "bfcl-subset-a",
        "bfcl-subset-b",
        "bfcl-subset-c",
        "bfcl-subset-d",
    ]
    .iter()
    .enumerate()
    {
        std::fs::create_dir_all(gate_dir(root, m)).unwrap();
        // `-d`'s record says it ran shard C. Same commit, same row counts.
        let overrides: &[(&str, f64)] = if *m == "bfcl-subset-d" {
            &[("shard.index", 2.0)]
        } else {
            &[]
        };
        plant_shard_with(root, m, SHA, 1_785_891_000 + i as u64, overrides);
    }
    match check_one(root, "bfcl-subset", SHA) {
        GateStatus::Fail(why) => {
            let joined = why.join(" ");
            assert!(joined.contains("partition"), "{joined}");
            assert!(joined.contains("[0, 1, 2, 2]"), "{joined}");
        }
        other => panic!("a duplicated shard must fail the group, got {other:?}"),
    }
}

/// ★ END TO END: a member that lost samples to the transport is refused. Those
/// samples were scored as "made no call", which is the CORRECT answer on the
/// irrelevance subsets, so a degraded shard can RAISE the aggregate while
/// measuring less of the draw — a failure that makes the number look better.
#[test]
fn a_member_degraded_by_transport_failures_is_refused() {
    let dir = scaffold();
    let root = dir.path();
    for (i, m) in [
        "bfcl-subset-a",
        "bfcl-subset-b",
        "bfcl-subset-c",
        "bfcl-subset-d",
    ]
    .iter()
    .enumerate()
    {
        std::fs::create_dir_all(gate_dir(root, m)).unwrap();
        let overrides: &[(&str, f64)] = if *m == "bfcl-subset-b" {
            &[("transport_errors", 7.0)]
        } else {
            &[]
        };
        plant_shard_with(root, m, SHA, 1_785_891_000 + i as u64, overrides);
    }
    match check_one(root, "bfcl-subset", SHA) {
        GateStatus::Fail(why) => {
            let joined = why.join(" ");
            assert!(joined.contains("bfcl-subset-b"), "{joined}");
            assert!(joined.contains("7 transport failures"), "{joined}");
        }
        other => panic!("a degraded member must fail the group, got {other:?}"),
    }
}

/// A clean run must still pass with the two new metrics present — the guards
/// refuse the bad cases without refusing the good one.
#[test]
fn four_clean_distinct_shards_still_pass() {
    let dir = scaffold();
    let root = dir.path();
    for (i, m) in [
        "bfcl-subset-a",
        "bfcl-subset-b",
        "bfcl-subset-c",
        "bfcl-subset-d",
    ]
    .iter()
    .enumerate()
    {
        std::fs::create_dir_all(gate_dir(root, m)).unwrap();
        plant_shard_with(root, m, SHA, 1_785_891_000 + i as u64, &[]);
    }
    assert!(
        matches!(check_one(root, "bfcl-subset", SHA), GateStatus::Pass),
        "{:?}",
        check_one(root, "bfcl-subset", SHA)
    );
}
