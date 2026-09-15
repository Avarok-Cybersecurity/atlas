// SPDX-License-Identifier: AGPL-3.0-only
//! `check_one`'s group path: four shards produce one verdict, and a partial set
//! produces none.
//!
//! Since 2026-09-13 the members are the ONLY path: a whole-draw record under
//! the group's own id is history, and every per-record rule a plain gate
//! applies (subject, frame, dirty tree, signature) is applied per member. The
//! negative controls below each plant one defective member among three clean
//! ones and expect the group to refuse it by name.

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

/// ★ THE RULE. A passing whole-draw record under the group's own id, with no
/// shards present, does NOT satisfy the group — and the verdict names the
/// record and says why it stopped counting, so nobody bisects a kernel change
/// looking for what re-opened a gate that was re-opened by policy.
#[test]
fn a_whole_draw_record_no_longer_satisfies_the_group() {
    let dir = scaffold();
    let root = dir.path();
    super::tests::plant(root, "bfcl-subset", SHA, 1_785_891_382, "PASS");
    match check_one(root, "bfcl-subset", SHA) {
        GateStatus::Missing(why) => {
            assert!(why.contains("4 of its members have no record"), "{why}");
            assert!(why.contains("whole-draw record(s)"), "{why}");
            assert!(why.contains("no longer satisfy"), "{why}");
            assert!(why.contains("2026-09-13"), "{why}");
        }
        other => panic!("a whole-draw record must not satisfy the group, got {other:?}"),
    }
}

/// With NO records at all the group reports its members as missing and says,
/// per member, that nothing was ever committed — the plain-gate wording is
/// gone with the plain-gate path.
#[test]
fn an_unsharded_group_with_no_records_names_every_member() {
    let dir = scaffold();
    match check_one(dir.path(), "bfcl-subset", SHA) {
        GateStatus::Missing(why) => {
            for m in [
                "bfcl-subset-a",
                "bfcl-subset-b",
                "bfcl-subset-c",
                "bfcl-subset-d",
            ] {
                assert!(why.contains(m), "{why}");
            }
            assert!(why.contains("no record has ever been committed"), "{why}");
            assert!(
                !why.contains("whole-draw"),
                "no whole-draw record exists: {why}"
            );
        }
        other => panic!("expected Missing, got {other:?}"),
    }
}

/// A member whose newest record was invalidated names the perf-path files
/// that did it — the 20-second-fix property the whole-draw arm had.
#[test]
fn a_member_invalidated_by_a_perf_path_names_the_file() {
    let dir = scaffold();
    let root = dir.path();
    use super::coverage_tests::scratch_repo;
    scratch_repo::init(root);
    scratch_repo::commit(root, "crates/lib.rs", "v1", "kernel v1");
    let old = scratch_repo::head(root);
    for m in [
        "bfcl-subset-a",
        "bfcl-subset-b",
        "bfcl-subset-c",
        "bfcl-subset-d",
    ] {
        std::fs::create_dir_all(gate_dir(root, m)).unwrap();
        plant_shard(root, m, &old, 1_785_891_000, 95, 100);
    }
    scratch_repo::commit(root, "crates/lib.rs", "v2", "kernel v2");
    let head = scratch_repo::head(root);
    match check_one(root, "bfcl-subset", &head) {
        GateStatus::Missing(why) => {
            assert!(why.contains("invalidated by"), "{why}");
            assert!(why.contains("crates/lib.rs"), "{why}");
        }
        other => panic!("expected Missing naming the file, got {other:?}"),
    }
}

/// Plant four clean shards, then let the caller break one of them.
fn four_shards(root: &std::path::Path, secs: u64) {
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
        plant_shard(root, m, SHA, secs + i as u64, 95, 100);
    }
}

fn rewrite_member(root: &std::path::Path, member: &str, edit: impl FnOnce(&mut GateRecord)) {
    let path = records_newest_first(root, member).remove(0);
    let mut r = read_record(&path).unwrap();
    edit(&mut r);
    std::fs::write(&path, serde_json::to_string_pretty(&r).unwrap()).unwrap();
}

/// ★ NEGATIVE CONTROL: a member recorded after the signature cutover with no
/// `.sig` is refused, by name. Before 2026-09-13 the whole-draw arm ran first
/// and no shard was ever asked for its signature.
#[test]
fn an_unsigned_member_after_the_cutover_fails_the_group() {
    let dir = scaffold();
    let root = dir.path();
    four_shards(root, super::signing::SIGNATURE_REQUIRED_AFTER + 10);
    match check_one(root, "bfcl-subset", SHA) {
        GateStatus::Fail(why) => {
            let joined = why.join(" ");
            assert!(joined.contains("bfcl-subset-a: "), "{joined}");
            assert!(joined.contains("no signature"), "{joined}");
        }
        other => panic!("an unsigned member must fail the group, got {other:?}"),
    }
}

/// ★ NEGATIVE CONTROL: a member measured from a dirty tree does not describe
/// its own sha, and the group must say so rather than fold it in.
#[test]
fn a_dirty_tree_member_fails_the_group() {
    let dir = scaffold();
    let root = dir.path();
    four_shards(root, 1_785_891_000);
    rewrite_member(root, "bfcl-subset-c", |r| {
        r.dirty_paths = vec!["crates/spark-model/src/lib.rs".to_string()];
    });
    match check_one(root, "bfcl-subset", SHA) {
        GateStatus::Fail(why) => {
            let joined = why.join(" ");
            assert!(
                joined.contains("bfcl-subset-c: measured from a dirty tree"),
                "{joined}"
            );
            assert!(joined.contains("crates/spark-model/src/lib.rs"), "{joined}");
        }
        other => panic!("a dirty member must fail the group, got {other:?}"),
    }
}

/// ★ NEGATIVE CONTROL: a member whose run did not complete is a failure with
/// the run's own reason, not a quarter of a measurement.
#[test]
fn a_failed_frame_member_fails_the_group() {
    let dir = scaffold();
    let root = dir.path();
    four_shards(root, 1_785_891_000);
    rewrite_member(root, "bfcl-subset-b", |r| {
        r.frame_status = crate::result::RunStatus::Failed;
        r.verdict_reason = "scorer crashed".to_string();
    });
    match check_one(root, "bfcl-subset", SHA) {
        GateStatus::Fail(why) => {
            let joined = why.join(" ");
            assert!(
                joined.contains("bfcl-subset-b: the run itself failed"),
                "{joined}"
            );
            assert!(joined.contains("scorer crashed"), "{joined}");
        }
        other => panic!("a failed member must fail the group, got {other:?}"),
    }
}

/// ★ NEGATIVE CONTROL: a member measured on a non-default variant is not the
/// gate's subject. It is ignored, so the member reads as MISSING — a pass on
/// the wrong checkpoint is not evidence for the required one.
#[test]
fn a_member_on_another_variant_is_not_the_subject() {
    let dir = scaffold();
    let root = dir.path();
    four_shards(root, 1_785_891_000);
    rewrite_member(root, "bfcl-subset-d", |r| {
        r.target_model = "some-other/checkpoint".to_string();
    });
    match check_one(root, "bfcl-subset", SHA) {
        GateStatus::Missing(why) => assert!(why.contains("bfcl-subset-d"), "{why}"),
        other => panic!("an off-subject member must not count, got {other:?}"),
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

/// `members_owed` is the planner's question — "which shards must a campaign
/// still run?" — and its answer is exactly the set the verdict would refuse:
/// a missing member, and a present member whose record does not count.
/// Anything else being skipped would leave the group un-certifiable; anything
/// else being run would re-measure a shard the gate already accepts.
#[test]
fn members_owed_names_exactly_the_shards_the_verdict_would_refuse() {
    let group = super::group::find("bfcl-subset").unwrap();
    let dir = scaffold();
    let root = dir.path();
    // Nothing banked: every member.
    assert_eq!(members_owed(root, group, SHA), group.members);
    // Four clean shards: nothing owed, and the verdict agrees.
    four_shards(root, 1_785_891_000);
    assert!(members_owed(root, group, SHA).is_empty());
    assert!(matches!(
        check_one(root, "bfcl-subset", SHA),
        GateStatus::Pass
    ));
    // One shard removed: that one, and only that one.
    let c = records_newest_first(root, "bfcl-subset-c").remove(0);
    std::fs::remove_file(&c).unwrap();
    assert_eq!(members_owed(root, group, SHA), ["bfcl-subset-c"]);
    // NEGATIVE CONTROL: a member that is PRESENT but would be refused (a
    // failed frame) is still owed — presence is not enough, the record must
    // count.
    rewrite_member(root, "bfcl-subset-b", |r| {
        r.frame_status = crate::result::RunStatus::Failed;
    });
    assert_eq!(
        members_owed(root, group, SHA),
        ["bfcl-subset-b", "bfcl-subset-c"]
    );
    assert!(matches!(
        check_one(root, "bfcl-subset", SHA),
        GateStatus::Missing(_)
    ));
}
