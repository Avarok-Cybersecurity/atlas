// SPDX-License-Identifier: AGPL-3.0-only
//! `check_one`'s group path: a complete partition of shards produces one
//! verdict, and a partial set produces none.
//!
//! Since 2026-09-13 shards are the ONLY path: a whole-draw record under the
//! group's own id is history, and every per-record rule a plain gate applies
//! (subject, frame, dirty tree, signature) is applied per shard. Since
//! 2026-09-15 the shard COUNT is whatever the campaign chose: the fixtures
//! here use several counts on purpose, so nothing passes by matching the
//! historical four. The negative controls each plant one defective shard
//! among clean ones and expect the group to refuse it by name.

use super::check::check_one;
use super::tests::{bfcl_baseline, tempdir};
use super::*;
use crate::result::Verdict;
use std::collections::BTreeMap;

const SHA: &str = "1111111111";
const G: &str = "bfcl-subset";

/// Plant one shard record of `G` carrying per-subset tallies, the way
/// `report.rs` writes them, with `overrides` applied last for the degraded
/// and mislabelled cases.
fn plant_shard_with(
    root: &std::path::Path,
    shard: (usize, usize),
    sha: &str,
    secs: u64,
    overrides: &[(&str, f64)],
) {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 90.0);
    metrics.insert("subset.simple_python.hits".to_string(), 95.0);
    metrics.insert("subset.simple_python.n".to_string(), 100.0);
    metrics.insert("shard.index".to_string(), shard.0 as f64);
    metrics.insert("shard.count".to_string(), shard.1 as f64);
    metrics.insert("transport_errors".to_string(), 0.0);
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
    gate.benchmark_id = G.to_string();
    gate.verdict = Some("PASS".to_string());
    gate.recorded_at = secs;
    write_record(root, &gate).unwrap();
}

fn plant_shard(root: &std::path::Path, shard: (usize, usize), sha: &str, secs: u64) {
    plant_shard_with(root, shard, sha, secs, &[]);
}

/// A complete `n`-way partition at `sha`, one second apart.
fn partition(root: &std::path::Path, n: usize, sha: &str, secs: u64) {
    for i in 0..n {
        plant_shard(root, (i, n), sha, secs + i as u64);
    }
}

fn scaffold() -> tempdir::Dir {
    let dir = tempdir::Dir::new();
    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(dir.path(), id)).unwrap();
        super::fixture_baseline::write_baseline(dir.path(), id, &bfcl_baseline());
    }
    dir
}

/// Edit the record of shard `index` in place (its signature, if any, no
/// longer matches — the tests that need one plant before the cutover).
fn rewrite_shard(root: &std::path::Path, index: usize, edit: impl FnOnce(&mut GateRecord)) {
    let path = records_newest_first(root, G)
        .into_iter()
        .find(|p| read_record(p).is_ok_and(|r| r.shard().is_some_and(|(i, _)| i == index)))
        .expect("a record for that shard");
    let mut r = read_record(&path).unwrap();
    edit(&mut r);
    std::fs::write(&path, serde_json::to_string_pretty(&r).unwrap()).unwrap();
}

/// A complete partition, one commit, aggregate clears the group's bar → Pass,
/// at any count. This is the whole feature: no whole-draw record exists.
#[test]
fn a_complete_partition_satisfies_the_group_at_any_count() {
    for n in [1usize, 3, 6] {
        let dir = scaffold();
        let root = dir.path();
        partition(root, n, SHA, 1_785_891_000);
        assert!(
            matches!(check_one(root, G, SHA), GateStatus::Pass),
            "{n}-way: {:?}",
            check_one(root, G, SHA)
        );
    }
}

/// n-1 shards is not (n-1)/n measured. It must not pass, and the reason
/// must name the missing index.
#[test]
fn an_incomplete_partition_does_not_satisfy_the_group() {
    let dir = scaffold();
    let root = dir.path();
    for i in [0, 1, 2, 4] {
        plant_shard(root, (i, 5), SHA, 1_785_891_000 + i as u64);
    }
    match check_one(root, G, SHA) {
        GateStatus::Missing(why) => {
            assert!(
                why.contains("5-way at 1111111111 holds [0, 1, 2, 4], missing 3"),
                "{why}"
            );
            assert!(why.contains("different measurement"), "{why}");
        }
        other => panic!("four of five must not satisfy the group, got {other:?}"),
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
    super::tests::plant(root, G, SHA, 1_785_891_382, "PASS");
    match check_one(root, G, SHA) {
        GateStatus::Missing(why) => {
            assert!(why.contains("no shard record"), "{why}");
            assert!(why.contains("1 whole-draw record(s)"), "{why}");
            assert!(why.contains("no longer satisfy"), "{why}");
            assert!(why.contains("2026-09-13"), "{why}");
        }
        other => panic!("a whole-draw record must not satisfy the group, got {other:?}"),
    }
}

/// With NO records at all the group says so — and does not mention
/// whole-draw records that do not exist.
#[test]
fn a_group_with_no_records_says_so() {
    let dir = scaffold();
    match check_one(dir.path(), G, SHA) {
        GateStatus::Missing(why) => {
            assert!(why.contains("no shard record"), "{why}");
            assert!(!why.contains("whole-draw"), "{why}");
        }
        other => panic!("expected Missing, got {other:?}"),
    }
}

/// A partition whose newest record was invalidated names the perf-path files
/// that did it — the 20-second-fix property the whole-draw arm had.
#[test]
fn a_partition_invalidated_by_a_perf_path_names_the_file() {
    let dir = scaffold();
    let root = dir.path();
    use super::coverage_tests::scratch_repo;
    scratch_repo::init(root);
    scratch_repo::commit(root, "crates/lib.rs", "v1", "kernel v1");
    let old = scratch_repo::head(root);
    partition(root, 4, &old, 1_785_891_000);
    scratch_repo::commit(root, "crates/lib.rs", "v2", "kernel v2");
    let head = scratch_repo::head(root);
    match check_one(root, G, &head) {
        GateStatus::Missing(why) => {
            assert!(why.contains("invalidated by"), "{why}");
            assert!(why.contains("crates/lib.rs"), "{why}");
        }
        other => panic!("expected Missing naming the file, got {other:?}"),
    }
}

/// ★ CONTENT RULE (#1086). A partition measured at an OLDER commit still
/// certifies a newer one when no perf path moved in between: the records
/// stand by content, not ancestry.
#[test]
fn a_standing_partition_at_an_older_commit_still_certifies() {
    let dir = scaffold();
    let root = dir.path();
    use super::coverage_tests::scratch_repo;
    scratch_repo::init(root);
    scratch_repo::commit(root, "crates/lib.rs", "v1", "kernel v1");
    let old = scratch_repo::head(root);
    partition(root, 3, &old, 1_785_891_000);
    scratch_repo::commit(root, "docs/notes.md", "hello", "docs only");
    let head = scratch_repo::head(root);
    assert!(
        matches!(check_one(root, G, &head), GateStatus::Pass),
        "{:?}",
        check_one(root, G, &head)
    );
}

/// ★ NEGATIVE CONTROL: a shard recorded after the signature cutover with no
/// `.sig` is refused, by name. Before 2026-09-13 the whole-draw arm ran first
/// and no shard was ever asked for its signature.
#[test]
fn an_unsigned_shard_after_the_cutover_fails_the_group() {
    let dir = scaffold();
    let root = dir.path();
    partition(root, 4, SHA, super::signing::SIGNATURE_REQUIRED_AFTER + 10);
    match check_one(root, G, SHA) {
        GateStatus::Fail(why) => {
            let joined = why.join(" ");
            assert!(joined.contains("bfcl-subset[0/4]: "), "{joined}");
            assert!(joined.contains("no signature"), "{joined}");
        }
        other => panic!("an unsigned shard must fail the group, got {other:?}"),
    }
}

/// ★ NEGATIVE CONTROL: a shard measured from a dirty tree does not describe
/// its own sha, and the group must say so rather than fold it in.
#[test]
fn a_dirty_tree_shard_fails_the_group() {
    let dir = scaffold();
    let root = dir.path();
    partition(root, 4, SHA, 1_785_891_000);
    rewrite_shard(root, 2, |r| {
        r.dirty_paths = vec!["crates/spark-model/src/lib.rs".to_string()];
    });
    match check_one(root, G, SHA) {
        GateStatus::Fail(why) => {
            let joined = why.join(" ");
            assert!(
                joined.contains("bfcl-subset[2/4]: measured from a dirty tree"),
                "{joined}"
            );
            assert!(joined.contains("crates/spark-model/src/lib.rs"), "{joined}");
        }
        other => panic!("a dirty shard must fail the group, got {other:?}"),
    }
}

/// ★ NEGATIVE CONTROL: a shard whose run did not complete is a failure with
/// the run's own reason, not a quarter of a measurement.
#[test]
fn a_failed_frame_shard_fails_the_group() {
    let dir = scaffold();
    let root = dir.path();
    partition(root, 4, SHA, 1_785_891_000);
    rewrite_shard(root, 1, |r| {
        r.frame_status = crate::result::RunStatus::Failed;
        r.verdict_reason = "scorer crashed".to_string();
    });
    match check_one(root, G, SHA) {
        GateStatus::Fail(why) => {
            let joined = why.join(" ");
            assert!(
                joined.contains("bfcl-subset[1/4]: the run itself failed"),
                "{joined}"
            );
            assert!(joined.contains("scorer crashed"), "{joined}");
        }
        other => panic!("a failed shard must fail the group, got {other:?}"),
    }
}

/// ★ NEGATIVE CONTROL: a shard measured on a non-default variant is not the
/// gate's subject. It is ignored, so the partition reads as INCOMPLETE — a
/// pass on the wrong checkpoint is not evidence for the required one.
#[test]
fn a_shard_on_another_variant_is_not_the_subject() {
    let dir = scaffold();
    let root = dir.path();
    partition(root, 4, SHA, 1_785_891_000);
    rewrite_shard(root, 3, |r| {
        r.target_model = "some-other/checkpoint".to_string();
    });
    match check_one(root, G, SHA) {
        GateStatus::Missing(why) => assert!(why.contains("missing 3"), "{why}"),
        other => panic!("an off-subject shard must not count, got {other:?}"),
    }
}

/// A shard measured by a binary that wrote no tallies cannot be folded in,
/// and counting it as empty would score the group over fewer samples than
/// the draw.
#[test]
fn a_shard_without_tallies_is_refused_rather_than_counted_as_empty() {
    let dir = scaffold();
    let root = dir.path();
    partition(root, 4, SHA, 1_785_891_000);
    rewrite_shard(root, 3, |r| {
        r.metrics.retain(|k, _| !k.starts_with("subset."));
    });
    match check_one(root, G, SHA) {
        GateStatus::Missing(why) => assert!(why.contains("no per-subset tallies"), "{why}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// ★ END TO END: two records ran shard 2, nobody ran shard 3. Four records,
/// all at one commit, and the row count is exactly what the draw expects —
/// so `samples` passes. Only the partition rule sees it. This exercises the
/// CALL SITE, not just the predicate: without the wiring in `check_group`
/// this returns Pass.
#[test]
fn two_records_of_the_same_shard_do_not_complete_the_partition() {
    let dir = scaffold();
    let root = dir.path();
    for i in 0..3 {
        plant_shard(root, (i, 4), SHA, 1_785_891_000 + i as u64);
    }
    // The fourth record says it ran shard 2. Same commit, same row counts.
    plant_shard_with(root, (3, 4), SHA, 1_785_891_003, &[("shard.index", 2.0)]);
    match check_one(root, G, SHA) {
        GateStatus::Missing(why) => {
            assert!(
                why.contains("4-way at 1111111111 holds [0, 1, 2], missing 3"),
                "{why}"
            );
        }
        other => panic!("a duplicated shard must not complete the group, got {other:?}"),
    }
}

/// ★ END TO END: a shard that lost samples to the transport is refused. Those
/// samples were scored as "made no call", which is the CORRECT answer on the
/// irrelevance subsets, so a degraded shard can RAISE the aggregate while
/// measuring less of the draw — a failure that makes the number look better.
#[test]
fn a_shard_degraded_by_transport_failures_is_refused() {
    let dir = scaffold();
    let root = dir.path();
    for i in 0..4 {
        let overrides: &[(&str, f64)] = if i == 1 {
            &[("transport_errors", 7.0)]
        } else {
            &[]
        };
        plant_shard_with(root, (i, 4), SHA, 1_785_891_000 + i as u64, overrides);
    }
    match check_one(root, G, SHA) {
        GateStatus::Fail(why) => {
            let joined = why.join(" ");
            assert!(joined.contains("bfcl-subset[1/4]"), "{joined}");
            assert!(joined.contains("7 transport failures"), "{joined}");
        }
        other => panic!("a degraded shard must fail the group, got {other:?}"),
    }
}

/// A shard re-run at a newer commit re-opens the group until its siblings
/// join it there: a partition is never assembled across commits. Both
/// commits stand here (the records name shas no tree has, so nothing is
/// invalidated), and still there is no complete partition at ONE.
#[test]
fn a_partition_is_never_assembled_across_commits() {
    let dir = scaffold();
    let root = dir.path();
    partition(root, 4, SHA, 1_785_891_000);
    plant_shard(root, (2, 4), "2222222222", 1_785_899_000);
    match check_one(root, G, "2222222222") {
        GateStatus::Missing(why) => {
            assert!(
                why.contains("4-way at 2222222222 holds [2], missing 0,1,3"),
                "{why}"
            );
            assert!(why.contains("ONE measurement"), "{why}");
        }
        other => panic!("expected Missing, got {other:?}"),
    }
}

/// `shards_owed` is the planner's question — "which shards must a campaign
/// still run?" — and its answer is exactly the set the verdict would refuse:
/// a missing index, and a present one whose record does not count. Anything
/// else being skipped would leave the group un-certifiable; anything else
/// being run would re-measure a shard the gate already accepts.
#[test]
fn shards_owed_names_exactly_the_shards_the_verdict_would_refuse() {
    let group = super::group::find(G).unwrap();
    let dir = scaffold();
    let root = dir.path();
    // Nothing banked: a fresh partition at the count the campaign wants.
    assert_eq!(
        shards_owed(root, group, SHA, 6),
        vec![(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6)]
    );
    // A complete partition — at ANOTHER count than the campaign would pick:
    // nothing owed, and the verdict agrees.
    partition(root, 4, SHA, 1_785_891_000);
    assert!(shards_owed(root, group, SHA, 6).is_empty());
    assert!(matches!(check_one(root, G, SHA), GateStatus::Pass));
    // One shard removed: that one, at the count already begun — never a
    // fresh 6-way.
    let c = records_newest_first(root, G)
        .into_iter()
        .find(|p| read_record(p).unwrap().shard() == Some((2, 4)))
        .unwrap();
    std::fs::remove_file(&c).unwrap();
    assert_eq!(shards_owed(root, group, SHA, 6), vec![(2, 4)]);
    // NEGATIVE CONTROL: a shard that is PRESENT but would be refused (a
    // failed frame) is still owed — presence is not enough, the record must
    // count.
    rewrite_shard(root, 1, |r| {
        r.frame_status = crate::result::RunStatus::Failed;
    });
    assert_eq!(shards_owed(root, group, SHA, 6), vec![(1, 4), (2, 4)]);
    assert!(matches!(check_one(root, G, SHA), GateStatus::Missing(_)));
}

/// A partition begun at ANOTHER commit cannot be finished at this one (the
/// verdict never assembles across commits), so the planner starts fresh
/// here rather than running one straggler that could never complete it.
#[test]
fn shards_owed_does_not_try_to_finish_another_commits_partition() {
    let group = super::group::find(G).unwrap();
    let dir = scaffold();
    let root = dir.path();
    for i in 0..3 {
        plant_shard(root, (i, 4), SHA, 1_785_891_000 + i as u64);
    }
    assert_eq!(shards_owed(root, group, SHA, 2), vec![(3, 4)]);
    assert_eq!(
        shards_owed(root, group, "2222222222", 2),
        vec![(0, 2), (1, 2)]
    );
}
