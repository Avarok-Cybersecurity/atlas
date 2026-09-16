// SPDX-License-Identifier: AGPL-3.0-only
//! A group is satisfied by a COMPLETE partition of shard records at ONE
//! commit, whatever its count. Three of four is not 75% measured — it is an
//! aggregate over a sample set the thresholds were never drawn against.

use super::group::{GroupFault, Partition, ShardRecord, select_partition};

/// `(index, count, sha, recorded_at)` → a shard record; the handle is the
/// position in the list, as `check_group` hands them over.
fn shards(rows: &[(usize, usize, &str, u64)]) -> Vec<ShardRecord> {
    rows.iter()
        .enumerate()
        .map(|(handle, (index, count, sha, at))| ShardRecord {
            index: *index,
            count: *count,
            git_sha: (*sha).to_string(),
            recorded_at: *at,
            handle,
        })
        .collect()
}

fn handles(p: &Partition) -> Vec<usize> {
    let mut h = p.handles.clone();
    h.sort_unstable();
    h
}

#[test]
fn a_complete_partition_at_one_commit_is_selected_at_any_count() {
    for n in [1usize, 2, 4, 6, 8, 13] {
        let rows: Vec<_> = (0..n).map(|i| (i, n, "abc", 100 + i as u64)).collect();
        let p = select_partition("g", &shards(&rows)).unwrap_or_else(|f| panic!("{n}-way: {f}"));
        assert_eq!(p.count, n);
        assert_eq!(p.git_sha, "abc");
        assert_eq!(handles(&p), (0..n).collect::<Vec<_>>());
    }
}

/// Order is not part of the rule — shards finish on any box in any order.
#[test]
fn order_of_arrival_does_not_matter() {
    let p = select_partition(
        "g",
        &shards(&[
            (3, 4, "abc", 1),
            (0, 4, "abc", 2),
            (2, 4, "abc", 3),
            (1, 4, "abc", 4),
        ]),
    )
    .unwrap();
    assert_eq!(p.count, 4);
    assert_eq!(handles(&p), vec![0, 1, 2, 3]);
}

/// THE RULE. A missing shard must be a named failure, never a quiet pass over
/// three quarters of the draw — and the message must say WHY, or an operator
/// who reads "3 of 4" will reasonably assume it is 75% of the evidence.
#[test]
fn a_missing_shard_is_refused_and_named() {
    let fault = select_partition(
        "bfcl-subset",
        &shards(&[(0, 4, "abc", 1), (1, 4, "abc", 1), (3, 4, "abc", 1)]),
    )
    .expect_err("three of four");
    match &fault {
        GroupFault::Missing { group, held } => {
            assert_eq!(*group, "bfcl-subset");
            assert_eq!(held, &vec![(4, "abc".to_string(), vec![0, 1, 3])]);
        }
    }
    let msg = fault.to_string();
    assert!(
        msg.contains("4-way at abc holds [0, 1, 3], missing 2"),
        "{msg}"
    );
    assert!(msg.contains("different measurement"), "{msg}");
}

#[test]
fn no_records_at_all_is_a_missing_fault_not_a_pass() {
    let fault = select_partition("g", &[]).expect_err("empty");
    assert!(matches!(&fault, GroupFault::Missing { held, .. } if held.is_empty()));
    assert!(fault.to_string().contains("no shard record"), "{fault}");
}

/// Shards may be signed by DIFFERENT boxes (they are Correctness-class), but
/// a partition is never assembled across COMMITS — that is one measurement
/// stitched from two trees. Here the 4-way has every index, but index 2 was
/// measured at another commit: no complete partition at ONE commit exists.
#[test]
fn a_partition_is_never_assembled_across_commits() {
    let fault = select_partition(
        "g",
        &shards(&[
            (0, 4, "abc", 1),
            (1, 4, "abc", 1),
            (2, 4, "def", 9),
            (3, 4, "abc", 1),
        ]),
    )
    .expect_err("index 2 is at another commit");
    let GroupFault::Missing { held, .. } = &fault;
    assert_eq!(
        held,
        &vec![
            (4, "abc".to_string(), vec![0, 1, 3]),
            (4, "def".to_string(), vec![2])
        ]
    );
    assert!(fault.to_string().contains("ONE measurement"), "{fault}");
}

/// ★ THE CASE THE `samples` PIN CANNOT SEE. Two records ran shard C; nobody
/// ran shard D. The union still holds ~995 rows, every per-subset tally still
/// looks plausible because the subsets are strided, and `samples` (pinned
/// min == max == 995) passes — while shard D was never measured. Newest-per-
/// index makes the duplicate a re-run, never a stand-in.
#[test]
fn a_duplicated_shard_does_not_stand_in_for_a_missing_one() {
    let fault = select_partition(
        "g",
        &shards(&[
            (0, 4, "abc", 1),
            (1, 4, "abc", 1),
            (2, 4, "abc", 1),
            (2, 4, "abc", 2),
        ]),
    )
    .expect_err("C twice, D never");
    let GroupFault::Missing { held, .. } = &fault;
    assert_eq!(held, &vec![(4, "abc".to_string(), vec![0, 1, 2])]);
}

/// A re-run shard replaces its older self: the newest record per index is
/// the one whose handle comes back.
#[test]
fn the_newest_record_per_index_is_the_one_selected() {
    let p = select_partition(
        "g",
        &shards(&[(0, 2, "abc", 1), (1, 2, "abc", 1), (0, 2, "abc", 5)]),
    )
    .unwrap();
    assert_eq!(handles(&p), vec![1, 2]);
}

/// A shard that thinks the draw is split a different number of ways is a
/// different slice of the draw entirely: it belongs to ITS count's partition,
/// and never completes another's.
#[test]
fn a_shard_of_another_count_never_completes_a_partition() {
    let fault = select_partition(
        "g",
        &shards(&[
            (0, 4, "abc", 1),
            (1, 4, "abc", 1),
            (2, 4, "abc", 1),
            (3, 8, "abc", 1),
        ]),
    )
    .expect_err("an 8-way shard does not finish a 4-way");
    let GroupFault::Missing { held, .. } = &fault;
    assert_eq!(held.len(), 2);
    assert!(
        fault.to_string().contains("8-way at abc holds [3]"),
        "{fault}"
    );
}

/// Two complete partitions: the one whose newest record is newer wins — the
/// branch's current word, like every other newest-first rule.
#[test]
fn of_two_complete_partitions_the_newer_wins() {
    let rows = [(0, 2, "abc", 10), (1, 2, "abc", 11), (0, 1, "abc", 5)];
    assert_eq!(select_partition("g", &shards(&rows)).unwrap().count, 2);
    let rows = [(0, 2, "abc", 10), (1, 2, "abc", 11), (0, 1, "def", 50)];
    let p = select_partition("g", &shards(&rows)).unwrap();
    assert_eq!((p.count, p.git_sha.as_str()), (1, "def"));
}

/// Every group id must be a REGISTERED benchmark, a required gate, and a
/// Correctness one: `check_record` judges the aggregate under the GROUP's
/// thresholds, and sharding recombines COUNTS — which preserves accuracy and
/// destroys timing, so a Speed benchmark cannot be split.
#[test]
fn every_group_is_a_registered_required_correctness_benchmark() {
    use crate::hardware::Sensitivity;
    for g in super::group::GROUPS {
        let d = crate::registry::find(g.id)
            .unwrap_or_else(|| panic!("group {} is not a registered benchmark", g.id));
        assert_eq!(d.sensitivity, Sensitivity::Correctness, "group {}", g.id);
        assert!(
            super::coverage::REQUIRED.iter().any(|r| r.id == g.id),
            "group {} is not in REQUIRED — then nothing asks for it",
            g.id
        );
        assert!(
            d.build().parameters().iter().any(|p| p.key == "shard"),
            "group {} has no `shard` parameter: nothing can run a slice of it",
            g.id
        );
    }
}

/// The legacy per-shard ids are gone: a shard is the group's own benchmark
/// run with `--param shard=i/n`, not a registered benchmark of its own.
#[test]
fn legacy_shard_ids_are_not_registered() {
    for id in ["bfcl-subset-a", "bfcl-subset-d", "bfcl-subset-echolp-a"] {
        assert!(
            crate::registry::find(id).is_none(),
            "{id} is still registered"
        );
        assert!(super::group::find(id).is_none(), "{id} is a group?");
    }
}
