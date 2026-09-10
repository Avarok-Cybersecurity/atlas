// SPDX-License-Identifier: AGPL-3.0-only
//! A group must be ALL of its members or none. Three of four is not 75%
//! measured — it is an aggregate over a sample set the thresholds were never
//! drawn against.

use super::group::{
    BenchmarkGroup, GroupFault, MemberRecord, composition_ok, member_of, partition_ok,
};

const G: BenchmarkGroup = BenchmarkGroup {
    id: "bfcl-subset",
    members: &[
        "bfcl-subset-a",
        "bfcl-subset-b",
        "bfcl-subset-c",
        "bfcl-subset-d",
    ],
};

fn recs(pairs: &[(&str, &str)]) -> Vec<MemberRecord> {
    pairs
        .iter()
        .map(|(id, sha)| MemberRecord {
            id: (*id).to_string(),
            git_sha: (*sha).to_string(),
        })
        .collect()
}

#[test]
fn all_four_members_at_one_commit_is_satisfied() {
    let r = recs(&[
        ("bfcl-subset-a", "abc"),
        ("bfcl-subset-b", "abc"),
        ("bfcl-subset-c", "abc"),
        ("bfcl-subset-d", "abc"),
    ]);
    assert_eq!(composition_ok(&G, &r), Ok(()));
}

/// THE RULE. A missing shard must be a named failure, never a quiet pass over
/// three quarters of the draw.
#[test]
fn a_missing_shard_is_refused_and_named() {
    let r = recs(&[
        ("bfcl-subset-a", "abc"),
        ("bfcl-subset-b", "abc"),
        ("bfcl-subset-d", "abc"),
    ]);
    match composition_ok(&G, &r) {
        Err(GroupFault::Missing { group, missing }) => {
            assert_eq!(group, "bfcl-subset");
            assert_eq!(missing, vec!["bfcl-subset-c"]);
        }
        other => panic!("expected a named missing shard, got {other:?}"),
    }
}

/// And the message must say WHY, not just that. An operator who reads "3 of 4"
/// will reasonably assume it is 75% of the evidence.
#[test]
fn the_missing_message_explains_that_a_subset_is_a_different_measurement() {
    let r = recs(&[("bfcl-subset-a", "abc")]);
    let msg = composition_ok(&G, &r).unwrap_err().to_string();
    assert!(msg.contains("bfcl-subset-b"), "{msg}");
    assert!(msg.contains("different measurement"), "{msg}");
}

#[test]
fn no_records_at_all_is_a_missing_shard_fault_not_a_pass() {
    match composition_ok(&G, &[]) {
        Err(GroupFault::Missing { missing, .. }) => assert_eq!(missing.len(), 4),
        other => panic!("an empty group must not pass, got {other:?}"),
    }
}

/// Members may be signed by DIFFERENT boxes (they are Correctness-class), but
/// they may not be measured at different COMMITS — that is one measurement
/// stitched from two trees.
#[test]
fn members_at_two_commits_are_refused() {
    let r = recs(&[
        ("bfcl-subset-a", "abc"),
        ("bfcl-subset-b", "abc"),
        ("bfcl-subset-c", "def"),
        ("bfcl-subset-d", "abc"),
    ]);
    match composition_ok(&G, &r) {
        Err(GroupFault::SpansCommits { commits, .. }) => assert_eq!(commits.len(), 2),
        other => panic!("expected a spans-commits fault, got {other:?}"),
    }
}

/// A record for something that is not a member must not be folded in — that
/// would let an unrelated run inflate or deflate the aggregate.
#[test]
fn a_foreign_member_is_refused() {
    let mut r = recs(&[
        ("bfcl-subset-a", "abc"),
        ("bfcl-subset-b", "abc"),
        ("bfcl-subset-c", "abc"),
        ("bfcl-subset-d", "abc"),
    ]);
    r.push(MemberRecord {
        id: "bfcl-subset-echolp-a".into(),
        git_sha: "abc".into(),
    });
    match composition_ok(&G, &r) {
        Err(GroupFault::Foreign { id, .. }) => assert_eq!(id, "bfcl-subset-echolp-a"),
        other => panic!("expected a foreign-member fault, got {other:?}"),
    }
}

/// Duplicates of the same member are tolerated by composition — the newest
/// record wins upstream, exactly as `records_newest_first` already decides for
/// a single gate. This pins that duplicates are not mistaken for completeness.
#[test]
fn a_duplicate_member_does_not_stand_in_for_a_missing_one() {
    let r = recs(&[
        ("bfcl-subset-a", "abc"),
        ("bfcl-subset-a", "abc"),
        ("bfcl-subset-b", "abc"),
        ("bfcl-subset-c", "abc"),
    ]);
    match composition_ok(&G, &r) {
        Err(GroupFault::Missing { missing, .. }) => assert_eq!(missing, vec!["bfcl-subset-d"]),
        other => panic!("two copies of A must not cover D, got {other:?}"),
    }
}

/// Every group id and every member id must be a REGISTERED benchmark, or the
/// group names something nothing can run.
#[test]
fn every_group_and_member_resolves_in_the_registry() {
    for g in super::group::GROUPS {
        assert!(
            crate::registry::find(g.id).is_some(),
            "group {} is not a registered benchmark",
            g.id
        );
        for m in g.members {
            assert!(
                crate::registry::find(m).is_some(),
                "{m} is a member of {} but is not registered",
                g.id
            );
        }
    }
}

/// A MEMBER must never be a required gate in its own right: `REQUIRED` names
/// the group, and requiring the members too would demand four records where the
/// group needs one verdict.
#[test]
fn no_group_member_is_itself_a_required_gate() {
    for g in super::group::GROUPS {
        for m in g.members {
            assert!(
                !super::coverage::REQUIRED.iter().any(|r| r.id == *m),
                "{m} is a group member AND a required gate"
            );
        }
        assert!(
            super::coverage::REQUIRED.iter().any(|r| r.id == g.id),
            "group {} is not in REQUIRED — then nothing asks for it",
            g.id
        );
    }
}

/// Members belong to exactly one group; a shard shared between two groups would
/// be counted twice.
#[test]
fn no_benchmark_is_a_member_of_two_groups() {
    let mut seen = std::collections::BTreeSet::new();
    for g in super::group::GROUPS {
        for m in g.members {
            assert!(seen.insert(*m), "{m} is a member of more than one group");
        }
    }
    assert!(
        member_of("decode-floor").is_none(),
        "a plain gate is not a member"
    );
    assert!(member_of("bfcl-subset-a").is_some(), "a shard IS a member");
}

/// The happy case: four members, four shards, indices 0..3 once each.
#[test]
fn four_distinct_shards_are_a_partition() {
    assert_eq!(partition_ok("g", &[(0, 4), (1, 4), (2, 4), (3, 4)]), Ok(()));
    // Order is not part of the rule — members may finish on any box in any
    // order, and the gate reads them in registry order regardless.
    assert_eq!(partition_ok("g", &[(3, 4), (0, 4), (2, 4), (1, 4)]), Ok(()));
}

/// ★ THE CASE THE `samples` PIN CANNOT SEE. Two members ran shard C; nobody ran
/// shard D. The union still holds ~995 rows, every per-subset tally still looks
/// plausible because the subsets are strided, and `samples` (pinned min == max
/// == 995) passes — while shard D was never measured. This is the silently
/// wrong green, and it is the reason this check exists separately.
#[test]
fn a_duplicated_shard_is_not_a_partition_even_though_the_count_is_right() {
    let fault = partition_ok("bfcl-subset", &[(0, 4), (1, 4), (2, 4), (2, 4)])
        .expect_err("two members ran shard C and none ran D");
    match &fault {
        GroupFault::NotAPartition { group, detail } => {
            assert_eq!(*group, "bfcl-subset");
            assert!(detail.contains("[0, 1, 2, 2]"), "{detail}");
        }
        other => panic!("wrong fault: {other:?}"),
    }
    // The reading must say why the row count is not a defence, or the next
    // reader will "fix" this by trusting `samples`.
    let msg = fault.to_string();
    assert!(msg.contains("EXACTLY once"), "{msg}");
    assert!(msg.contains("samples"), "{msg}");
}

/// A member that thinks the draw is split a different number of ways cannot be
/// folded in with members that think otherwise — its rows are a different slice
/// of the draw entirely.
#[test]
fn a_member_reporting_a_different_shard_count_is_refused() {
    let fault = partition_ok("g", &[(0, 4), (1, 4), (2, 4), (3, 8)])
        .expect_err("one member ran an 8-way split");
    match fault {
        GroupFault::NotAPartition { detail, .. } => {
            assert!(detail.contains("8 shards"), "{detail}");
            assert!(detail.contains("4 members"), "{detail}");
        }
        other => panic!("wrong fault: {other:?}"),
    }
}

/// An index outside the range is refused even when the indices are distinct —
/// distinctness alone would let {0,1,2,7} through, and shard 3 would be unmeasured.
#[test]
fn distinct_but_out_of_range_indices_are_refused() {
    assert!(partition_ok("g", &[(0, 4), (1, 4), (2, 4), (7, 4)]).is_err());
}

/// Anti-vacuity: the rule must not report success on an empty member list. A
/// group with no members has measured nothing, and `partition_ok` returning
/// `Ok` there would make "no shards ran" indistinguishable from "all did".
#[test]
fn no_members_is_not_a_partition_of_anything() {
    // Zero members means every member declared count 0 == n, and the index
    // list is trivially equal to the empty expectation -- so this DOES return
    // Ok, and the emptiness must be caught before here. `check_group` returns
    // `Missing` on an empty group before ever calling this, and the group
    // registry has no empty groups; this test pins that division of labour so
    // a future refactor cannot quietly make this the only guard.
    assert_eq!(partition_ok("g", &[]), Ok(()));
    for g in super::group::GROUPS {
        assert!(
            g.members.len() >= 2,
            "{} has {} members; a group of <2 makes the emptiness path \
             reachable and this rule is not the guard for it",
            g.id,
            g.members.len()
        );
    }
}

/// ★ SHARDING IS ONLY SOUND FOR A KNOWN-ANSWER TEST. Splitting a draw across
/// boxes and recombining the counts preserves an ACCURACY measurement — every
/// sample is scored once, wherever it ran. It destroys a SPEED measurement,
/// because for those the timing IS the number: four quarter-length runs on
/// three boxes have a different wall, a different TTFT distribution and a
/// different concurrency profile from one serial run, and no arithmetic
/// recovers the original.
///
/// So every member of every group must be `Sensitivity::Correctness`. This is
/// checked against the REGISTRY rather than a new descriptor flag, because the
/// registry already carries the fact — a second marker could disagree with it.
#[test]
fn every_group_member_is_a_correctness_benchmark() {
    use crate::hardware::Sensitivity;
    for g in super::group::GROUPS {
        for m in g.members {
            let d = crate::registry::find(m)
                .unwrap_or_else(|| panic!("{} names member {m}, which is not registered", g.id));
            assert_eq!(
                d.sensitivity,
                Sensitivity::Correctness,
                "{m} is a member of group {} but is {:?}. Sharding recombines \
                 COUNTS, which preserves accuracy and destroys timing — a Speed \
                 benchmark cannot be split.",
                g.id,
                d.sensitivity
            );
        }
    }
}

/// The group id must itself be a registered benchmark, and a Correctness one:
/// `check_record` judges the aggregate under the GROUP's thresholds, so a group
/// whose id resolved to nothing (or to a Speed gate) would be judging the
/// merged counts against bars that were never drawn for them.
#[test]
fn every_group_id_resolves_to_a_correctness_benchmark() {
    use crate::hardware::Sensitivity;
    for g in super::group::GROUPS {
        let d = crate::registry::find(g.id)
            .unwrap_or_else(|| panic!("group {} is not a registered benchmark", g.id));
        assert_eq!(d.sensitivity, Sensitivity::Correctness, "group {}", g.id);
    }
}

/// Membership must not recurse: a member that were itself a group would make
/// "the aggregate" ambiguous — `check_group` would have to aggregate an
/// aggregate, over a sample set neither level's thresholds describe.
///
/// (Belonging to two groups is already covered by
/// `no_benchmark_is_a_member_of_two_groups`; this is only the recursion.)
#[test]
fn a_member_is_never_itself_a_group() {
    for g in super::group::GROUPS {
        for m in g.members {
            assert!(
                !super::group::GROUPS.iter().any(|other| other.id == *m),
                "{m} is both a group and a member of {}",
                g.id
            );
        }
    }
}
