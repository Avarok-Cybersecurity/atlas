// SPDX-License-Identifier: AGPL-3.0-only
//! A group must be ALL of its members or none. Three of four is not 75%
//! measured — it is an aggregate over a sample set the thresholds were never
//! drawn against.

use super::group::{BenchmarkGroup, GroupFault, MemberRecord, composition_ok, member_of};

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
