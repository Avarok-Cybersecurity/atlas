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

/// Until the shard descriptors land there are no groups, and nothing may be
/// treated as a member. Pinned so an empty GROUPS table cannot silently mean
/// "everything is a member of nothing in particular".
#[test]
fn nothing_is_a_group_member_yet() {
    assert!(member_of("bfcl-subset-a").is_none());
    assert!(member_of("decode-floor").is_none());
}
