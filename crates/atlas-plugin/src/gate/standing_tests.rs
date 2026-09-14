// SPDX-License-Identifier: AGPL-3.0-only
//! `record_standing`: the one answer the gate verdict and the agreement rule
//! share (#1086). A certified commit stays certified for every successor
//! that touches nothing its gate measures — by CONTENT, never ancestry
//! (`coverage_squash_tests`): a record from before a perf-path change does
//! not stand, and a commit git cannot diff cannot be judged at all.

use super::check::{Standing, record_standing};
use super::coverage_tests::{any_gate, scratch_repo};
use super::tests::{hw, run_record, tempdir};
use super::*;
use crate::result::Verdict;
use std::collections::BTreeMap;

fn record_at(sha: &str) -> GateRecord {
    GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        sha.to_string(),
        Vec::new(),
        None,
    )
    .unwrap()
}

#[test]
fn a_record_stands_across_harmless_commits_and_falls_to_a_perf_path_or_an_unknown_commit() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    scratch_repo::commit(
        root,
        "crates/x/src/lib.rs",
        "// measured",
        "the measured tree",
    );
    let measured = scratch_repo::head(root);
    let record = record_at(&measured);
    let gate = any_gate();

    // At its own commit, and after a docs-only commit: stands.
    assert_eq!(
        record_standing(root, &measured, &record, &gate),
        Standing::Stands
    );
    scratch_repo::commit(root, "docs/notes.md", "words", "docs only");
    let docs = scratch_repo::head(root);
    assert_eq!(
        record_standing(root, &docs, &record, &gate),
        Standing::Stands
    );

    // Content, not ancestry: a record from a side branch whose perf-path
    // content equals the head's stands (the squash-merge shape), and one
    // from a commit this repository does not have cannot be judged.
    let main_branch = scratch_repo::current_branch(root);
    scratch_repo::branch(root, "side");
    scratch_repo::commit(root, "docs/other.md", "aside", "side branch");
    let side = scratch_repo::head(root);
    scratch_repo::checkout_default(root, &main_branch);
    assert_eq!(
        record_standing(root, &docs, &record_at(&side), &gate),
        Standing::Stands
    );
    assert_eq!(
        record_standing(root, &docs, &record_at("0000000000"), &gate),
        Standing::Unknown
    );

    // NEGATIVE CONTROL: a later commit touches a perf path the strictest gate
    // reads — the record no longer stands, and the path is named.
    scratch_repo::commit(root, "crates/x/src/lib.rs", "// changed", "perf path");
    let moved = scratch_repo::head(root);
    assert_eq!(
        record_standing(root, &moved, &record, &gate),
        Standing::Invalidated(vec!["crates/x/src/lib.rs".to_string()])
    );
}
