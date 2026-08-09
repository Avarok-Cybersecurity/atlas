// SPDX-License-Identifier: AGPL-3.0-only

//! Squash-merge coverage: a record must survive its own PR landing.
//!
//! Split from `coverage_tests.rs` for the 500-LoC cap.
//!
//! # The outage these pin
//!
//! Atlas squash-merges. A gate record is written on a PR branch against a
//! commit on that branch; the squash lands a NEW commit on main with a
//! different sha and no parent link back. Under the old
//! `merge-base --is-ancestor` guard, every record a PR paid GPU hours for
//! stopped covering anything the moment it merged.
//!
//! That is not hypothetical. `.benchmarks/*/2026-08-09-b0be4ba0e6.json` are
//! five real passing records for #389; after #389 squash-landed as `dd2ac46d5`
//! the gate reported "not an ancestor of this commit" for all five, main went
//! red, and every PR opened afterwards inherited a demand for five fresh GPU
//! legs.
//!
//! The rule now is content, not ancestry — the same lesson the kernel-lever
//! audits learned: never ask `merge-base --is-ancestor` of a repo that
//! squashes, diff the content instead.

use super::coverage_tests::{any_gate, scratch_repo};
use super::tests::{tempdir, *};
use super::*;

/// A squash-merge: same tree, unrelated commit. The record must still cover.
#[test]
fn a_record_survives_its_pr_being_squash_merged() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);

    // Fixtures first, so `git add .` inside `commit` cannot sweep scaffolding
    // into the commit under test (the trap `coverage_tests` documents).
    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(root, id)).unwrap();
        write_baseline(root, id, &bfcl_baseline());
    }
    scratch_repo::commit(root, "docs/seed.md", "seed", "baseline fixtures");
    let default_branch = scratch_repo::current_branch(root);

    // The PR branch: a real code change, measured there.
    scratch_repo::branch(root, "pr");
    scratch_repo::commit(root, "crates/feature.rs", "// the change", "the feature");
    let branch_tip = scratch_repo::head(root);
    for id in REQUIRED_GATES {
        plant(root, id, &branch_tip, 1_785_891_382, "PASS");
    }

    // The squash: main gets the SAME file contents under a brand-new commit
    // whose history has nothing to do with `branch_tip`.
    scratch_repo::checkout_default(root, &default_branch);
    scratch_repo::commit(
        root,
        "crates/feature.rs",
        "// the change",
        "the feature (#1)",
    );
    let squashed = scratch_repo::head(root);

    assert!(
        !scratch_repo::is_ancestor(root, &branch_tip, &squashed),
        "fixture must reproduce the real shape: the record's commit is NOT an \
         ancestor of the squash"
    );
    assert!(
        record_covers(root, &squashed, &branch_tip, &any_gate()),
        "the squash has byte-identical perf-path content — the record that \
         measured it still speaks for it"
    );
}

/// The guard the change must not remove: an unrelated commit that actually
/// DIFFERS on a perf path is still rejected. "Drop ancestry" must not become
/// "accept anything".
#[test]
fn an_unrelated_commit_that_differs_is_still_not_covered() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(root, id)).unwrap();
        write_baseline(root, id, &bfcl_baseline());
    }
    scratch_repo::commit(root, "docs/seed.md", "seed", "baseline fixtures");
    let default_branch = scratch_repo::current_branch(root);

    scratch_repo::branch(root, "pr");
    scratch_repo::commit(root, "crates/feature.rs", "// version A", "feature A");
    let branch_tip = scratch_repo::head(root);
    for id in REQUIRED_GATES {
        plant(root, id, &branch_tip, 1_785_891_382, "PASS");
    }

    // Same path, DIFFERENT contents: not the code that was measured.
    scratch_repo::checkout_default(root, &default_branch);
    scratch_repo::commit(root, "crates/feature.rs", "// version B", "feature B");
    let other = scratch_repo::head(root);

    assert!(
        !record_covers(root, &other, &branch_tip, &any_gate()),
        "different perf-path content must still invalidate — content is the \
         test, and this content differs"
    );
}

/// A record for a commit this clone does not have is fail-closed, not covered.
/// This is the one thing the ancestry check caught incidentally; the diff has
/// to keep catching it.
#[test]
fn a_record_for_an_unknown_commit_is_not_covered() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(root, id)).unwrap();
        write_baseline(root, id, &bfcl_baseline());
    }
    scratch_repo::commit(root, "docs/seed.md", "seed", "baseline fixtures");
    let head = scratch_repo::head(root);

    assert!(
        !record_covers(root, &head, "deadbeefcafe", &any_gate()),
        "git cannot diff a commit that is not here; that must read as \
         not-covered, never as a pass"
    );
}
