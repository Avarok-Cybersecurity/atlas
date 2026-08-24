// SPDX-License-Identifier: AGPL-3.0-only

//! The dirty-working-tree guard: a record must not name a commit whose sources
//! did not build the binary that produced it.
//!
//! Split from `coverage_tests.rs` for the 500-LoC cap. Like those, these need a
//! real git repo — the question is what `git status` says about the working
//! tree, and that cannot be faked with fixture files.

use super::coverage_tests::scratch_repo;
use super::tests::{tempdir, *};
use super::*;
use crate::result::Verdict;
use std::collections::BTreeMap;

/// The guard has to be silent for the file a campaign dirties every single
/// time, and loud for the one that changes the answer.
///
/// `.benchmarks/` is deliberately absent from `PERF_PATHS` — the records ARE
/// the verdict, not its subject — and during a gate campaign the previous
/// bench's record file is uncommitted in the tree for hours. A guard that
/// fired on that would be dismissed by the second run and would then be
/// dismissed on the one occasion it mattered.
#[test]
fn a_dirty_record_file_is_silent_and_a_dirty_crate_is_not() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    scratch_repo::commit(root, "crates/a/src/lib.rs", "fn a() {}", "add a");

    assert!(
        dirty_perf_paths(root).unwrap().is_empty(),
        "a freshly committed tree has nothing uncommitted"
    );

    std::fs::create_dir_all(root.join(".benchmarks/bfcl-subset")).unwrap();
    std::fs::write(
        root.join(".benchmarks/bfcl-subset/2026-08-07-abc.json"),
        "{}",
    )
    .unwrap();
    assert!(
        dirty_perf_paths(root).unwrap().is_empty(),
        "an uncommitted gate record is the normal state of a campaign"
    );

    std::fs::write(root.join("crates/a/src/lib.rs"), "fn a() { todo!() }").unwrap();
    assert_eq!(dirty_perf_paths(root).unwrap(), ["crates/a/src/lib.rs"]);
}

/// An untracked file is as invisible to the sha as a modified one, and a build
/// that globs — `kernels/` does — will happily compile it. Over-broad costs a
/// re-run; under-broad is a lie.
#[test]
fn an_untracked_kernel_counts_but_an_ignored_file_does_not() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    std::fs::create_dir_all(root.join("kernels")).unwrap();
    std::fs::write(root.join("kernels/new.cu"), "__global__ void k() {}").unwrap();
    assert_eq!(dirty_perf_paths(root).unwrap(), ["kernels/new.cu"]);

    scratch_repo::commit(root, ".gitignore", "kernels/*.cu\n", "ignore built kernels");
    assert!(
        dirty_perf_paths(root).unwrap().is_empty(),
        "an ignored file is not evidence of an unrecorded source change"
    );
}

/// "Could not tell" must not render as "nothing to disclose".
#[test]
fn a_non_checkout_errs_rather_than_reporting_a_clean_tree() {
    let dir = tempdir::Dir::new();
    assert!(
        dirty_perf_paths(dir.path()).is_err(),
        "no git metadata means the question is unanswered, not answered clean"
    );
}

/// The record carries the dirt, so a reader six weeks later can see it without
/// having watched the console — and the check rejects it.
///
/// `record_covers` proves nothing changed between the record's commit and head.
/// That proof cannot see a change that was never committed, which is precisely
/// how a passing agentic record came to name a commit that did not contain the
/// truncation fix its binary carried.
#[test]
fn a_record_measured_from_a_dirty_tree_fails_the_gate() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    std::fs::create_dir_all(gate_dir(root, "bfcl-subset")).unwrap();
    write_baseline(root, "bfcl-subset", &bfcl_baseline());
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 90.0);

    let mut gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        vec!["crates/spark-model/src/layers/gdn.rs".to_string()],
        None,
        Default::default(),
    )
    .unwrap();
    gate.recorded_at = 1_785_891_382;
    write_record(root, &gate).unwrap();

    // Every number in it clears the baseline and the run's own verdict is PASS.
    assert!(gate.verdict_passes());
    assert!(check_record(&gate, &bfcl_baseline()).is_none());

    match &check_gates(root, SHA)["bfcl-subset"] {
        GateStatus::Fail(reasons) => assert_eq!(
            reasons,
            &[format!(
                "measured from a dirty tree — 1 uncommitted invalidation-set file(s) \
                 when the run started (crates/spark-model/src/layers/gdn.rs), so the binary \
                 was not {SHA}"
            )]
        ),
        other => panic!("a record that names no commit is not a pass: {other:?}"),
    }
}

/// A clean record must not grow a key, and a record written before the field
/// existed must still parse — otherwise the guard would retroactively
/// invalidate the committed history it was added to protect.
#[test]
fn the_field_is_absent_when_clean_and_optional_when_reading() {
    let gate = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    let json = serde_json::to_string(&gate).unwrap();
    assert!(!json.contains("dirty_paths"), "{json}");

    let older: GateRecord = serde_json::from_str(&json).expect("an older record still parses");
    assert!(older.dirty_paths.is_empty());
}
