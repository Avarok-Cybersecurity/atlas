// SPDX-License-Identifier: AGPL-3.0-only

//! Coverage tests: which committed record still speaks for HEAD.
//!
//! Split from `tests.rs` for the 500-LoC cap. These are the ones that
//! need a real git repo, because `record_covers` walks ancestry: a record
//! can never be written AT head (committing moves head), so an ancestor
//! must be able to speak for it — until a perf path changes between them.

use super::tests::{tempdir, *};
use super::*;
use crate::result::{RunStatus, Verdict};
use std::collections::BTreeMap;

mod scratch_repo {
    use std::path::Path;
    use std::process::Command;

    fn git(root: &Path, args: &[&str]) {
        let out = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .env("GIT_AUTHOR_NAME", "test")
            .env("GIT_AUTHOR_EMAIL", "t@t")
            .env("GIT_COMMITTER_NAME", "test")
            .env("GIT_COMMITTER_EMAIL", "t@t")
            .output()
            .expect("git runs");
        assert!(out.status.success(), "git {args:?}: {:?}", out);
    }

    pub fn init(root: &Path) {
        git(root, &["init", "-q"]);
        std::fs::write(root.join("README.md"), "first").unwrap();
        git(root, &["add", "."]);
        git(root, &["commit", "-q", "-m", "first"]);
    }

    pub fn head(root: &Path) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["rev-parse", "--short=10", "HEAD"])
            .output()
            .expect("git runs");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    pub fn commit(root: &Path, file: &str, contents: &str, message: &str) {
        let path = root.join(file);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, contents).unwrap();
        git(root, &["add", "."]);
        git(root, &["commit", "-q", "-m", message]);
    }
}

#[test]
fn an_ancestor_record_covers_head_until_a_perf_path_changes() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    let sha_a = scratch_repo::head(root);

    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(root, id)).unwrap();
        std::fs::write(
            baseline_path(root, id),
            serde_json::to_string_pretty(&bfcl_baseline()).unwrap(),
        )
        .unwrap();
    }
    for id in REQUIRED_GATES {
        plant(root, id, &sha_a, 1_785_891_382, "PASS");
    }

    // A docs-only commit afterwards: every record still covers head.
    scratch_repo::commit(root, "docs/notes.md", "hello", "docs only");
    let sha_b = scratch_repo::head(root);
    assert!(
        record_covers(root, &sha_b, &sha_a),
        "docs-only diff is inert"
    );
    let gates = check_gates(root, &sha_b);
    for id in REQUIRED_GATES {
        assert!(
            matches!(gates[id], GateStatus::Pass),
            "{id}: {:?}",
            gates[id]
        );
    }

    // A change under crates/ invalidates every earlier record.
    scratch_repo::commit(root, "crates/x.rs", "// code", "touch a crate");
    let sha_c = scratch_repo::head(root);
    assert!(
        !record_covers(root, &sha_c, &sha_a),
        "crates/ diff invalidates"
    );
    let gates = check_gates(root, &sha_c);
    for id in REQUIRED_GATES {
        assert!(
            matches!(&gates[id], GateStatus::Missing(m) if m.contains(&sha_a)),
            "{id}: {:?}",
            gates[id]
        );
    }
}

#[test]
fn a_failed_frame_fails_the_gate_even_with_passing_numbers() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    std::fs::create_dir_all(gate_dir(root, "bfcl-subset")).unwrap();
    std::fs::write(
        baseline_path(root, "bfcl-subset"),
        serde_json::to_string_pretty(&bfcl_baseline()).unwrap(),
    )
    .unwrap();
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 90.0);
    let mut record = run_record(metrics.clone(), Verdict::fail("scoring crashed"));
    record.frame = frame(RunStatus::Failed, metrics, Verdict::fail("scoring crashed"));
    let mut gate = GateRecord::from_run(&record, hw(), SHA.into(), None).unwrap();
    gate.recorded_at = 1_785_891_382;
    write_record(root, &gate).unwrap();

    let gates = check_gates(root, SHA);
    match &gates["bfcl-subset"] {
        GateStatus::Fail(reasons) => assert!(reasons.iter().any(|r| r.contains("failed"))),
        other => panic!("wanted Fail, got {other:?}"),
    }
}

#[test]
fn the_summary_names_the_model_the_numbers_and_the_verdict() {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 87.74);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        None,
    )
    .unwrap();
    assert!(gate.summary.contains(MODEL), "{}", gate.summary);
    assert!(
        gate.summary.contains("overall_accuracy=87.74"),
        "{}",
        gate.summary
    );
    assert!(gate.summary.contains("Pass"), "{}", gate.summary);
}

#[test]
fn required_gates_are_registered_benchmarks() {
    for id in REQUIRED_GATES {
        assert!(
            crate::registry::find(id).is_some(),
            "{id} is not registered"
        );
    }
}

/// A two-sided bound is a RANGE, and an equal pair is an EXACT pin.
///
/// ★ Regression: only the one-sided arms existed, so `{"min": 995, "max": 995}`
/// fell through to "malformed bound". Fail-closed, so nothing scored leniently
/// — but the gate then failed every run and blamed the baseline's syntax
/// instead of the measurement, which makes an exact pin unusable. The BFCL draw
/// size is pinned exactly this way.
#[test]
fn an_exact_pin_passes_only_on_the_pinned_value() {
    use crate::gate::{Bound, Comparison, compare};

    let pin = Bound {
        min: Some(1004.0),
        max: Some(1004.0),
        noise: None,
    };
    assert!(matches!(compare("samples", 1004.0, &pin), Comparison::Pass));

    // The exact failure this pin exists to catch: the echolp draw silently
    // becoming 972 because a subset floor defaulted to 0.
    let Comparison::Fail(msg) = compare("samples", 972.0, &pin) else {
        panic!("a draw of 972 against a pin of 1004 must FAIL, not pass or skip");
    };
    assert!(msg.contains("972") && msg.contains("1004"), "{msg}");
    assert!(
        !msg.contains("malformed"),
        "must blame the measurement, not the baseline's syntax: {msg}"
    );
}

#[test]
fn a_two_sided_range_accepts_its_interior_and_rejects_outside() {
    use crate::gate::{Bound, Comparison, compare};

    let range = Bound {
        min: Some(10.0),
        max: Some(20.0),
        noise: None,
    };
    for v in [10.0, 15.0, 20.0] {
        assert!(
            matches!(compare("m", v, &range), Comparison::Pass),
            "{v} is inside [10, 20]"
        );
    }
    for v in [9.0, 21.0] {
        assert!(
            matches!(compare("m", v, &range), Comparison::Fail(_)),
            "{v} is outside [10, 20]"
        );
    }
}

#[test]
fn a_bound_with_no_side_at_all_is_still_reported() {
    use crate::gate::{Bound, Comparison, compare};

    let empty = Bound {
        min: None,
        max: None,
        noise: None,
    };
    assert!(matches!(compare("m", 1.0, &empty), Comparison::Skip(_)));
}
