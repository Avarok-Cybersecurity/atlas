// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the pull-request gate records.

use super::*;
use crate::hardware::Hardware;
use crate::history::{RunRecord, RunSource};
use crate::result::{BenchmarkResult, RunStatus, Verdict};
use std::collections::BTreeMap;

const MODEL: &str = "Qwen/Qwen3.6-35B-A3B-FP8";
const SHA: &str = "b72dad1893";

mod tempdir {
    use std::path::{Path, PathBuf};
    pub struct Dir(PathBuf);
    impl Dir {
        pub fn new() -> Self {
            let n = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default();
            let p = std::env::temp_dir()
                .join(format!("atlas-gate-{n}-{:?}", std::thread::current().id()));
            std::fs::create_dir_all(&p).expect("scratch dir");
            Self(p)
        }
        pub fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn frame(status: RunStatus, metrics: BTreeMap<String, f64>, verdict: Verdict) -> BenchmarkResult {
    let mut f = BenchmarkResult::completed("done", std::time::Duration::ZERO);
    f.status = status;
    f.with_metrics(metrics).with_verdict(verdict)
}

fn run_record(metrics: BTreeMap<String, f64>, verdict: Verdict) -> RunRecord {
    let mut params = BTreeMap::new();
    params.insert("repeats".to_string(), "12".to_string());
    RunRecord {
        schema: 1,
        run_id: "run-1".to_string(),
        benchmark_id: "bfcl-subset".to_string(),
        benchmark_name: "BFCL (subset)".to_string(),
        recorded_at: 1_785_891_382,
        target_url: "http://127.0.0.1:8888".to_string(),
        target_model: MODEL.to_string(),
        params,
        source: RunSource::Cli,
        atlas_version: "test".to_string(),
        frame: frame(RunStatus::Completed, metrics, verdict),
    }
}

fn bfcl_baseline() -> GateBaseline {
    let mut metrics = BTreeMap::new();
    metrics.insert(
        "overall_accuracy".to_string(),
        Bound {
            min: Some(83.64),
            ..Bound::default()
        },
    );
    GateBaseline {
        model: MODEL.to_string(),
        note: "MLPerf floor".to_string(),
        metrics,
    }
}

#[test]
fn date_of_matches_the_utc_civil_calendar() {
    assert_eq!(date_of(0), "1970-01-01");
    assert_eq!(date_of(1_785_891_382), "2026-08-05");
    // Leap-year boundary.
    assert_eq!(date_of(1_709_251_200), "2024-03-01");
    // The last second of a year.
    assert_eq!(date_of(1_735_689_599), "2024-12-31");
}

#[test]
fn the_record_path_is_date_and_sha_and_replaces_a_same_day_rerun() {
    let dir = tempdir::Dir::new();
    let p1 = record_path(dir.path(), "bfcl-subset", 1_785_891_382, SHA);
    assert!(p1.ends_with(".benchmarks/bfcl-subset/2026-08-05-b72dad1893.json"));
    let p2 = record_path(dir.path(), "bfcl-subset", 1_785_891_382 + 3_600, SHA);
    assert_eq!(p1, p2, "same sha + UTC day = same file");
}

#[test]
fn from_run_rejects_a_missing_sha_and_a_non_terminal_frame() {
    let record = run_record(BTreeMap::new(), Verdict::pass("ok"));
    assert!(GateRecord::from_run(&record, Hardware::unknown(), String::new()).is_err());

    let mut running = record.clone();
    running.frame.status = RunStatus::Running;
    assert!(GateRecord::from_run(&running, Hardware::unknown(), SHA.into()).is_err());
}

#[test]
fn from_run_reconstructs_the_exact_cli_command() {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 87.74);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        Hardware::unknown(),
        SHA.into(),
    )
    .unwrap();
    let joined = gate.command.join(" ");
    assert!(
        joined.starts_with("spark benchmark run bfcl-subset"),
        "{joined}"
    );
    assert!(
        joined.contains("--model Qwen/Qwen3.6-35B-A3B-FP8"),
        "{joined}"
    );
    assert!(joined.contains("--param repeats=12"), "{joined}");
    assert!(joined.ends_with("--pull-request-gate"), "{joined}");
    assert_eq!(gate.verdict.as_deref(), Some("PASS"));
    assert_eq!(gate.frame_status, RunStatus::Completed);
}

#[test]
fn the_agentic_bench_needs_yes_in_its_command() {
    let mut record = run_record(BTreeMap::new(), Verdict::pass("ok"));
    record.benchmark_id = "agentic-webserver".to_string();
    let gate = GateRecord::from_run(&record, Hardware::unknown(), SHA.into()).unwrap();
    assert!(gate.command.contains(&"--yes".to_string()));
}

#[test]
fn a_failed_frame_is_recorded_but_never_passes() {
    let record = RunRecord {
        frame: frame(
            RunStatus::Failed,
            BTreeMap::new(),
            Verdict::fail("scoring crashed"),
        ),
        ..run_record(BTreeMap::new(), Verdict::fail("scoring crashed"))
    };
    let gate = GateRecord::from_run(&record, Hardware::unknown(), SHA.into()).unwrap();
    assert!(gate.frame_status_failed());
    assert!(!gate.verdict_passes());
}

#[test]
fn compare_enforces_min_max_and_noise() {
    let floor = Bound {
        min: Some(83.64),
        noise: Some(0.4),
        ..Bound::default()
    };
    assert!(matches!(compare("x", 84.0, &floor), Comparison::Pass));
    assert!(
        matches!(compare("x", 83.3, &floor), Comparison::Pass),
        "noise covers the dip"
    );
    assert!(matches!(compare("x", 83.0, &floor), Comparison::Fail(_)));

    let ceiling = Bound {
        max: Some(1300.0),
        ..Bound::default()
    };
    assert!(matches!(compare("wall", 978.0, &ceiling), Comparison::Pass));
    assert!(matches!(
        compare("wall", 1400.0, &ceiling),
        Comparison::Fail(_)
    ));

    let malformed = Bound {
        min: Some(1.0),
        max: Some(2.0),
        ..Bound::default()
    };
    assert!(matches!(compare("x", 1.5, &malformed), Comparison::Skip(_)));
}

#[test]
fn check_record_refuses_a_cross_checkpoint_comparison() {
    let gate = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        Hardware::unknown(),
        SHA.into(),
    )
    .unwrap();
    let mut baseline = bfcl_baseline();
    baseline.model = "some-other-model".to_string();
    let problems = check_record(&gate, &baseline).expect("refused");
    assert!(problems[0].contains("some-other-model"), "{}", problems[0]);
}

#[test]
fn check_record_scores_every_bound_and_missing_metric() {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 87.74);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        Hardware::unknown(),
        SHA.into(),
    )
    .unwrap();
    let mut baseline = bfcl_baseline();
    baseline.metrics.insert(
        "samples".to_string(),
        Bound {
            min: Some(995.0),
            ..Bound::default()
        },
    );
    let problems = check_record(&gate, &baseline).expect("samples missing");
    assert!(
        problems.iter().any(|p| p.starts_with("samples")),
        "{problems:?}"
    );

    let passing = bfcl_baseline();
    assert!(check_record(&gate, &passing).is_none());
}

#[test]
fn write_and_read_round_trip_through_the_repo_layout() {
    let dir = tempdir::Dir::new();
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 87.74);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        Hardware::unknown(),
        SHA.into(),
    )
    .unwrap();
    let path = write_record(dir.path(), &gate).unwrap();
    assert!(path.starts_with(dir.path().join(".benchmarks")));
    let back = read_record(&path).unwrap();
    assert_eq!(back.git_sha, SHA);
    assert_eq!(back.metrics["overall_accuracy"], 87.74);
}

fn plant(root: &Path, id: &str, sha: &str, secs: u64, verdict: &str) {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 90.0);
    let record = run_record(metrics, Verdict::pass("ok"));
    let mut gate = GateRecord::from_run(&record, Hardware::unknown(), sha.to_string()).unwrap();
    gate.benchmark_id = id.to_string();
    gate.verdict = Some(verdict.to_string());
    gate.recorded_at = secs;
    write_record(root, &gate).unwrap();
}

#[test]
fn check_gates_reports_each_required_bench() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(root, id)).unwrap();
        let baseline = bfcl_baseline();
        std::fs::write(
            baseline_path(root, id),
            serde_json::to_string_pretty(&baseline).unwrap(),
        )
        .unwrap();
    }
    // Passing record for this sha.
    plant(root, "bfcl-subset", SHA, 1_785_891_382, "PASS");
    // Record for ANOTHER sha.
    plant(root, "ttft-warm-gate", "aaaaaaaaaa", 1_785_891_382, "PASS");
    // Failing record for this sha.
    plant(root, "agentic-webserver", SHA, 1_785_891_382, "FAIL");
    // ttft-cold-gate: nothing planted at all.

    let gates = check_gates(root, SHA);
    assert!(matches!(gates["bfcl-subset"], GateStatus::Pass));
    assert!(
        matches!(&gates["ttft-warm-gate"], GateStatus::Missing(m) if m.contains("aaaaaaaaaa")),
        "{:?}",
        gates["ttft-warm-gate"]
    );
    assert!(matches!(gates["ttft-cold-gate"], GateStatus::Missing(_)));
    match &gates["agentic-webserver"] {
        GateStatus::Fail(reasons) => assert!(reasons.iter().any(|r| r.contains("not PASS"))),
        other => panic!("wanted Fail, got {other:?}"),
    }
}

/// A scratch git repo, so the ancestry + invalidation rules can be tested
/// against real commits rather than mocked shas.
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
    let mut gate = GateRecord::from_run(&record, Hardware::unknown(), SHA.into()).unwrap();
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
        Hardware::unknown(),
        SHA.into(),
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
