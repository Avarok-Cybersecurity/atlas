// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the pull-request gate records.

use super::*;
use crate::hardware::Hardware;
use crate::history::{RunRecord, RunSource};
use crate::result::{BenchmarkResult, RunStatus, Verdict};
use std::collections::BTreeMap;

pub(super) const MODEL: &str = "Qwen/Qwen3.6-35B-A3B-FP8";
/// The box class the fixtures report, and the key their baselines are under.
pub(super) const TEST_HW: &str = "gb10";

/// A realistic fingerprint, so the tests exercise the real `gate_key()`
/// derivation rather than the degenerate "unknown" path — which has its own
/// test below, because an unknown box must FAIL to resolve rather than
/// quietly borrow some other box's thresholds.
pub(super) fn hw() -> Hardware {
    Hardware {
        gpu: "NVIDIA GB10".to_string(),
        driver: "580.126.09".to_string(),
        sm_clock_mhz: Some(2405.0),
        source: "nvidia-smi".to_string(),
    }
}
pub(super) const SHA: &str = "b72dad1893";

pub(super) mod tempdir {
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

pub(super) fn frame(
    status: RunStatus,
    metrics: BTreeMap<String, f64>,
    verdict: Verdict,
) -> BenchmarkResult {
    let mut f = BenchmarkResult::completed("done", std::time::Duration::ZERO);
    f.status = status;
    f.with_metrics(metrics).with_verdict(verdict)
}

pub(super) fn run_record(metrics: BTreeMap<String, f64>, verdict: Verdict) -> RunRecord {
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

pub(super) use super::fixture_baseline::write_baseline;

pub(super) fn bfcl_baseline() -> GateBaseline {
    let mut metrics = BTreeMap::new();
    metrics.insert(
        "overall_accuracy".to_string(),
        Bound {
            min: Some(83.64),
            ..Bound::default()
        },
    );
    baseline_for(MODEL, metrics)
}

/// A schema-v2 baseline with one hardware class and one model.
///
/// The hardware key must match what `Hardware::gate_key()` derives from the
/// record under test — `TEST_HW` below is the fingerprint the fixtures carry,
/// so a mismatch here shows up as an unresolved baseline rather than a silent
/// pass.
pub(super) fn baseline_for(model: &str, metrics: BTreeMap<String, Bound>) -> GateBaseline {
    let mut models = BTreeMap::new();
    models.insert(
        model.to_string(),
        crate::gate::ModelBaseline {
            recipe: Some("qwen3.6/test-recipe".to_string()),
            note: "MLPerf floor".to_string(),
            metrics,
        },
    );
    let mut hardware = BTreeMap::new();
    hardware.insert(
        TEST_HW.to_string(),
        crate::gate::HardwareBaseline {
            default: model.to_string(),
            models,
        },
    );
    GateBaseline {
        schema: 2,
        hardware,
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
    assert!(
        GateRecord::from_run(
            &record,
            hw(),
            String::new(),
            Vec::new(),
            None,
            Default::default()
        )
        .is_err()
    );

    let mut running = record.clone();
    running.frame.status = RunStatus::Running;
    assert!(
        GateRecord::from_run(
            &running,
            hw(),
            SHA.into(),
            Vec::new(),
            None,
            Default::default()
        )
        .is_err()
    );
}

#[test]
fn from_run_reconstructs_the_exact_cli_command() {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 87.74);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        Default::default(),
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
fn a_self_provisioned_run_records_the_recipe_not_a_dead_url() {
    // The gate served this itself on an ephemeral port. Naming that port would
    // give a command that drives nothing (or, worse, whatever else is on 8888
    // later), and a --model the caller never chose. What actually determined
    // the config is the recipe, so that is what the record carries — and the
    // command replays by asking for the same benchmark again.
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 87.74);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth".to_string()),
        Default::default(),
    )
    .unwrap();
    let joined = gate.command.join(" ");
    assert!(!joined.contains("--url"), "no dead endpoint: {joined}");
    assert!(!joined.contains("--model"), "the recipe chose it: {joined}");
    assert!(joined.ends_with("--pull-request-gate"), "{joined}");
    assert_eq!(
        gate.served_by.as_deref(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth")
    );
    // The model is still recorded on its own field — the run must always be
    // able to say what it measured, however the endpoint was obtained.
    assert_eq!(gate.target_model, MODEL);
}

#[test]
fn the_agentic_bench_needs_yes_in_its_command() {
    let mut record = run_record(BTreeMap::new(), Verdict::pass("ok"));
    record.benchmark_id = "agentic-webserver".to_string();
    let gate = GateRecord::from_run(
        &record,
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
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
    let gate = GateRecord::from_run(
        &record,
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
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

    // A two-sided bound is a RANGE, not a malformed entry. It used to be
    // rejected, which made an exact pin unusable: Skip is counted as a problem,
    // so such a bound failed every run and blamed the baseline's syntax rather
    // than the measurement. Nothing could have depended on the old behaviour
    // for that reason. The BFCL draw size is pinned this way — see
    // `an_exact_pin_passes_only_on_the_pinned_value` in coverage_tests.
    let range = Bound {
        min: Some(1.0),
        max: Some(2.0),
        ..Bound::default()
    };
    assert!(matches!(compare("x", 1.5, &range), Comparison::Pass));
    assert!(matches!(compare("x", 2.5, &range), Comparison::Fail(_)));

    // A bound with NO side is the genuinely malformed case.
    let no_side = Bound::default();
    assert!(matches!(compare("x", 1.5, &no_side), Comparison::Skip(_)));
}

#[test]
fn check_record_refuses_a_cross_checkpoint_comparison() {
    let gate = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    // The baseline knows only another checkpoint, so the record's model does
    // not resolve — refused, not scored against the wrong thresholds.
    let mut metrics = BTreeMap::new();
    metrics.insert(
        "overall_accuracy".to_string(),
        Bound {
            min: Some(83.64),
            ..Bound::default()
        },
    );
    let baseline = baseline_for("some-other-model", metrics);
    let problems = check_record(&gate, &baseline).expect("refused");
    assert!(problems[0].contains(MODEL), "{}", problems[0]);
    assert!(problems[0].contains("some-other-model"), "{}", problems[0]);
}

#[test]
fn check_record_refuses_a_cross_hardware_comparison() {
    // A TTFT ceiling measured on one box says nothing about another, so a
    // record from an unrecognised box must fail to resolve rather than borrow
    // whatever entry happens to be present.
    let mut gate = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    gate.hardware = Hardware {
        gpu: "AMD Instinct MI300X".to_string(),
        ..Hardware::default()
    };
    let problems = check_record(&gate, &bfcl_baseline()).expect("refused");
    assert!(problems[0].contains("instinctmi300x"), "{}", problems[0]);
    assert!(problems[0].contains(TEST_HW), "{}", problems[0]);
}

#[test]
fn an_unknown_fingerprint_never_silently_matches() {
    // `fetch_hardware` degrades to `Hardware::unknown()` on EVERY error path
    // without surfacing one, so a torn-down or unreachable endpoint yields a
    // record with no fingerprint. That must not resolve to some box's entry.
    let mut gate = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    gate.hardware = Hardware::unknown();
    assert_eq!(gate.hardware.gate_key(), "unknown");
    let problems = check_record(&gate, &bfcl_baseline()).expect("refused");
    assert!(problems[0].contains("unknown"), "{}", problems[0]);
}

#[test]
fn check_record_scores_every_bound_and_missing_metric() {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 87.74);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    let mut metrics = BTreeMap::new();
    metrics.insert(
        "overall_accuracy".to_string(),
        Bound {
            min: Some(83.64),
            ..Bound::default()
        },
    );
    metrics.insert(
        "samples".to_string(),
        Bound {
            min: Some(995.0),
            ..Bound::default()
        },
    );
    let baseline = baseline_for(MODEL, metrics);
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
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    let path = write_record(dir.path(), &gate).unwrap();
    assert!(path.starts_with(dir.path().join(".benchmarks")));
    let back = read_record(&path).unwrap();
    assert_eq!(back.git_sha, SHA);
    assert_eq!(back.metrics["overall_accuracy"], 87.74);
}

pub(super) fn plant(root: &Path, id: &str, sha: &str, secs: u64, verdict: &str) {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 90.0);
    let record = run_record(metrics, Verdict::pass("ok"));
    let mut gate = GateRecord::from_run(
        &record,
        hw(),
        sha.to_string(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
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
        write_baseline(root, id, &bfcl_baseline());
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
