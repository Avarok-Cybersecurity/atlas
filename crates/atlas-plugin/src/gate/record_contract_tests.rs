// SPDX-License-Identifier: AGPL-3.0-only

//! Construction and replay contracts for committed gate records.

use super::tests::{MODEL, SHA, frame, hw, run_record, tempdir};
use super::*;
use crate::history::RunRecord;
use crate::result::{RunStatus, Verdict};
use std::collections::BTreeMap;

#[test]
fn date_of_matches_the_utc_civil_calendar() {
    assert_eq!(
        [
            0,
            1_709_164_799,
            1_709_164_800,
            1_709_251_199,
            1_709_251_200,
            1_735_689_599,
            1_735_689_600,
        ]
        .map(date_of),
        [
            "1970-01-01",
            "2024-02-28",
            "2024-02-29",
            "2024-02-29",
            "2024-03-01",
            "2024-12-31",
            "2025-01-01",
        ]
    );
}

#[test]
fn the_record_path_is_date_and_sha_and_replaces_a_same_day_rerun() {
    let dir = tempdir::Dir::new();
    let p1 = record_path(dir.path(), "bfcl-subset", 1_785_891_382, SHA);
    assert_eq!(
        p1,
        dir.path()
            .join(".benchmarks/bfcl-subset/2026-08-05-b72dad1893.json")
    );
    let p2 = record_path(dir.path(), "bfcl-subset", 1_785_891_382 + 3_600, SHA);
    assert_eq!(p1, p2, "same sha + UTC day = same file");
    assert_eq!(
        record_path(dir.path(), "bfcl-subset", 1_785_974_400, SHA),
        dir.path()
            .join(".benchmarks/bfcl-subset/2026-08-06-b72dad1893.json")
    );
}

#[test]
fn from_run_rejects_a_missing_sha_and_a_non_terminal_frame() {
    let record = run_record(BTreeMap::new(), Verdict::pass("ok"));
    for missing in ["", " \t\n"] {
        assert_eq!(
            GateRecord::from_run(
                &record,
                hw(),
                missing.into(),
                Vec::new(),
                None,
                Default::default(),
            )
            .unwrap_err()
            .to_string(),
            "a gate record needs the commit sha it was measured from"
        );
    }

    let mut running = record.clone();
    running.frame.status = RunStatus::Running;
    assert_eq!(
        GateRecord::from_run(
            &running,
            hw(),
            SHA.into(),
            Vec::new(),
            None,
            Default::default(),
        )
        .unwrap_err()
        .to_string(),
        "the run never reached a terminal frame — nothing to gate"
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
    assert_eq!(
        gate.command,
        [
            "spark",
            "benchmark",
            "run",
            "bfcl-subset",
            "--url",
            "http://127.0.0.1:8888",
            "--model",
            MODEL,
            "--param",
            "repeats=12",
            "--pull-request-gate",
        ]
    );
    assert_eq!(gate.verdict.as_deref(), Some("PASS"));
    assert_eq!(gate.frame_status, RunStatus::Completed);
}

#[test]
fn a_self_provisioned_run_records_the_recipe_not_a_dead_url() {
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
    assert_eq!(
        gate.command,
        [
            "spark",
            "benchmark",
            "run",
            "bfcl-subset",
            "--param",
            "repeats=12",
            "--pull-request-gate",
        ]
    );
    assert_eq!(
        gate.served_by.as_deref(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth")
    );
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
    assert_eq!(
        gate.command,
        [
            "spark",
            "benchmark",
            "run",
            "agentic-webserver",
            "--url",
            "http://127.0.0.1:8888",
            "--model",
            MODEL,
            "--param",
            "repeats=12",
            "--yes",
            "--pull-request-gate",
        ]
    );
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
    assert_eq!(gate.frame_status, RunStatus::Failed);
    assert_eq!(gate.verdict.as_deref(), Some("FAIL"));
    assert_eq!(gate.verdict_reason, "scoring crashed");
    assert!(gate.frame_status_failed());
    assert!(!gate.verdict_passes());
}
