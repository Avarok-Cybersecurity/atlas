// SPDX-License-Identifier: AGPL-3.0-only

//! Construction and replay contracts for committed gate records.

use super::record::resolve_perf_env;
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

// ── scheduler performance-control provenance (atlas#812) ───────────────────
//
// The defect these pin: `--pull-request-gate` starts the server inside this
// process, so the scheduler reads ATLAS_PREFILL_CODISPATCH* from the inherited
// environment. Nothing pinned or recorded them, so two records could share a
// tree, a recipe and a full set of serve overrides while executing different
// admission behaviour — and the investigation that needed to tell those apart
// found the records could not.

#[test]
fn an_unset_control_is_recorded_as_the_default_the_scheduler_would_apply() {
    // "absent" and "explicitly set to the default" are the SAME run. A record
    // that showed one as blank and the other as a number would invite a reader
    // to infer a difference that did not exist.
    let resolved = resolve_perf_env(|_| None);
    assert_eq!(
        resolved.get("ATLAS_PREFILL_CODISPATCH").map(String::as_str),
        Some("0")
    );
    assert_eq!(
        resolved
            .get("ATLAS_PREFILL_CODISPATCH_WINDOW_MS")
            .map(String::as_str),
        Some("100")
    );
    assert_eq!(
        resolved
            .get("ATLAS_PREFILL_CODISPATCH_SETTLE_MS")
            .map(String::as_str),
        Some("10")
    );
}

#[test]
fn a_set_control_wins_and_an_empty_one_does_not() {
    // An exported-but-empty variable is how a shell spells "I did not set
    // this"; the scheduler's own parse falls back to the default for it, so
    // recording the empty string would misreport the run.
    let resolved = resolve_perf_env(|k| match k {
        "ATLAS_PREFILL_CODISPATCH" => Some("1".into()),
        "ATLAS_PREFILL_CODISPATCH_WINDOW_MS" => Some("   ".into()),
        _ => None,
    });
    assert_eq!(
        resolved.get("ATLAS_PREFILL_CODISPATCH").map(String::as_str),
        Some("1")
    );
    assert_eq!(
        resolved
            .get("ATLAS_PREFILL_CODISPATCH_WINDOW_MS")
            .map(String::as_str),
        Some("100"),
        "an empty value must resolve to the default, not to the empty string"
    );
}

/// The defaults above are duplicated from `scheduler::mod_helpers` because
/// `atlas-plugin` does not depend on `spark-server`. This is the test that
/// makes the duplication safe: it reads the scheduler's own source and fails
/// if a default moves there without moving here, which would silently make
/// every record disclose a value the server never used.
#[test]
fn perf_env_defaults_match_the_scheduler() {
    let path = repo_root().join("crates/spark-server/src/scheduler/mod_helpers.rs");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    // Match the RESOLUTION, not the first mention: each control is named in a
    // doc comment before it is read, so anchoring on the name alone would
    // assert against prose and pass whatever the code did.
    let resolution = |var: &str| -> String {
        let at = src
            .find(&format!("std::env::var(\"{var}\")"))
            .unwrap_or_else(|| panic!("{var} is not read in mod_helpers.rs"));
        src[at..].chars().take(220).collect()
    };
    assert!(
        resolution("ATLAS_PREFILL_CODISPATCH_WINDOW_MS").contains("unwrap_or(100)"),
        "the scheduler's co-dispatch WINDOW default moved; PERF_CONTROLS in record.rs still \
         says 100 and every record would disclose a value the server never used"
    );
    assert!(
        resolution("ATLAS_PREFILL_CODISPATCH_SETTLE_MS").contains("unwrap_or(10)"),
        "the scheduler's co-dispatch SETTLE default moved; PERF_CONTROLS in record.rs still \
         says 10"
    );
    let enable = resolution("ATLAS_PREFILL_CODISPATCH");
    assert!(
        enable.contains("unwrap_or(false)"),
        "the scheduler's co-dispatch ENABLE default moved; the record's \"0\" default is only \
         correct while an unset variable means off"
    );
}

fn repo_root() -> std::path::PathBuf {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    while !d.join(".git").exists() {
        assert!(d.pop(), "no repo root above CARGO_MANIFEST_DIR");
    }
    d
}
