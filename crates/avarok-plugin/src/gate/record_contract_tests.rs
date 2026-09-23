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
            GateRecord::from_run(&record, hw(), missing.into(), Vec::new(), None,)
                .unwrap_err()
                .to_string(),
            "a gate record needs the commit sha it was measured from"
        );
    }

    let mut running = record.clone();
    running.frame.status = RunStatus::Running;
    assert_eq!(
        GateRecord::from_run(&running, hw(), SHA.into(), Vec::new(), None,)
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
    let gate = GateRecord::from_run(&record, hw(), SHA.into(), Vec::new(), None).unwrap();
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
    let gate = GateRecord::from_run(&record, hw(), SHA.into(), Vec::new(), None).unwrap();
    assert_eq!(gate.frame_status, RunStatus::Failed);
    assert_eq!(gate.verdict.as_deref(), Some("FAIL"));
    assert_eq!(gate.verdict_reason, "scoring crashed");
    assert!(gate.frame_status_failed());
    assert!(!gate.verdict_passes());
}

// ── scheduler performance-control provenance (avarok#812) ───────────────────
//
// The defect these pin: `--pull-request-gate` starts the server inside this
// process, so the scheduler reads AVAROK_PREFILL_CODISPATCH* from the inherited
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
        resolved
            .get("AVAROK_PREFILL_CODISPATCH_WINDOW_MS")
            .map(String::as_str),
        Some("100")
    );
    assert_eq!(
        resolved
            .get("AVAROK_PREFILL_CODISPATCH_SETTLE_MS")
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
        "AVAROK_PREFILL_CODISPATCH_SETTLE_MS" => Some("25".into()),
        "AVAROK_PREFILL_CODISPATCH_WINDOW_MS" => Some("   ".into()),
        _ => None,
    });
    assert_eq!(
        resolved
            .get("AVAROK_PREFILL_CODISPATCH_SETTLE_MS")
            .map(String::as_str),
        Some("25")
    );
    assert_eq!(
        resolved
            .get("AVAROK_PREFILL_CODISPATCH_WINDOW_MS")
            .map(String::as_str),
        Some("100"),
        "an empty value must resolve to the default, not to the empty string"
    );
}

/// The defaults above are duplicated from `scheduler::mod_helpers` because
/// `avarok-plugin` does not depend on `spark-server`. This is the test that
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
        resolution("AVAROK_PREFILL_CODISPATCH_WINDOW_MS").contains("unwrap_or(100)"),
        "the scheduler's co-dispatch WINDOW default moved; PERF_CONTROLS in record_env.rs still \
         says 100 and every record would disclose a value the server never used"
    );
    assert!(
        resolution("AVAROK_PREFILL_CODISPATCH_SETTLE_MS").contains("unwrap_or(10)"),
        "the scheduler's co-dispatch SETTLE default moved; PERF_CONTROLS in record_env.rs still \
         says 10"
    );
    // ★ THE ENABLE READ MOVED, 2026-09-22, and this assertion followed it to the
    // SSOT rather than being deleted. `AVAROK_PREFILL_CODISPATCH` became the
    // `--prefill-codispatch` flag, so mod_helpers.rs no longer reads the
    // variable at all — it asks `prefill_codispatch_enabled()`, which resolves
    // the flag first and falls back to the variable. ★ G22 (2026-09-23): the
    // record no longer discloses the enable from `perf_env` at all — a flag
    // never reaches the environment, so the table default "0" contradicted
    // every `--prefill-codispatch true` serve. It is disclosed from the
    // rendered flag in `serve_resolved`, ABSENT when the flag was absent, and
    // that absence reads "off" only while an UNSET variable still means off
    // (a declared legacy variable is in `serve_env`). Asserting that where it
    // is now decided is the point; in mod_helpers.rs it would pass on prose.
    let ssot = repo_root().join("crates/spark-model/src/layers/ops/dispatch_helpers.rs");
    let ssot_src =
        std::fs::read_to_string(&ssot).unwrap_or_else(|e| panic!("{}: {e}", ssot.display()));
    let at = ssot_src
        .find("pub fn prefill_codispatch_enabled()")
        .expect("the codispatch SSOT is gone; PERF_CONTROLS has nothing to agree with");
    let body: String = ssot_src[at..].chars().take(260).collect();
    assert!(
        body.contains("std::env::var(\"AVAROK_PREFILL_CODISPATCH\")"),
        "the env FALLBACK is gone, so an operator setting the documented variable \
         gets nothing while the record still discloses it: {body}"
    );
    assert!(
        body.contains("bool_value_enabled"),
        "the truthiness rule changed; an absent disclosure assumes unset means off: {body}"
    );
    // G22 regression pin: the enable must never be re-resolved from the
    // environment into `perf_env`, where its table default would contradict
    // the flag the serve actually ran.
    assert!(
        !crate::gate::record::resolve_perf_env(|_| None).contains_key("AVAROK_PREFILL_CODISPATCH"),
        "perf_env re-discloses the codispatch enable from the environment; the flag is \
         disclosed in serve_resolved (record_serve::PREFILL_CODISPATCH)"
    );
    // The scheduler must no longer read the variable behind the SSOT's back —
    // two readers of one lever is what made this lever hard to reason about.
    assert!(
        !src.contains("std::env::var(\"AVAROK_PREFILL_CODISPATCH\")"),
        "mod_helpers.rs reads the codispatch variable directly again, bypassing the \
         flag: a --prefill-codispatch that the scheduler ignores is worse than no flag"
    );
}

fn repo_root() -> std::path::PathBuf {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    while !d.join(".git").exists() {
        assert!(d.pop(), "no repo root above CARGO_MANIFEST_DIR");
    }
    d
}

/// The gate record's regime is DERIVED from the run's, not passed beside it.
///
/// Both records used to be handed the same map by one caller, which made them
/// agree by convention — one future edit away from a history record and a gate
/// record describing different regimes for the same run, with nothing anywhere
/// saying which was true. Deriving makes that state impossible to express, and
/// this pins it so the parameter cannot come back.
#[test]
fn the_gate_records_regime_is_the_runs_regime() {
    let mut record = run_record(BTreeMap::new(), Verdict::pass("ok"));
    record.serve_overrides = [
        ("hermetic".to_string(), "true".to_string()),
        ("ssm_cache_slots".to_string(), "0".to_string()),
    ]
    .into_iter()
    .collect();

    let gate = GateRecord::from_run(&record, hw(), SHA.to_string(), Vec::new(), None).unwrap();

    assert_eq!(
        gate.serve_overrides, record.serve_overrides,
        "the gate record must carry the regime the run was measured under"
    );
    // And it must reach the REPLAY command, or the record describes a server
    // nobody can start again.
    let cmd = gate.command.join(" ");
    assert!(
        cmd.contains("--serve-override hermetic=true"),
        "the replay command must reproduce the regime: {cmd}"
    );
    assert!(cmd.contains("--serve-override ssm_cache_slots=0"), "{cmd}");
}

/// A run with no recorded regime produces a gate record with none — an empty
/// map, not a missing one, and no stray `--serve-override` in the command.
#[test]
fn a_run_with_no_recorded_regime_claims_none() {
    let record = run_record(BTreeMap::new(), Verdict::pass("ok"));
    let gate = GateRecord::from_run(&record, hw(), SHA.to_string(), Vec::new(), None).unwrap();
    assert!(gate.serve_overrides.is_empty());
    assert!(!gate.command.join(" ").contains("--serve-override"));
}

/// ★ Oracle: the committed corpus itself. Every record in `.benchmarks/`
/// written before `Hardware::gpu_count` existed must still load, and must
/// load as UNMEASURED — `None`, because reading those as single-GPU would be
/// inventing a topology reading that was never taken. A record written since
/// (stack 1089308's campaign, 2026-09-14, was the first) carries the count
/// its box reported, and that count is a positive number.
///
/// `gpu_count` was added additively (schema stays 1, `#[serde(default)]`,
/// omitted when absent), following `dataset_fingerprint`. The claim that
/// buys — that no migration is needed — is only worth anything if a real old
/// record is read back and still resolves. A hand-written fixture would prove
/// serde's defaulting, not the corpus's compatibility. The split is made on
/// the record's own text: a file without the key is an old record, whatever
/// its date.
#[test]
fn every_committed_record_still_loads_without_a_gpu_count() {
    let benchmarks = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.benchmarks");
    let mut read = 0;
    for gate in std::fs::read_dir(&benchmarks)
        .expect(".benchmarks/ is in the tree")
        .flatten()
    {
        for record in std::fs::read_dir(gate.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            let path = record.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let loaded = read_record(&path).unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
            let text = std::fs::read_to_string(&path).unwrap();
            if text.contains("\"gpu_count\"") {
                assert!(
                    loaded.hardware.gpu_count.is_some_and(|n| n >= 1),
                    "{} wrote a gpu_count that is not a measured width",
                    path.display()
                );
            } else {
                assert_eq!(
                    loaded.hardware.gpu_count,
                    None,
                    "{} carries a width nothing measured",
                    path.display()
                );
            }
            assert_eq!(loaded.schema, 1, "{}", path.display());
            read += 1;
        }
    }
    assert!(
        read > 100,
        "read only {read} records — the walk is broken, not the format"
    );
}

/// The box is recorded the way #569 records it — inside `hardware_state`,
/// captured by the executor and carried through `from_run` untouched. None of
/// the six failing records in #1159 could say which box produced them; every
/// record built from a frame that captured its state can.
#[test]
fn the_record_carries_the_box_identity_the_frame_captured() {
    use crate::hardware::policy::Sensitivity;
    use crate::hardware::state::MachineIdentity;
    use crate::hardware::{HardwareState, HardwareStateReport};
    let mut with_state = run_record(BTreeMap::new(), Verdict::pass("ok"));
    let before = HardwareState {
        captured_at: 1_000,
        machine: MachineIdentity {
            hostname: Some("spark-28c2".into()),
            machine_id: Some("7af66f30966a49b6886e00e2fce4b42f".into()),
            gpu: Some("NVIDIA GB10".into()),
            driver: Some("580.159.03".into()),
        },
        ..HardwareState::default()
    };
    with_state.frame.hardware_state = Some(HardwareStateReport::opened(
        Sensitivity::Correctness,
        before,
        None,
    ));
    let record = GateRecord::from_run(&with_state, hw(), SHA.into(), Vec::new(), None).unwrap();
    let machine = &record
        .hardware_state
        .as_ref()
        .expect("the captured state travels with the record")
        .before
        .machine;
    assert_eq!(
        machine.machine_id.as_deref(),
        Some("7af66f30966a49b6886e00e2fce4b42f")
    );
    assert_eq!(machine.hostname.as_deref(), Some("spark-28c2"));
    let json = serde_json::to_value(&record).unwrap();
    assert_eq!(
        json["hardware_state"]["before"]["machine"]["machine_id"],
        "7af66f30966a49b6886e00e2fce4b42f"
    );

    // Absent means UNMEASURED, and the record says so by carrying nothing —
    // never a machine invented at write time.
    let without = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
    )
    .unwrap();
    assert!(without.hardware_state.is_none());
    assert!(
        serde_json::to_value(&without)
            .unwrap()
            .get("hardware_state")
            .is_none()
    );
}
