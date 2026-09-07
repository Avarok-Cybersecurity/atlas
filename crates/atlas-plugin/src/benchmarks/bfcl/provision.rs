// SPDX-License-Identifier: AGPL-3.0-only

//! `~/.atlas/artifacts/bfcl` — everything BFCL needs, fetched on `load()`.
//!
//! Steps, in order, each one reported before it runs so a slow `pip` is visible
//! rather than a hang:
//!
//! 1. **Preflight** — a `python3` new enough, with `venv` importable.
//! 2. **venv** at `artifacts/bfcl/venv`.
//! 3. **pip install** the pinned `requirements.txt` (this is the download).
//! 4. **Materialize** the single-turn table to `dataset.jsonl` via the
//!    committed `provision.py`, which reads bfcl-eval's own data files.
//!
//! Steps 2–4 are skipped when a [`Stamp`] over the pinned inputs matches, so a
//! changed pin re-provisions by itself and an unchanged one costs a file read.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::artifacts::{ArtifactStore, Stamp, write_asset};
use crate::plugin::PluginHandle;
use crate::python;

pub const PLUGIN_ID: &str = "bfcl";

const REQUIREMENTS: &str = include_str!("../../../assets/bfcl/requirements.txt");
const PROVISION_PY: &str = include_str!("../../../assets/bfcl/provision.py");
const SCORE_PY: &str = include_str!("../../../assets/bfcl/score.py");

/// Minimum interpreter. bfcl-eval and the scorer both use `match` and modern
/// typing syntax.
const MIN_PYTHON: (u32, u32) = (3, 10);

fn stamp_value(requirements: &str, provision: &str, scorer: &str) -> String {
    use sha2::{Digest, Sha256};

    let mut digest = Sha256::new();
    for asset in [requirements, provision, scorer] {
        digest.update((asset.len() as u64).to_le_bytes());
        digest.update(asset.as_bytes());
    }
    format!(
        "v2 py>={}.{} assets={:x}",
        MIN_PYTHON.0,
        MIN_PYTHON.1,
        digest.finalize()
    )
}

/// The provisioned artifact set.
#[derive(Clone, Debug)]
pub struct Artifacts {
    pub dir: PathBuf,
    /// Interpreter inside the venv — the one that can import bfcl-eval.
    pub python: PathBuf,
    pub dataset: PathBuf,
    pub scorer: PathBuf,
    /// Per-subset row counts of the materialized table. The draw is computed
    /// from these.
    pub subset_totals: std::collections::BTreeMap<String, usize>,
}

#[derive(Deserialize)]
struct ProvisionSummary {
    total: usize,
    sha256: String,
    subsets: std::collections::BTreeMap<String, usize>,
}

/// Provision (or verify) the BFCL artifacts. Idempotent.
pub async fn ensure(store: &ArtifactStore, handle: &PluginHandle) -> Result<Artifacts> {
    let dir = store.plugin_dir(PLUGIN_ID)?;
    // Scripts are rewritten whenever the shipped bytes differ, so an Atlas
    // upgrade that changes the scorer cannot leave the previous release's copy
    // scoring runs in ~/.atlas.
    write_asset(&dir, "requirements.txt", REQUIREMENTS)?;
    write_asset(&dir, "provision.py", PROVISION_PY)?;
    write_asset(&dir, "score.py", SCORE_PY)?;

    let dataset = dir.join("dataset.jsonl");
    let scorer = dir.join("score.py");
    let venv = dir.join("venv");
    let interpreter = python::venv_python(&venv);
    // The stamp covers every input that can change the materialized data: the
    // pins and both scripts.
    let stamp = Stamp::new(
        &dir,
        ".provisioned",
        stamp_value(REQUIREMENTS, PROVISION_PY, SCORE_PY),
    );

    if stamp.is_current() && dataset.is_file() && interpreter.is_file() {
        handle.info("BFCL artifacts already provisioned");
        let totals = read_totals(&dir)?;
        return Ok(Artifacts {
            dir,
            python: interpreter,
            dataset,
            scorer,
            subset_totals: totals,
        });
    }

    handle.status("BFCL: checking for python");
    let system_python = python::find_python(MIN_PYTHON.0, MIN_PYTHON.1).await?;
    handle.info(format!("python: {}", system_python.display()));

    handle.status("BFCL: creating venv");
    let interpreter = python::ensure_venv(&system_python, &venv).await?;

    handle.status("BFCL: downloading bfcl-eval (needs network)");
    python::pip_install(&interpreter, &dir.join("requirements.txt")).await?;

    handle.status("BFCL: materializing the single-turn dataset");
    let out = python::run(
        &interpreter,
        &[
            dir.join("provision.py")
                .to_str()
                .context("artifact path is not valid UTF-8")?,
            "--out",
            dataset
                .to_str()
                .context("dataset path is not valid UTF-8")?,
        ],
        Some(&dir),
    )
    .await
    .context("materializing the BFCL dataset")?;

    let summary: ProvisionSummary = serde_json::from_str(out.stdout.trim())
        .with_context(|| format!("provision.py printed unexpected output: {}", out.stdout))?;
    handle.info(format!(
        "BFCL dataset: {} samples across {} subsets (sha256 {}…)",
        summary.total,
        summary.subsets.len(),
        &summary.sha256[..12.min(summary.sha256.len())]
    ));
    std::fs::write(
        dir.join("dataset_summary.json"),
        serde_json::to_string_pretty(&summary.subsets)?,
    )?;
    // Committed last: a stamp written before the data exists turns a
    // half-provisioned directory into a permanent "already done".
    stamp.commit()?;

    Ok(Artifacts {
        dir,
        python: interpreter,
        dataset,
        scorer,
        subset_totals: summary.subsets,
    })
}

fn read_totals(dir: &Path) -> Result<std::collections::BTreeMap<String, usize>> {
    let text = std::fs::read_to_string(dir.join("dataset_summary.json")).context(
        "dataset_summary.json is missing — delete ~/.atlas/artifacts/bfcl to re-provision",
    )?;
    Ok(serde_json::from_str(&text)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "atlas-bfcl-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn python() -> &'static str {
        ["python3", "python"]
            .into_iter()
            .find(|candidate| {
                std::process::Command::new(candidate)
                    .arg("--version")
                    .output()
                    .is_ok_and(|out| out.status.success())
            })
            .expect("the committed Python assets require a Python interpreter")
    }

    fn write_jsonl(path: &Path, rows: &[serde_json::Value]) {
        let text = rows
            .iter()
            .map(serde_json::Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(path, format!("{text}\n")).unwrap();
    }

    #[test]
    fn the_committed_python_assets_are_non_empty_and_own_their_cli() {
        assert!(!PROVISION_PY.trim().is_empty());
        assert!(!SCORE_PY.trim().is_empty());
        let pins: Vec<&str> = REQUIREMENTS
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect();
        // EXACT, not "contains": every pin here is a scorer input, and a
        // dependency that floats can move the score between runs and silently
        // invalidate a stored baseline. soundfile is a TRANSITIVE import
        // bfcl-eval does not declare (bfcl-eval -> qwen_agent, whose utils.py
        // does a bare `import soundfile as sf` at module scope), so without it
        // the venv provisions "successfully" and the scorer then fails to
        // import — which made the gate unrunnable from a clean provision.
        assert_eq!(pins, ["bfcl-eval==2026.3.23", "soundfile==0.14.0"]);

        let dir = temp_dir("cli");
        let provision = dir.join("provision.py");
        let scorer = dir.join("score.py");
        std::fs::write(&provision, PROVISION_PY).unwrap();
        std::fs::write(&scorer, SCORE_PY).unwrap();

        let out = std::process::Command::new(python())
            .arg(&provision)
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2), "provision.py accepted no --out");
        let stderr = String::from_utf8(out.stderr).unwrap();
        assert!(
            stderr.contains("the following arguments are required: --out"),
            "provision.py did not require --out: {stderr}"
        );

        for (provided, missing) in [
            (["--dataset", "unused"], "--responses"),
            (["--responses", "unused"], "--dataset"),
        ] {
            let out = std::process::Command::new(python())
                .arg(&scorer)
                .args(provided)
                .output()
                .unwrap();
            assert_eq!(out.status.code(), Some(2), "score.py accepted no {missing}");
            let stderr = String::from_utf8(out.stderr).unwrap();
            assert!(
                stderr.contains(&format!("the following arguments are required: {missing}")),
                "score.py did not require {missing}: {stderr}"
            );
        }
    }

    #[test]
    fn the_scorer_reproduces_the_reference_aggregation_strategies() {
        let dir = temp_dir("score");
        let fake = dir.join("bfcl_eval");
        for package in [
            fake.clone(),
            fake.join("constants"),
            fake.join("eval_checker"),
            fake.join("eval_checker/ast_eval"),
        ] {
            std::fs::create_dir_all(&package).unwrap();
            std::fs::write(package.join("__init__.py"), "").unwrap();
        }
        std::fs::write(
            fake.join("constants/enums.py"),
            "from enum import Enum\nclass Language(Enum):\n    PYTHON='PYTHON'\n    JAVA='JAVA'\n    JAVASCRIPT='JAVASCRIPT'\n",
        )
        .unwrap();
        std::fs::write(
            fake.join("eval_checker/ast_eval/ast_checker.py"),
            "def ast_checker(**kwargs):\n    return {'valid': bool(kwargs['model_output']) and kwargs['model_name'] == 'gpt-4o-2024-11-20-FC'}\n",
        )
        .unwrap();

        let mut dataset = Vec::new();
        let mut responses = Vec::new();
        let cases = [
            ("simple_python", [true, true, false]),
            ("simple_java", [false, false, false]),
            ("multiple", [false, false, false]),
            ("live_simple", [true, true, true]),
            ("live_multiple", [false, false, false]),
            ("irrelevance", [true, true, true]),
            ("live_irrelevance", [false, false, false]),
        ];
        let counts = [2, 1, 1, 3, 1, 3, 1];
        for ((subset, outcomes), count) in cases.into_iter().zip(counts) {
            for (index, passes) in outcomes.into_iter().take(count).enumerate() {
                let id = format!("{subset}-{index}");
                dataset.push(serde_json::json!({
                    "sample_id": id,
                    "subset": subset,
                    "ground_truth": "[1]",
                    "func_description": "[]"
                }));
                responses.push(serde_json::json!({
                    "sample_id": id,
                    "has_tool_calls": if subset.contains("irrelevance") { !passes } else { passes },
                    "tool_calls": if passes && !subset.contains("irrelevance") {
                        serde_json::json!([{"name": "f", "arguments": {}}])
                    } else {
                        serde_json::json!([])
                    }
                }));
            }
        }
        let dataset_path = dir.join("dataset.jsonl");
        let responses_path = dir.join("responses.jsonl");
        let scorer_path = dir.join("score.py");
        write_jsonl(&dataset_path, &dataset);
        write_jsonl(&responses_path, &responses);
        std::fs::write(&scorer_path, SCORE_PY).unwrap();

        let out = std::process::Command::new(python())
            .args([
                scorer_path.as_os_str(),
                "--dataset".as_ref(),
                dataset_path.as_os_str(),
                "--responses".as_ref(),
                responses_path.as_os_str(),
            ])
            .env("PYTHONPATH", &dir)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "scorer failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let result: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(result["category_scores"]["non_live"], 25.0);
        assert_eq!(result["category_scores"]["live"], 75.0);
        assert_eq!(result["category_scores"]["hallucination"], 50.0);
        assert_eq!(result["normalized_single_turn_score"], 50.0);
        assert_eq!(result["overall_accuracy"], 66.67);
        assert_eq!(result["total_samples"], 12);
        assert_eq!(result["unmatched_responses"], 0);

        // ── Cross-language conformance ────────────────────────────────────
        // The four-way shard split recombines per-subset (hits, n) counts in
        // Rust and applies score.py's weighting there. That is a second
        // implementation of the aggregation, so the two must be pinned to each
        // other on real scorer output -- not on a hand-written fixture, which
        // would only prove the Rust agrees with my reading of the Python.
        let totals = result["subset_totals"]
            .as_object()
            .expect("score.py must emit subset_totals for the shard aggregator");
        let tallies: std::collections::BTreeMap<String, super::super::aggregate::Tally> = totals
            .iter()
            .map(|(k, v)| {
                let a = v.as_array().expect("[hits, n]");
                (
                    k.clone(),
                    super::super::aggregate::Tally {
                        hits: a[0].as_u64().expect("hits"),
                        n: a[1].as_u64().expect("n"),
                    },
                )
            })
            .collect();
        let rust = super::super::aggregate::aggregate(&tallies);
        assert_eq!(
            rust.overall_accuracy,
            result["overall_accuracy"].as_f64().unwrap(),
            "the Rust aggregator disagrees with score.py on overall_accuracy"
        );
        assert_eq!(
            rust.normalized_single_turn_score,
            result["normalized_single_turn_score"].as_f64().unwrap(),
            "the Rust aggregator disagrees with score.py on normalized"
        );
        assert_eq!(
            rust.total_samples,
            result["total_samples"].as_u64().unwrap(),
            "sample count disagrees"
        );
        for (cat, v) in result["category_scores"].as_object().unwrap() {
            assert_eq!(
                rust.category_scores.get(cat).copied(),
                v.as_f64(),
                "category {cat} disagrees between score.py and the Rust aggregator"
            );
        }
    }

    #[test]
    fn the_stamp_changes_when_any_shipped_asset_changes() {
        let dir = temp_dir("stamp");
        let original = stamp_value(REQUIREMENTS, PROVISION_PY, SCORE_PY);
        let a = Stamp::new(&dir, ".provisioned", &original);
        a.commit().unwrap();
        assert!(a.is_current());
        for changed in [
            stamp_value(&REQUIREMENTS.replacen('#', "!", 1), PROVISION_PY, SCORE_PY),
            stamp_value(REQUIREMENTS, &PROVISION_PY.replacen('#', "!", 1), SCORE_PY),
            stamp_value(REQUIREMENTS, PROVISION_PY, &SCORE_PY.replacen('#', "!", 1)),
        ] {
            assert_eq!(changed.len(), original.len());
            assert_ne!(changed, original);
            assert!(!Stamp::new(&dir, ".provisioned", changed).is_current());
        }
    }
}
