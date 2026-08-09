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
        format!(
            "v1 py>={}.{} req={} prov={} score={}",
            MIN_PYTHON.0,
            MIN_PYTHON.1,
            REQUIREMENTS.len(),
            PROVISION_PY.len(),
            SCORE_PY.len()
        ),
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

    #[test]
    fn the_committed_python_assets_are_non_empty_and_own_their_cli() {
        assert!(PROVISION_PY.contains("--out"));
        assert!(SCORE_PY.contains("--dataset") && SCORE_PY.contains("--responses"));
        assert!(
            REQUIREMENTS.contains("bfcl-eval=="),
            "the pin must be exact"
        );
    }

    #[test]
    fn the_scorer_reproduces_the_reference_aggregation_strategies() {
        // These three strings ARE the normalized score. A silent edit to any of
        // them would move every recorded baseline without any other signal.
        assert!(SCORE_PY.contains("\"live\": \"sample_weighted\""));
        assert!(SCORE_PY.contains("\"non_live\": \"hierarchical\""));
        assert!(SCORE_PY.contains("\"hallucination\": \"unweighted\""));
        assert!(SCORE_PY.contains("gpt-4o-2024-11-20-FC"));
    }

    #[test]
    fn the_stamp_changes_when_a_shipped_script_changes() {
        let dir = std::env::temp_dir().join(format!("atlas-bfcl-stamp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let a = Stamp::new(&dir, ".provisioned", "v1 req=10 prov=20");
        a.commit().unwrap();
        assert!(a.is_current());
        assert!(!Stamp::new(&dir, ".provisioned", "v1 req=10 prov=21").is_current());
    }
}
