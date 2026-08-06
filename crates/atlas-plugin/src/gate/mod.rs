// SPDX-License-Identifier: AGPL-3.0-only

//! The pull-request gate — benchmark records committed with the code they
//! measured.
//!
//! A run stored only in `~/.atlas` answers for one box and one person. A gate
//! record lives in the repo's `.benchmarks/<id>/` directory, one file per run
//! (`YYYY-MM-DD-<sha>.json`), so the question "did this branch pass its
//! benchmarks?" can be answered from the branch itself — by a human reading
//! the diff and by CI's `--pull-request-gate-check`.
//!
//! Two files per benchmark matter:
//!
//! * a run record — this run's metrics, verdict, hardware and command, derived
//!   from the [`crate::RunRecord`] that history already writes, plus the git
//!   sha and a one-line summary ([`record`]);
//! * `BASELINE.json` — the thresholds a pass must meet, with the same
//!   comparison the run-time verdict uses: minimum for scores, maximum for
//!   latencies and wall time, plus optional per-metric noise allowances. The
//!   committed records are checked against this file alone, so the check
//!   carries no per-box state ([`check`]).

pub mod check;
pub mod record;

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};

pub use check::{
    Comparison, GateStatus, check_gates, check_record, compare, record_covers, records_newest_first,
};
pub use record::{
    Bound, GateBaseline, GateRecord, HardwareBaseline, ModelBaseline, date_of, now_secs,
    read_baseline, read_record, record_path, write_record,
};

/// The five benches whose records must pass for the branch to be gated.
///
/// Gate A (agentic webserver), Gate C (warm + cold TTFT), and the two BFCL
/// gates. Every id is a registered benchmark, and registration is tested
/// against this list.
///
/// ★ **There are two BFCL entries because there are two draws, and a score from
/// one is not comparable to a threshold from the other.** The dense 27B is
/// gated on the golden n=995 MLPerf draw (`bfcl-subset`, ratchet 87.44/88.59);
/// the 35B MoE is gated on the echolp n=1004 draw (`bfcl-subset-echolp`,
/// ratchet 84.66/83.32) because that is the only draw its recorded history is
/// on. The two draws land `overall_accuracy` in the same place while
/// `normalized_single_turn_score` differs by ~1.8 points purely from category
/// mix — which is exactly what makes crossing them so easy to miss. Each
/// bench's `BASELINE.json` pins its own model, and a model mismatch is a hard
/// fail in `check_record`.
pub const REQUIRED_GATES: [&str; 5] = [
    "agentic-webserver",
    "ttft-warm-gate",
    "ttft-cold-gate",
    "bfcl-subset",
    "bfcl-subset-echolp",
];

/// The wall-clock timeout a gate run gives the endpoint's `/hardware` fetch.
pub const HARDWARE_TIMEOUT: Duration = Duration::from_secs(10);

/// `.benchmarks/<benchmark_id>` under `root`.
pub fn gate_dir(root: &Path, benchmark_id: &str) -> PathBuf {
    root.join(".benchmarks").join(benchmark_id)
}

/// `.benchmarks/<benchmark_id>/BASELINE.json` under `root`.
pub fn baseline_path(root: &Path, benchmark_id: &str) -> PathBuf {
    gate_dir(root, benchmark_id).join("BASELINE.json")
}

/// The short commit id for this working tree. `ATLAS_GATE_SHA` overrides —
/// the escape hatch for a checkout without git metadata.
pub fn git_sha(root: &Path) -> Result<String> {
    if let Some(explicit) = std::env::var_os("ATLAS_GATE_SHA") {
        let sha = explicit.to_string_lossy().trim().to_string();
        if sha.is_empty() {
            bail!("ATLAS_GATE_SHA is set but empty");
        }
        return Ok(sha);
    }
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--short=10", "HEAD"])
        .stdin(std::process::Stdio::null())
        .output()
        .context("running git rev-parse")?;
    if !out.status.success() {
        bail!(
            "git rev-parse failed — {} is not a git checkout (or git is \
             missing); set ATLAS_GATE_SHA to record a gate run",
            root.display()
        );
    }
    let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if sha.is_empty() {
        bail!("git rev-parse printed nothing");
    }
    Ok(sha)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
