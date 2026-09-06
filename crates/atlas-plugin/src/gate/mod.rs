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

/// The one-time, content-pinned 2026-08-16 invalidation amnesty.
pub mod amnesty;
pub mod bench;
pub mod card;
pub mod check;
mod check_fmt;
pub mod closure;
pub mod codeowners;
pub mod coverage;
pub mod record;
mod record_path;
pub mod scoring;
pub mod signing;
pub mod taxon;

/// The PR INTENT taxonomy — what a change is FOR, and the benchmarks that
/// implies. Distinct from [`taxon`], which is derived from paths.
pub mod pr_taxonomy;
pub mod required;
pub mod telemetry;

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};

pub use check::{
    Comparison, GateStatus, check_gates, check_record, compare, record_covers, records_newest_first,
};
pub use record::{
    Bound, GateBaseline, GateRecord, HardwareBaseline, ModelBaseline, date_of,
    merge_serve_overrides, now_secs, read_baseline, read_record, record_path, record_path_for,
    variant_slug, write_record,
};

/// The eleven benches whose records must pass for the branch to be gated.
///
/// Agentic webserver, vision and video fidelity, warm and cold TTFT, two BFCL
/// draws, SSM state poisoning, decode floor, and the concurrency sweep in both
/// its plain and DFlash2 forms. Every id is a registered benchmark, and
/// registration is tested against this list.
///
/// ★ **There are two concurrency entries because one of them speculates.**
/// `concurrency-sweep` serves a no-drafter recipe and
/// `concurrency-sweep-dflash2` serves the same ladder with the DFlash2 drafter
/// armed. Their aggregates are not comparable — speculation moves low-C
/// throughput by more than any regression either bar is set to catch — so each
/// carries its own BENCH.toml bounds, exactly like the two BFCL draws below.
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
pub const REQUIRED_GATES: [&str; 11] = [
    coverage::REQUIRED[0].id,
    coverage::REQUIRED[1].id,
    coverage::REQUIRED[2].id,
    coverage::REQUIRED[3].id,
    coverage::REQUIRED[4].id,
    coverage::REQUIRED[5].id,
    coverage::REQUIRED[6].id,
    coverage::REQUIRED[7].id,
    coverage::REQUIRED[8].id,
    coverage::REQUIRED[9].id,
    coverage::REQUIRED[10].id,
];

/// The wall-clock timeout a gate run gives the endpoint's `/hardware` fetch.
pub const HARDWARE_TIMEOUT: Duration = Duration::from_secs(10);

/// The tracked paths that determine what a gate run measures. A diff touching
/// any of them between a record's commit and `head` invalidates that record —
/// see [`check::record_covers`].
///
/// ★ **Over-broad costs a re-run; under-broad is a lie.** A path missing from
/// this list does not fail loudly — it makes a stale record keep speaking for a
/// commit it never measured, which is the one outcome a gate must never
/// produce. So the bar for adding a path is "could changing it move a number?",
/// not "is it code?".
///
/// * `crates`, `kernels`, `Cargo.toml`, `Cargo.lock`, `vendor` — the binary.
///   `crates` also carries the BFCL dataset provisioner and AST scorer
///   (`crates/atlas-plugin/assets/bfcl/*.py`), which define the score itself.
/// * `jinja-templates` — **not build input, runtime input.** The server loads
///   `jinja-templates/<model_type>.jinja` from the repo root at startup and it
///   OVERRIDES the checkpoint's own chat template, so editing one changes the
///   exact bytes every prompt is rendered to. A tool-schema change in a
///   template has already been measured moving BFCL by +2.70 points; without
///   this entry that edit would have inherited the previous run's record.
/// * `rust-toolchain.toml` — pins the compiler. A bump rebuilds every kernel
///   launch path from the same sources into a different binary.
///
/// Deliberately NOT here: `.benchmarks` (the records and thresholds are the
/// verdict, not the subject), `bench/` and `scripts/` (developer tooling that
/// no gate drives), and docs.
pub use coverage::PERF_PATHS;

/// `.benchmarks/<benchmark_id>` under `root`.
pub fn gate_dir(root: &Path, benchmark_id: &str) -> PathBuf {
    root.join(".benchmarks").join(benchmark_id)
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

/// The uncommitted [`PERF_PATHS`] files in this working tree, sorted.
///
/// ★ **`sha_at_start` stops HEAD moving out from under a run; this stops the
/// binary having never been HEAD in the first place.** A gate record names a
/// commit, and a reader takes that to mean "this commit's sources built the
/// binary that produced these numbers". Nothing enforced it: build with an
/// uncommitted fix in `crates/`, run the gate, and the record confidently
/// stamps the commit that does NOT contain the change it measured. That has
/// happened for real — a passing agentic record named `b75394fb` while the
/// binary carried an uncommitted truncation fix.
///
/// The intersection with [`PERF_PATHS`] is the whole point, and it is the same
/// invalidation set [`check::record_covers`] uses between two commits: a dirty
/// tree is just that diff taken against the index instead. During a campaign
/// the tree is *routinely* dirty with the previous gate's own record file, so
/// a guard that fired on any modification would be noise nobody reads —
/// `.benchmarks` is deliberately not a perf path, and this stays silent for it.
/// A dirty `crates/` file means the record is lying.
///
/// Untracked-but-not-ignored files count. A new `kernels/*.cu` picked up by a
/// glob is exactly as invisible to the sha as an edited one, and over-broad
/// costs a re-run while under-broad is a lie. `--untracked-files=all` rather
/// than the default `normal`, which collapses a wholly-untracked directory to
/// `kernels/` — the record has to name the file a reader would go open, not
/// the directory it is somewhere under.
///
/// Errs when git cannot answer (no metadata, no git) rather than reporting a
/// clean tree — "could not tell" must never render as "nothing to disclose".
pub fn dirty_perf_paths(root: &Path) -> Result<Vec<String>> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["status", "--porcelain", "--untracked-files=all", "--"])
        .args(PERF_PATHS)
        .stdin(std::process::Stdio::null())
        .output()
        .context("running git status")?;
    if !out.status.success() {
        bail!(
            "git status failed in {} — cannot tell whether the measured binary \
             matches the commit being stamped",
            root.display()
        );
    }
    let mut paths: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| line.get(3..))
        // A rename reads `R  old -> new`; the destination is the file that is
        // in the tree now, and the one a reader would go look at.
        .map(|entry| match entry.split_once(" -> ") {
            Some((_, dest)) => dest.trim().to_string(),
            None => entry.trim().to_string(),
        })
        .filter(|p| !p.is_empty())
        .collect();
    paths.sort();
    paths.dedup();
    Ok(paths)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

/// Construction and replay contracts split from `tests.rs` for its LoC cap.
#[cfg(test)]
#[path = "record_contract_tests.rs"]
mod record_contract_tests;

#[cfg(test)]
#[path = "variant_tests.rs"]
mod variant_tests;

/// Split from `tests.rs` for the 500-LoC cap: writing a fixture baseline into
/// the kernel tree, where the thresholds now live.
#[cfg(test)]
#[path = "fixture_baseline.rs"]
mod fixture_baseline;

/// Split from `coverage_tests.rs` for the 500-LoC cap: the deterministic floor.
#[cfg(test)]
#[path = "coverage_map_tests.rs"]
mod coverage_map_tests;

/// Proofs for the exact, `#[cfg(test)]`-guarded Rust module exemption.
#[cfg(test)]
#[path = "test_only_coverage_tests.rs"]
mod test_only_coverage_tests;

#[cfg(test)]
#[path = "coverage_promotion_tests.rs"]
mod coverage_promotion_tests;
#[cfg(test)]
#[path = "coverage_tests.rs"]
mod coverage_tests;

/// Squash-merge coverage. Split from `coverage_tests.rs` for the 500-LoC cap.
#[cfg(test)]
#[path = "coverage_squash_tests.rs"]
mod coverage_squash_tests;

/// Three holes an adversarial review found. Split for the 500-LoC cap.
#[cfg(test)]
#[path = "hardening_tests.rs"]
mod hardening_tests;

/// Result cards: the specs must name metrics that exist.
#[cfg(test)]
#[path = "card_tests.rs"]
mod card_tests;

/// Ed25519 record signatures: round trips and, mostly, negative controls.
pub mod agreement;

#[cfg(test)]
#[path = "agreement_tests.rs"]
mod agreement_tests;

#[cfg(test)]
#[path = "signer_registry_tests.rs"]
mod signer_registry_tests;

#[cfg(test)]
#[path = "signing_tests.rs"]
mod signing_tests;

/// The one-time 2026-08-16 amnesty: pinned grant, fail-closed, expiry.
#[cfg(test)]
#[path = "amnesty_tests.rs"]
mod amnesty_tests;

#[cfg(test)]
#[path = "dirty_tests.rs"]
mod dirty_tests;

/// Split from `tests.rs` for the 500-LoC cap: `--serve-override` provenance.
#[cfg(test)]
#[path = "override_tests.rs"]
mod override_tests;
