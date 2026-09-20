// SPDX-License-Identifier: AGPL-3.0-only

//! Writing one gate record without erasing a failure.
//!
//! Moved out of `record.rs` (exact piecewise copy of `write_record`, which
//! sits on the size-cap allow list) and extended with the one rule this file
//! exists for: **a failing record is never overwritten**.
//!
//! Records are keyed `<date>-<sha>.json`, so a same-day re-run at the same
//! commit used to REPLACE the failing record with the passing one. Verified
//! on 2026-08-29: `agentic-webserver` scored 9/10 and then 10/10 twice, and the
//! committed record is a clean 10/10 with no trace of the 9/10 — which made
//! every failure rate quoted from `.benchmarks/` a count of failures that
//! happened to survive, not of failures (#1159).
//!
//! A PASS arriving later is new information, not a correction. So when the
//! canonical name holds a FAIL the newcomer is written beside it under the
//! re-run name ([`super::record_path::rerun_path`]) and the failure stays on
//! disk as history. Gating is unaffected: `records_newest_first` orders by
//! `recorded_at`, so the newest record still decides. Deleting a failing
//! record is a `git rm` a human types, which is exactly the visibility the
//! rule wants.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use super::gate_dir;
use super::record::{GateRecord, read_record, record_path_for};
use super::record_path::rerun_path;

/// Write one gate record. Returns the path; the parent directory is created,
/// but never committed on the writer's behalf — that stays the caller's
/// explicit act.
///
/// The path is the canonical `record_path_for` name unless that name already
/// holds a failing record, in which case it is the re-run name beside it —
/// see the module doc and the private `preserving_path` helper.
pub fn write_record(root: &Path, record: &GateRecord) -> Result<PathBuf> {
    let path = preserving_path(&record_path_for(root, record), record)?;
    std::fs::create_dir_all(path.parent().expect("record path has a parent")).with_context(
        || {
            format!(
                "creating {}",
                gate_dir(root, &record.benchmark_id).display()
            )
        },
    )?;
    let json = serde_json::to_string_pretty(record).context("serializing the gate record")?;
    std::fs::write(&path, json + "\n").with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// The path `record` may be written to without erasing a failure.
///
/// `canonical` when it is free or holds a record that is not a FAIL — a
/// passing record replaced by a same-day re-run is the behaviour every
/// consumer was built on, and it hides no failure. The re-run name when the
/// canonical holds a FAIL. An error when the re-run name holds a FAIL too:
/// two failures in the same second at one commit is a re-write of one run,
/// and the rule refuses rather than guesses.
fn preserving_path(canonical: &Path, record: &GateRecord) -> Result<PathBuf> {
    if !holds_failure(canonical)? {
        return Ok(canonical.to_path_buf());
    }
    let rerun = rerun_path(canonical, record.recorded_at);
    if holds_failure(&rerun)? {
        bail!(
            "{} and {} both hold FAILING records for this commit; neither is overwritten. \
             A failing record is evidence — remove one with `git rm` if it must go.",
            canonical.display(),
            rerun.display()
        );
    }
    Ok(rerun)
}

/// Whether `path` holds a record whose verdict is FAIL.
///
/// Fail-closed on a file that exists but cannot be read as a record: what
/// cannot be classified is not overwritten either, because the one thing this
/// module promises is that nothing failing disappears silently.
fn holds_failure(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let existing = read_record(path).with_context(|| {
        format!(
            "{} exists but is not a readable gate record, so it is not overwritten",
            path.display()
        )
    })?;
    Ok(existing.verdict.as_deref() == Some("FAIL"))
}

#[cfg(test)]
#[path = "record_write_tests.rs"]
mod record_write_tests;
