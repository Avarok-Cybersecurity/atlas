// SPDX-License-Identifier: AGPL-3.0-only
//! A fetched record becomes a repository record only after it is checked.
//!
//! atlasctl already verified size and sha256 against what the node
//! promised; this side verifies what MATTERS to certification: the record is
//! for this unit, at the anchor, on the class being certified, completed and
//! clean, and its signature checks under a committed key. Anything else is
//! removed again and reported as a harness failure that no retry will fix —
//! a node that hands back the wrong record will do it twice.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use atlas_plugin::gate::{self, signing};

use super::atlasctl::FetchedFile;

/// The record and its sidecar, where the repository keeps them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    pub record: PathBuf,
    pub signature: PathBuf,
    /// The child's log, under the campaign's log dir.
    pub log: Option<PathBuf>,
}

/// What a placed record must satisfy.
pub struct Expect<'a> {
    pub unit_id: &'a str,
    pub anchor: &'a str,
    pub hardware: &'a str,
}

/// Pure: which fetched files are the record, its signature and the log.
///
/// # Errors
/// When the set has no record, no signature, or a file whose path does not
/// belong to this unit.
pub fn sort_files(
    files: &[FetchedFile],
    unit_id: &str,
) -> Result<(FetchedFile, FetchedFile, Option<FetchedFile>)> {
    let record_dir = format!(".benchmarks/{unit_id}/");
    let mut record = None;
    let mut sig = None;
    let mut log = None;
    for f in files {
        let rel = f.relative_path.as_str();
        if rel.starts_with(&record_dir) && rel.ends_with(".json") {
            if record.replace(f.clone()).is_some() {
                bail!("the node returned two records for {unit_id}");
            }
        } else if rel.starts_with(&record_dir) && rel.ends_with(".json.sig") {
            if sig.replace(f.clone()).is_some() {
                bail!("the node returned two signatures for {unit_id}");
            }
        } else if rel.starts_with(".certify/") && rel.ends_with(".log") {
            log = Some(f.clone());
        } else {
            bail!(
                "the node returned {rel:?}, which is not a record, signature or log of {unit_id}"
            );
        }
    }
    let Some(record) = record else {
        bail!("the node returned no record for {unit_id}");
    };
    let Some(sig) = sig else {
        bail!("the node returned no signature for {unit_id}");
    };
    if sig.relative_path != format!("{}.sig", record.relative_path) {
        bail!(
            "signature {} does not belong to record {}",
            sig.relative_path,
            record.relative_path
        );
    }
    Ok((record, sig, log))
}

fn copy_new(from: &Path, to: &Path) -> Result<()> {
    if let Some(p) = to.parent() {
        std::fs::create_dir_all(p).with_context(|| format!("creating {}", p.display()))?;
    }
    let bytes = std::fs::read(from).with_context(|| format!("reading {}", from.display()))?;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(to)
        .with_context(|| {
            format!(
                "{} already exists — a record is never overwritten",
                to.display()
            )
        })?;
    std::io::Write::write_all(&mut f, &bytes).with_context(|| format!("writing {}", to.display()))
}

/// Check the fetched record and move it, with its signature, into the
/// repository. On any failure nothing is left under `.benchmarks/`.
///
/// # Errors
/// Named, and never retryable by the caller.
pub fn place(
    root: &Path,
    log_dir: &Path,
    files: &[FetchedFile],
    expect: &Expect,
) -> Result<Placed> {
    let (rec, sig, log) = sort_files(files, expect.unit_id)?;
    let parsed = gate::read_record(&rec.path)
        .with_context(|| format!("parsing the fetched record {}", rec.path.display()))?;
    if parsed.benchmark_id != expect.unit_id {
        bail!(
            "the fetched record is for {}, not {}",
            parsed.benchmark_id,
            expect.unit_id
        );
    }
    if !(parsed.git_sha.starts_with(expect.anchor) || expect.anchor.starts_with(&parsed.git_sha)) {
        bail!(
            "the fetched record names commit {}, not the anchor {}",
            parsed.git_sha,
            expect.anchor
        );
    }
    let class = parsed.hardware.gate_key();
    if class != expect.hardware {
        bail!(
            "the fetched record was measured on class {class}, this campaign certifies {}",
            expect.hardware
        );
    }
    if parsed.frame_status_failed() {
        bail!("the fetched record says its own run failed (frame status Failed)");
    }
    if !parsed.dirty_paths.is_empty() {
        bail!(
            "the fetched record was measured on a dirty tree ({})",
            parsed.dirty_paths.join(", ")
        );
    }
    let record_to = root.join(&rec.relative_path);
    let sig_to = root.join(&sig.relative_path);
    copy_new(&rec.path, &record_to)?;
    if let Err(e) = copy_new(&sig.path, &sig_to) {
        let _ = std::fs::remove_file(&record_to);
        return Err(e);
    }
    if let Err(e) = signing::verify_record(root, &record_to, &parsed.git_sha, parsed.recorded_at) {
        let _ = std::fs::remove_file(&record_to);
        let _ = std::fs::remove_file(&sig_to);
        return Err(
            e.context("the fetched record's signature does not verify under a committed key")
        );
    }
    let log_to = match &log {
        Some(l) => {
            let name = Path::new(&l.relative_path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| format!("{}.log", expect.unit_id));
            let to = log_dir.join(format!("{}.remote.{name}", expect.unit_id));
            match std::fs::copy(&l.path, &to) {
                Ok(_) => Some(to),
                Err(_) => None,
            }
        }
        None => None,
    };
    Ok(Placed {
        record: record_to,
        signature: sig_to,
        log: log_to,
    })
}

#[cfg(test)]
#[path = "place_tests.rs"]
mod place_tests;
