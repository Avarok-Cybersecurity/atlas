// SPDX-License-Identifier: AGPL-3.0-only

//! Run history — the one place a benchmark result is written and read.
//!
//! The dashboard used to persist runs itself: it wrote the terminal
//! [`BenchmarkResult`] and nothing else, under `run-<unix_secs>.json`. That had
//! two defects worth naming, because both destroy information silently.
//!
//! * **No configuration.** A number with no parameters and no target is not a
//!   result — you cannot tell what was measured, against which endpoint, or
//!   reproduce it. So a record now carries every parameter (defaults included,
//!   not just overrides), the target, the source, and the Atlas version.
//! * **One-second filenames.** Two runs of the same benchmark in the same
//!   second overwrote each other. The second one just vanished. Records are now
//!   keyed by nanosecond with an explicit collision guard, so that is
//!   structurally impossible rather than merely unlikely.
//!
//! Both the CLI and the TUI write through [`save`] and read through [`load`] /
//! [`load_all`], so the two cannot drift into different formats or different
//! directories.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::artifacts::ArtifactStore;
use crate::benchmark::BenchmarkDescriptor;
use crate::params::{ParamSpec, ParamValues};
use crate::plugin::TargetEndpoint;
use crate::result::{BenchmarkResult, VerdictKind};

/// Current record schema. `0` means a pre-schema bare-frame file.
pub const SCHEMA: u32 = 1;

/// Where a run was started from.
///
/// Recorded because "the CLI number and the dashboard number disagree" is a
/// question someone eventually asks, and it is unanswerable after the fact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunSource {
    Tui,
    Cli,
    /// A legacy bare-frame file, which carried no provenance.
    #[default]
    Unknown,
}

/// One completed run, as stored.
///
/// Only `benchmark_id`, `recorded_at` and `frame` are required — every other
/// field defaults. That is deliberate twice over: it lets an older binary read
/// a newer file, and it makes this shape unambiguously distinguishable from a
/// legacy bare [`BenchmarkResult`] (which has no `frame` key at all).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunRecord {
    #[serde(default)]
    pub schema: u32,
    /// Unique per run, and equal to the file stem, so it addresses the record.
    #[serde(default)]
    pub run_id: String,
    pub benchmark_id: String,
    #[serde(default)]
    pub benchmark_name: String,
    /// Unix seconds at which the terminal frame was written.
    pub recorded_at: u64,
    /// Stored flat rather than as a nested `TargetEndpoint`: that type enforces
    /// a no-trailing-slash invariant in its constructor, and a serde-built one
    /// would bypass it. [`RunRecord::target`] rebuilds it properly.
    #[serde(default)]
    pub target_url: String,
    #[serde(default)]
    pub target_model: String,
    /// EVERY parameter, not just the overridden ones.
    #[serde(default)]
    pub params: BTreeMap<String, String>,
    #[serde(default)]
    pub source: RunSource,
    #[serde(default)]
    pub atlas_version: String,
    /// The terminal frame, byte-identical to what the run pane rendered.
    pub frame: BenchmarkResult,
}

impl RunRecord {
    /// Assemble a record from what a caller holds when the run ends.
    ///
    /// Taking the descriptor, values and target together rather than a builder
    /// means no call site can forget one — none of them is optional.
    pub fn new(
        descriptor: &BenchmarkDescriptor,
        values: &ParamValues,
        target: &TargetEndpoint,
        source: RunSource,
        atlas_version: &str,
        frame: BenchmarkResult,
    ) -> Self {
        Self {
            schema: SCHEMA,
            run_id: String::new(), // stamped by `save`, from the name it picks
            benchmark_id: descriptor.id.to_string(),
            benchmark_name: descriptor.name.to_string(),
            recorded_at: now_secs(),
            target_url: target.base_url.clone(),
            target_model: target.model.clone(),
            params: values.to_strings(),
            source,
            atlas_version: atlas_version.to_string(),
            frame,
        }
    }

    /// The endpoint, rebuilt through its constructor so its invariants hold.
    pub fn target(&self) -> TargetEndpoint {
        TargetEndpoint::new(&self.target_url, &self.target_model)
    }

    /// Rehydrate the stored parameters against a live schema.
    ///
    /// Routed through each spec's `ParamKind`, so a value from an older Atlas
    /// whose bounds have since tightened is reported rather than accepted.
    pub fn values(&self, specs: &[ParamSpec]) -> Result<ParamValues> {
        let pairs = self
            .params
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect::<Vec<_>>();
        ParamValues::from_overrides(specs, pairs)
    }

    pub fn verdict_kind(&self) -> Option<VerdictKind> {
        self.frame.verdict.as_ref().map(|v| v.kind)
    }

    /// True for a pre-schema file: no params, no target, no provenance.
    pub fn is_legacy(&self) -> bool {
        self.schema == 0
    }

    /// Compact age, e.g. `3m ago`. Rendered by the History pane.
    pub fn age_text(&self) -> String {
        let secs = now_secs().saturating_sub(self.recorded_at);
        match secs {
            0..=59 => format!("{secs}s ago"),
            60..=3599 => format!("{}m ago", secs / 60),
            3600..=86_399 => format!("{}h ago", secs / 3600),
            _ => format!("{}d ago", secs / 86_400),
        }
    }

    fn from_legacy(benchmark_id: &str, run_id: &str, frame: BenchmarkResult) -> Self {
        Self {
            schema: 0,
            recorded_at: run_id
                .trim_start_matches("run-")
                .parse()
                .unwrap_or_default(),
            run_id: run_id.to_string(),
            benchmark_id: benchmark_id.to_string(),
            benchmark_name: String::new(),
            target_url: String::new(),
            target_model: String::new(),
            params: BTreeMap::new(),
            source: RunSource::Unknown,
            atlas_version: String::new(),
            frame,
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// Write `record`, stamping its `run_id` from the filename chosen. Returns the path.
///
/// The name is `run-<unix_nanos>` zero-padded to 19 digits: fixed width keeps
/// lexical order chronological until the year 2262, so a plain filename sort is
/// a time sort. If that name is somehow taken the nanos are bumped until it is
/// free, which is what makes "two runs never collide" a property rather than a
/// probability.
pub fn save(store: &ArtifactStore, record: &mut RunRecord) -> Result<PathBuf> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or_default();
    save_at(store, record, nanos, |_| {})
}

/// The deterministic seam for collision and publication checks. Production
/// supplies the current clock and no hook; tests can force the same first
/// candidate and inspect the claimed placeholder before it is published.
fn save_at<F>(
    store: &ArtifactStore,
    record: &mut RunRecord,
    mut nanos: u64,
    after_claim: F,
) -> Result<PathBuf>
where
    F: FnOnce(&Path),
{
    let dir = store.runs_dir(&record.benchmark_id)?;
    // Claim the id with `create_new`, not `exists()`. This directory is shared:
    // the dashboard and `spark benchmark` are separate processes writing the
    // same tree, and check-then-write lets both conclude the same name is free
    // and one silently overwrite the other's run. `create_new` is atomic in the
    // kernel, so exactly one claimant can win.
    let (run_id, path) = loop {
        let id = format!("run-{nanos:019}");
        let path = dir.join(format!("{id}.json"));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(_) => break (id, path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                nanos = nanos.saturating_add(1);
            }
            Err(e) => {
                return Err(e).with_context(|| format!("claiming {}", path.display()));
            }
        }
    };
    record.run_id = run_id;
    after_claim(&path);
    let json = serde_json::to_string_pretty(&record).context("serializing the run record")?;
    // Write beside it and rename over the claim. `rename` is atomic, so the
    // History pane reads either the empty placeholder or the finished record —
    // never half of one. Writing in place would let a poll land mid-write, and
    // the reader caches, so a torn read is not merely retried.
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("publishing {}", path.display()))?;
    Ok(path)
}

/// Every run for one benchmark, newest first.
///
/// Tolerant by design: an unreadable or corrupt file is skipped, never fatal.
/// History is a convenience, and one bad file must not hide the rest.
pub fn load(store: &ArtifactStore, benchmark_id: &str) -> Vec<RunRecord> {
    let Ok(dir) = store.runs_dir(benchmark_id) else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<RunRecord> = entries
        .flatten()
        .filter_map(|e| read_one(benchmark_id, &e.path()))
        .collect();
    sort_newest_first(&mut out);
    out
}

/// Every run for every registered benchmark, newest first across all of them.
pub fn load_all(store: &ArtifactStore) -> Vec<RunRecord> {
    let mut out: Vec<RunRecord> = crate::registry::all()
        .iter()
        .flat_map(|d| load(store, d.id))
        .collect();
    sort_newest_first(&mut out);
    out
}

/// One run by its `run_id`, across every benchmark.
pub fn find(store: &ArtifactStore, run_id: &str) -> Option<RunRecord> {
    load_all(store).into_iter().find(|r| r.run_id == run_id)
}

fn sort_newest_first(records: &mut [RunRecord]) {
    records.sort_by(|a, b| {
        b.recorded_at
            .cmp(&a.recorded_at)
            .then_with(|| b.run_id.cmp(&a.run_id))
    });
}

/// Parse one file, accepting both the record shape and the legacy bare frame.
///
/// The two cannot be confused: a bare frame has no `frame` key, so the record
/// parse fails on a missing required field; a record has no top-level `status`,
/// so the frame parse fails the same way.
fn read_one(benchmark_id: &str, path: &Path) -> Option<RunRecord> {
    let stem = path.file_stem()?.to_str()?;
    if !stem.starts_with("run-") || path.extension()? != "json" {
        return None; // baseline.json and the agentic sandbox live here too
    }
    let text = std::fs::read_to_string(path).ok()?;
    if let Ok(mut record) = serde_json::from_str::<RunRecord>(&text) {
        if record.run_id.is_empty() {
            record.run_id = stem.to_string();
        }
        return Some(record);
    }
    let frame = serde_json::from_str::<BenchmarkResult>(&text).ok()?;
    Some(RunRecord::from_legacy(benchmark_id, stem, frame))
}

#[cfg(test)]
#[path = "history_tests.rs"]
mod tests;
