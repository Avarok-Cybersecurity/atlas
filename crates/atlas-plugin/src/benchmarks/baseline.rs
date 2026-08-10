// SPDX-License-Identifier: AGPL-3.0-only

//! Per-benchmark baselines, stored beside the runs in `~/.atlas`.
//!
//! A regression gate needs something to regress against. Storing that here —
//! typed, written by the benchmark itself at the end of a clean run — keeps the
//! comparison self-contained: the pane never has to reverse-engineer a number
//! out of a rendered table.
//!
//! Baselines are **box-local and config-local by construction** (`~/.atlas` is
//! not shared, and the key records the endpoint the numbers came from), because
//! a TTFT baseline carried across boxes or serve configs manufactures wins.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::artifacts::ArtifactStore;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Baseline {
    /// Unix seconds. Shown so a stale baseline is visible rather than implied.
    pub recorded_at: u64,
    /// The endpoint + model the numbers were measured against. A baseline from
    /// a different target is reported, never silently compared.
    pub target: String,
    pub model: String,
    pub metrics: BTreeMap<String, f64>,
}

impl Baseline {
    pub fn get(&self, key: &str) -> Option<f64> {
        self.metrics.get(key).copied()
    }

    /// Human age, for the "vs baseline (4 h old)" line.
    pub fn age_text(&self) -> String {
        let now = now_secs();
        let secs = now.saturating_sub(self.recorded_at);
        match secs {
            0..=90 => "just now".into(),
            91..=5400 => format!("{} min old", secs / 60),
            5401..=172_800 => format!("{} h old", secs / 3600),
            _ => format!("{} d old", secs / 86_400),
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn path(store: &ArtifactStore, benchmark_id: &str) -> Result<std::path::PathBuf> {
    Ok(store.runs_dir(benchmark_id)?.join("baseline.json"))
}

/// Read the stored baseline, if any. A corrupt file is treated as absent —
/// running without a baseline is a degraded but correct mode, while refusing to
/// start because of an unreadable cache file is not.
pub fn load(store: &ArtifactStore, benchmark_id: &str) -> Option<Baseline> {
    let p = path(store, benchmark_id).ok()?;
    let text = std::fs::read_to_string(p).ok()?;
    serde_json::from_str(&text).ok()
}

/// Record a new baseline. Call only after a run that is trustworthy — a gate
/// that stores the numbers from a failed or partial leg poisons every later run.
pub fn save(
    store: &ArtifactStore,
    benchmark_id: &str,
    target: &str,
    model: &str,
    metrics: BTreeMap<String, f64>,
) -> Result<()> {
    let baseline = Baseline {
        recorded_at: now_secs(),
        target: target.to_string(),
        model: model.to_string(),
        metrics,
    };
    let p = path(store, benchmark_id)?;
    std::fs::write(&p, serde_json::to_string_pretty(&baseline)?)
        .with_context(|| format!("writing {}", p.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(name: &str) -> ArtifactStore {
        let d = std::env::temp_dir().join(format!("atlas-baseline-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        ArtifactStore::with_root(d)
    }

    #[test]
    fn round_trips_and_reports_the_target_it_came_from() {
        let s = store("rt");
        assert!(load(&s, "ttft-warm").is_none());
        let mut m = BTreeMap::new();
        m.insert("median_ms".into(), 812.5);
        save(&s, "ttft-warm", "http://127.0.0.1:8888", "qwen", m).unwrap();
        let b = load(&s, "ttft-warm").unwrap();
        assert_eq!(b.get("median_ms"), Some(812.5));
        assert_eq!(b.model, "qwen");
        assert_eq!(b.age_text(), "just now");
    }

    #[test]
    fn a_corrupt_baseline_reads_as_absent() {
        let s = store("corrupt");
        let p = s.runs_dir("x").unwrap().join("baseline.json");
        std::fs::write(p, "{ not json").unwrap();
        assert!(load(&s, "x").is_none());
    }
}
