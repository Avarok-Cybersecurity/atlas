// SPDX-License-Identifier: AGPL-3.0-only

//! `kernels/<hw>/<model>/BENCH.toml` — a model's benchmarks, beside the model.
//!
//! # Why the numbers moved
//!
//! They used to live in `.benchmarks/<gate>/BASELINE.json`, indexed
//! gate → hardware → model. That put a model's thresholds five directories away
//! from the model, and split one model's numbers across five files. Asking
//! "what is gated on the 27B, and at what?" meant opening all of them.
//!
//! `BENCH.toml` transposes the index: one file per model, `[[benchmarks]]`
//! entries naming the gate. `MODEL.toml` is its sibling, so hardware and model
//! are implied by the path and cannot disagree with their own contents.
//!
//! # It is deliberately outside the closure hash
//!
//! [`super::taxon::configs`] does not list `BENCH.toml`, so editing a threshold
//! does not change any target's closure hash. A ratchet therefore does not
//! invalidate the very record that justified it — the records are the verdict,
//! not its subject. `BENCH.toml` is still under `kernels/`, so it maps to a
//! target and gets excused through the normal rung-0 path.
//!
//! # A guess can never go green
//!
//! `status = "unmeasured"` entries carry no `[benchmarks.metrics]` table at all.
//! Absence is the TODO: a gate with no metrics reports *ungated*, never PASS.
//! Writing a plausible-looking guessed threshold would be worse than writing
//! nothing, because a run clearing it would look verified.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::record::{GateBaseline, HardwareBaseline, ModelBaseline};
use super::taxon;

/// One `[[benchmarks]]` entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchEntry {
    /// Quantisation this applies to — the first key, matching the directory
    /// under the model that holds its kernels.
    pub quant: String,
    /// The served checkpoint. Two checkpoints of one model can differ by
    /// several BFCL points, so thresholds are per checkpoint, never per model.
    pub checkpoint: String,
    /// Benchmark id, e.g. `bfcl-subset`.
    pub gate: String,
    /// `<family>/<stem>` in `atlas-recipes`, when one serves this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
    /// Whether this is the checkpoint the gate runs when none is named.
    #[serde(default)]
    pub default: bool,
    /// `measured` or `unmeasured`.
    pub status: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
    /// Thresholds. Absent for `unmeasured` — see the module docs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<BTreeMap<String, super::record::Bound>>,
}

#[derive(Debug, Default, Deserialize)]
struct BenchFile {
    #[serde(default)]
    benchmarks: Vec<BenchEntry>,
}

/// Every benchmark entry in the tree, with the target each belongs to.
pub fn load_all(root: &Path) -> Result<Vec<(taxon::Target, BenchEntry)>> {
    let mut out = Vec::new();
    let mut seen_models = std::collections::BTreeSet::new();
    for target in taxon::walk(root) {
        // One BENCH.toml per MODEL, but `walk` yields one entry per quant, so
        // the file would otherwise be parsed and its entries duplicated once
        // per quant directory.
        if !seen_models.insert((target.hardware.clone(), target.model.clone())) {
            continue;
        }
        let path = root
            .join("kernels")
            .join(&target.hardware)
            .join(&target.model)
            .join("BENCH.toml");
        if !path.exists() {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let parsed: BenchFile =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        for entry in parsed.benchmarks {
            if entry.status != "measured" && entry.status != "unmeasured" {
                bail!(
                    "{}: status must be \"measured\" or \"unmeasured\", got {:?}",
                    path.display(),
                    entry.status
                );
            }
            if entry.status == "unmeasured" && entry.metrics.is_some() {
                bail!(
                    "{}: {} / {} is unmeasured but carries thresholds. A guessed \
                     number a run can clear is worse than no number — it reports \
                     PASS for something nobody measured.",
                    path.display(),
                    entry.gate,
                    entry.checkpoint
                );
            }
            if entry.status == "measured" && entry.metrics.is_none() {
                bail!(
                    "{}: {} / {} claims to be measured but declares no metrics",
                    path.display(),
                    entry.gate,
                    entry.checkpoint
                );
            }
            out.push((
                taxon::Target {
                    hardware: target.hardware.clone(),
                    model: target.model.clone(),
                    quant: entry.quant.clone(),
                },
                entry,
            ));
        }
    }
    Ok(out)
}

/// Assemble the baseline for one gate from every `BENCH.toml` in the tree.
///
/// The shape returned is exactly what `BASELINE.json` used to hold, so nothing
/// downstream of `GateBaseline` had to change when the storage moved.
///
/// Entries with `status = "unmeasured"` are DROPPED rather than included with
/// empty thresholds: `resolve()` failing loudly with "no baseline for model X"
/// is the honest answer, where an entry with no bounds would pass every check
/// it was given.
pub fn baseline_for(root: &Path, benchmark_id: &str) -> Result<GateBaseline> {
    let mut hardware: BTreeMap<String, HardwareBaseline> = BTreeMap::new();
    let mut defaults: BTreeMap<String, (String, String)> = BTreeMap::new();

    for (target, entry) in load_all(root)? {
        if entry.gate != benchmark_id || entry.status != "measured" {
            continue;
        }
        let Some(metrics) = entry.metrics.clone() else {
            continue;
        };
        if entry.default {
            // Two defaults is a silent coin-flip over which checkpoint a gate
            // scores, and the two can differ by several points.
            if let Some((prior, prior_model)) = defaults.get(&target.hardware) {
                bail!(
                    "{benchmark_id}: both {prior} (in {prior_model}) and {} (in {}) \
                     claim to be the default on {}",
                    entry.checkpoint,
                    target.model,
                    target.hardware
                );
            }
            defaults.insert(
                target.hardware.clone(),
                (entry.checkpoint.clone(), target.model.clone()),
            );
        }
        let hw = hardware
            .entry(target.hardware.clone())
            .or_insert_with(|| HardwareBaseline {
                default: String::new(),
                models: BTreeMap::new(),
            });
        if hw
            .models
            .insert(
                entry.checkpoint.clone(),
                ModelBaseline {
                    recipe: entry.recipe.clone(),
                    note: entry.note.clone(),
                    metrics,
                },
            )
            .is_some()
        {
            bail!(
                "{benchmark_id}: {} is declared twice on {}",
                entry.checkpoint,
                target.hardware
            );
        }
    }

    for (hw_name, hw) in &mut hardware {
        match defaults.get(hw_name) {
            Some((checkpoint, _)) => hw.default = checkpoint.clone(),
            // No implicit "the only one wins": a second checkpoint added later
            // would silently move which one the gate scores.
            None => bail!(
                "{benchmark_id}: no checkpoint on {hw_name} sets `default = true`; \
                 one must, or the gate has no defined subject"
            ),
        }
    }
    Ok(GateBaseline {
        schema: 2,
        hardware,
    })
}

#[cfg(test)]
#[path = "bench_tests.rs"]
mod bench_tests;
