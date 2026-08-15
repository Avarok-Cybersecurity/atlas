// SPDX-License-Identifier: AGPL-3.0-only

//! The model-variant step of the Benchmarks section.
//!
//! A benchmark with `BENCH.toml` entries for more than one checkpoint is not
//! one measurement — each variant carries its own serve recipe and its own
//! thresholds, and numbers never compare ACROSS variants. So the Suite flow
//! mirrors the Library's Models → recipes shape: pick the benchmark, then pick
//! WHICH model it runs on, then the parameters. A benchmark with a single
//! variant still passes through the step — the card is where the variant's
//! measured rationale (its `note`) is readable, exactly as a model with one
//! recipe still shows its card.
//!
//! The rows are DERIVED from the same assembled baseline the gate uses
//! (`gate::read_baseline` over every `kernels/<hw>/<model>/BENCH.toml`), never
//! from a second list: a variant the TUI offered but the gate refused, or vice
//! versa, would be two disagreeing catalogues of one fact. Outside a checkout
//! (an installed binary far from the repo) there is no baseline to read; the
//! flow then skips straight to the parameters, exactly as before this step
//! existed.

use atlas_plugin::gate;

use super::bench_state::{BenchState, View};

/// One selectable variant: a (hardware, checkpoint) baseline entry.
#[derive(Clone, Debug)]
pub struct VariantRow {
    pub hardware: String,
    pub checkpoint: String,
    /// The entry's `label`, or the checkpoint id when it carries none.
    pub title: String,
    pub recipe: Option<String>,
    /// Whether this is the checkpoint the gate runs when none is named.
    pub is_default: bool,
    /// The entry's `note` — why these are the thresholds.
    pub note: String,
    /// The committed bounds, for the detail pane.
    pub metrics: Vec<(String, gate::Bound)>,
}

/// The variants a benchmark is defined on, defaults first within each box
/// class. Empty when there is no checkout or no baseline — that is the
/// "measures whatever it is pointed at" case, not an error.
pub fn variants_for(benchmark_id: &str) -> Vec<VariantRow> {
    let Ok(root) = crate::cli::bench_run::repo_root() else {
        return Vec::new();
    };
    let baseline = match gate::read_baseline(&root, benchmark_id) {
        Ok(b) => b,
        Err(e) => {
            tracing::debug!("no variant baseline for {benchmark_id}: {e:#}");
            return Vec::new();
        }
    };
    let mut rows = Vec::new();
    for (hardware, hw) in &baseline.hardware {
        let mut here: Vec<VariantRow> = hw
            .models
            .iter()
            .map(|(checkpoint, entry)| VariantRow {
                hardware: hardware.clone(),
                checkpoint: checkpoint.clone(),
                title: if entry.label.is_empty() {
                    checkpoint.clone()
                } else {
                    entry.label.clone()
                },
                recipe: entry.recipe.clone(),
                is_default: *checkpoint == hw.default,
                note: entry.note.clone(),
                metrics: entry
                    .metrics
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            })
            .collect();
        // The gate's declared subject leads; the rest keep the map's stable
        // (sorted) order.
        here.sort_by_key(|r| !r.is_default);
        rows.extend(here);
    }
    rows
}

impl BenchState {
    /// Enter the selected benchmark from the Suite list: through the variant
    /// step when it has declared variants, straight to the form otherwise.
    pub fn enter_selected(&mut self) {
        let Some(descriptor) = self.descriptor() else {
            return;
        };
        self.variants = variants_for(descriptor.id);
        self.variant_row = self.variant_row.min(self.variants.len().saturating_sub(1));
        self.view = if self.variants.is_empty() {
            View::Params
        } else {
            View::Variants
        };
    }

    /// Adopt the selected variant and open the form.
    ///
    /// Two things change, and they travel together or not at all: the target
    /// model (pinned, so `follow_live_model` cannot swap it back), and every
    /// parameter the descriptor couples to a committed threshold — the agentic
    /// Σ-wall budget is the selected variant's own ceiling, not the schema
    /// default that is only right for one of them.
    pub fn choose_variant(&mut self, index: usize) {
        let Some(row) = self.variants.get(index).cloned() else {
            return;
        };
        self.variant_row = index;
        self.target =
            atlas_plugin::TargetEndpoint::new(self.target.base_url.clone(), &row.checkpoint);
        self.target_model_pinned = true;
        self.variant_pinned = true;
        if let Some(descriptor) = self.descriptor() {
            for (param, metric) in descriptor.threshold_params {
                let Some(max) = row
                    .metrics
                    .iter()
                    .find(|(k, _)| k == metric)
                    .and_then(|(_, b)| b.max)
                else {
                    continue;
                };
                let Some(pos) = self.specs.iter().position(|s| s.key == *param) else {
                    continue;
                };
                // Through the spec's own parser, exactly like a typed value —
                // a derived number must not bypass the kind's bounds.
                match self.specs[pos].kind.parse(&format!("{max}")) {
                    Ok(value) => {
                        self.values.set(param.to_string(), value);
                        if let Some(buf) = self.edit.get_mut(pos) {
                            *buf = format!("{max}");
                        }
                        self.errors.remove(*param);
                    }
                    Err(e) => {
                        tracing::warn!("variant {} bound for {param} rejected: {e:#}", row.title);
                    }
                }
            }
        }
        // The model row of the form shows the adopted checkpoint.
        if let Some(buf) = self.edit.get_mut(self.specs.len() + 1) {
            *buf = row.checkpoint.clone();
        }
        self.view = View::Params;
    }

    /// Keys for the variant list: the same j/k/Enter/Esc grammar as the Suite
    /// list it sits under.
    pub(super) fn variants_key(&mut self, key: crossterm::event::KeyEvent) {
        use crossterm::event::KeyCode;
        let n = self.variants.len();
        match key.code {
            KeyCode::Down | KeyCode::Char('j') if n > 0 => {
                self.variant_row = (self.variant_row + 1).min(n - 1);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.variant_row = self.variant_row.saturating_sub(1);
            }
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                self.choose_variant(self.variant_row);
            }
            KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') => self.view = View::List,
            _ => {}
        }
    }
}

#[cfg(test)]
#[path = "bench_variants_tests.rs"]
mod tests;
