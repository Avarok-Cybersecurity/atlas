// SPDX-License-Identifier: AGPL-3.0-only

//! Accumulate routing per category, then pick each layer's expert set.
//!
//! Pure: no I/O, no transport, no clock. The benchmark's `next()` feeds it
//! one parsed response at a time and asks for the result at the end, which
//! is what lets the selection rule be tested without a GPU.
//!
//! ## The selection rule
//!
//! Per (category, layer), keep the smallest set of experts whose summed
//! routing MASS covers `coverage` of that layer's total — MoE-Spec expert
//! budgeting, `atlas_core::moe_policy::budget_experts`. Mass rather than
//! activation frequency because mass is what the layer's output is actually
//! made of: an expert selected in every token at weight 0.02 contributes
//! less than one selected in a tenth of them at weight 0.5, and loading the
//! former while dropping the latter is the mistake this ordering avoids.

use std::collections::BTreeMap;

use super::usage::Activation;

/// `(count, mass)` accumulated for one expert.
type ExpertTally = (u64, f64);
/// One layer's experts, keyed by id so a fold is a lookup.
type LayerTally = BTreeMap<u32, ExpertTally>;
/// `(expert_id, count, mass)` — one layer's distribution, flattened.
pub type ExpertMass = (u32, u32, f64);

/// Per-category, per-layer summed mass and counts.
#[derive(Debug, Default, Clone)]
pub struct Accumulator {
    /// `category -> layer -> expert -> (count, mass)`.
    per_category: BTreeMap<String, BTreeMap<usize, LayerTally>>,
    /// `category -> (prompts folded, token positions routed, unattributed)`.
    totals: BTreeMap<String, CategoryTotals>,
    top_k: u32,
    num_experts: u32,
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct CategoryTotals {
    pub prompts: u64,
    pub tokens_routed: u64,
    pub unattributed_rows: u64,
}

/// One category's measured expert sets.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoryBudget {
    pub category: String,
    pub coverage: f64,
    pub totals: CategoryTotals,
    /// `(layer, expert ids ascending)` — layers ascending, only layers that
    /// routed.
    pub layers: Vec<(usize, Vec<u32>)>,
}

impl Accumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn top_k(&self) -> u32 {
        self.top_k
    }

    pub fn num_experts(&self) -> u32 {
        self.num_experts
    }

    /// Fold one response into `category`.
    ///
    /// Returns an error if this response's routing geometry disagrees with
    /// what earlier responses reported: summing mass across two different
    /// expert spaces would produce a table describing neither model.
    pub fn feed(&mut self, category: &str, act: &Activation) -> anyhow::Result<()> {
        if self.top_k == 0 {
            self.top_k = act.top_k;
            self.num_experts = act.num_experts;
        } else if self.top_k != act.top_k || self.num_experts != act.num_experts {
            anyhow::bail!(
                "routing geometry changed mid-run: was top_k={} num_experts={}, now top_k={} \
                 num_experts={}. The served model changed under the benchmark; the partial \
                 aggregate describes neither.",
                self.top_k,
                self.num_experts,
                act.top_k,
                act.num_experts
            );
        }

        let cat = self.per_category.entry(category.to_string()).or_default();
        for layer in &act.layers {
            let slot = cat.entry(layer.layer).or_default();
            for &(id, count, mass) in &layer.experts {
                let e = slot.entry(id).or_insert((0, 0.0));
                e.0 += u64::from(count);
                e.1 += mass;
            }
        }
        let t = self.totals.entry(category.to_string()).or_default();
        t.prompts += 1;
        t.tokens_routed += act.tokens_routed;
        t.unattributed_rows += act.unattributed_rows;
        Ok(())
    }

    /// Categories folded so far, in a stable order.
    pub fn categories(&self) -> Vec<&str> {
        self.per_category.keys().map(String::as_str).collect()
    }

    pub fn totals(&self, category: &str) -> CategoryTotals {
        self.totals.get(category).copied().unwrap_or_default()
    }

    /// Per-layer summed mass for one category — the evidence behind a
    /// budget, kept for the stats artifact.
    pub fn layer_mass(&self, category: &str) -> Vec<(usize, Vec<ExpertMass>)> {
        self.per_category
            .get(category)
            .map(|layers| {
                layers
                    .iter()
                    .map(|(&l, experts)| {
                        (
                            l,
                            experts
                                .iter()
                                .map(|(&e, &(c, m))| (e, c as u32, m))
                                .collect(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Apply the coverage rule to every category.
    pub fn budgets(&self, coverage: f64) -> Vec<CategoryBudget> {
        self.per_category
            .iter()
            .map(|(category, layers)| {
                let budgeted = layers
                    .iter()
                    .filter_map(|(&layer, experts)| {
                        let weights: Vec<(u32, f32)> =
                            experts.iter().map(|(&e, &(_, m))| (e, m as f32)).collect();
                        let kept =
                            atlas_core::moe_policy::budget_experts(&weights, coverage as f32);
                        if kept.is_empty() {
                            // A layer whose total mass was zero. It routed
                            // nothing, so it contributes no experts — and it
                            // must not appear as an empty list, which BEL
                            // would read as "load no experts here".
                            return None;
                        }
                        // budget_experts returns descending by weight; the
                        // table is written ascending by id so two runs of the
                        // same measurement diff cleanly.
                        let mut ids: Vec<u32> = kept.into_iter().map(|(e, _)| e).collect();
                        ids.sort_unstable();
                        Some((layer, ids))
                    })
                    .collect();
                CategoryBudget {
                    category: category.clone(),
                    coverage,
                    totals: self.totals(category),
                    layers: budgeted,
                }
            })
            .collect()
    }
}

/// Jaccard overlap of two categories' budgeted sets, averaged over the
/// layers they share. This is the number that says whether the categories
/// discriminate at all: two categories that route identically cannot be
/// given different expert sets, however clean the measurement was.
pub fn mean_jaccard(a: &CategoryBudget, b: &CategoryBudget) -> f64 {
    let bl: BTreeMap<usize, &Vec<u32>> = b.layers.iter().map(|(l, ids)| (*l, ids)).collect();
    let mut sum = 0.0;
    let mut n = 0usize;
    for (layer, ids_a) in &a.layers {
        let Some(ids_b) = bl.get(layer) else {
            continue;
        };
        let sa: std::collections::BTreeSet<u32> = ids_a.iter().copied().collect();
        let sb: std::collections::BTreeSet<u32> = ids_b.iter().copied().collect();
        let union = sa.union(&sb).count();
        if union == 0 {
            continue;
        }
        sum += sa.intersection(&sb).count() as f64 / union as f64;
        n += 1;
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

#[cfg(test)]
#[path = "aggregate_tests.rs"]
mod tests;
