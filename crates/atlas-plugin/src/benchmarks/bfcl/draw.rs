// SPDX-License-Identifier: AGPL-3.0-only

//! The BFCL sample draw.
//!
//! **The draw is the whole ballgame.** `normalized_single_turn_score` is
//! category-mix sensitive: the same model on the same checkpoint scores 89.39
//! on the golden 62/10/10 draw and 87.60 on a different one, while
//! `overall_accuracy` lands in the same place — which makes a wrong draw easy
//! to miss and impossible to compare.
//!
//! Two rules produce it, and both are deterministic and RNG-free:
//!
//! 1. **Selection by category.** Asking for `[non_live, live, hallucination]`
//!    expands to those categories' subsets — and `live_relevance` belongs to
//!    none of them, so it drops out. That single exclusion is the difference
//!    between n = 1011 and the golden **n = 995**.
//! 2. **Per-subset head(n).** `n = total if total <= subset_floor else
//!    max(1, int(total * pct / 100))`, taking the FIRST `n` rows, concatenated
//!    in sorted subset order.

use std::collections::BTreeMap;

/// Every single-turn subset, in the order `provision.py` writes them.
pub const SINGLE_TURN_SUBSETS: [&str; 13] = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "irrelevance",
    "live_irrelevance",
    "live_relevance",
];

/// The three scored categories.
pub const CATEGORIES: [&str; 3] = ["non_live", "live", "hallucination"];

/// Which category a subset belongs to.
///
/// `live_relevance` deliberately maps to `None`: the reference scores it
/// per-sample but excludes it from every category aggregate, and a category
/// selection therefore leaves it out entirely.
pub fn category_of(subset: &str) -> Option<&'static str> {
    match subset {
        "simple_python" | "simple_java" | "simple_javascript" | "multiple" | "parallel"
        | "parallel_multiple" => Some("non_live"),
        "live_simple" | "live_multiple" | "live_parallel" | "live_parallel_multiple" => {
            Some("live")
        }
        "irrelevance" | "live_irrelevance" => Some("hallucination"),
        _ => None,
    }
}

/// Sampling configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct DrawSpec {
    /// Categories to include. Empty means every single-turn subset, including
    /// the uncategorised `live_relevance`.
    pub categories: Vec<String>,
    /// Per-category percentage. A selected subset whose category has no entry
    /// is taken whole.
    pub category_pct: BTreeMap<String, f64>,
    /// Any subset this small or smaller is taken in full, bypassing the
    /// percentage — it stops `live_parallel` collapsing to two noisy samples.
    pub subset_floor: Option<usize>,
}

impl DrawSpec {
    fn categories_or_all(categories: &[&str]) -> Vec<String> {
        categories.iter().map(|s| s.to_string()).collect()
    }

    /// The golden MLPerf-edge draw: the three categories at 62/10/10 with a
    /// floor of 25. On the full v4 single-turn table this is exactly **995**.
    pub fn golden() -> Self {
        Self {
            categories: Self::categories_or_all(&CATEGORIES),
            category_pct: [
                ("non_live".to_string(), 62.0),
                ("live".to_string(), 10.0),
                ("hallucination".to_string(), 10.0),
            ]
            .into_iter()
            .collect(),
            subset_floor: Some(25),
        }
    }

    /// Every sample of the three scored categories, no sampling.
    ///
    /// The same COMPOSITION as `golden`, so the normalized score stays directly
    /// comparable — a "full" run that also swept in `live_relevance` would move
    /// `overall_accuracy` against a category the aggregate does not score.
    pub fn full() -> Self {
        Self {
            categories: Self::categories_or_all(&CATEGORIES),
            category_pct: BTreeMap::new(),
            subset_floor: None,
        }
    }

    /// The `echolp` draw: the three categories at 46/23/12 with a floor of 25.
    /// On the full v4 single-turn table this is exactly **1004**.
    ///
    /// This is a DIFFERENT COMPOSITION from `golden`, and that is the point —
    /// it weights `live` more than twice as heavily (23 % vs 10 %). The two
    /// draws land `overall_accuracy` in the same place (~87.44 vs 87.45) while
    /// `normalized_single_turn_score` differs by ~1.8 points purely from the
    /// category mix. A score from one draw is therefore NOT comparable to a
    /// threshold from the other; each has its own baseline.
    pub fn echolp() -> Self {
        Self {
            categories: Self::categories_or_all(&CATEGORIES),
            category_pct: [
                ("non_live".to_string(), 46.0),
                ("live".to_string(), 23.0),
                ("hallucination".to_string(), 12.0),
            ]
            .into_iter()
            .collect(),
            subset_floor: Some(25),
        }
    }

    /// Is this subset in the selection?
    pub fn includes(&self, subset: &str) -> bool {
        if self.categories.is_empty() {
            return true;
        }
        category_of(subset).is_some_and(|c| self.categories.iter().any(|sel| sel == c))
    }

    /// How many rows to keep from a subset holding `total` rows.
    pub fn take_count(&self, subset: &str, total: usize) -> usize {
        if total == 0 || !self.includes(subset) {
            return 0;
        }
        if self.subset_floor.is_some_and(|floor| total <= floor) {
            return total;
        }
        match category_of(subset).and_then(|c| self.category_pct.get(c).copied()) {
            // `int()` truncates in the reference; `as usize` on a non-negative
            // f64 does the same. `max(1)` keeps a subset from vanishing.
            Some(pct) => ((total as f64 * pct / 100.0) as usize).max(1),
            None => total,
        }
    }
}

/// Apply the draw to per-subset totals, returning `(subset, take)` in the
/// reference's concatenation order — sorted subset name, as `pandas.groupby`
/// yields.
pub fn plan(spec: &DrawSpec, totals: &BTreeMap<String, usize>) -> Vec<(String, usize)> {
    totals
        .iter()
        .map(|(subset, total)| (subset.clone(), spec.take_count(subset, *total)))
        .filter(|(_, take)| *take > 0)
        .collect()
}

/// Total sample count for a plan — what the params pane shows live, so a
/// mis-set percentage is caught before the run instead of three hours later.
pub fn total(plan: &[(String, usize)]) -> usize {
    plan.iter().map(|(_, n)| n).sum()
}

#[cfg(test)]
#[path = "draw_tests.rs"]
mod tests;
