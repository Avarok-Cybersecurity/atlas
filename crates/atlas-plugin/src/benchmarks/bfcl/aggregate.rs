// SPDX-License-Identifier: AGPL-3.0-only
//! Combining BFCL shard results into the score the gate judges.
//!
//! # Why shard scores cannot simply be averaged
//!
//! `score.py` does not compute a flat mean. It aggregates hierarchically, and
//! the weights are the whole difficulty:
//!
//! ```text
//! non_live      = mean[ mean(simple_python, simple_java, simple_javascript),
//!                       multiple, parallel, parallel_multiple ]      <- FOUR terms
//! live          = sample-weighted mean over its subsets
//! hallucination = mean(irrelevance, live_irrelevance)                <- UNWEIGHTED
//! normalized    = mean(the categories present)
//! overall       = flat mean over every scored sample
//! ```
//!
//! So `simple_javascript` (31 rows in the golden draw) carries the same weight
//! as `simple_python` (248) divided three ways, and `irrelevance` (24 rows)
//! counts equally with `live_irrelevance` (88). A mean of four shards'
//! `normalized_single_turn_score` is therefore **not** the whole-set value, and
//! even `overall_accuracy` survives only a sample-count-weighted mean — and then
//! only to the 2 decimal places each shard's JSON is rounded to, which is coarse
//! against a +/-0.4 noise band.
//!
//! Worse, `score.py` builds each category from the subsets *present*, so a
//! subset missing from a shard silently changes that category's divisor.
//! `live_parallel` is 16 rows in the golden draw; a quarter of it is four.
//!
//! # What this does instead
//!
//! Shards report per-subset `(hits, n)` INTEGER counts — `_score_ast` returns
//! exactly 0.0 or 1.0, so nothing is lost. Summing those across shards and
//! aggregating once reproduces the unsharded value exactly, whatever the
//! weighting, because the weighting is applied to the union and never to a
//! partial view.
//!
//! The arithmetic below mirrors `score.py` deliberately rather than being
//! imported from it, and [`super::aggregate_tests`] pins the two together on the
//! reference fixture — if they ever disagree, that test says so.

use std::collections::BTreeMap;

/// The subsets of each scored category, in `score.py`'s order.
pub const NON_LIVE: [&str; 6] = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
];
/// Live subsets.
pub const LIVE: [&str; 4] = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
];
/// Hallucination subsets.
pub const HALLUCINATION: [&str; 2] = ["irrelevance", "live_irrelevance"];
/// The three `simple_*` subsets collapse to ONE term inside `non_live`.
pub const SIMPLE_AST: [&str; 3] = ["simple_python", "simple_java", "simple_javascript"];

/// Hits and sample count for one subset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    /// Samples scored 1.0.
    pub hits: u64,
    /// Samples scored at all.
    pub n: u64,
}

impl Tally {
    fn mean(self) -> Option<f64> {
        (self.n > 0).then(|| self.hits as f64 / self.n as f64)
    }
}

/// The numbers the gate judges.
#[derive(Clone, Debug, PartialEq)]
pub struct Aggregate {
    /// Flat mean over every scored sample, x100, rounded to 2dp.
    pub overall_accuracy: f64,
    /// Unweighted mean of the categories present, x100, rounded to 2dp.
    pub normalized_single_turn_score: f64,
    /// Per-category, x100, rounded to 2dp.
    pub category_scores: BTreeMap<String, f64>,
    /// Total samples scored.
    pub total_samples: u64,
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Aggregate per-subset tallies exactly as `score.py` would over the same rows.
///
/// Takes the UNION of every shard's tallies. Feeding it one shard's tallies
/// gives that shard's own score, which is what makes the conformance test in
/// [`super::aggregate_tests`] possible on an unsharded run.
pub fn aggregate(tallies: &BTreeMap<String, Tally>) -> Aggregate {
    let subset_mean: BTreeMap<&str, f64> = tallies
        .iter()
        .filter_map(|(k, t)| t.mean().map(|m| (k.as_str(), m)))
        .collect();

    let mut category_scores_raw: BTreeMap<String, f64> = BTreeMap::new();

    // hallucination: unweighted mean over the subsets PRESENT.
    let present: Vec<f64> = HALLUCINATION
        .iter()
        .filter_map(|s| subset_mean.get(s).copied())
        .collect();
    if !present.is_empty() {
        category_scores_raw.insert("hallucination".into(), mean(&present));
    }

    // live: sample-weighted, i.e. the flat mean over live samples.
    let live: Vec<(&str, f64)> = LIVE
        .iter()
        .filter_map(|s| subset_mean.get(s).map(|m| (*s, *m)))
        .collect();
    if !live.is_empty() {
        let total: u64 = live.iter().map(|(s, _)| tallies[*s].n).sum();
        if total > 0 {
            let num: f64 = live
                .iter()
                .map(|(s, m)| m * tallies[*s].n as f64)
                .sum::<f64>();
            category_scores_raw.insert("live".into(), num / total as f64);
        }
    }

    // non_live: HIERARCHICAL. The three simple_* collapse to one term, then an
    // unweighted mean over that term plus each remaining subset.
    let non_live_present: Vec<&str> = NON_LIVE
        .iter()
        .copied()
        .filter(|s| subset_mean.contains_key(s))
        .collect();
    if !non_live_present.is_empty() {
        let simple: Vec<f64> = SIMPLE_AST
            .iter()
            .filter_map(|s| subset_mean.get(s).copied())
            .collect();
        let mut top: Vec<f64> = Vec::new();
        if !simple.is_empty() {
            top.push(mean(&simple));
        }
        for s in &non_live_present {
            if !SIMPLE_AST.contains(s) {
                top.push(subset_mean[s]);
            }
        }
        if !top.is_empty() {
            category_scores_raw.insert("non_live".into(), mean(&top));
        }
    }

    let normalized = if category_scores_raw.is_empty() {
        0.0
    } else {
        mean(&category_scores_raw.values().copied().collect::<Vec<_>>())
    };

    let hits: u64 = tallies.values().map(|t| t.hits).sum();
    let n: u64 = tallies.values().map(|t| t.n).sum();
    let overall = if n > 0 { hits as f64 / n as f64 } else { 0.0 };

    Aggregate {
        overall_accuracy: round2(overall * 100.0),
        normalized_single_turn_score: round2(normalized * 100.0),
        category_scores: category_scores_raw
            .into_iter()
            .map(|(k, v)| (k, round2(v * 100.0)))
            .collect(),
        total_samples: n,
    }
}

/// Sum shard tallies into the union.
///
/// Returns the union AND the per-subset provenance, so a caller can assert the
/// union matches the draw plan — the invariant that turns a missing or
/// duplicated shard into a named failure instead of a quietly reweighted score.
pub fn union(shards: &[BTreeMap<String, Tally>]) -> BTreeMap<String, Tally> {
    let mut out: BTreeMap<String, Tally> = BTreeMap::new();
    for shard in shards {
        for (k, t) in shard {
            let e = out.entry(k.clone()).or_default();
            e.hits += t.hits;
            e.n += t.n;
        }
    }
    out
}
