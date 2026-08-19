// SPDX-License-Identifier: AGPL-3.0-only

//! The intent eigenvector: a derived view answering "where is this repository
//! going?" from the same canonical JSONL the ledger already holds.
//!
//! `materialize()` builds the other derived view — a graph, for traversals.
//! This one is spectral. Each PR's intent becomes a point in embedding space;
//! the principal components of that cloud are the axes the work is spread
//! along, and the angle between the older and newer halves' principal
//! directions says whether those axes are turning.
//!
//! # The division of labour with the model
//!
//! The mathematics finds the axes. A language model is used for exactly one
//! thing: putting a NAME to an axis that has already been found, given the PRs
//! at each of its poles. That ordering matters — a model asked to summarise a
//! hundred intents will produce something plausible whether or not any
//! structure exists, whereas an eigendecomposition either finds a direction or
//! reports that it did not. Naming is therefore optional by construction: if
//! the call fails or the free tier is spent, the report still renders with its
//! axes shown by their poles.

pub mod spectral;

#[cfg(test)]
#[path = "spectral_tests.rs"]
mod spectral_tests;

pub mod render;

use serde::{Deserialize, Serialize};

/// How many PRs to show at each end of an axis. Five is enough to see what the
/// pole has in common and few enough that three axes still fit in a comment.
const POLE_SIZE: usize = 5;

/// One PR's intent, as harvested and then embedded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentDoc {
    pub pr: u64,
    pub title: String,
    /// The taxonomy path the classifier settled on, e.g. `performance/decode`.
    pub intent: String,
    /// Newest event timestamp for this PR, used to split the drift halves.
    pub at: i64,
    pub embedding: Vec<f64>,
}

/// What the workflow hands the analyser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeInput {
    /// The ledger window these documents came from, echoed into the report so a
    /// reader knows the view is rolling and not all-time.
    pub window: usize,
    pub docs: Vec<IntentDoc>,
}

/// A PR at one end of an axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pole {
    pub pr: u64,
    pub title: String,
    pub intent: String,
    pub loading: f64,
}

/// One named-or-unnamed axis, with the evidence for it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Axis {
    pub explained: f64,
    /// PRs furthest along the positive direction, strongest first.
    pub positive: Vec<Pole>,
    /// PRs furthest along the negative direction, strongest first.
    pub negative: Vec<Pole>,
}

/// The analyser's output — everything the renderer needs except the names.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Analysis {
    pub n: usize,
    pub dim: usize,
    pub window: usize,
    pub coherence: Option<f64>,
    pub drift_degrees: Option<f64>,
    pub axes: Vec<Axis>,
}

/// What the model contributes, if it answered at all.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Naming {
    /// Parallel to `Analysis::axes`. A shorter list names only the axes it
    /// covers; a longer one is truncated by the renderer.
    #[serde(default)]
    pub axes: Vec<AxisName>,
    #[serde(default)]
    pub trajectory: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisName {
    pub label: String,
    pub gloss: String,
}

/// What the renderer is handed: the analysis, plus whatever naming survived.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderInput {
    #[serde(flatten)]
    pub analysis: Analysis,
    #[serde(default)]
    pub naming: Option<Naming>,
}

/// Rank the documents along one component and take the extremes.
fn poles(docs: &[IntentDoc], loadings: &[f64]) -> (Vec<Pole>, Vec<Pole>) {
    let mut ranked: Vec<Pole> = docs
        .iter()
        .zip(loadings)
        .map(|(d, &loading)| Pole {
            pr: d.pr,
            title: d.title.clone(),
            intent: d.intent.clone(),
            loading,
        })
        .collect();
    // Descending by loading, ties broken by PR number so the order is total and
    // the rendered comment does not shuffle between runs.
    ranked.sort_by(|a, b| {
        b.loading
            .partial_cmp(&a.loading)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.pr.cmp(&b.pr))
    });

    let positive: Vec<Pole> = ranked.iter().take(POLE_SIZE).cloned().collect();
    let mut negative: Vec<Pole> = ranked.iter().rev().take(POLE_SIZE).cloned().collect();
    // A pole is only meaningful if the two ends are different PRs. With fewer
    // than 2·POLE_SIZE documents the two lists would otherwise overlap and the
    // axis would appear to have the same PR at both ends.
    let taken: std::collections::HashSet<u64> = positive.iter().map(|p| p.pr).collect();
    negative.retain(|p| !taken.contains(&p.pr));
    (positive, negative)
}

/// Decompose a window of intents into at most `k` axes.
pub fn analyze(input: &AnalyzeInput, k: usize) -> Analysis {
    let rows: Vec<Vec<f64>> = input.docs.iter().map(|d| d.embedding.clone()).collect();
    let stamps: Vec<i64> = input.docs.iter().map(|d| d.at).collect();
    let spectrum = spectral::decompose(&rows, &stamps, k);

    let axes = spectrum
        .components
        .iter()
        .map(|c| {
            let (positive, negative) = poles(&input.docs, &c.loadings);
            Axis {
                explained: c.explained,
                positive,
                negative,
            }
        })
        .collect();

    Analysis {
        n: spectrum.n,
        dim: spectrum.dim,
        window: input.window,
        coherence: spectrum.coherence,
        drift_degrees: spectrum.drift_degrees,
        axes,
    }
}
