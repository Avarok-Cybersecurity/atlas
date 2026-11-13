// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. Evaluation design after 80-routing.md section 5 item 4 (the B0 gate) and section 8.
//! Responsibility: the arms that are not a plain logistic head: the dataset prior, the nested best-single-candidate
//! selection over the Stage-1 probabilities, the neutral fill for unscored rows, and the within-source AUROC.

use std::collections::{BTreeMap, BTreeSet};

use crate::route_gate::error::GResult;
use crate::route_gate::mat::Mat;
use crate::route_gate::metrics::{auroc, stratified_folds};

/// NaN scores (a row with no finite candidate) become the median so they neither help nor hurt; returns the count.
pub fn neutral_fill(score: &[f64]) -> (Vec<f64>, usize) {
    let bad = score.iter().filter(|v| !v.is_finite()).count();
    if bad == 0 {
        return (score.to_vec(), 0);
    }
    let mut finite: Vec<f64> = score.iter().copied().filter(|v| !v.is_nan()).collect();
    finite.sort_by(f64::total_cmp);
    let fill = if bad < score.len() && !finite.is_empty() {
        let mid = finite.len() / 2;
        if finite.len() % 2 == 1 {
            finite[mid]
        } else {
            (finite[mid - 1] + finite[mid]) / 2.0
        }
    } else {
        0.5
    };
    (
        score
            .iter()
            .map(|&v| if v.is_finite() { v } else { fill })
            .collect(),
        bad,
    )
}

/// Out-of-fold failure rate (Laplace smoothed) of each row's dataset, from the training rows only.
pub fn source_prior_scores(sources: &[String], y: &[bool], folds: usize, seed: u64) -> Vec<f64> {
    let mut oof = vec![f64::NAN; y.len()];
    for (tr, te) in stratified_folds(y, folds, seed) {
        let names: BTreeSet<&String> = te.iter().map(|&i| &sources[i]).collect();
        for s in names {
            let train: Vec<usize> = tr.iter().copied().filter(|&i| &sources[i] == s).collect();
            let fails = train.iter().filter(|&&i| y[i]).count() as f64;
            let rate = (fails + 1.0) / (train.len() as f64 + 2.0);
            for &i in te.iter().filter(|&&i| &sources[i] == s) {
                oof[i] = rate;
            }
        }
    }
    oof
}

/// Result of [`cv_best_candidate`].
pub struct BestCandidate {
    pub oof: Vec<f64>,
    pub aurocs: Vec<f64>,
    /// The column picked in each fold (`-1` when no column qualified).
    pub picks: Vec<i64>,
}

/// Nested selection of one candidate: in each fold choose the column with the best TRAIN AUROC of `1 - p` and score
/// the test rows with it. `p` is `[n, C]` success probabilities with possible NaN.
pub fn cv_best_candidate(
    p: &Mat,
    y: &[bool],
    folds: usize,
    repeats: usize,
    seed: u64,
) -> GResult<BestCandidate> {
    let n = p.rows;
    let (mut oofs, mut aucs, mut picks) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..repeats {
        let mut oof = vec![f64::NAN; n];
        for (tr, te) in stratified_folds(y, folds, seed + 1000 * r as u64) {
            let ytr: Vec<bool> = tr.iter().map(|&i| y[i]).collect();
            let (mut best, mut best_auc) = (-1i64, -1.0);
            for c in 0..p.cols {
                let col: Vec<f64> = tr.iter().map(|&i| p.at(i, c)).collect();
                let (lo, hi) = col
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
                        (l.min(v), h.max(v))
                    });
                if !col.iter().all(|v| v.is_finite()) || hi - lo < 1e-12 {
                    continue;
                }
                let a = auroc(&ytr, &col.iter().map(|v| 1.0 - v).collect::<Vec<_>>())?;
                if a > best_auc {
                    best = c as i64;
                    best_auc = a;
                }
            }
            picks.push(best);
            for &i in &te {
                oof[i] = if best >= 0 {
                    1.0 - p.at(i, best as usize)
                } else {
                    0.5
                };
            }
        }
        let (filled, _) = neutral_fill(&oof);
        aucs.push(auroc(y, &filled)?);
        oofs.push(filled);
    }
    let oof = (0..n)
        .map(|i| oofs.iter().map(|o| o[i]).sum::<f64>() / repeats as f64)
        .collect();
    Ok(BestCandidate {
        oof,
        aurocs: aucs,
        picks,
    })
}

/// AUROC inside each source that has at least `min_n` rows and both outcomes.
pub fn within_source_auroc(
    y: &[bool],
    score: &[f64],
    sources: &[String],
    min_n: usize,
) -> GResult<BTreeMap<String, f64>> {
    let mut out = BTreeMap::new();
    let names: BTreeSet<&String> = sources.iter().collect();
    for s in names {
        let idx: Vec<usize> = (0..y.len()).filter(|&i| &sources[i] == s).collect();
        let fails = idx.iter().filter(|&&i| y[i]).count();
        if idx.len() >= min_n && fails > 0 && fails < idx.len() {
            let yy: Vec<bool> = idx.iter().map(|&i| y[i]).collect();
            let ss: Vec<f64> = idx.iter().map(|&i| score[i]).collect();
            out.insert(s.clone(), auroc(&yy, &ss)?);
        }
    }
    Ok(out)
}
