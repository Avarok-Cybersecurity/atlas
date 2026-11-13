// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. Standard definitions only (rank-sum AUROC, Newton-Raphson ridge logistic regression, stratified
// k-fold, percentile bootstrap, quantile-bin calibration); no upstream code copied.
//! Responsibility: statistics for the routing gate (B0), plain Rust (no scipy, no scikit-learn, no NumPy).
//!
//! Convention used everywhere: `y = true` means "the 4B model fails the prompt" (the event the router must catch),
//! and a score is higher when failure is more likely. AUROC is the probability that a random failing prompt scores
//! above a random passing one (ties count half).

mod cv;
mod logistic;

pub use cv::{
    BootCi, BootDiff, Bootstrap, CvResult, bootstrap_auroc, choose_l2, cv_logistic, cv_platt,
    stratified_folds,
};
pub use logistic::{Standardizer, fit_logistic, fit_predict_logistic};

use super::error::{GResult, value_err};

/// 1-based ranks with ties averaged (`scipy.stats.rankdata(method="average")`).
pub fn rankdata(a: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..a.len()).collect();
    order.sort_by(|&i, &j| a[i].total_cmp(&a[j]));
    let mut ranks = vec![0.0; a.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start;
        while end + 1 < order.len() && a[order[end + 1]] == a[order[start]] {
            end += 1;
        }
        let mean_rank = (start + end) as f64 / 2.0 + 1.0;
        for &i in &order[start..=end] {
            ranks[i] = mean_rank;
        }
        start = end + 1;
    }
    ranks
}

/// Rank-sum AUROC. NaN when only one class is present. Non-finite scores are an error, never silently dropped.
pub fn auroc(y: &[bool], score: &[f64]) -> GResult<f64> {
    if y.len() != score.len() {
        return value_err(format!("y {} and score {} differ", y.len(), score.len()));
    }
    if !score.iter().all(|s| s.is_finite()) {
        return value_err("auroc: non-finite scores");
    }
    let n_pos = y.iter().filter(|&&v| v).count();
    let n_neg = y.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return Ok(f64::NAN);
    }
    let ranks = rankdata(score);
    let pos_sum: f64 = y
        .iter()
        .zip(&ranks)
        .filter(|(yy, _)| **yy)
        .map(|(_, r)| r)
        .sum();
    let np = n_pos as f64;
    Ok((pos_sum - np * (np + 1.0) / 2.0) / (np * n_neg as f64))
}

/// O(n^2) pair count, the definition. Used by the tests as the oracle for [`auroc`].
pub fn auroc_bruteforce(y: &[bool], score: &[f64]) -> f64 {
    let pos: Vec<f64> = y
        .iter()
        .zip(score)
        .filter(|(yy, _)| **yy)
        .map(|(_, s)| *s)
        .collect();
    let neg: Vec<f64> = y
        .iter()
        .zip(score)
        .filter(|(yy, _)| !**yy)
        .map(|(_, s)| *s)
        .collect();
    if pos.is_empty() || neg.is_empty() {
        return f64::NAN;
    }
    let (mut gt, mut eq) = (0usize, 0usize);
    for &p in &pos {
        for &q in &neg {
            if p > q {
                gt += 1;
            } else if p == q {
                eq += 1;
            }
        }
    }
    (gt as f64 + 0.5 * eq as f64) / (pos.len() * neg.len()) as f64
}

/// One calibration bin.
#[derive(Clone, Debug, PartialEq)]
pub struct Bin {
    pub n: usize,
    pub mean_pred: f64,
    pub frac_pos: f64,
}

/// Equal-count bins over the predicted probability, lowest prediction first (`np.array_split` sizes).
pub fn calibration_bins(y: &[bool], p: &[f64], bins: usize) -> Vec<Bin> {
    let mut order: Vec<usize> = (0..p.len()).collect();
    order.sort_by(|&i, &j| p[i].total_cmp(&p[j]));
    let k = bins.min(order.len().max(1));
    let (base, extra) = (order.len() / k, order.len() % k);
    let mut out = Vec::new();
    let mut start = 0;
    for b in 0..k {
        let size = base + usize::from(b < extra);
        let chunk = &order[start..start + size];
        start += size;
        if chunk.is_empty() {
            continue;
        }
        let m = chunk.len() as f64;
        out.push(Bin {
            n: chunk.len(),
            mean_pred: chunk.iter().map(|&i| p[i]).sum::<f64>() / m,
            frac_pos: chunk.iter().filter(|&&i| y[i]).count() as f64 / m,
        });
    }
    out
}

/// Expected calibration error over equal-count bins: weighted mean `|frac_pos - mean_pred|`.
pub fn ece(bins: &[Bin]) -> f64 {
    let total: usize = bins.iter().map(|b| b.n).sum();
    if total == 0 {
        return f64::NAN;
    }
    bins.iter()
        .map(|b| b.n as f64 * (b.frac_pos - b.mean_pred).abs())
        .sum::<f64>()
        / total as f64
}

/// Brier score `mean((p - y)^2)`.
pub fn brier(y: &[bool], p: &[f64]) -> f64 {
    let s: f64 = y
        .iter()
        .zip(p)
        .map(|(&yy, &pp)| (pp - f64::from(u8::from(yy))).powi(2))
        .sum();
    s / y.len() as f64
}
