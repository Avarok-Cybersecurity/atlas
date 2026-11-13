// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. Stratified k-fold, percentile bootstrap and nested cross-validation; no upstream code.
//! Responsibility: stratified folds, the stratified percentile bootstrap, and repeated nested cross-validation of the
//! ridge logistic head (the ridge strength is chosen inside each training fold only).

use rayon::prelude::*;

use super::auroc;
use super::logistic::fit_predict_logistic;
use crate::route_gate::error::GResult;
use crate::route_gate::mat::Mat;
use crate::route_gate::mt::NumpyRandom;
use crate::route_gate::stage_one::linear_quantile_row;

/// A `(train, test)` index pair.
pub type Fold = (Vec<usize>, Vec<usize>);

/// `k` `(train_idx, test_idx)` pairs. Each class is shuffled and dealt round-robin, so every fold keeps the class
/// ratio.
pub fn stratified_folds(y: &[bool], k: usize, seed: u64) -> Vec<Fold> {
    let n_pos = y.iter().filter(|&&v| v).count();
    let n_neg = y.len() - n_pos;
    let k = k
        .min(if n_pos == 0 { 2 } else { n_pos })
        .min(if n_neg == 0 { 2 } else { n_neg })
        .max(2);
    let mut rng = NumpyRandom::new(seed);
    let mut fold_of = vec![0usize; y.len()];
    for cls in [true, false] {
        let mut idx: Vec<usize> = (0..y.len()).filter(|&i| y[i] == cls).collect();
        rng.shuffle(&mut idx);
        for (pos, &i) in idx.iter().enumerate() {
            fold_of[i] = pos % k;
        }
    }
    (0..k)
        .map(|f| {
            let test = (0..y.len()).filter(|&i| fold_of[i] == f).collect();
            let train = (0..y.len()).filter(|&i| fold_of[i] != f).collect();
            (train, test)
        })
        .collect()
}

/// A percentile interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BootCi {
    pub lo: f64,
    pub hi: f64,
}

/// Paired difference to the baseline on the same resamples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BootDiff {
    pub mean: f64,
    pub lo: f64,
    pub hi: f64,
}

/// Result of [`bootstrap_auroc`]: an interval per arm and, per non-baseline arm, the paired difference.
#[derive(Clone, Debug, Default)]
pub struct Bootstrap {
    pub arms: Vec<(String, BootCi)>,
    pub diff: Vec<(String, BootDiff)>,
}

impl Bootstrap {
    pub fn ci(&self, arm: &str) -> Option<BootCi> {
        self.arms.iter().find(|(n, _)| n == arm).map(|(_, c)| *c)
    }

    pub fn diff_of(&self, arm: &str) -> Option<BootDiff> {
        self.diff.iter().find(|(n, _)| n == arm).map(|(_, d)| *d)
    }
}

fn sorted_copy(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s
}

/// `numpy.percentile(sorted, q)` with linear interpolation; `sorted` must be ascending.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    linear_quantile_row(sorted, sorted.len(), q / 100.0)
}

/// Stratified percentile bootstrap (resample failures and passes separately, so every replicate has both classes).
///
/// `scores` maps arm name to score vector. When `baseline` names an arm, the paired difference `AUROC(arm) -
/// AUROC(baseline)` is computed on the SAME resamples.
pub fn bootstrap_auroc(
    y: &[bool],
    scores: &[(String, Vec<f64>)],
    reps: usize,
    seed: u64,
    baseline: Option<&str>,
) -> GResult<Bootstrap> {
    let pos: Vec<usize> = (0..y.len()).filter(|&i| y[i]).collect();
    let neg: Vec<usize> = (0..y.len()).filter(|&i| !y[i]).collect();
    if pos.is_empty() || neg.is_empty() {
        let nan = BootCi {
            lo: f64::NAN,
            hi: f64::NAN,
        };
        return Ok(Bootstrap {
            arms: scores.iter().map(|(a, _)| (a.clone(), nan)).collect(),
            diff: Vec::new(),
        });
    }
    let mut rng = NumpyRandom::new(seed);
    let mut resamples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut idx: Vec<usize> = rng
            .choice_indices(pos.len(), pos.len())
            .into_iter()
            .map(|j| pos[j])
            .collect();
        idx.extend(
            rng.choice_indices(neg.len(), neg.len())
                .into_iter()
                .map(|j| neg[j]),
        );
        resamples.push(idx);
    }
    // samples[arm][rep]
    let samples: Vec<Vec<f64>> = scores
        .par_iter()
        .map(|(_, s)| {
            resamples
                .iter()
                .map(|idx| {
                    let yy: Vec<bool> = idx.iter().map(|&i| y[i]).collect();
                    let ss: Vec<f64> = idx.iter().map(|&i| s[i]).collect();
                    auroc(&yy, &ss)
                })
                .collect::<GResult<Vec<f64>>>()
        })
        .collect::<GResult<Vec<_>>>()?;
    let mut out = Bootstrap::default();
    for ((name, _), v) in scores.iter().zip(&samples) {
        let sorted = sorted_copy(v);
        out.arms.push((
            name.clone(),
            BootCi {
                lo: percentile(&sorted, 2.5),
                hi: percentile(&sorted, 97.5),
            },
        ));
    }
    if let Some(base) = baseline
        && let Some(bi) = scores.iter().position(|(n, _)| n == base)
    {
        for (ai, (name, _)) in scores.iter().enumerate() {
            if ai == bi {
                continue;
            }
            let d: Vec<f64> = samples[ai]
                .iter()
                .zip(&samples[bi])
                .map(|(a, b)| a - b)
                .collect();
            let mean = d.iter().sum::<f64>() / d.len() as f64;
            let sorted = sorted_copy(&d);
            out.diff.push((
                name.clone(),
                BootDiff {
                    mean,
                    lo: percentile(&sorted, 2.5),
                    hi: percentile(&sorted, 97.5),
                },
            ));
        }
    }
    Ok(out)
}

fn to_f64(y: &[bool]) -> Vec<f64> {
    y.iter().map(|&v| f64::from(u8::from(v))).collect()
}

/// Picks the ridge strength by inner cross-validated AUROC on the training rows (ties go to the larger l2).
pub fn choose_l2(x: &Mat, y: &[bool], grid: &[f64], folds: usize, seed: u64) -> GResult<f64> {
    let splits = stratified_folds(y, folds, seed);
    let yf = to_f64(y);
    let aucs: Vec<GResult<f64>> = grid
        .par_iter()
        .map(|&l2| {
            let mut oof = vec![f64::NAN; y.len()];
            for (tr, te) in &splits {
                let ytr: Vec<f64> = tr.iter().map(|&i| yf[i]).collect();
                let p = fit_predict_logistic(&x.select_rows(tr), &ytr, &x.select_rows(te), l2);
                for (&i, v) in te.iter().zip(p) {
                    oof[i] = v;
                }
            }
            auroc(y, &oof)
        })
        .collect();
    let mut best = grid[grid.len() - 1];
    let mut best_auc = -1.0;
    for (&l2, a) in grid.iter().zip(aucs) {
        let a = a?;
        if a.is_nan() {
            continue;
        }
        if a > best_auc + 1e-12 || ((a - best_auc).abs() <= 1e-12 && l2 > best) {
            best = l2;
            best_auc = a;
        }
    }
    Ok(best)
}

/// Result of [`cv_logistic`].
#[derive(Clone, Debug)]
pub struct CvResult {
    /// Mean out-of-fold probability over the repeats.
    pub oof: Vec<f64>,
    /// AUROC of each repeat.
    pub aurocs: Vec<f64>,
    /// The ridge strength chosen in every fold of every repeat.
    pub l2: Vec<f64>,
}

/// Repeated nested cross-validation of an L2 logistic head. The ridge strength is chosen inside each training fold
/// only, so no test row influences its own prediction or its own regularization.
pub fn cv_logistic(
    x: &Mat,
    y: &[bool],
    grid: &[f64],
    folds: usize,
    repeats: usize,
    seed: u64,
    inner_folds: usize,
) -> GResult<CvResult> {
    let yf = to_f64(y);
    let mut tasks = Vec::new();
    for r in 0..repeats {
        for (fi, split) in stratified_folds(y, folds, seed + 1000 * r as u64)
            .into_iter()
            .enumerate()
        {
            tasks.push((r, fi, split));
        }
    }
    let done: Vec<GResult<(f64, Vec<f64>)>> = tasks
        .par_iter()
        .map(|(r, fi, (tr, te))| {
            let xtr = x.select_rows(tr);
            let ytr_b: Vec<bool> = tr.iter().map(|&i| y[i]).collect();
            let l2 = choose_l2(
                &xtr,
                &ytr_b,
                grid,
                inner_folds,
                seed + 1000 * *r as u64 + *fi as u64 + 1,
            )?;
            let ytr: Vec<f64> = tr.iter().map(|&i| yf[i]).collect();
            Ok((l2, fit_predict_logistic(&xtr, &ytr, &x.select_rows(te), l2)))
        })
        .collect();
    let mut oofs = vec![vec![f64::NAN; y.len()]; repeats];
    let mut chosen = Vec::with_capacity(tasks.len());
    for ((r, _, (_, te)), res) in tasks.iter().zip(done) {
        let (l2, p) = res?;
        chosen.push(l2);
        for (&i, v) in te.iter().zip(p) {
            oofs[*r][i] = v;
        }
    }
    let aurocs = oofs
        .iter()
        .map(|o| auroc(y, o))
        .collect::<GResult<Vec<f64>>>()?;
    let oof = (0..y.len())
        .map(|i| oofs.iter().map(|o| o[i]).sum::<f64>() / repeats as f64)
        .collect();
    Ok(CvResult {
        oof,
        aurocs,
        l2: chosen,
    })
}

/// Cross-validated 1-D logistic calibration of a raw score into P(fail), so a score with no probability meaning
/// (for example `1 - p`) can be drawn on a calibration curve. AUROC is unchanged up to ties.
pub fn cv_platt(score: &[f64], y: &[bool], folds: usize, repeats: usize, seed: u64) -> Vec<f64> {
    let s = Mat {
        rows: score.len(),
        cols: 1,
        data: score.to_vec(),
    };
    let yf = to_f64(y);
    let mut outs = vec![vec![f64::NAN; y.len()]; repeats];
    for (r, oof) in outs.iter_mut().enumerate() {
        for (tr, te) in stratified_folds(y, folds, seed + 1000 * r as u64) {
            let ytr: Vec<f64> = tr.iter().map(|&i| yf[i]).collect();
            let p = fit_predict_logistic(&s.select_rows(&tr), &ytr, &s.select_rows(&te), 1e-3);
            for (&i, v) in te.iter().zip(p) {
                oof[i] = v;
            }
        }
    }
    (0..y.len())
        .map(|i| outs.iter().map(|o| o[i]).sum::<f64>() / repeats as f64)
        .collect()
}
