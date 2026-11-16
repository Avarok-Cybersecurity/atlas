// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: unit tests of `metrics` (AUROC, folds, ridge logistic, nested CV, bootstrap, calibration), ported
//! from the Python `MetricsTests`; seeded data comes from the port's own NumPy-compatible `NumpyRandom`.

use super::mat::{Mat, dot};
use super::metrics::{
    auroc, auroc_bruteforce, bootstrap_auroc, brier, calibration_bins, cv_logistic, cv_platt, ece,
    fit_logistic, rankdata, stratified_folds,
};
use super::mt::NumpyRandom;

fn near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} vs {b} (tol {tol})");
}

/// `rng.rand(n) < p`.
fn rand_bools(rng: &mut NumpyRandom, n: usize, p: f64) -> Vec<bool> {
    rng.rand_vec(n).into_iter().map(|v| v < p).collect()
}

fn as_f64(y: &[bool]) -> Vec<f64> {
    y.iter().map(|&v| f64::from(u8::from(v))).collect()
}

fn mat(rows: usize, cols: usize, data: Vec<f64>) -> Mat {
    Mat::new(rows, cols, data).unwrap()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn auroc_known_values_and_ties() {
    near(
        auroc(&[false, false, true, true], &[0.1, 0.4, 0.35, 0.8]).unwrap(),
        0.75,
        1e-7,
    );
    assert_eq!(auroc(&[false, true], &[0.1, 0.9]).unwrap(), 1.0);
    assert_eq!(auroc(&[false, true], &[0.9, 0.1]).unwrap(), 0.0);
    assert_eq!(
        auroc(&[false, false, true, true], &[0.5, 0.5, 0.5, 0.5]).unwrap(),
        0.5
    );
    assert!(
        auroc(&[true, true, true], &[0.1, 0.2, 0.3])
            .unwrap()
            .is_nan()
    );
    assert!(auroc(&[false, true], &[0.1, f64::NAN]).is_err());
    let ranks = rankdata(&[10.0, 20.0, 20.0, 30.0]);
    for (got, want) in ranks.iter().zip([1.0, 2.5, 2.5, 4.0]) {
        near(*got, want, 1e-7);
    }
}

#[test]
fn auroc_matches_bruteforce() {
    let mut rng = NumpyRandom::new(0);
    for _ in 0..20 {
        let y = rand_bools(&mut rng, 60, 0.4);
        let s: Vec<f64> = (0..60).map(|_| rng.randint(0, 6) as f64).collect(); // many ties
        near(auroc(&y, &s).unwrap(), auroc_bruteforce(&y, &s), 1e-12);
    }
}

#[test]
fn stratified_folds_partition() {
    let mut rng = NumpyRandom::new(1);
    let y = rand_bools(&mut rng, 103, 0.3);
    let mean = y.iter().filter(|&&v| v).count() as f64 / y.len() as f64;
    let folds = stratified_folds(&y, 5, 0);
    let mut tests: Vec<usize> = folds
        .iter()
        .flat_map(|(_, te)| te.iter().copied())
        .collect();
    tests.sort_unstable();
    assert_eq!(tests, (0..103).collect::<Vec<_>>());
    for (tr, te) in &folds {
        assert!(tr.iter().all(|i| !te.contains(i)));
        assert!(te.iter().any(|&i| y[i]) && te.iter().any(|&i| !y[i]));
        let frac = te.iter().filter(|&&i| y[i]).count() as f64 / te.len() as f64;
        near(frac, mean, 0.06);
    }
    let again = stratified_folds(&y, 5, 0);
    let te_a: Vec<&Vec<usize>> = folds.iter().map(|(_, te)| te).collect();
    let te_b: Vec<&Vec<usize>> = again.iter().map(|(_, te)| te).collect();
    assert_eq!(te_a, te_b);
}

#[test]
fn logistic_recovers_direction_and_ridge_shrinks() {
    let mut rng = NumpyRandom::new(2);
    let x = mat(300, 5, rng.randn_vec(300 * 5));
    let noise = rng.randn_vec(300);
    let y: Vec<bool> = (0..300)
        .map(|i| x.at(i, 0) * 2.0 - x.at(i, 1) + noise[i] * 0.5 > 0.0)
        .collect();
    let yf = as_f64(&y);
    let (w, b) = fit_logistic(&x, &yf, 1.0, 60, 1e-9);
    assert!(w[0] > 0.5, "w0 {}", w[0]);
    assert!(w[1] < -0.2, "w1 {}", w[1]);
    let score: Vec<f64> = (0..300).map(|i| dot(x.row(i), &w) + b).collect();
    assert!(auroc(&y, &score).unwrap() > 0.95);
    let (w_big, _) = fit_logistic(&x, &yf, 1e4, 60, 1e-9);
    assert!(
        norm(&w_big) < 0.2 * norm(&w),
        "{} vs {}",
        norm(&w_big),
        norm(&w)
    );
}

#[test]
fn cv_logistic_signal_vs_noise() {
    let mut rng = NumpyRandom::new(3);
    let x = mat(200, 20, rng.randn_vec(200 * 20));
    let noise = rng.randn_vec(200);
    let y: Vec<bool> = (0..200)
        .map(|i| x.at(i, 0) + noise[i] * 0.7 > 0.0)
        .collect();
    let grid = [1.0, 100.0];
    let good = cv_logistic(&x, &y, &grid, 4, 1, 0, 3).unwrap();
    assert!(good.aurocs[0] > 0.8, "good {}", good.aurocs[0]);
    let y_noise = rand_bools(&mut rng, 200, 0.5);
    let bad = cv_logistic(&x, &y_noise, &grid, 4, 1, 0, 3).unwrap();
    assert!(bad.aurocs[0] < 0.65, "bad {}", bad.aurocs[0]);
    assert!(good.oof.iter().all(|v| v.is_finite()));
}

#[test]
fn bootstrap_and_paired_difference() {
    let mut rng = NumpyRandom::new(4);
    let y = rand_bools(&mut rng, 300, 0.4);
    let yf = as_f64(&y);
    let n1 = rng.randn_vec(300);
    let strong: Vec<f64> = (0..300).map(|i| yf[i] * 1.5 + n1[i]).collect();
    let n2 = rng.randn_vec(300);
    let weak: Vec<f64> = (0..300).map(|i| yf[i] * 0.2 + n2[i]).collect();
    let point = auroc(&y, &strong).unwrap();
    let scores = vec![("strong".to_string(), strong), ("weak".to_string(), weak)];
    let out = bootstrap_auroc(&y, &scores, 200, 1, Some("weak")).unwrap();
    let ci = out.ci("strong").unwrap();
    assert!(ci.lo < point, "lo {} point {point}", ci.lo);
    assert!(ci.hi > point, "hi {} point {point}", ci.hi);
    assert!(out.diff_of("strong").unwrap().lo > 0.0);
    assert!(out.diff_of("weak").is_none());
}

#[test]
fn calibration() {
    let mut rng = NumpyRandom::new(5);
    let p = rng.rand_vec(4000);
    let u = rng.rand_vec(4000);
    let y: Vec<bool> = (0..4000).map(|i| u[i] < p[i]).collect();
    let bins = calibration_bins(&y, &p, 10);
    assert_eq!(bins.iter().map(|b| b.n).sum::<usize>(), 4000);
    assert!(ece(&bins) < 0.05, "ece {}", ece(&bins));
    let flipped: Vec<f64> = p.iter().map(|v| 1.0 - v).collect();
    assert!(ece(&calibration_bins(&y, &flipped, 10)) > 0.2);
    near(brier(&[false, true], &[0.0, 1.0]), 0.0, 1e-7);
    near(brier(&[false, true], &[1.0, 0.0]), 1.0, 1e-7);
    let platt = cv_platt(&p, &y, 4, 1, 0);
    assert_eq!(platt.len(), 4000);
    assert!(platt.iter().all(|v| v.is_finite()));
}
