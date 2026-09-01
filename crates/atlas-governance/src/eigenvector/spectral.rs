// SPDX-License-Identifier: AGPL-3.0-only

//! Principal-component decomposition of intent-space.
//!
//! The ledger says what each individual change was for. Stacked together those
//! answers form a cloud of points in embedding space, and the directions that
//! cloud is stretched along are the axes the repository is actually working on.
//! This module finds them.
//!
//! # Why the Gram matrix and not the covariance matrix
//!
//! The natural formulation is the `D × D` covariance `XᵀX`, where `D` is the
//! embedding width — a few thousand. But the ledger window holds at most a
//! hundred PRs, so `n ≪ D` and the `n × n` Gram matrix `XXᵀ` has the same
//! non-zero spectrum for a ten-thousandth of the work. Its eigenvectors are
//! *per-sample loadings*, which is also the more useful form: ranking PRs along
//! an axis is exactly what makes the axis interpretable, and it needs `u`
//! directly. The feature-space direction is recovered as `Xᵀu` only where it is
//! genuinely needed — the drift angle.
//!
//! # Why determinism is a requirement and not a nicety
//!
//! The output is upserted into a Discussion comment on a schedule. If the same
//! ledger produced a different body each run, the comment would rewrite itself
//! forever and every reader would see churn that means nothing. So: a fixed
//! iteration count with no convergence-based early exit, a fixed start vector,
//! and a fixed sign convention. Eigenvectors are only defined up to sign, and
//! left to itself power iteration will happily return `-u` one day and `u` the
//! next.

use serde::{Deserialize, Serialize};

/// Power-iteration steps per component. Fixed rather than convergence-gated:
/// stopping on a tolerance makes the step count depend on the data, and two
/// runs that stop at different iterations can differ in the last decimal —
/// which is enough to rewrite the comment.
const ITERATIONS: usize = 256;

/// Below this, a component is treated as numerical noise rather than a real
/// direction. An all-identical corpus has zero variance in every direction and
/// must yield no components at all, not three arbitrary ones.
const EIGENVALUE_FLOOR: f64 = 1e-9;

/// One axis of intent-space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Component {
    /// Share of total variance this axis explains, in `[0, 1]`.
    pub explained: f64,
    /// Per-sample loading, in the caller's input order. Sign is meaningful only
    /// relative to the other samples: the two ends of this list are the two
    /// poles of the axis.
    pub loadings: Vec<f64>,
}

/// The decomposition of one ledger window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spectrum {
    pub n: usize,
    pub dim: usize,
    pub components: Vec<Component>,
    /// Variance explained by the first component — how single-minded the
    /// corpus is. `None` when there is no variance to explain.
    pub coherence: Option<f64>,
    /// Angle in degrees between the principal direction of the older half of
    /// the window and that of the newer half. `None` when either half is too
    /// small or degenerate to have a direction.
    ///
    /// 0° means the repository is pushing the same way it was; 90° means the
    /// recent work is orthogonal to what preceded it.
    pub drift_degrees: Option<f64>,
}

/// Subtract the mean sample. PCA without centring finds the direction of the
/// mean rather than the direction of the variation, which for embeddings — all
/// of which point broadly the same way — is a component that says nothing.
fn centered(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = rows.len();
    let dim = rows.first().map_or(0, Vec::len);
    let mut mean = vec![0.0; dim];
    for row in rows {
        for (m, v) in mean.iter_mut().zip(row) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    rows.iter()
        .map(|row| row.iter().zip(&mean).map(|(v, m)| v - m).collect())
        .collect()
}

fn gram(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = rows.len();
    let mut g = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let dot: f64 = rows[i].iter().zip(&rows[j]).map(|(a, b)| a * b).sum();
            g[i][j] = dot;
            g[j][i] = dot;
        }
    }
    g
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// A deterministic, data-independent start vector. All-ones is the obvious
/// choice and the wrong one: it is exactly orthogonal to the leading
/// eigenvector of some symmetric matrices, and power iteration started there
/// never rotates into it. `sin(i+1)` has no such relationship to anything.
fn start_vector(n: usize) -> Vec<f64> {
    let v: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();
    let s = norm(&v);
    v.into_iter().map(|x| x / s).collect()
}

/// Fix the sign so the same input always yields the same output: the
/// largest-magnitude coordinate is made positive. Ties broken by index, since
/// two coordinates of equal magnitude would otherwise flip on rounding.
fn canonical_sign(v: &mut [f64]) {
    let pivot = v
        .iter()
        .enumerate()
        .fold((0usize, 0.0f64), |(bi, bv), (i, &x)| {
            if x.abs() > bv + 1e-12 {
                (i, x.abs())
            } else {
                (bi, bv)
            }
        })
        .0;
    if v.get(pivot).is_some_and(|x| *x < 0.0) {
        for x in v.iter_mut() {
            *x = -*x;
        }
    }
}

/// Leading eigenpair of a symmetric matrix by power iteration.
fn leading_eigenpair(m: &[Vec<f64>]) -> Option<(f64, Vec<f64>)> {
    let n = m.len();
    if n == 0 {
        return None;
    }
    let mut v = start_vector(n);
    for _ in 0..ITERATIONS {
        let mut next = vec![0.0; n];
        for (i, row) in m.iter().enumerate() {
            next[i] = row.iter().zip(&v).map(|(a, b)| a * b).sum();
        }
        let s = norm(&next);
        if s < EIGENVALUE_FLOOR {
            return None; // the matrix annihilates our vector: no direction here
        }
        v = next.into_iter().map(|x| x / s).collect();
    }
    // Rayleigh quotient. With `v` unit-length this is vᵀMv.
    let mut mv = vec![0.0; n];
    for (i, row) in m.iter().enumerate() {
        mv[i] = row.iter().zip(&v).map(|(a, b)| a * b).sum();
    }
    let lambda: f64 = v.iter().zip(&mv).map(|(a, b)| a * b).sum();
    if lambda <= EIGENVALUE_FLOOR {
        return None;
    }
    canonical_sign(&mut v);
    Some((lambda, v))
}

/// Remove a known eigenpair so the next iteration finds the next-largest.
fn deflate(m: &mut [Vec<f64>], lambda: f64, v: &[f64]) {
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell -= lambda * v[i] * v[j];
        }
    }
}

/// Unit feature-space direction `Xᵀu` for a loading vector `u`.
fn feature_direction(rows: &[Vec<f64>], u: &[f64]) -> Option<Vec<f64>> {
    let dim = rows.first()?.len();
    let mut w = vec![0.0; dim];
    for (row, &coeff) in rows.iter().zip(u) {
        for (acc, v) in w.iter_mut().zip(row) {
            *acc += coeff * v;
        }
    }
    let s = norm(&w);
    if s < EIGENVALUE_FLOOR {
        return None;
    }
    Some(w.into_iter().map(|x| x / s).collect())
}

/// Principal feature-space direction of a set of rows, or `None` if they have
/// no variation to speak of.
fn principal_direction(rows: &[Vec<f64>]) -> Option<Vec<f64>> {
    if rows.len() < 2 {
        return None;
    }
    let c = centered(rows);
    let (_, u) = leading_eigenpair(&gram(&c))?;
    feature_direction(&c, &u)
}

/// Angle between two unit vectors, in degrees, folded into `[0°, 90°]`.
///
/// Folded because an eigenvector's sign is arbitrary: `w` and `-w` describe the
/// same axis, so an unfolded angle would report 180° for two identical
/// directions that happened to be signed differently.
fn axis_angle_degrees(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot.abs().clamp(0.0, 1.0).acos().to_degrees()
}

/// Split the window by timestamp and measure how far the principal direction
/// has turned between the two halves.
fn drift(rows: &[Vec<f64>], timestamps: &[i64]) -> Option<f64> {
    // Four is the smallest window that gives each half two points, which is the
    // minimum for a direction to exist at all.
    if rows.len() < 4 || rows.len() != timestamps.len() {
        return None;
    }
    let mut order: Vec<usize> = (0..rows.len()).collect();
    // Ties broken by index so the split is stable when several PRs share a
    // timestamp — otherwise the drift angle moves without the data moving.
    order.sort_by_key(|&i| (timestamps[i], i));
    let mid = order.len() / 2;
    let take = |idx: &[usize]| -> Vec<Vec<f64>> { idx.iter().map(|&i| rows[i].clone()).collect() };
    let older = principal_direction(&take(&order[..mid]))?;
    let newer = principal_direction(&take(&order[mid..]))?;
    Some(axis_angle_degrees(&older, &newer))
}

/// Decompose a window of embeddings into its top `k` axes.
///
/// Returns an empty component list rather than an error when the corpus has no
/// variance — a single PR, or a hundred identical ones, is a legitimate state
/// of a young ledger and must render as "no axes yet", not as a failure.
pub fn decompose(rows: &[Vec<f64>], timestamps: &[i64], k: usize) -> Spectrum {
    let n = rows.len();
    let dim = rows.first().map_or(0, Vec::len);
    let empty = Spectrum {
        n,
        dim,
        components: Vec::new(),
        coherence: None,
        drift_degrees: None,
    };
    if n < 2 || dim == 0 {
        return empty;
    }

    let c = centered(rows);
    let mut g = gram(&c);
    // Total variance, before any deflation, is the denominator every explained
    // ratio is measured against.
    let trace: f64 = (0..n).map(|i| g[i][i]).sum();
    if trace <= EIGENVALUE_FLOOR {
        return empty;
    }

    let mut components = Vec::new();
    for _ in 0..k.min(n) {
        let Some((lambda, u)) = leading_eigenpair(&g) else {
            break;
        };
        deflate(&mut g, lambda, &u);
        components.push(Component {
            explained: lambda / trace,
            loadings: u,
        });
    }

    Spectrum {
        n,
        dim,
        coherence: components.first().map(|c| c.explained),
        drift_degrees: drift(rows, timestamps),
        components,
    }
}
