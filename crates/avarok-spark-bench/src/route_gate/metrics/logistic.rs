// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. Newton-Raphson ridge logistic regression with a backtracking line search; no upstream code.
//! Responsibility: ridge logistic regression by damped Newton, the standardizer, and `fit_predict_logistic`.

use rayon::prelude::*;

use crate::route_gate::mat::{Mat, dot, logaddexp, sigmoid};

fn loss(x: &Mat, y: &[f64], w: &[f64], b: f64, l2: f64) -> f64 {
    let mut s = 0.0;
    for (i, &yi) in y.iter().enumerate().take(x.rows) {
        let z = dot(x.row(i), w) + b;
        s += logaddexp(0.0, z) - yi * z;
    }
    s + 0.5 * l2 * dot(w, w)
}

/// Solves `a x = g` for a dense `size x size` matrix by Gaussian elimination with partial pivoting. A singular
/// system is retried with a growing diagonal shift (the Python version fell back to least squares).
fn solve(a: Vec<f64>, g: &[f64], size: usize) -> Vec<f64> {
    let mut shift = 0.0;
    loop {
        let mut m = a.clone();
        for i in 0..size {
            m[i * size + i] += shift;
        }
        let mut rhs = g.to_vec();
        if let Some(x) = eliminate(&mut m, &mut rhs, size) {
            return x;
        }
        shift = if shift == 0.0 { 1e-8 } else { shift * 100.0 };
        if shift > 1e12 {
            return vec![0.0; size];
        }
    }
}

fn eliminate(m: &mut [f64], rhs: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if m[r * n + col].abs() > m[piv * n + col].abs() {
                piv = r;
            }
        }
        if m[piv * n + col].abs() < 1e-300 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                m.swap(col * n + c, piv * n + c);
            }
            rhs.swap(col, piv);
        }
        let d = m[col * n + col];
        let (head, tail) = m.split_at_mut((col + 1) * n);
        let prow = &head[col * n..(col + 1) * n];
        let prhs = rhs[col];
        tail.par_chunks_mut(n)
            .zip(rhs[col + 1..].par_iter_mut())
            .for_each(|(row, r)| {
                let f = row[col] / d;
                if f != 0.0 {
                    for c in col..n {
                        row[c] -= f * prow[c];
                    }
                    *r -= f * prhs;
                }
            });
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for j in i + 1..n {
            s -= m[i * n + j] * x[j];
        }
        x[i] = s / m[i * n + i];
    }
    Some(x)
}

/// Minimizes `sum(log(1 + e^z) - y z) + l2/2 |w|^2` (bias unpenalized) by damped Newton. Returns `(w, b)`.
pub fn fit_logistic(x: &Mat, y: &[f64], l2: f64, max_iter: usize, tol: f64) -> (Vec<f64>, f64) {
    let (n, d) = (x.rows, x.cols);
    let mut w = vec![0.0; d];
    let prior = (y.iter().sum::<f64>() / n as f64).clamp(1e-3, 1.0 - 1e-3);
    let mut b = (prior / (1.0 - prior)).ln();
    let mut cur = loss(x, y, &w, b, l2);
    // Column-major copy so the Hessian is a set of contiguous dot products.
    let xt: Vec<Vec<f64>> = (0..d).map(|j| x.col(j)).collect();
    let size = d + 1;
    for _ in 0..max_iter {
        let p: Vec<f64> = (0..n).map(|i| sigmoid(dot(x.row(i), &w) + b)).collect();
        let s: Vec<f64> = p.iter().map(|&q| (q * (1.0 - q)).max(1e-10)).collect();
        let resid: Vec<f64> = p.iter().zip(y).map(|(a, t)| a - t).collect();
        let mut g = vec![0.0; size];
        for j in 0..d {
            g[j] = dot(&xt[j], &resid) + l2 * w[j];
        }
        g[d] = resid.iter().sum();
        let sxt: Vec<Vec<f64>> = xt
            .par_iter()
            .map(|col| col.iter().zip(&s).map(|(a, b)| a * b).collect())
            .collect();
        let rows: Vec<Vec<f64>> = (0..d)
            .into_par_iter()
            .map(|i| (i..d).map(|j| dot(&sxt[i], &xt[j])).collect())
            .collect();
        let mut h = vec![0.0; size * size];
        for i in 0..d {
            for (off, v) in rows[i].iter().enumerate() {
                let j = i + off;
                h[i * size + j] = *v;
                h[j * size + i] = *v;
            }
            h[i * size + i] += l2;
            let xs: f64 = sxt[i].iter().sum();
            h[i * size + d] = xs;
            h[d * size + i] = xs;
        }
        h[d * size + d] = s.iter().sum();
        for i in 0..size {
            h[i * size + i] += 1e-10;
        }
        let step = solve(h, &g, size);
        let (mut t, mut improved) = (1.0, false);
        let (mut w2, mut b2, mut new) = (w.clone(), b, cur);
        for _ in 0..30 {
            w2 = w.iter().zip(&step[..d]).map(|(a, st)| a - t * st).collect();
            b2 = b - t * step[d];
            new = loss(x, y, &w2, b2, l2);
            if new <= cur + 1e-12 {
                improved = true;
                break;
            }
            t *= 0.5;
        }
        if !improved {
            break;
        }
        let done = (cur - new).abs() < tol * (1.0 + cur.abs());
        w = w2;
        b = b2;
        cur = new;
        if done {
            break;
        }
    }
    (w, b)
}

/// Per-feature mean and standard deviation from the training rows; constant features get std 1 so they contribute 0.
pub struct Standardizer {
    pub mu: Vec<f64>,
    pub sd: Vec<f64>,
}

impl Standardizer {
    pub fn new(x: &Mat) -> Self {
        let mu = x.col_means();
        let n = x.rows.max(1) as f64;
        let sd = (0..x.cols)
            .map(|j| {
                let var = (0..x.rows)
                    .map(|i| (x.at(i, j) - mu[j]).powi(2))
                    .sum::<f64>()
                    / n;
                let s = var.sqrt();
                if s > 1e-12 { s } else { 1.0 }
            })
            .collect();
        Self { mu, sd }
    }

    pub fn apply(&self, x: &Mat) -> Mat {
        let data = (0..x.data.len())
            .map(|q| (x.data[q] - self.mu[q % x.cols]) / self.sd[q % x.cols])
            .collect();
        Mat {
            rows: x.rows,
            cols: x.cols,
            data,
        }
    }
}

/// Fits on the standardized training rows and returns the probabilities of the test rows.
pub fn fit_predict_logistic(xtr: &Mat, ytr: &[f64], xte: &Mat, l2: f64) -> Vec<f64> {
    let sc = Standardizer::new(xtr);
    let (w, b) = fit_logistic(&sc.apply(xtr), ytr, l2, 60, 1e-9);
    let z = sc.apply(xte);
    (0..z.rows)
        .map(|i| sigmoid(dot(z.row(i), &w) + b))
        .collect()
}
