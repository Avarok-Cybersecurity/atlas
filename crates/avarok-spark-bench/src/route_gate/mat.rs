// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: a minimal row-major f64 matrix, the one array type the statistics and the heads share (there is
//! no NumPy; everything else is plain slices).

use super::error::{GResult, value_err};

/// Row-major `rows x cols` matrix of f64.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    /// Wraps `data`, which must hold exactly `rows * cols` values.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> GResult<Self> {
        if data.len() != rows * cols {
            return value_err(format!(
                "matrix {rows}x{cols} needs {} values, got {}",
                rows * cols,
                data.len()
            ));
        }
        Ok(Self { rows, cols, data })
    }

    /// All zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Stacks equal-length rows (an empty list gives a 0x0 matrix).
    pub fn from_rows(rows: &[Vec<f64>]) -> GResult<Self> {
        let cols = rows.first().map_or(0, Vec::len);
        let mut data = Vec::with_capacity(rows.len() * cols);
        for r in rows {
            if r.len() != cols {
                return value_err("from_rows: rows differ in length");
            }
            data.extend_from_slice(r);
        }
        Ok(Self {
            rows: rows.len(),
            cols,
            data,
        })
    }

    /// Row `i` as a slice.
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Element `(i, j)`.
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    /// The listed rows, in order.
    pub fn select_rows(&self, idx: &[usize]) -> Mat {
        let mut data = Vec::with_capacity(idx.len() * self.cols);
        for &i in idx {
            data.extend_from_slice(self.row(i));
        }
        Mat {
            rows: idx.len(),
            cols: self.cols,
            data,
        }
    }

    /// The listed columns, in order.
    pub fn select_cols(&self, idx: &[usize]) -> Mat {
        let mut data = Vec::with_capacity(self.rows * idx.len());
        for i in 0..self.rows {
            for &j in idx {
                data.push(self.at(i, j));
            }
        }
        Mat {
            rows: self.rows,
            cols: idx.len(),
            data,
        }
    }

    /// Column `j` as a vector.
    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.at(i, j)).collect()
    }

    /// Mean of each row over a subset of rows (`[cols]`), or zeros for an empty matrix.
    pub fn col_means(&self) -> Vec<f64> {
        let mut m = vec![0.0; self.cols];
        for i in 0..self.rows {
            for (a, v) in m.iter_mut().zip(self.row(i)) {
                *a += v;
            }
        }
        if self.rows > 0 {
            for a in &mut m {
                *a /= self.rows as f64;
            }
        }
        m
    }

    /// True when every value is finite.
    pub fn all_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }
}

/// Dot product of two equal-length slices.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Logistic function, accurate in both tails; NaN stays NaN.
pub fn sigmoid(x: f64) -> f64 {
    let e = (-x.abs()).exp();
    if x >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    }
}

/// `ln(exp(a) + exp(b))` without overflow (NumPy's `logaddexp`).
pub fn logaddexp(a: f64, b: f64) -> f64 {
    if a == b {
        return a + std::f64::consts::LN_2;
    }
    let m = a.max(b);
    if m.is_infinite() {
        return m;
    }
    m + (-(a - b).abs()).exp().ln_1p()
}

/// NumPy's `maximum`: a NaN in either argument gives NaN (Rust's `f64::max` would ignore it).
pub fn np_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

/// NumPy's `minimum`: a NaN in either argument gives NaN.
pub fn np_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a <= b {
        a
    } else {
        b
    }
}

/// NumPy's `clip(x, lo, hi)`: NaN stays NaN.
pub fn np_clip(x: f64, lo: f64, hi: f64) -> f64 {
    np_min(np_max(x, lo), hi)
}
