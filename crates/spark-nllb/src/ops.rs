// SPDX-License-Identifier: AGPL-3.0-only

//! Minimal CPU tensor ops for the NLLB reference forward pass.
//!
//! All tensors are row-major `f32` slices. Row-parallelised with rayon; the
//! matmuls dominate runtime so per-row parallelism is enough.

use rayon::prelude::*;

/// LayerNorm eps used by `torch.nn.LayerNorm` (HF default).
pub const LN_EPS: f32 = 1e-5;

/// `y[rows, out] = x[rows, in] @ w[out, in]^T + bias`.
///
/// `w` is in PyTorch `nn.Linear` layout `[out_features, in_features]`.
pub fn linear(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    w: &[f32],
    out_dim: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * in_dim);
    debug_assert_eq!(w.len(), out_dim * in_dim);
    let mut y = vec![0f32; rows * out_dim];
    y.par_chunks_mut(out_dim)
        .zip(x.par_chunks(in_dim))
        .for_each(|(yr, xr)| {
            for o in 0..out_dim {
                let wr = &w[o * in_dim..o * in_dim + in_dim];
                let mut acc = 0f32;
                for i in 0..in_dim {
                    acc += xr[i] * wr[i];
                }
                yr[o] = acc + bias.map_or(0.0, |b| b[o]);
            }
        });
    y
}

/// In-place LayerNorm over the last dim with affine `weight`/`bias`.
pub fn layer_norm_inplace(x: &mut [f32], _rows: usize, dim: usize, weight: &[f32], bias: &[f32]) {
    x.par_chunks_mut(dim).for_each(|row| {
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / dim as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for j in 0..dim {
            row[j] = (row[j] - mean) * inv * weight[j] + bias[j];
        }
    });
}

/// In-place ReLU.
pub fn relu_inplace(x: &mut [f32]) {
    x.par_iter_mut().for_each(|v| {
        if *v < 0.0 {
            *v = 0.0;
        }
    });
}

/// In-place elementwise add: `a += b`.
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    a.par_iter_mut()
        .zip(b.par_iter())
        .for_each(|(x, y)| *x += *y);
}

/// Row-wise numerically-stable softmax over the last dim, in place.
pub fn softmax_rows_inplace(x: &mut [f32], rows: usize, cols: usize) {
    let _ = rows;
    x.par_chunks_mut(cols).for_each(|row| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    });
}

/// Argmax over a slice.
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// In-place log-softmax over a single row: `x[i] -= logsumexp(x)`.
pub fn log_softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for &v in x.iter() {
        sum += (v - max).exp();
    }
    let lse = max + sum.ln();
    for v in x.iter_mut() {
        *v -= lse;
    }
}

/// Indices of the `k` largest values, in descending order of value.
///
/// Ties break toward the lower index (matches PyTorch `topk` on CPU, which
/// keeps the earlier index for equal values). `k` is clamped to `x.len()`.
pub fn top_k_indices(x: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(x.len());
    let mut idx: Vec<usize> = (0..x.len()).collect();
    // Partial sort: select the top-k by value desc, index asc on ties.
    idx.select_nth_unstable_by(k.saturating_sub(1).min(x.len() - 1), |&a, &b| {
        x[b].partial_cmp(&x[a]).unwrap().then(a.cmp(&b))
    });
    idx.truncate(k);
    idx.sort_by(|&a, &b| x[b].partial_cmp(&x[a]).unwrap().then(a.cmp(&b)));
    idx
}
