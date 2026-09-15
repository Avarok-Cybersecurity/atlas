// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DeepSeek-V4.1 **block scaffold**: hyper-connections (`hc_mixes` with the Sinkhorn split,
//! `hc_pre`, `hc_post`) and RMSNorm, CPU reference against the golden.
//!
//! The residual stream is `hc_mult` parallel copies `[tokens, hc, dim]`. A sub-layer sits
//! between `hc_pre` (collapse the copies to one input, weighted by a `pre` mix) and `hc_post`
//! (expand its output back out, mixing the residual in through a doubly-stochastic `comb`).
//! `hc_mixes` derives all three coefficient sets from the stream itself, one projection of the
//! flattened `hc * dim` stream normalised by its RMS, then split by `hc_split_sinkhorn`:
//! `pre = sigmoid(.) + eps`, `post = 2 sigmoid(.)`, `comb` = row softmax + eps, then alternating
//! column / row normalisations (`sinkhorn_iters` in total).
//!
//! The mixes are DELAYED: a block's attention uses the `pre` its predecessor's FFN produced, and
//! its FFN uses the `pre` its own attention produced (`Block.forward`). The tests reproduce that
//! order exactly and check every intermediate the golden captured.
//!
//! Orientation of `comb`, the one place this is easy to get backwards: with `comb[j][k]`, the
//! output copy is `k` and the residual copies `j` are summed,
//! `y[k] = post[k] * x + sum_j comb[j][k] * residual[j]` (the reference's
//! `sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`).

use super::to_bf16_rne;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `ds41_ref_shims.hc_split_sinkhorn` for one token. `mixes`: `[(2 + hc) * hc]`. Returns
/// (`pre[hc]`, `post[hc]`, `comb[hc * hc]` row-major `[j][k]`), all f32.
pub fn split_sinkhorn(mixes: &[f32], scale: &[f32], base: &[f32], hc: usize, iters: usize, eps: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pre: Vec<f32> = (0..hc).map(|j| sigmoid(mixes[j] * scale[0] + base[j]) + eps).collect();
    let post: Vec<f32> = (0..hc).map(|j| 2.0 * sigmoid(mixes[hc + j] * scale[1] + base[hc + j])).collect();
    let mut comb: Vec<f32> = (0..hc * hc).map(|i| mixes[2 * hc + i] * scale[2] + base[2 * hc + i]).collect();
    // row softmax + eps
    for j in 0..hc {
        let row = &mut comb[j * hc..(j + 1) * hc];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v = *v / sum + eps;
        }
    }
    let col_norm = |c: &mut [f32]| {
        for k in 0..hc {
            let s: f32 = (0..hc).map(|j| c[j * hc + k]).sum();
            for j in 0..hc {
                c[j * hc + k] /= s + eps;
            }
        }
    };
    let row_norm = |c: &mut [f32]| {
        for j in 0..hc {
            let s: f32 = c[j * hc..(j + 1) * hc].iter().sum();
            for v in &mut c[j * hc..(j + 1) * hc] {
                *v /= s + eps;
            }
        }
    };
    col_norm(&mut comb);
    for _ in 1..iters {
        row_norm(&mut comb);
        col_norm(&mut comb);
    }
    (pre, post, comb)
}

/// `Block.hc_mixes`. `x`: `[tokens, hc, dim]` (bf16 values); `hc_fn`: `[(2+hc)*hc, hc*dim]` f32.
/// Returns (`pre[tokens, hc]`, `post[tokens, hc]`, `comb[tokens, hc, hc]`).
pub fn hc_mixes(
    x: &[f32],
    tokens: usize,
    hc: usize,
    dim: usize,
    hc_fn: &[f32],
    scale: &[f32],
    base: &[f32],
    iters: usize,
    hc_eps: f32,
    norm_eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = hc * dim;
    let mix_hc = (2 + hc) * hc;
    let (mut pre, mut post, mut comb) = (Vec::with_capacity(tokens * hc), Vec::with_capacity(tokens * hc), Vec::with_capacity(tokens * hc * hc));
    for t in 0..tokens {
        let xf = &x[t * n..(t + 1) * n];
        // one RMS over the whole flattened hc*dim stream, per token
        let rsqrt = (xf.iter().map(|v| v * v).sum::<f32>() / n as f32 + norm_eps).sqrt().recip();
        let mixes: Vec<f32> = (0..mix_hc)
            .map(|m| hc_fn[m * n..(m + 1) * n].iter().zip(xf).map(|(a, b)| a * b).sum::<f32>() * rsqrt)
            .collect();
        let (p, q, c) = split_sinkhorn(&mixes, scale, base, hc, iters, hc_eps);
        pre.extend(p);
        post.extend(q);
        comb.extend(c);
    }
    (pre, post, comb)
}

/// `Block.hc_pre`: collapse the copies, `y[d] = sum_c pre[c] * x[c][d]`, back to bf16.
pub fn hc_pre(x: &[f32], pre: &[f32], tokens: usize, hc: usize, dim: usize) -> Vec<f32> {
    let mut y = vec![0f32; tokens * dim];
    for t in 0..tokens {
        for d in 0..dim {
            let mut acc = 0f32;
            for c in 0..hc {
                acc += pre[t * hc + c] * x[(t * hc + c) * dim + d];
            }
            y[t * dim + d] = to_bf16_rne(acc);
        }
    }
    y
}

/// `Block.hc_post`: `y[k][d] = post[k] * x[d] + sum_j comb[j][k] * residual[j][d]`, to bf16.
pub fn hc_post(x: &[f32], residual: &[f32], post: &[f32], comb: &[f32], tokens: usize, hc: usize, dim: usize) -> Vec<f32> {
    let mut y = vec![0f32; tokens * hc * dim];
    for t in 0..tokens {
        for k in 0..hc {
            for d in 0..dim {
                let mut acc = post[t * hc + k] * x[t * dim + d];
                for j in 0..hc {
                    acc += comb[(t * hc + j) * hc + k] * residual[(t * hc + j) * dim + d];
                }
                y[(t * hc + k) * dim + d] = to_bf16_rne(acc);
            }
        }
    }
    y
}

/// `RMSNorm.forward`: `w * (x * rsqrt(mean(x^2) + eps))` in f32, back to bf16.
pub fn rms_norm(x: &[f32], w: &[f32], tokens: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; tokens * dim];
    for t in 0..tokens {
        let row = &x[t * dim..(t + 1) * dim];
        let rsqrt = (row.iter().map(|v| v * v).sum::<f32>() / dim as f32 + eps).sqrt().recip();
        for d in 0..dim {
            y[t * dim + d] = to_bf16_rne(w[d] * (row[d] * rsqrt));
        }
    }
    y
}

#[cfg(test)]
#[path = "hc_tests.rs"]
mod tests;
