// SPDX-License-Identifier: AGPL-3.0-only
// Gate design after TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// docs/kernel-contract.md (gate 3: a result within |want| * half-ulp(bf16) + gamma(depth) * magnitude of a
// float64 reference). Copyright 2026 TensorFold contributors. Licensed under the Apache License,
// Version 2.0; code written before TensorFold 0.6.0 is also available under the MIT License. See
// TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: no code copied; the reference and the depth of the lane GEMV's fp32 chain are
// derived here for the affine kernels.

//! Responsibility: the float64 reference of the affine lane GEMV and the error bound a GPU result must
//! meet against it.
//!
//! `want[r][c] = sum_g ( s_g * sum_i x_i q_i + b_g * sum_i x_i )` in f64 with the stored bf16 scale and
//! bias, which is `sum_i x_i * (s_g q_i + b_g)` exactly. `mag` is the same sum of absolute terms.
//!
//! The GPU computes, per group, an fp32 MMA over the group's products (exact: a bf16 input times a code
//! below 16 fits fp32), then two fp32 FMAs (`fma(xs, b, fma(dot, s, acc))`, where `xs` is an fp32 sum of
//! the group's inputs), then adds the K slices in fp32 and rounds once to bf16. The chain's depth is at
//! most `group` (the MMA's and the sums' additions) + `2 * kg` (the epilogue FMAs) + `sk` (the slice
//! sum) + 1; the bound doubles it because the tensor core's internal accumulation is not specified to
//! round like a sequence of IEEE additions. With `u = 2^-24` and `gamma(d) = d u / (1 - d u)`:
//!
//! `|got - want| <= |want| * 2^-8 + gamma(2 * depth) * mag * (1 + 2^-8)`.

use super::reference::{Mlx, f32_of};

/// The float64 reference of `rows` input rows against one weight: `want` and `mag`, `[rows, n]`.
#[derive(Clone, Debug)]
pub struct GemvRef {
    pub rows: usize,
    pub n: usize,
    pub want: Vec<f64>,
    pub mag: Vec<f64>,
}

/// `x` is `[rows, k]` bf16 bits. Columns are split over `threads` scoped threads (0 or 1: inline).
pub fn gemv_f64(x: &[u16], rows: usize, w: &Mlx, threads: usize) -> GemvRef {
    let (n, k, group) = (w.n, w.k, w.group);
    let kg = k / group;
    let xf: Vec<f64> = x[..rows * k]
        .iter()
        .map(|&b| f64::from(f32_of(b)))
        .collect();
    let mut want = vec![0f64; rows * n];
    let mut mag = vec![0f64; rows * n];
    let column = |c: usize, want: &mut [f64], mag: &mut [f64]| {
        for r in 0..rows {
            let xr = &xf[r * k..(r + 1) * k];
            let (mut acc, mut abs) = (0f64, 0f64);
            for g in 0..kg {
                let s = f64::from(f32_of(w.scales[c * kg + g]));
                let b = f64::from(f32_of(w.biases[c * kg + g]));
                let (mut dot, mut adot, mut sum, mut asum) = (0f64, 0f64, 0f64, 0f64);
                for i in g * group..(g + 1) * group {
                    let q = f64::from(w.code(c, i));
                    dot += xr[i] * q;
                    adot += xr[i].abs() * q;
                    sum += xr[i];
                    asum += xr[i].abs();
                }
                acc += s * dot + b * sum;
                abs += s.abs() * adot + b.abs() * asum;
            }
            want[r] = acc;
            mag[r] = abs;
        }
    };
    // Column-major scratch per column, then scattered: each thread owns a disjoint column range.
    let mut cols_want = vec![0f64; n * rows];
    let mut cols_mag = vec![0f64; n * rows];
    let threads = threads.max(1).min(n.max(1));
    let per = n.div_ceil(threads);
    std::thread::scope(|scope| {
        for (t, (wch, mch)) in cols_want
            .chunks_mut(per * rows)
            .zip(cols_mag.chunks_mut(per * rows))
            .enumerate()
        {
            let column = &column;
            scope.spawn(move || {
                for (i, (wc, mc)) in wch.chunks_mut(rows).zip(mch.chunks_mut(rows)).enumerate() {
                    column(t * per + i, wc, mc);
                }
            });
        }
    });
    for c in 0..n {
        for r in 0..rows {
            want[r * n + c] = cols_want[c * rows + r];
            mag[r * n + c] = cols_mag[c * rows + r];
        }
    }
    GemvRef { rows, n, want, mag }
}

/// The largest `|got - want|` the lane GEMV may produce for an output with reference `want` and term
/// magnitude `mag` (module docs).
pub fn gemv_bound(want: f64, mag: f64, k: usize, group: usize, sk: usize) -> f64 {
    let depth = 2.0 * (group + 2 * (k / group) + sk + 1) as f64;
    let u = (2f64).powi(-24);
    let gamma = depth * u / (1.0 - depth * u);
    let half_ulp = (2f64).powi(-8);
    want.abs() * half_ulp + gamma * mag * (1.0 + half_ulp)
}

/// Outputs of `got` (`[rows, n]` bf16 bits) outside the bound against `r`, and the worst ratio of error
/// to bound (above 1 means outside).
pub fn count_outside(got: &[u16], r: &GemvRef, k: usize, group: usize, sk: usize) -> (usize, f64) {
    let mut bad = 0;
    let mut worst = 0f64;
    for (i, &g) in got.iter().enumerate().take(r.rows * r.n) {
        let gv = f64::from(f32_of(g));
        let bound = gemv_bound(r.want[i], r.mag[i], k, group, sk);
        let err = (gv - r.want[i]).abs();
        let ratio = if bound > 0.0 {
            err / bound
        } else if err == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        if ratio.is_nan() || ratio > 1.0 {
            bad += 1;
        }
        worst = worst.max(ratio);
    }
    (bad, worst)
}
