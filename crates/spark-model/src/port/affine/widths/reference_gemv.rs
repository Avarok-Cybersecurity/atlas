// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/kernels/cuda/qwen35.cu:12-23 (group_sum) and the gate-3 bound of docs/kernel-contract.md
// (`|want| * half-ulp(bf16) + gamma(depth) * magnitude`). Copyright 2026 TensorFold contributors.
// Licensed under the Apache License, Version 2.0; code written before TensorFold 0.6.0 is also
// available under the MIT License. See TENSORFOLD-ATTRIBUTION.md. Changes for Atlas: Rust; the
// float64 reference over MLX codes and the fp32 model of `affine_n_ada.cu`'s GEMV are Atlas code.
//! Responsibility: the GEMV half of the widths oracle (lane L17c): group sums, the float64 reference
//! GEMV with its per-output tolerance, and an fp32 model of the kernel's arithmetic in the unsplit and
//! split orders. Re-exported through `widths::reference`.

use super::layout;
use super::reference::{MlxWeight, Tiles, bf16_f32, code_at, lane_codes};

/// fp32 group sums of `rows` bf16 rows `[rows][k]`, entries added in order (`affine_n_sums_ada`).
pub fn group_sums(x: &[f32], rows: usize, k: usize) -> Vec<f32> {
    let kg = layout::groups(k);
    let mut out = vec![0f32; rows * kg];
    for r in 0..rows {
        for g in 0..kg {
            let mut s = 0f32;
            for v in &x[r * k + g * 64..r * k + (g + 1) * 64] {
                s += v;
            }
            out[r * kg + g] = s;
        }
    }
    out
}

/// float64 reference `y[r][c] = sum_i x[r][i] * (s * q + b)` over the MLX weight, with each output's
/// gate tolerance: `|y| * 2^-8` (the bf16 output rounding) plus `gamma(k + 2 * kg + 64) * mag` with
/// a unit roundoff of 2^-22 (twice fp32's, for the MMA's accumulation) and `mag` the sum of the terms'
/// magnitudes, `sum_g |s| * sum_i |x_i q_i| + |b| * sum_i |x_i|`. Returns `(want, tol)` row-major.
pub fn gemv_f64(w: &MlxWeight, x: &[f32], rows: usize) -> (Vec<f64>, Vec<f64>) {
    let (n, k) = (w.n, w.k);
    let (kg, rw) = (layout::groups(k), layout::mlx_row_words(k, w.bits));
    let depth = (k + 2 * kg + 64) as f64;
    let u = 2f64.powi(-22);
    let gamma = depth * u / (1.0 - depth * u);
    let (mut want, mut tol) = (vec![0f64; rows * n], vec![0f64; rows * n]);
    for c in 0..n {
        let words = &w.words[c * rw..(c + 1) * rw];
        for r in 0..rows {
            let xr = &x[r * k..(r + 1) * k];
            let (mut y, mut mag) = (0f64, 0f64);
            for g in 0..kg {
                let s = f64::from(bf16_f32(w.scales[c * kg + g]));
                let b = f64::from(bf16_f32(w.biases[c * kg + g]));
                let (mut dot, mut dmag, mut sum, mut smag) = (0f64, 0f64, 0f64, 0f64);
                for i in g * 64..(g + 1) * 64 {
                    let (xv, q) = (f64::from(xr[i]), f64::from(code_at(words, i, w.bits)));
                    dot += xv * q;
                    dmag += (xv * q).abs();
                    sum += xv;
                    smag += xv.abs();
                }
                y += s * dot + b * sum;
                mag += s.abs() * dmag + b.abs() * smag;
            }
            want[r * n + c] = y;
            tol[r * n + c] = y.abs() * 2f64.powi(-8) + gamma * mag;
        }
    }
    (want, tol)
}

/// An fp32 model of the GEMV over the packed tiles: operands read through [`lane_codes`], the group
/// dot in input order, the epilogue `acc = fma(sum, b, fma(dot, s, acc))`, K slices of `split_k`
/// summed in slice order. `split = false` follows the gemv kernel (`tot = first ? acc : tot + acc`),
/// `split = true` the split kernel (slice partials, then `p0 + p1 + ...`). Outputs before the bf16
/// rounding, `[rows][n]`.
pub fn gemv_model(t: &Tiles, x: &[f32], rows: usize, split: bool) -> Vec<f32> {
    let (n, k) = (t.n, t.k);
    let (kg, np) = (layout::groups(k), layout::npad(n));
    let sk = layout::split_k(n, k);
    let per = kg / sk;
    let xs = group_sums(x, rows, k);
    let mut out = vec![0f32; rows * n];
    for tile in 0..layout::tiles(n) {
        let codes: Vec<Vec<[u8; 64]>> = (0..kg).map(|g| lane_codes(t, tile, g)).collect();
        for cc in 0..64 {
            let col = tile * 64 + cc;
            if col >= n {
                continue;
            }
            for r in 0..rows {
                let mut parts = vec![0f32; sk];
                for (g, block) in codes.iter().enumerate() {
                    let xr = &x[r * k + g * 64..r * k + (g + 1) * 64];
                    let mut dot = 0f32;
                    for (xv, &q) in xr.iter().zip(block[cc].iter()) {
                        dot += xv * f32::from(q);
                    }
                    let s = bf16_f32(t.scales_t[g * np + col]);
                    let b = bf16_f32(t.biases_t[g * np + col]);
                    let acc = &mut parts[g / per];
                    *acc = xs[r * kg + g].mul_add(b, dot.mul_add(s, *acc));
                }
                out[r * n + col] = if split {
                    // The split kernel's last CTA: slice 0's partial, then each later one added.
                    let mut y = parts[0];
                    for p in &parts[1..] {
                        y += p;
                    }
                    y
                } else {
                    // The gemv kernel: tot = first ? acc : tot + acc at every slice end.
                    let mut y = 0f32;
                    for (i, p) in parts.iter().enumerate() {
                        y = if i == 0 { *p } else { y + p };
                    }
                    y
                };
            }
        }
    }
    out
}
