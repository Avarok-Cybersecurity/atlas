// SPDX-License-Identifier: AGPL-3.0-only
// Portions follow the ARC-1.0 release documentation and rayline-ai/vllm-arc (examples/readouts.py), Copyright 2026
// Atlas Futures, Inc., licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). The
// operators are the standard Qwen3.5 text-stack operators written as plain CPU f32 loops. By Atlas contributors.
//! Responsibility: the CPU operators of the encoder: BF16-weight matmul (weights stay as `u16` bit patterns and are
//! widened one row at a time), RMSNorm in both of its Qwen3.5 forms, the gated RMSNorm, SiLU, softplus, L2 norm and
//! the BF16 rounding used by the `bf16` emulation mode.

use rayon::prelude::*;

/// Rounds each value to the nearest BF16 (ties to even) and back.
pub fn round_bf16(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = half::bf16::from_f32(*v).to_f32();
    }
}

/// `round_bf16` of a copy, when `on`.
pub fn maybe_round(mut x: Vec<f32>, on: bool) -> Vec<f32> {
    if on {
        round_bf16(&mut x);
    }
    x
}

/// BF16 bit pattern of an f32 (round to nearest even).
pub fn to_bf16_bits(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}

/// `x . y` with eight partial sums so the loop vectorizes.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// A linear layer `y = x W^T` whose `[out, inp]` weights are kept as BF16 bit patterns.
pub struct Lin {
    pub out: usize,
    pub inp: usize,
    pub w: Vec<u16>,
}

impl Lin {
    /// Wraps BF16 bits, checking the size.
    pub fn from_bits(out: usize, inp: usize, w: Vec<u16>) -> Self {
        assert_eq!(w.len(), out * inp, "weight size");
        Self { out, inp, w }
    }

    /// Rounds f32 weights to BF16 (what the checkpoint stores).
    pub fn from_f32(out: usize, inp: usize, w: &[f32]) -> Self {
        Self::from_bits(out, inp, w.iter().map(|&v| to_bf16_bits(v)).collect())
    }

    /// `x` is `[t, inp]`; the result is `[t, out]`, rounded to BF16 when `bf16` (a bf16 matmul output).
    pub fn forward(&self, x: &[f32], t: usize, bf16: bool) -> Vec<f32> {
        let (o, i) = (self.out, self.inp);
        assert_eq!(x.len(), t * i, "matmul input size");
        let mut cols = vec![0f32; o * t];
        cols.par_chunks_mut(t.max(1))
            .zip(self.w.par_chunks(i))
            .for_each_init(
                || vec![0f32; i],
                |wrow, (col, bits)| {
                    for (d, &b) in wrow.iter_mut().zip(bits) {
                        *d = f32::from_bits(u32::from(b) << 16);
                    }
                    for (tt, c) in col.iter_mut().enumerate() {
                        *c = dot_f32(wrow, &x[tt * i..(tt + 1) * i]);
                    }
                },
            );
        let mut out = vec![0f32; t * o];
        for (row, col) in cols.chunks(t.max(1)).enumerate() {
            for (tt, v) in col.iter().enumerate() {
                out[tt * o + row] = *v;
            }
        }
        maybe_round(out, bf16)
    }
}

/// Qwen3.5 RMSNorm over rows of width `w.len()`: `x * rsqrt(mean(x^2) + eps) * (1 + w)` in f32, rounded once at the end
/// when `bf16`.
pub fn rms_norm_1p(x: &[f32], w: &[f32], eps: f32, bf16: bool) -> Vec<f32> {
    let h = w.len();
    let mut out = vec![0f32; x.len()];
    out.par_chunks_mut(h)
        .zip(x.par_chunks(h))
        .for_each(|(o, r)| {
            let var = r.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / h as f64;
            let inv = (1.0 / (var + f64::from(eps)).sqrt()) as f32;
            for k in 0..h {
                o[k] = r[k] * inv * (1.0 + w[k]);
            }
        });
    maybe_round(out, bf16)
}

/// `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `F.softplus` with torch's threshold 20.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// `x * rsqrt(sum(x^2) + eps)` over one vector, in place.
pub fn l2_normalize(x: &mut [f32], eps: f32) {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Gated RMSNorm of the linear-attention output: per row of width `w.len()`,
/// `(w * round(x * rsqrt(mean(x^2) + eps))) * silu(z)`; `z` has the same shape as `x`. Plain `w` (no `1 +`).
pub fn rms_norm_gated(x: &[f32], z: &[f32], w: &[f32], eps: f32, bf16: bool) -> Vec<f32> {
    let h = w.len();
    let mut out = vec![0f32; x.len()];
    out.par_chunks_mut(h)
        .zip(x.par_chunks(h).zip(z.par_chunks(h)))
        .for_each(|(o, (r, g))| {
            let var = r.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / h as f64;
            let inv = (1.0 / (var + f64::from(eps)).sqrt()) as f32;
            for k in 0..h {
                let mut y = r[k] * inv;
                if bf16 {
                    y = half::bf16::from_f32(y).to_f32();
                    y = half::bf16::from_f32(w[k] * y).to_f32();
                } else {
                    y *= w[k];
                }
                o[k] = y * silu(g[k]);
            }
        });
    maybe_round(out, bf16)
}

/// Elementwise `a + b`, rounded to BF16 when `bf16` (a residual add in a bf16 model).
pub fn add(a: &[f32], b: &[f32], bf16: bool) -> Vec<f32> {
    maybe_round(a.iter().zip(b).map(|(x, y)| x + y).collect(), bf16)
}
