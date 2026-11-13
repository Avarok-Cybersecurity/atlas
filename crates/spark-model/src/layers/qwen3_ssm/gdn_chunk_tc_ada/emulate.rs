// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU emulation of the two GDN prefill routes' ARITHMETIC
//! PROFILES, for calibrating the oracle criteria without a GPU:
//!   * [`tc_route`]: what `gdn_chunk_tc_ada.cu` rounds and where (FP32
//!     accumulation; K K^T, Q K^T, Q S on BF16 operands; T_inv, beta V,
//!     beta exp(g) K, S as an MM1 operand, v_new * decay and the decayed Q K^T
//!     rounded to TF32 to nearest; W, U, v_new, S_c and the output rounded to
//!     BF16; the gate cumsum in FP32 in token order);
//!   * [`scalar_route`]: the token-sequential FP32 recurrence with a BF16
//!     output, the profile of the scalar spine the TC route must agree with.
//!
//! Summation order is not emulated (the MMA tree is not specified); the
//! rounding points are what set the error class the criteria must admit.

// Index loops mirror the kernel's tile algebra one to one.
#![allow(clippy::needless_range_loop)]

use super::oracle::{GATE_FLOOR, Problem, Reference, bf16_round};

/// Round to the nearest TF32, ties away from zero (`cvt.rna.tf32.f32`).
pub(crate) fn tf32_round(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    f32::from_bits(x.to_bits().wrapping_add(0x1000) & !0x1FFF)
}

fn kh(p: &Problem, vh: usize) -> usize {
    vh / (p.nv / p.nk)
}

fn row(x: &[f32], t: usize, heads: usize, h: usize, d: usize) -> &[f32] {
    let o = (t * heads + h) * d;
    &x[o..o + d]
}

/// The tensor-core route's profile, chunk 64, last chunk partial.
pub(crate) fn tc_route(p: &Problem) -> Reference {
    const C: usize = 64;
    let (kd, vd) = (p.kd, p.vd);
    let scale = 1.0f32 / (kd as f32).sqrt();
    let mut out = vec![0f64; p.t * p.nv * vd];
    let mut state_f32: Vec<f32> = p.h0.clone();
    for vh in 0..p.nv {
        let h = kh(p, vh);
        let s = &mut state_f32[vh * kd * vd..(vh + 1) * kd * vd];
        let mut cs = 0;
        while cs < p.t {
            let ce = (p.t - cs).min(C);
            let k: Vec<&[f32]> = (0..ce).map(|i| row(&p.k, cs + i, p.nk, h, kd)).collect();
            let q: Vec<&[f32]> = (0..ce).map(|i| row(&p.q, cs + i, p.nk, h, kd)).collect();
            let v: Vec<&[f32]> = (0..ce).map(|i| row(&p.v, cs + i, p.nv, vh, vd)).collect();
            let beta: Vec<f32> = (0..ce).map(|i| p.beta[(cs + i) * p.nv + vh]).collect();
            let mut gc = vec![0f32; ce];
            let mut acc = 0f32;
            for i in 0..ce {
                acc += p.gate[(cs + i) * p.nv + vh].max(GATE_FLOOR as f32).ln();
                gc[i] = acc;
            }
            let dotf = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
            // T_inv = (I + L)^-1, L[i][l] = beta_i exp(gc_i - gc_l) <k_l, k_i>, by columns.
            let mut lm = vec![0f32; ce * ce];
            for i in 0..ce {
                for l in 0..i {
                    lm[i * ce + l] = beta[i] * (gc[i] - gc[l]).exp() * dotf(k[l], k[i]);
                }
            }
            let mut tinv = vec![0f32; ce * ce];
            for j in 0..ce {
                tinv[j * ce + j] = 1.0;
                for i in j + 1..ce {
                    let mut x = 0f32;
                    for l in j..i {
                        x -= lm[i * ce + l] * tinv[l * ce + j];
                    }
                    tinv[i * ce + j] = x;
                }
            }
            for x in tinv.iter_mut() {
                *x = tf32_round(*x);
            }
            // U = bf16(T_inv tf32(beta V)), W = bf16(T_inv tf32(beta exp(gc) K)).
            let mut u = vec![0f32; ce * vd];
            let mut w = vec![0f32; ce * kd];
            for i in 0..ce {
                for c in 0..vd {
                    let x: f32 = (0..=i)
                        .map(|l| tinv[i * ce + l] * tf32_round(beta[l] * v[l][c]))
                        .sum();
                    u[i * vd + c] = bf16_round(x);
                }
                for r in 0..kd {
                    let x: f32 = (0..=i)
                        .map(|l| tinv[i * ce + l] * tf32_round(beta[l] * gc[l].exp() * k[l][r]))
                        .sum();
                    w[i * kd + r] = bf16_round(x);
                }
            }
            // S_c as stage 3 reads it, and v_new = U - W tf32(S_c) in FP32.
            let hc: Vec<f32> = s.iter().map(|x| bf16_round(*x)).collect();
            let st: Vec<f32> = s.iter().map(|x| tf32_round(*x)).collect();
            let mut vn = vec![0f32; ce * vd];
            for i in 0..ce {
                for c in 0..vd {
                    let ws: f32 = (0..kd).map(|r| w[i * kd + r] * st[r * vd + c]).sum();
                    vn[i * vd + c] = u[i * vd + c] - ws;
                }
            }
            // Stage 3: O = scale (exp(gc_i) q_i^T bf16(S_c) + tf32(A_i) bf16(v_new)).
            for i in 0..ce {
                let a: Vec<f32> = (0..=i)
                    .map(|l| tf32_round(dotf(q[i], k[l]) * (gc[i] - gc[l]).exp()))
                    .collect();
                let eg = gc[i].exp();
                for c in 0..vd {
                    let qs: f32 = (0..kd).map(|r| q[i][r] * hc[r * vd + c]).sum();
                    let av: f32 = (0..=i).map(|l| a[l] * bf16_round(vn[l * vd + c])).sum();
                    let o = bf16_round(scale * (eg * qs + av));
                    out[((cs + i) * p.nv + vh) * vd + c] = o as f64;
                }
            }
            // Stage 2's update: S = exp(gc_last) S + K^T tf32(v_new exp(gc_last - gc)).
            let gl = gc[ce - 1];
            let gamma = gl.exp();
            for x in s.iter_mut() {
                *x *= gamma;
            }
            for i in 0..ce {
                let d = (gl - gc[i]).exp();
                for c in 0..vd {
                    let vdc = tf32_round(vn[i * vd + c] * d);
                    for r in 0..kd {
                        s[r * vd + c] += k[i][r] * vdc;
                    }
                }
            }
            cs += ce;
        }
    }
    Reference {
        out,
        state: state_f32.iter().map(|x| *x as f64).collect(),
    }
}

/// The scalar spine's profile: FP32 token-sequential, BF16 output.
pub(crate) fn scalar_route(p: &Problem) -> Reference {
    let (kd, vd) = (p.kd, p.vd);
    let inv = 1.0f32 / (kd as f32).sqrt();
    let mut out = vec![0f64; p.t * p.nv * vd];
    let mut state_f32: Vec<f32> = p.h0.clone();
    for vh in 0..p.nv {
        let h = kh(p, vh);
        let s = &mut state_f32[vh * kd * vd..(vh + 1) * kd * vd];
        for t in 0..p.t {
            let (q, k, v) = (
                row(&p.q, t, p.nk, h, kd),
                row(&p.k, t, p.nk, h, kd),
                row(&p.v, t, p.nv, vh, vd),
            );
            let (g, b) = (p.gate[t * p.nv + vh], p.beta[t * p.nv + vh]);
            for c in 0..vd {
                let hk: f32 = (0..kd).map(|r| s[r * vd + c] * k[r]).sum();
                let vn = (v[c] - g * hk) * b;
                let mut qd = 0f32;
                for r in 0..kd {
                    let x = g * s[r * vd + c] + k[r] * vn;
                    s[r * vd + c] = x;
                    qd += x * q[r];
                }
                out[(t * p.nv + vh) * vd + c] = bf16_round(qd * inv) as f64;
            }
        }
    }
    Reference {
        out,
        state: state_f32.iter().map(|x| *x as f64).collect(),
    }
}
